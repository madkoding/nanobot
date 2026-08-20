"""Turn state-machine mixin for AgentLoop (extracted from agent/loop.py)."""

from __future__ import annotations

import dataclasses
import time
from functools import partial
from typing import Any

from loguru import logger

from nanobot.agent.loop_types import TurnContext, TurnKind, TurnState
from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.outbound_events import StreamedResponseEvent
from nanobot.command import CommandContext
from nanobot.runtime_context import RUNTIME_CONTEXT_HISTORY_META, RUNTIME_CONTEXT_MESSAGE_META
from nanobot.session import turn_continuation
from nanobot.session.automation_turns import automation_history_overrides
from nanobot.session.manager import replay_max_messages_for_context
from nanobot.utils.document import extract_documents, reference_non_image_attachments
from nanobot.utils.helpers import image_placeholder_text
from nanobot.utils.helpers import truncate_text as truncate_text_fn
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE


class TurnStateMixin:
    """Turn FSM handlers (RESTORE/COMPACT/COMMAND/BUILD/RUN/SAVE/RESPOND).

    Mixin: ``AgentLoop`` inherits this so ``self`` resolves against the full
    loop instance (``self.sessions``, ``self.auto_compact``, ``self.tools``,
    ``self._run_agent_loop``, etc.).
    """

    def _assemble_outbound(
        self,
        msg: InboundMessage,
        final_content: str,
        all_msgs: list[dict[str, Any]],
        stop_reason: str,
        had_injections: bool,
        streamed_content: bool,
        *,
        turn_latency_ms: int | None = None,
    ) -> OutboundMessage | None:
        """Assemble the final outbound message from turn results."""
        # MessageTool suppression
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        event = None
        meta = dict(msg.metadata or {})
        if streamed_content and stop_reason not in {"error", "tool_error"}:
            event = StreamedResponseEvent()
        if turn_latency_ms is not None:
            meta["latency_ms"] = int(turn_latency_ms)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            event=event,
            metadata=meta,
        )

    async def _state_restore(self, ctx: TurnContext) -> TurnState:
        """Restore checkpoint / pending user turn; extract documents."""
        msg = ctx.msg

        if ctx.kind is TurnKind.USER and msg.media:
            new_content, image_only = self._prepare_message_media(msg.content, msg.media)
            ctx.msg = dataclasses.replace(msg, content=new_content, media=image_only)
            msg = ctx.msg

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        if ctx.kind is TurnKind.SYSTEM:
            logger.info("Processing system message from {}", msg.sender_id)
        else:
            logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # Session is already fetched by the caller (_process_message) but
        # ensure it exists in case this handler is invoked independently.
        if ctx.session is None:
            ctx.session = self.sessions.get_or_create(ctx.session_key)
        await ctx.delivery.started()
        if ctx.kind is TurnKind.USER:
            self.workspace_scopes.persist_message_scope(ctx.session, msg)

        if self._restore_runtime_checkpoint(ctx.session):
            self.sessions.save(ctx.session)
        if self._restore_pending_user_turn(ctx.session):
            self.sessions.save(ctx.session)

        return "ok"

    def _prepare_message_media(self, content: str, media: list[str]) -> tuple[str, list[str]]:
        if self._should_extract_document_text():
            return extract_documents(content, media)
        return reference_non_image_attachments(content, media)

    def _should_extract_document_text(self) -> bool:
        if self.channels_config is None:
            return True
        return self.channels_config.extract_document_text

    async def _state_compact(self, ctx: TurnContext) -> str:
        ctx.session, pending = self.auto_compact.prepare_session(ctx.session, ctx.session_key)
        ctx.pending_summary = pending
        if pending:
            # Show a one-shot archived-context notice in the WebUI thread so the
            # user sees where the older conversation was summarized. Guarded by a
            # per-summary marker (metadata returns the summary every turn after a
            # restart, so we must not re-emit the notice on every message).
            summary_meta = ctx.session.metadata.get("_last_summary")
            summary_id = (
                summary_meta.get("last_active")
                if isinstance(summary_meta, dict) and summary_meta.get("last_active")
                else pending
            )
            if ctx.session.metadata.get("_summary_notice_active") != summary_id:
                self._write_transcript_archived_notice(ctx, pending)
                ctx.session.metadata["_summary_notice_active"] = summary_id
                self.sessions.save(ctx.session)
        return "ok"

    def _write_transcript_archived_notice(self, ctx, summary: str) -> None:
        """Emit a visible ``system`` notice in the WebUI transcript thread."""
        try:
            from nanobot.webui.transcript import append_transcript_object
        except Exception:
            return
        try:
            webui_key = self._webui_session_key(ctx.session_key)
            _, chat_id = self._channel_chat_id_from_session_key(ctx.session_key)
        except Exception:
            return
        append_transcript_object(
            webui_key,
            {
                "event": "message",
                "chat_id": chat_id,
                "role": "system",
                "kind": "notice",
                "text": f"[Contexto archivado]\n{summary}",
                "created_at_ms": int(time.time() * 1000),
            },
        )

    async def _state_command(self, ctx: TurnContext) -> str:
        if ctx.kind is TurnKind.SYSTEM:
            return "dispatch"
        raw = ctx.msg.content.strip()
        _, automation_metadata = automation_history_overrides(ctx.msg.metadata)
        is_user_turn = (
            ctx.original_user_text is not None
            and not automation_metadata
            and ctx.msg.channel != "system"
            and ctx.msg.sender_id != "subagent"
        )
        cmd_ctx = CommandContext(
            msg=ctx.msg,
            session=ctx.session,
            key=ctx.session_key,
            raw=raw,
            loop=self,
            runtime=ctx.runtime,
            is_user_turn=is_user_turn,
            turn_scopes=ctx.turn_scopes,
        )
        result = await self.commands.dispatch(cmd_ctx)
        if result is not None:
            ctx.outbound = result
            # Shortcut commands skip BUILD and SAVE, so we must persist the
            # turn here so WebUI history hydration after _turn_end sees the
            # message.  Mark messages with _command so get_history can filter
            # them out of LLM context.  /new is excluded because it
            # intentionally clears the session.
            if cmd_ctx.raw.lower() != "/new":
                ctx.input_persisted_early = self._persist_user_message_early(
                    ctx.msg, ctx.session, _command=True
                )
                ctx.session.add_message(
                    "assistant", result.content, _command=True
                )
                self.sessions.save(ctx.session)
                self._clear_pending_user_turn(ctx.session)
            return "shortcut"
        return "dispatch"

    async def _state_build(self, ctx: TurnContext) -> str:
        runtime = ctx.runtime
        if runtime is None:
            runtime = self.runtime_for_session(ctx.session)
            ctx.runtime = runtime
        if ctx.on_runtime_admitted is not None:
            await ctx.on_runtime_admitted(runtime)
        replay_max_messages = replay_max_messages_for_context(
            runtime.context_window_tokens
        )
        if not ctx.ephemeral:
            consolidation_workspace = self._turn_workspace(ctx)
            store = self.context.memory_store_for(consolidation_workspace)
            await self.consolidator.maybe_consolidate_by_tokens(
                ctx.session,
                runtime=runtime,
                replay_max_messages=replay_max_messages,
                store=store,
                sender_id=ctx.msg.sender_id,
            )
        is_subagent = ctx.kind is TurnKind.SYSTEM and ctx.msg.sender_id == "subagent"

        if ctx.kind is TurnKind.USER and (message_tool := self.tools.get("message")):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        _hist_kwargs: dict[str, Any] = {
            "max_messages": replay_max_messages,
            "max_tokens": self._replay_token_budget(runtime),
            "extend_to_user": is_subagent,
        }
        ctx.history = ctx.session.get_history(**_hist_kwargs)
        if ctx.session.metadata.pop("_skip_recent_history_once", None):
            self.sessions.save(ctx.session)
        if is_subagent:
            # Keep the durable internal delivery as an assistant record, but
            # present this completion to the model as fresh follow-up input.
            # Providers without assistant-prefill support drop trailing
            # assistant messages, so using the persisted record as the current
            # prompt would hide an independently dispatched subagent result.
            if self._persist_subagent_followup(ctx.session, ctx.msg):
                logger.debug("Subagent result persisted for session {}", ctx.session_key)
                self.sessions.save(ctx.session)
            ctx.input_persisted_early = True
        ctx.delivery.record_runtime(ctx.runtime)

        ctx.request_context = self._request_context_for_turn(ctx)
        if ctx.kind is TurnKind.USER:
            ctx.runtime_context_blocks = await self._resolve_runtime_context_for_turn(ctx)
        ctx.initial_messages = self._build_initial_messages(ctx)
        if ctx.kind is TurnKind.USER:
            ctx.input_persisted_early = self._persist_user_message_early(
                ctx.msg,
                ctx.session,
                runtime_context_blocks=ctx.runtime_context_blocks,
            )

        if ctx.on_progress is None:
            ctx.on_progress = ctx.delivery.progress_callback()
        if ctx.on_retry_wait is None:
            ctx.on_retry_wait = ctx.delivery.retry_wait_callback()

        return "ok"

    async def _state_run(self, ctx: TurnContext) -> str:
        if ctx.visible_run_started_at is None:
            ctx.visible_run_started_at = time.time()
        await ctx.delivery.running(started_at=ctx.visible_run_started_at)
        result = await self._run_agent_loop(
            ctx.initial_messages,
            runtime=ctx.runtime,
            on_progress=ctx.on_progress,
            on_stream=ctx.on_stream,
            on_stream_end=ctx.on_stream_end,
            on_retry_wait=ctx.on_retry_wait,
            session=ctx.session,
            channel=ctx.delivery.route.channel,
            chat_id=ctx.delivery.route.chat_id,
            message_id=ctx.msg.metadata.get("message_id"),
            metadata=ctx.msg.metadata,
            session_key=ctx.session_key,
            original_user_text=ctx.original_user_text,
            pending_queue=ctx.pending_queue,
            ephemeral=ctx.ephemeral,
            run_extra_hooks_for_ephemeral=ctx.run_extra_hooks_for_ephemeral,
            hooks=ctx.hooks,
            hook_factories=ctx.hook_factories,
            turn_scopes=ctx.turn_scopes,
            tools=ctx.tools,
            request_context=ctx.request_context,
            sender_id=ctx.msg.sender_id,
        )
        final_content, tools_used, all_msgs, stop_reason, had_injections = result
        ctx.final_content = final_content
        ctx.tools_used = tools_used
        ctx.all_messages = all_msgs
        ctx.stop_reason = stop_reason
        ctx.had_injections = had_injections
        if ctx.kind is TurnKind.USER:
            await turn_continuation.maybe_continue_turn(ctx)
        return "ok"

    async def _state_save(self, ctx: TurnContext) -> str:
        turn_continuation.prepare_save_boundary(ctx)

        if (
            ctx.kind is TurnKind.USER
            and (ctx.final_content is None or not ctx.final_content.strip())
            and not ctx.suppress_response
        ):
            ctx.final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        latency_started_at = (
            ctx.visible_run_started_at
            if (
                ctx.kind is TurnKind.SYSTEM
                or turn_continuation.internal_continuation_inbound(ctx.msg.metadata)
            )
            and ctx.visible_run_started_at is not None
            else ctx.turn_wall_started_at
        )
        ctx.turn_latency_ms = max(0, int((time.time() - latency_started_at) * 1000))
        self._save_turn(
            ctx.session, ctx.all_messages, ctx.save_skip,
            turn_latency_ms=ctx.turn_latency_ms,
        )
        ctx.delivery.record_latency(ctx.turn_latency_ms)
        if not ctx.ephemeral:
            ctx.session.enforce_file_cap(
                on_archive=partial(
                    self._archive_file_cap,
                    session_key=ctx.session_key,
                    sender_id=ctx.msg.sender_id,
                )
            )
        self._clear_pending_user_turn(ctx.session)
        self._clear_runtime_checkpoint(ctx.session)
        self.sessions.save(ctx.session)
        return "ok"

    async def _state_respond(self, ctx: TurnContext) -> str:
        if ctx.suppress_response:
            ctx.outbound = None
            return "ok"
        if ctx.kind is TurnKind.SYSTEM:
            ctx.outbound = ctx.delivery.background_response(
                ctx.final_content,
                stop_reason=ctx.stop_reason,
                streamed=ctx.streamed_content,
                latency_ms=ctx.turn_latency_ms,
            )
            return "ok"
        ctx.outbound = self._assemble_outbound(
            ctx.msg,
            ctx.final_content,
            ctx.all_messages,
            ctx.stop_reason,
            ctx.had_injections,
            ctx.streamed_content,
            turn_latency_ms=ctx.turn_latency_ms,
        )
        if ctx.ephemeral and ctx.outbound is not None:
            ctx.outbound.metadata["_stop_reason"] = ctx.stop_reason
        return "ok"

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(
        self,
        session,
        messages: list[dict],
        skip: int,
        *,
        turn_latency_ms: int | None = None,
    ) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        declared_tool_call_ids = {
            str(tc["id"])
            for m in session.messages
            if m.get("role") == "assistant"
            for tc in m.get("tool_calls") or []
            if isinstance(tc, dict) and tc.get("id")
        }
        fulfilled_tool_call_ids = {
            str(m["tool_call_id"])
            for m in session.messages
            if m.get("role") == "tool" and m.get("tool_call_id")
        }
        last_assistant_idx: int | None = None
        for m in messages[skip:]:
            entry = dict(m)
            internal_meta = entry.pop("_meta", None)
            runtime_context_meta = (
                internal_meta.get(RUNTIME_CONTEXT_MESSAGE_META)
                if isinstance(internal_meta, dict)
                else None
            )
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                tool_call_id = entry.get("tool_call_id")
                tool_call_id_str = str(tool_call_id) if tool_call_id else ""
                if (
                    not tool_call_id_str
                    or tool_call_id_str not in declared_tool_call_ids
                    or tool_call_id_str in fulfilled_tool_call_ids
                ):
                    # Undeclared tool results corrupt future provider requests.
                    logger.warning(
                        "Dropping invalid tool result {} from session {} during persistence",
                        tool_call_id_str or "(missing id)",
                        session.key,
                    )
                    continue
                fulfilled_tool_call_ids.add(tool_call_id_str)
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        # Preserve the tool_call/result pair after block filtering.
                        filtered = [
                            {"type": "text", "text": "[tool result omitted during persistence]"}
                        ]
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content)
                    if not filtered:
                        continue
                    entry["content"] = filtered
                if isinstance(runtime_context_meta, dict):
                    entry[RUNTIME_CONTEXT_HISTORY_META] = runtime_context_meta
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
            if role == "assistant":
                last_assistant_idx = len(session.messages) - 1
                declared_tool_call_ids.update(
                    str(tc["id"])
                    for tc in entry.get("tool_calls") or []
                    if isinstance(tc, dict) and tc.get("id")
                )
        if turn_latency_ms is not None and last_assistant_idx is not None:
            session.messages[last_assistant_idx]["latency_ms"] = int(turn_latency_ms)
        session.updated_at = datetime.now()

    def _persist_subagent_followup(self, session, msg: InboundMessage) -> bool:
        """Persist subagent follow-ups before prompt assembly so history stays durable.

        Returns True if a new entry was appended; False if the follow-up was
        deduped (same ``subagent_task_id`` already in session) or carries no
        content worth persisting.
        """
        if not msg.content:
            return False
        task_id = msg.metadata.get("subagent_task_id") if isinstance(msg.metadata, dict) else None
        if task_id and any(
            m.get("injected_event") == "subagent_result" and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event="subagent_result",
            subagent_task_id=task_id,
        )
        return True
