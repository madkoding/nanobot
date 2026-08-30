"""Agent run-loop mixin for AgentLoop (extracted from agent/loop.py)."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentTurnHookFactory
from nanobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunSpec
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.file_state import bind_file_states, reset_file_states
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
from nanobot.bus.events import InboundMessage
from nanobot.runtime_context import RUNTIME_CONTEXT_MESSAGE_META, append_runtime_context
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)
from nanobot.session import turn_continuation
from nanobot.session.goal_state import (
    goal_state_runtime_lines,
    runner_wall_llm_timeout_s,
    sustained_goal_active,
)
from nanobot.session.history_visibility import HIDDEN_HISTORY_META
from nanobot.session.manager import Session
from nanobot.utils.helpers import is_owner_match
from nanobot.utils.llm_runtime import LLMRuntime


class RunLoopMixin:
    """Single-turn agent run loop (builds context, runs the runner).

    Mixin: ``AgentLoop`` inherits this so ``self`` resolves against the full
    loop instance (``self.runner``, ``self.subagents``, ``self.workspace_scopes``,
    ``self.context``, etc.).
    """

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
        *,
        runtime: LLMRuntime,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        original_user_text: str | None = None,
        pending_queue: asyncio.Queue | None = None,
        ephemeral: bool = False,
        run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        hook_factories: list[AgentTurnHookFactory] | None = None,
        turn_scopes: list[Any] | None = None,
        tools: ToolRegistry | None = None,
        request_context: RequestContext | None = None,
        sender_id: str | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.

        Returns (final_content, tools_used, messages, stop_reason, had_injections).
        """
        self._sync_subagent_runtime_limits()

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """Drain follow-up messages from the pending queue.

            When no messages are immediately available but sub-agents
            spawned in this dispatch are still running, blocks until at
            least one result arrives (or timeout).  This keeps the runner
            loop alive so subsequent sub-agent completions are consumed
            in-order rather than dispatched separately.
            """
            if pending_queue is None:
                return []

            async def _to_user_message(pending_msg: InboundMessage) -> dict[str, Any]:
                content = pending_msg.content
                media = pending_msg.media if pending_msg.media else None
                if media:
                    content, media = self._prepare_message_media(content, media)
                    media = media or None
                user_content = self.context._build_user_content(content, media)
                row: dict[str, Any] = {"role": "user", "content": user_content}
                metadata = pending_msg.metadata if isinstance(pending_msg.metadata, dict) else {}
                if pending_msg.channel != "system":
                    scope = self.workspace_scopes.for_turn(
                        channel=pending_msg.channel,
                        message_metadata=metadata,
                        session_metadata=session.metadata if session is not None else None,
                    )
                    pending_request = RequestContext(
                        channel=pending_msg.channel,
                        chat_id=pending_msg.chat_id,
                        message_id=metadata.get("message_id"),
                        session_key=active_session_key,
                        original_user_text=pending_msg.content,
                        runtime=runtime,
                        metadata=dict(metadata),
                        sender_id=pending_msg.sender_id,
                        turn_id=request_ctx.turn_id,
                        workspace=scope.project_path,
                    )
                    blocks = await self._resolve_runtime_context_for_request(
                        pending_request,
                        effective_tools,
                    )
                    row["content"], marker = append_runtime_context(user_content, blocks)
                    if marker is not None:
                        row["_meta"] = {RUNTIME_CONTEXT_MESSAGE_META: marker}
                if (
                    pending_msg.sender_id == "subagent"
                    and metadata.get("injected_event") == "subagent_result"
                ):
                    marker: dict[str, Any] = {"kind": "subagent_result"}
                    task_id = metadata.get("subagent_task_id")
                    if isinstance(task_id, str) and task_id:
                        marker["subagent_task_id"] = task_id
                        row["subagent_task_id"] = task_id
                    row[HIDDEN_HISTORY_META] = marker
                    row["injected_event"] = "subagent_result"
                return row

            items: list[dict[str, Any]] = []
            while len(items) < limit:
                try:
                    items.append(await _to_user_message(pending_queue.get_nowait()))
                except asyncio.QueueEmpty:
                    break

            # Block if nothing drained but sub-agents spawned in this dispatch
            # are still running.  Keeps the runner loop alive so subsequent
            # completions are injected in-order rather than dispatched separately.
            if (not items
                    and session is not None
                    and self.subagents.get_running_count_by_session(session.key) > 0):
                try:
                    msg = await asyncio.wait_for(pending_queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for sub-agent completion in session {}",
                        session.key,
                    )
                    return items
                items.append(await _to_user_message(msg))
                while len(items) < limit:
                    try:
                        items.append(await _to_user_message(pending_queue.get_nowait()))
                    except asyncio.QueueEmpty:
                        break

            return items

        active_session_key = session.key if session else session_key
        effective_scope = self.workspace_scopes.for_turn(
            channel=channel,
            message_metadata=metadata,
            session_metadata=session.metadata if session is not None else None,
        )
        is_owner = is_owner_match(sender_id, self._owner_id)
        if is_owner:
            # Owner gets full filesystem access regardless of channel defaults.
            effective_scope = build_workspace_scope(
                effective_scope.project_path,
                "full",
                source_channel=effective_scope.source_channel,
            )
        effective_tools = tools or self.tools
        if self._owner_id and sender_id and not is_owner:
            effective_tools = effective_tools.filtered_view(self._NON_OWNER_ALLOWED_TOOLS)
        request_ctx = request_context or RequestContext(
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=active_session_key,
            sender_id=sender_id,
            original_user_text=original_user_text,
            runtime=runtime,
            metadata=dict(metadata or {}),
            workspace=effective_scope.project_path,
        )
        file_state_token = bind_file_states(self._file_state_store.for_session(active_session_key))
        request_token = bind_request_context(request_ctx)
        workspace_token = bind_workspace_scope(effective_scope)
        turn_scope_stack = ExitStack()
        # Compute lazily because create_goal may create goal metadata during this run.
        def _goal_continue() -> str | None:
            _goal_lines = goal_state_runtime_lines(session.metadata if session is not None else None)
            if not _goal_lines:
                return None
            return (
                "You have an active sustained goal:\n\n"
                + "\n".join(_goal_lines)
                + "\n\nPlease continue working toward the objective using your tools, "
                "or call update_goal with action='complete' if the work is truly finished."
            )

        session_metadata = session.metadata if session is not None else None
        try:
            for scope in turn_scopes or ():
                turn_scope_stack.enter_context(scope)
            hook = build_agent_turn_hook(AgentTurnHookSpec(
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                channel=channel,
                chat_id=chat_id,
                message_id=message_id,
                metadata=metadata,
                session_key=active_session_key,
                workspace=effective_scope.project_path,
                tool_hint_max_length=self.tool_hint_max_length,
                on_iteration=lambda iteration: setattr(self, "_current_iteration", iteration),
                registered_hook_factories=self._hook_factories,
                turn_hook_factories=list(hook_factories or []),
                registered_hooks=self._extra_hooks,
                turn_hooks=list(hooks or []),
                ephemeral=ephemeral,
                run_extra_hooks_for_ephemeral=run_extra_hooks_for_ephemeral,
            ))
            result = await self.runner.run(AgentRunSpec(
                initial_messages=initial_messages,
                tools=effective_tools,
                runtime=runtime,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                hook=hook,
                error_message="Sorry, I encountered an error calling the AI model.",
                concurrent_tools=True,
                workspace=effective_scope.project_path,
                session_key=session.key if session else None,
                context_block_limit=self.context_block_limit,
                provider_retry_mode=self.provider_retry_mode,
                progress_callback=on_progress,
                stream_progress_deltas=on_stream is not None,
                retry_wait_callback=on_retry_wait,
                checkpoint_callback=_checkpoint,
                injection_callback=_drain_pending,
                # Sustained goals may legitimately exceed NANOBOT_LLM_TIMEOUT_S; idle stall
                # is still capped by NANOBOT_STREAM_IDLE_TIMEOUT_S in streaming providers.
                llm_timeout_s=runner_wall_llm_timeout_s(
                    self.sessions,
                    session.key if session is not None else session_key,
                    metadata=session_metadata,
                    message_metadata=metadata,
                ),
                goal_active_predicate=lambda: sustained_goal_active(session.metadata) if session is not None else False,
                goal_continue_message=_goal_continue,
                finalize_on_max_iterations=turn_continuation.should_finalize_on_max_iterations(
                    pending_queue_available=pending_queue is not None and session is not None,
                    session_metadata=session_metadata,
                    message_metadata=metadata,
                ),
                on_snip=self._archive_sniped,
                tool_repeat_nudge_after=self._agent_defaults.tool_repeat_nudge_after,
                tool_repeat_hard_stop_after=self._agent_defaults.tool_repeat_hard_stop_after,
                content_repeat_nudge_after=self._agent_defaults.content_repeat_nudge_after,
                content_repeat_hard_stop_after=self._agent_defaults.content_repeat_hard_stop_after,
                alternating_pattern_nudge_after=self._agent_defaults.alternating_pattern_nudge_after,
                alternating_pattern_hard_stop_after=self._agent_defaults.alternating_pattern_hard_stop_after,
            ))
        finally:
            turn_scope_stack.close()
            reset_workspace_scope(workspace_token)
            reset_request_context(request_token)
            reset_file_states(file_state_token)
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            should_stream = turn_continuation.should_stream_budget_response(
                stop_reason=result.stop_reason,
                pending_queue_available=pending_queue is not None and session is not None,
                session_metadata=session_metadata,
                message_metadata=metadata,
            )
            # Push final content through stream so streaming channels (e.g. Feishu)
            # update the card instead of leaving it empty.
            if on_stream and on_stream_end and should_stream:
                await on_stream(result.final_content or "")
                await on_stream_end(resuming=False)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections
