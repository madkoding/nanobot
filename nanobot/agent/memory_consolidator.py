"""Consolidator: lightweight token-budget triggered consolidation into history.jsonl."""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.memory_store import (
    _ARCHIVE_SUMMARY_MAX_CHARS,
    _RAW_ARCHIVE_MAX_CHARS,
    MemoryStore,
)
from nanobot.runtime_context import public_history_messages
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import (
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    recent_message_start_index,
    truncate_text,
    truncate_text_to_tokens,
)
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.utils.llm_runtime import LLMRuntime


class Consolidator:
    """Lightweight consolidation: summarizes evicted messages into history.jsonl."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        store: MemoryStore,
        sessions: SessionManager,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        consolidation_ratio: float = 0.5,
        unified_session: bool = False,
        shared_locks: "weakref.WeakValueDictionary[str, asyncio.Lock] | None" = None,
    ):
        self.store = store
        self.sessions = sessions
        self.consolidation_ratio = consolidation_ratio
        self.unified_session = unified_session
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        # ponytail: when ``shared_locks`` is provided, consolidate through the
        # same per-session lock the dispatcher uses; otherwise fall back to a
        # private dict so the consolidator still works in isolation (tests).
        self._locks = shared_locks or weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    @staticmethod
    def _full_unconsolidated_history(
        session: Session,
    ) -> list[dict[str, Any]]:
        """Return the whole unconsolidated tail for consolidation decisions."""
        unconsolidated_count = len(session.messages) - session.last_consolidated
        if unconsolidated_count <= 0:
            return []
        return session.get_history(max_messages=unconsolidated_count)

    @staticmethod
    def _replay_overflow_boundary(
        session: Session,
        replay_max_messages: int | None,
    ) -> int | None:
        if not replay_max_messages or replay_max_messages <= 0:
            return None
        tail = list(enumerate(session.messages[session.last_consolidated:], session.last_consolidated))
        if len(tail) <= replay_max_messages:
            return None

        tail_messages = [message for _idx, message in tail]
        start_idx = recent_message_start_index(
            tail_messages,
            replay_max_messages,
            extend_to_user=True,
        )
        sliced = tail[start_idx:]
        for i, (_idx, message) in enumerate(sliced):
            if message.get("role") == "user":
                start = i
                if i > 0 and sliced[i - 1][1].get("_channel_delivery"):
                    start = i - 1
                sliced = sliced[start:]
                break

        legal_start = find_legal_message_start([message for _idx, message in sliced])
        if legal_start:
            sliced = sliced[legal_start:]
        if not sliced:
            return len(session.messages)

        first_visible_idx = sliced[0][0]
        if first_visible_idx <= session.last_consolidated:
            return None
        return first_visible_idx

    async def _consolidate_replay_overflow(
        self,
        session: Session,
        replay_max_messages: int | None,
        *,
        runtime: LLMRuntime,
        store: MemoryStore | None = None,
        sender_id: str | None = None,
    ) -> str | None:
        """Archive messages that would be hidden by the replay message window."""
        end_idx = self._replay_overflow_boundary(session, replay_max_messages)
        if end_idx is None:
            return None
        chunk = session.messages[session.last_consolidated:end_idx]
        if not chunk:
            return None
        logger.info(
            "Replay-window consolidation for {}: chunk={} msgs, replay_max={}",
            session.key,
            len(chunk),
            replay_max_messages,
        )
        summary = await self.archive(
            chunk,
            runtime=runtime,
            session_key=session.key,
            store=store,
            sender_id=sender_id,
        )
        session.last_consolidated = end_idx
        self.sessions.save(session)
        return summary

    def _persist_last_summary(self, session: Session, summary: str | None) -> None:
        if summary and summary != "(nothing)":
            session.metadata["_last_summary"] = {
                "text": summary,
                "last_active": session.updated_at.isoformat(),
            }
            self.sessions.save(session)

    def estimate_session_prompt_tokens(
        self,
        session: Session,
        *,
        runtime: LLMRuntime,
    ) -> tuple[int, str]:
        """Estimate prompt size from the full unconsolidated session tail."""
        history = self._full_unconsolidated_history(session)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        # Include archived summary in estimation so the budget accounts for it.
        meta = session.metadata.get("_last_summary")
        summary = meta.get("text") if isinstance(meta, dict) else (meta if isinstance(meta, str) else None)
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
            sender_id=None,
            session_summary=summary,
            session_metadata=session.metadata,
            session_key=session.key,
            unified_session=self.unified_session,
        )
        return estimate_prompt_tokens_chain(
            runtime.provider,
            runtime.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    def _input_token_budget(self, runtime: LLMRuntime) -> int:
        """Available input token budget for consolidation LLM."""
        return (
            runtime.context_window_tokens
            - runtime.generation.max_tokens
            - self._SAFETY_BUFFER
        )

    def _truncate_to_token_budget(self, text: str, *, runtime: LLMRuntime) -> str:
        """Truncate text so it fits within the consolidation LLM's token budget."""
        budget = self._input_token_budget(runtime)
        if budget <= 0:
            return truncate_text(text, _RAW_ARCHIVE_MAX_CHARS)
        return truncate_text_to_tokens(text, budget)

    async def archive(
        self,
        messages: list[dict],
        *,
        runtime: LLMRuntime,
        session_key: str | None = None,
        summary_messages: list[dict] | None = None,
        store: MemoryStore | None = None,
        sender_id: str | None = None,
    ) -> str | None:
        """Summarize messages via LLM and append to history.jsonl.

        ``messages`` are the messages being archived (removed from the live
        session); they are what gets raw-dumped if the LLM call fails.
        ``summary_messages``, when given, lets callers include retained
        messages in the summary without archiving them.

        ``store`` overrides the default MemoryStore so consolidation writes to
        the workspace that owns the session (e.g. a WhatsApp group workspace).
        """
        if not messages:
            return None
        store = store or self.store
        messages_to_summarize = public_history_messages(
            summary_messages if summary_messages is not None else messages
        )
        formatted = MemoryStore._format_messages(messages_to_summarize)
        formatted = self._truncate_to_token_budget(formatted, runtime=runtime)
        system_prompt = render_template(
            "agent/consolidator_archive.md",
            strip=True,
        )
        try:
            response = await runtime.provider.chat_with_retry(
                model=runtime.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": formatted},
                ],
                tools=None,
                tool_choice=None,
                temperature=runtime.generation.temperature,
                max_tokens=runtime.generation.max_tokens,
                reasoning_effort=runtime.generation.reasoning_effort,
            )
        except Exception:
            logger.warning("Consolidation provider call failed, raw-dumping to history")
            store.raw_archive(messages, session_key=session_key, sender_id=sender_id)
            return None
        if response.finish_reason == "error":
            logger.warning("Consolidation provider returned an error, raw-dumping to history")
            store.raw_archive(messages, session_key=session_key, sender_id=sender_id)
            return None
        summary = response.content or "[no summary]"
        store.append_history(
            summary,
            max_chars=_ARCHIVE_SUMMARY_MAX_CHARS,
            session_key=session_key,
            sender_id=sender_id,
        )
        return summary

    async def maybe_consolidate_by_tokens(
        self,
        session: Session,
        *,
        runtime: LLMRuntime,
        replay_max_messages: int | None = None,
        store: MemoryStore | None = None,
        sender_id: str | None = None,
    ) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.

        ``store`` overrides the default MemoryStore so consolidation writes to
        the workspace that owns the session (e.g. a WhatsApp group workspace).
        """
        if runtime.context_window_tokens <= 0:
            return

        store = store or self.store

        lock = self.get_lock(session.key)
        async with lock:
            # Refresh session reference: AutoCompact may have replaced it.
            fresh = self.sessions.get_or_create(session.key)
            if fresh is not session:
                session = fresh
            if not session.messages:
                return

            budget = self._input_token_budget(runtime)
            target = int(budget * self.consolidation_ratio)
            last_summary = await self._consolidate_replay_overflow(
                session,
                replay_max_messages,
                runtime=runtime,
                store=store,
                sender_id=sender_id,
            )
            estimated, source = self.estimate_session_prompt_tokens(
                session,
                runtime=runtime,
            )
            if estimated <= 0:
                self._persist_last_summary(session, last_summary)
                return
            if estimated < budget:
                unconsolidated_count = len(session.messages) - session.last_consolidated
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}, msgs={}",
                    session.key,
                    estimated,
                    runtime.context_window_tokens,
                    source,
                    unconsolidated_count,
                )
                self._persist_last_summary(session, last_summary)
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    break

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    break

                end_idx = boundary[0]

                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    break

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    runtime.context_window_tokens,
                    source,
                    len(chunk),
                )
                summary = await self.archive(
                    chunk,
                    runtime=runtime,
                    session_key=session.key,
                    store=store,
                    sender_id=sender_id,
                )
                # Advance the cursor either way: on success the chunk was
                # summarized; on failure archive() already raw-archived it as
                # a breadcrumb. Re-archiving the same chunk on the next call
                # would just emit duplicate [RAW] entries.
                if summary:
                    last_summary = summary
                session.last_consolidated = end_idx
                self.sessions.save(session)
                if not summary:
                    # LLM is degraded — stop hammering it this call;
                    # the next invocation can retry a fresh chunk.
                    break

                estimated, source = self.estimate_session_prompt_tokens(
                    session,
                    runtime=runtime,
                )
                if estimated <= 0:
                    break

            # Persist the last summary to session metadata so it can be injected
            # into the runtime context on the next prepare_session() call, aligning
            # the summary injection strategy with AutoCompact._archive().
            self._persist_last_summary(session, last_summary)

    async def compact_idle_session(
        self,
        session_key: str,
        *,
        runtime: LLMRuntime,
        max_suffix: int = 8,
        store: MemoryStore | None = None,
        sender_id: str | None = None,
    ) -> str | None:
        """Archive an idle session's prefix without deleting the messages from disk.

        Used by AutoCompact so all session mutation goes through a single
        lock-protected path.  Returns the summary text on success, ``None``
        if the LLM failed (raw_archive fallback), or ``""`` if there was
        nothing to archive.

        The archived prefix is preserved in ``session.messages`` (so the WebUI
        can paginate it and nothing is lost) and excluded from the LLM context
        by advancing ``last_consolidated`` instead of truncating the list.  The
        physical file cap (``enforce_file_cap`` / ``FILE_MAX_MESSAGES``) remains
        the only thing that ever removes old rows.
        """
        lock = self.get_lock(session_key)
        async with lock:
            self.sessions.invalidate(session_key)
            session = self.sessions.get_or_create(session_key)

            messages_to_summarize = list(session.messages[session.last_consolidated:])
            if not messages_to_summarize:
                self.sessions.save(session)
                return ""

            probe = Session(
                key=session.key,
                messages=messages_to_summarize.copy(),
                created_at=session.created_at,
                updated_at=session.updated_at,
                metadata={},
                last_consolidated=0,
            )
            result = probe.retain_recent_legal_suffix(max_suffix, extend_to_user=True)
            messages_to_keep = probe.messages
            messages_to_remove = result.dropped[result.already_consolidated_count:]

            if not messages_to_remove and not messages_to_keep:
                self.sessions.save(session)
                return ""

            summary: str | None = ""
            if messages_to_remove:
                # Summarize the retained suffix too, but only remove/raw-dump
                # the messages that are no longer kept in the live session.
                summary = await self.archive(
                    messages_to_remove,
                    runtime=runtime,
                    session_key=session_key,
                    summary_messages=messages_to_summarize,
                    store=store,
                    sender_id=sender_id,
                )

            if summary and summary != "(nothing)":
                self._persist_last_summary(session, summary)

            # Only advance last_consolidated if the summary succeeded (truthy) or
            # there was nothing to archive. If archive() returned None (LLM failed),
            # keep last_consolidated where it was so the next turn doesn't lose
            # context without a summary to inject.
            if summary is not None:
                if messages_to_remove:
                    # Start of the retained suffix = end of the archived prefix.
                    session.last_consolidated = (
                        session.last_consolidated + len(messages_to_remove)
                    )
                self.sessions.save(session)

            if messages_to_remove:
                logger.info(
                    "Idle-session compact for {}: archived={}, kept={}, summary={}",
                    session_key,
                    len(messages_to_remove),
                    len(messages_to_keep),
                    bool(summary),
                )

            return summary
