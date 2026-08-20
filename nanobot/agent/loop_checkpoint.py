"""Runtime checkpoint/resume mixin for AgentLoop (extracted from agent/loop.py)."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.session import turn_continuation
from nanobot.session.history_visibility import HIDDEN_HISTORY_META
from nanobot.session.keys import UNIFIED_SESSION_KEY


class CheckpointMixin:
    """Runtime turn checkpointing and interrupted-session resume.

    Mixin: ``AgentLoop`` inherits this so ``self`` resolves against the full
    loop instance (``self.sessions``, ``self.bus``, ``self._runtime_events()``).
    """

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"
    _PENDING_INJECTIONS_KEY = "_pending_injections"

    def _set_runtime_checkpoint(self, session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _mark_pending_user_turn(self, session) -> None:
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session) -> None:
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _append_pending_injection(self, session, msg: InboundMessage) -> None:
        payload = {
            "channel": msg.channel,
            "sender_id": msg.sender_id,
            "chat_id": msg.chat_id,
            "content": msg.content,
            "media": list(msg.media or []),
            "metadata": dict(msg.metadata or {}),
            "session_key_override": msg.session_key_override,
        }
        session.metadata.setdefault(self._PENDING_INJECTIONS_KEY, []).append(payload)

    def _clear_pending_injections(self, session) -> None:
        session.metadata.pop(self._PENDING_INJECTIONS_KEY, None)

    def _pending_injections(self, session) -> list[dict[str, Any]]:
        value = session.metadata.get(self._PENDING_INJECTIONS_KEY)
        if isinstance(value, list):
            return value
        return []

    def _clear_runtime_checkpoint(self, session) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _materialize_runtime_checkpoint(
        self,
        session,
        *,
        synthesize_missing: bool = True,
    ) -> list[dict[str, Any]]:
        """Build the message list that restores an unfinished turn.

        When ``synthesize_missing`` is True, pending tool calls are closed with
        an error placeholder (used on the next user turn or after /stop). When
        False, the checkpoint is materialized without placeholders so the turn
        can be resumed by re-injecting a system resume message.
        """
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return []

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": (
                        "Error: Task interrupted before this tool finished."
                        if synthesize_missing
                        else "The gateway restarted before this tool call completed. "
                             "Re-emit only if still necessary."
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        return restored_messages

    def _append_checkpoint_messages(
        self,
        session,
        restored_messages: list[dict[str, Any]],
    ) -> None:
        """Append checkpoint materialization to session history, avoiding duplication."""
        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

    def _restore_runtime_checkpoint(
        self,
        session,
        *,
        synthesize_missing: bool = True,
    ) -> bool:
        """Materialize an unfinished turn into session history before a new request."""
        restored_messages = self._materialize_runtime_checkpoint(
            session,
            synthesize_missing=synthesize_missing,
        )
        if not restored_messages:
            return False
        self._append_checkpoint_messages(session, restored_messages)
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _restore_pending_user_turn(self, session) -> bool:
        """Close a turn that only persisted the user message before crashing."""
        from datetime import datetime

        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                    HIDDEN_HISTORY_META: True,
                }
            )
            session.updated_at = datetime.now()

        self._clear_pending_user_turn(session)
        return True

    @staticmethod
    def _reconstruct_pending_injection(payload: dict[str, Any]) -> InboundMessage | None:
        try:
            return InboundMessage(
                channel=payload.get("channel", "cli"),
                sender_id=str(payload.get("sender_id", "user")),
                chat_id=payload.get("chat_id", "direct"),
                content=payload.get("content", ""),
                media=list(payload.get("media") or []),
                metadata=dict(payload.get("metadata") or {}),
                session_key_override=payload.get("session_key_override"),
            )
        except Exception:
            logger.warning("Skipping malformed pending injection: {}", payload)
            return None

    @staticmethod
    def _channel_chat_id_from_session_key(session_key: str) -> tuple[str, str]:
        """Return a sensible (channel, chat_id) pair derived from a session key."""
        if session_key == UNIFIED_SESSION_KEY:
            return ("cli", "direct")
        if ":" in session_key:
            channel, chat_id = session_key.split(":", 1)
            return (channel, chat_id)
        return ("cli", session_key)

    def _webui_session_key(self, session_key: str) -> str:
        """Map a session key to the websocket transcript namespace."""
        if session_key.startswith("websocket:"):
            return session_key
        _, chat_id = self._channel_chat_id_from_session_key(session_key)
        return f"websocket:{chat_id}"

    def _write_transcript_resume_events(
        self,
        session_key: str,
        *,
        resumed: bool,
        closed: bool,
    ) -> None:
        """Write synthetic transcript events to clear phantom spinners."""
        try:
            from nanobot.webui.transcript import append_transcript_object
        except Exception:
            return
        webui_key = self._webui_session_key(session_key)
        _, chat_id = self._channel_chat_id_from_session_key(session_key)
        now_ms = int(time.time() * 1000)
        append_transcript_object(
            webui_key,
            {
                "event": "turn_end",
                "chat_id": chat_id,
                "reason": "gateway_restart",
                "created_at_ms": now_ms,
            },
        )
        if resumed:
            append_transcript_object(
                webui_key,
                {
                    "event": "message",
                    "chat_id": chat_id,
                    "role": "system",
                    "text": "Turno reanudado tras reinicio del gateway.",
                    "kind": "notice",
                    "created_at_ms": now_ms,
                },
            )
        elif closed:
            append_transcript_object(
                webui_key,
                {
                    "event": "message",
                    "chat_id": chat_id,
                    "role": "system",
                    "text": "El gateway se reinició; el turno anterior se cerró.",
                    "kind": "notice",
                    "created_at_ms": now_ms,
                },
            )

    async def _restore_interrupted_sessions(self) -> int:
        """Resume sessions whose last turn was interrupted by a gateway restart.

        Finds sessions with ``runtime_checkpoint`` or ``pending_user_turn`` flags,
        materializes the checkpoint into session history, and re-injects a resume
        message so the agent loop continues the interrupted turn. Sessions that
        only have a pending user turn are closed without a retry.
        """
        count = 0
        for info in self.sessions.list_sessions():
            if not info.get("interrupted"):
                continue
            session_key = info["key"]
            try:
                session = self.sessions.get_or_create(session_key)
            except Exception:
                logger.exception("Could not load session {} for restart resume", session_key)
                continue
            try:
                checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
                has_checkpoint = isinstance(checkpoint, dict)
                if has_checkpoint:
                    # Materialize the checkpoint *without* error placeholders so the
                    # resumed turn can continue the LLM's pending tool calls.
                    checkpoint_messages = self._materialize_runtime_checkpoint(
                        session,
                        synthesize_missing=False,
                    )
                    if checkpoint_messages:
                        self._append_checkpoint_messages(session, checkpoint_messages)
                        # Clear the flags now so the subsequent resume message is
                        # processed as a fresh turn and does not re-synthesize
                        # placeholder tool errors.
                        self._clear_runtime_checkpoint(session)
                        self._clear_pending_user_turn(session)
                        self.sessions.save(session)
                        channel, chat_id = self._channel_chat_id_from_session_key(session_key)
                        resumed_msg = InboundMessage(
                            channel="system",
                            sender_id=turn_continuation._GATEWAY_RESUME_SENDER,
                            chat_id=f"{channel}:{chat_id}",
                            content=turn_continuation.gateway_resume_prompt(),
                            metadata=turn_continuation.gateway_resume_metadata(
                                original_channel=channel,
                                original_chat_id=chat_id,
                            ),
                            session_key_override=session_key,
                        )
                    await self.bus.publish_inbound(resumed_msg)
                    for payload in self._pending_injections(session):
                        injection = self._reconstruct_pending_injection(payload)
                        if injection is not None:
                            await self.bus.publish_inbound(injection)
                    self._clear_pending_injections(session)
                    self.sessions.save(session)
                    self._write_transcript_resume_events(
                        session_key,
                        resumed=True,
                        closed=False,
                    )
                    count += 1
                    continue

                if self._restore_pending_user_turn(session):
                    self.sessions.save(session)
                    await self._runtime_events().run_status_changed(
                        InboundMessage(
                            channel="cli",
                            sender_id="system",
                            chat_id="direct",
                            content="",
                        ),
                        session_key,
                        "idle",
                    )
                    self._write_transcript_resume_events(
                        session_key,
                        resumed=False,
                        closed=True,
                    )
                    count += 1
            except Exception:
                logger.exception("Failed to restore interrupted session {}", session_key)
        return count
