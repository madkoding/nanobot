"""Base channel interface for chat platforms."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.pairing import (
    PAIRING_CODE_META_KEY,
    format_pairing_reply,
    generate_code,
    is_approved,
)


class BaseChannel(ABC):
    """
    Abstract base class for chat channel implementations.

    Each channel (Telegram, Discord, etc.) should implement this interface
    to integrate with the nanobot message bus.
    """

    name: str = "base"
    display_name: str = "Base"
    send_progress: bool = True
    send_tool_hints: bool = True
    show_reasoning: bool = True

    def __init__(
        self,
        config: Any,
        bus: MessageBus,
        *,
        owner_id: str | list[str] | None = None,
    ):
        """
        Initialize the channel.

        Args:
            config: Channel-specific configuration.
            bus: The message bus for communication.
            owner_id: Operator identity (or list) used to resolve the
                owner DM chat for error notifications. Subclasses decide
                which identity maps to their DM format.
        """
        self.config = config
        self.logger = logger.bind(channel=self.name)
        self.bus = bus
        self._owner_id = owner_id
        self._running = False
        # ponytail: last_activity_at lets the manager watchdog detect a "live
        # but silent" channel (idle() blocked on a dead socket, group queue
        # task died, etc.) and force a restart. Subclasses should bump this
        # whenever they receive any inbound event or successfully publish a
        # message; BaseChannel bumps it on every publish_inbound. Default 0
        # means "never had activity" so the watchdog never fires during the
        # login grace window.
        self.last_activity_at: float = 0.0

    def owner_chat_id(self) -> str | None:
        """Return the operator's DM chat id for this channel, or None.

        Used to route error notifications to the operator's private DM
        instead of spamming the originating group. Subclasses override
        this to translate the global ``owner_id`` into their channel's
        DM chat format (e.g. WhatsApp ``<phone>@s.whatsapp.net``).
        """
        return None

    def _touch_activity(self) -> None:
        """Record that this channel just did work, for the manager watchdog."""
        import time as _time
        self.last_activity_at = _time.monotonic()

    async def transcribe_audio(self, file_path: str | Path) -> str:
        """Transcribe an audio file via Whisper (OpenAI or Groq). Returns empty string on failure."""
        try:
            from nanobot.audio.transcription import (
                resolve_transcription_config,
                transcribe_audio_file,
            )
            from nanobot.config.loader import load_config

            return await transcribe_audio_file(file_path, resolve_transcription_config(load_config()))
        except Exception:
            self.logger.exception("Audio transcription failed")
            return ""

    async def login(self, force: bool = False) -> bool:
        """
        Perform channel-specific interactive login (e.g. QR code scan).

        Args:
            force: If True, ignore existing credentials and force re-authentication.

        Returns True if already authenticated or login succeeds.
        Override in subclasses that support interactive login.
        """
        return True

    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel and begin listening for messages.

        This should be a long-running async task that:
        1. Connects to the chat platform
        2. Listens for incoming messages
        3. Forwards messages to the bus via _handle_message()
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        pass

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """
        Send a message through this channel.

        Args:
            msg: The message to send.

        Implementations should raise on delivery failure so the channel manager
        can apply any retry policy in one place.
        """
        pass

    async def send_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
        stream_end: bool = False,
        resuming: bool = False,
    ) -> None:
        """Deliver a streaming text chunk.

        Override in subclasses to enable streaming. Implementations should
        raise on delivery failure so the channel manager can retry.

        Stateful implementations should key buffers by ``stream_id`` rather
        than only by ``chat_id`` when it is provided.
        """
        pass

    async def send_reasoning_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
    ) -> None:
        """Stream a chunk of model reasoning/thinking content.

        Default is no-op. Channels with a native low-emphasis primitive
        (Slack context block, Telegram expandable blockquote, Discord
        subtext, WebUI italic bubble, ...) override to render reasoning
        as a subordinate trace that updates in place as the model thinks.

        Streaming contract mirrors :meth:`send_delta`: stateful implementations
        should key buffers by ``stream_id`` rather than only by ``chat_id``.
        """
        return

    async def send_reasoning_end(
        self,
        chat_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
    ) -> None:
        """Mark the end of a reasoning stream segment.

        Default is no-op. Channels that buffer ``send_reasoning_delta``
        chunks for in-place updates use this signal to flush and freeze
        the rendered group; one-shot channels can ignore it entirely.
        """
        return

    async def send_file_edit_events(
        self,
        chat_id: str,
        edits: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Deliver structured live file-edit events.

        Default is no-op. Channels with a rich activity surface can override
        this to render editing progress without receiving empty text messages.
        """
        return

    async def send_reasoning(self, msg: OutboundMessage) -> None:
        """Deliver a complete reasoning block.

        Default implementation reuses the streaming pair so plugins only
        need to override the delta/end methods. Equivalent to one delta
        with the full content followed immediately by an end marker —
        keeps a single rendering path for both streamed and one-shot
        reasoning (e.g. DeepSeek-R1's final-response ``reasoning_content``).
        """
        if not msg.content:
            return
        stream_id = getattr(msg.event, "stream_id", None)
        await self.send_reasoning_delta(
            msg.chat_id,
            msg.content,
            msg.metadata,
            stream_id=stream_id,
        )
        await self.send_reasoning_end(
            msg.chat_id,
            msg.metadata,
            stream_id=stream_id,
        )

    @property
    def supports_streaming(self) -> bool:
        """True when config enables streaming AND this subclass implements send_delta."""
        cfg = self.config
        streaming = cfg.get("streaming", False) if isinstance(cfg, dict) else getattr(cfg, "streaming", False)
        return bool(streaming) and type(self).send_delta is not BaseChannel.send_delta

    def is_allowed(self, sender_id: str) -> bool:
        """Check sender permission: star > allowlist > pairing store > deny."""
        if isinstance(self.config, dict):
            allow_list = self.config.get("allow_from") or self.config.get("allowFrom") or []
        else:
            allow_list = getattr(self.config, "allow_from", None) or []
        if "*" in allow_list:
            return True
        # allowFrom entries are opaque tokens — must match exactly.
        if str(sender_id) in allow_list:
            return True
        if is_approved(self.name, str(sender_id)):
            return True
        return False

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        is_dm: bool = False,
        authorization_id: str | None = None,
    ) -> None:
        """Handle a message after checking its authorization subject.

        ``sender_id`` is the identity recorded on the inbound message.  Channels
        where access is scoped to another entity (for example, a group or room)
        can pass that entity as ``authorization_id`` without changing the
        sender's identity.  When omitted, authorization remains sender-based.
        """
        permission_id = authorization_id if authorization_id is not None else sender_id
        if not self.is_allowed(permission_id):
            if is_dm:
                code = generate_code(self.name, str(sender_id))
                try:
                    await self.send(
                        OutboundMessage(
                            channel=self.name,
                            chat_id=str(chat_id),
                            content=format_pairing_reply(code),
                            metadata={PAIRING_CODE_META_KEY: code},
                        )
                    )
                except Exception:
                    # ponytail: never let a send failure (e.g. WhatsApp
                    # 463 throttle, disconnected socket) propagate out
                    # of the inbound handler — that would crash the
                    # event listener and silently drop every subsequent
                    # message. Log it so the user has a clue to retry.
                    self.logger.exception(
                        "Failed to send pairing code {} to sender {} in chat {}",
                        code, sender_id, chat_id,
                    )
                else:
                    self.logger.info(
                        "Sent pairing code {} to sender {} in chat {}",
                        code, sender_id, chat_id,
                    )
            else:
                self.logger.warning(
                    "Access denied for sender {}. "
                    "Add them to allowFrom list in config to grant access.",
                    sender_id,
                )
            return

        meta = metadata or {}
        if self.supports_streaming:
            meta = {**meta, "_wants_stream": True}

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=meta,
            session_key_override=session_key,
        )

        await self.bus.publish_inbound(msg)
        self._touch_activity()

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Return default config for onboard. Override in plugins to auto-populate config.json."""
        return {"enabled": False}

    @classmethod
    def refresh_feature_metadata(
        cls,
        config_path: Path,
        *,
        instance_id: str = "default",
    ) -> bool:
        """Refresh persisted display metadata after an explicit settings action."""
        return False

    @property
    def is_running(self) -> bool:
        """Check if the channel is running."""
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _require_ready(self) -> None:
        """Raise if the channel has not been started.

        Call at the top of :meth:`send` so the channel manager's retry policy
        can classify the failure consistently.
        """
        if not self._running:
            raise RuntimeError("channel not started")

    @classmethod
    def build_kwargs(cls, manager: Any) -> dict[str, Any]:
        """Return extra constructor kwargs supplied by the channel manager.

        Override this in channel subclasses that need runtime wiring from the
        manager (e.g. the WebSocket channel needs gateway services). The default
        is empty so ordinary channels require no special construction args.
        """
        return {}

    def accepts_outbound(self, msg: OutboundMessage) -> bool:
        """Return True when this channel should consume an outbound message.

        Most channels accept every message directed at them. The WebSocket
        channel uses this to keep :class:`RuntimeModelUpdatedEvent` fan-out
        alive even when no websocket runtime is currently enabled.
        """
        return True

    # ------------------------------------------------------------------
    # Shared runtime helpers
    # ------------------------------------------------------------------

    def _bounded_set(self, maxlen: int) -> "_BoundedSet":
        """Return a bounded set of string ids suitable for inbound dedup caches."""
        return _BoundedSet(maxlen)

    async def _download_to_media_dir(
        self,
        url: str,
        filename_hint: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        marker_type: str = "image",
    ) -> tuple[Path | None, str]:
        """Download an inbound media file to the channel's media directory.

        Returns the saved path and a content marker string. On failure it
        returns ``(None, marker)`` where marker describes the failure, so the
        upstream caller can still log what was dropped.
        """
        from nanobot.config.paths import get_media_dir
        from nanobot.utils.helpers import safe_filename

        media_dir = get_media_dir(self.name)
        media_dir.mkdir(parents=True, exist_ok=True)
        safe_name = safe_filename(filename_hint) or "attachment"
        path = media_dir / safe_name

        client: httpx.AsyncClient | None = None
        try:
            client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            path.write_bytes(response.content)
            marker = f"[{marker_type}: {safe_name}]"
            return path, marker
        except Exception as exc:
            self.logger.warning(
                "Failed to download {} media from {}: {}", marker_type, url, exc
            )
            return None, f"[{marker_type}: {safe_name} - download failed]"
        finally:
            if client is not None:
                await client.aclose()


class BoundedSet:
    """Bounded set of string ids used for inbound message deduplication."""

    def __init__(self, maxlen: int) -> None:
        self._maxlen = maxlen
        self._deque: deque[str] = deque(maxlen=maxlen)
        self._set: set[str] = set()

    def add(self, key: str) -> None:
        if key in self._set:
            return
        if len(self._deque) == self._maxlen:
            oldest = self._deque.popleft()
            self._set.discard(oldest)
        self._deque.append(key)
        self._set.add(key)

    def __contains__(self, key: str) -> bool:
        return key in self._set

    def __setitem__(self, key: str, value: object) -> None:
        """Support legacy OrderedDict-style insertion: ids[message_id] = None."""
        self.add(key)

    def clear(self) -> None:
        self._deque.clear()
        self._set.clear()

    def __len__(self) -> int:
        return len(self._set)

    def __iter__(self):
        return iter(self._deque)

    def keys(self) -> list[str]:
        return list(self._deque)


_BoundedSet = BoundedSet  # compat alias for internal type hints


class TypingIndicator:
    """Periodic typing action helper for channels that support it.

    Usage:
        indicator = TypingIndicator(interval=4.0)
        indicator.start(chat_id, send_action)
        # ... do work ...
        indicator.stop(chat_id)
    """

    def __init__(self, interval: float = 4.0) -> None:
        self._interval = interval
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def start(
        self,
        chat_id: str,
        send_action: Callable[[], Awaitable[object]],
    ) -> None:
        """Start a typing task for *chat_id*."""
        self.stop(chat_id)
        self._tasks[chat_id] = asyncio.create_task(self._loop(chat_id, send_action))

    def stop(self, chat_id: str) -> None:
        """Cancel the typing task for *chat_id*, if any."""
        task = self._tasks.pop(chat_id, None)
        if task is not None and not task.done():
            task.cancel()

    def stop_all(self) -> None:
        """Cancel all typing tasks."""
        for chat_id in list(self._tasks):
            self.stop(chat_id)

    async def _loop(self, chat_id: str, send_action: Callable[[], Awaitable[object]]) -> None:
        from contextlib import suppress

        try:
            while True:
                with suppress(Exception):
                    await send_action()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass


async def reconnect_loop(
    connect: Callable[[], Awaitable[None]],
    should_run: Callable[[], bool],
    *,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    logger: Any = None,
    label: str = "channel",
) -> None:
    """Run *connect* repeatedly with exponential backoff until *should_run* is False.

    This is meant for simple WebSocket/long-poll channels that need a single
    connection loop. Channels with complex state (e.g. WhatsApp neonize,
    Weixin iLink) should keep their own specialized loops.
    """
    delay = base_delay
    while should_run():
        try:
            await connect()
            if not should_run():
                break
            delay = base_delay
            continue
        except asyncio.CancelledError:
            break
        except Exception as exc:
            if logger is not None:
                logger.warning("{} connection failed: {}", label, exc)
            if not should_run():
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
