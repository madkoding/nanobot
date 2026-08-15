"""Filesystem-backed durable queues for the message bus.

Mirrors the durable queue pattern used by ``LocalTriggerStore``:
- ``inbox/``: messages waiting to be consumed
- ``processing/``: messages claimed by a consumer but not yet acknowledged

On startup, ``recover()`` moves any leftover ``processing/`` files back to
``inbox/`` so they can be redelivered.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import uuid
from collections.abc import Awaitable
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.outbound_events import (
    AutomationUpdateEvent,
    GoalStateSyncEvent,
    GoalStatusEvent,
    OutboundEvent,
    ProgressEvent,
    RetryWaitEvent,
    RuntimeModelUpdatedEvent,
    SessionUpdatedEvent,
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
    TurnEndEvent,
    TurnModelUpdatedEvent,
)
from nanobot.utils.atomic_write import atomic_write_text


def _encode_datetime(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _decode_datetime(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def _deep_decode_datetimes(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_decode_datetimes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_decode_datetimes(v) for v in obj]
    return _decode_datetime(obj)


_EVENT_REGISTRY: dict[str, type[OutboundEvent]] = {
    cls.__name__: cls
    for cls in (
        ProgressEvent,
        RetryWaitEvent,
        StreamDeltaEvent,
        StreamEndEvent,
        StreamedResponseEvent,
        TurnEndEvent,
        GoalStatusEvent,
        GoalStateSyncEvent,
        SessionUpdatedEvent,
        RuntimeModelUpdatedEvent,
        TurnModelUpdatedEvent,
        AutomationUpdateEvent,
    )
}


def _event_to_dict(event: OutboundEvent | None) -> dict[str, Any] | None:
    if event is None:
        return None
    cls = type(event)
    data: dict[str, Any] = {"_event_type": cls.__name__}
    for field in dataclasses.fields(event):
        data[field.name] = getattr(event, field.name)
    return data


def _event_from_dict(data: dict[str, Any] | None) -> OutboundEvent | None:
    if data is None:
        return None
    if "_event_type" not in data:
        return None
    cls = _EVENT_REGISTRY.get(data.pop("_event_type"))
    if cls is None:
        return None
    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return cls(**kwargs)


def _inbound_to_dict(msg: InboundMessage) -> dict[str, Any]:
    data = dataclasses.asdict(msg)
    data["_delivery_kind"] = "inbound"
    return data


def _outbound_to_dict(msg: OutboundMessage) -> dict[str, Any]:
    data = dataclasses.asdict(msg)
    data["_delivery_kind"] = "outbound"
    data["event"] = _event_to_dict(msg.event)
    return data


def _dict_to_inbound(data: dict[str, Any]) -> InboundMessage:
    data.pop("_delivery_kind", None)
    data.pop("_delivery_id", None)
    data = _deep_decode_datetimes(data)
    return InboundMessage(**data)


def _durable_session_key(data: dict[str, Any]) -> str:
    """Derive the session key for a persisted inbound message.

    Mirrors ``InboundMessage.session_key`` (override wins), falling back to
    ``{channel}:{chat_id}``.
    """
    override = data.get("session_key_override")
    if isinstance(override, str) and override:
        return override
    channel = data.get("channel")
    chat_id = data.get("chat_id")
    if isinstance(channel, str) and isinstance(chat_id, str):
        return f"{channel}:{chat_id}"
    return ""


def _dict_to_outbound(data: dict[str, Any]) -> OutboundMessage:
    data.pop("_delivery_kind", None)
    data.pop("_delivery_id", None)
    data["event"] = _event_from_dict(data.get("event"))
    data = _deep_decode_datetimes(data)
    # Tolerate fields written by older nanobot versions that this version's
    # OutboundMessage no longer accepts (e.g. `rich`). Dropping unknown keys
    # keeps the outbound dispatcher alive instead of crashing on a stale
    # durable message.
    known = set(OutboundMessage.__dataclass_fields__)
    data = {k: v for k, v in data.items() if k in known}
    return OutboundMessage(**data)


class DurableMessageQueue:
    """Generic filesystem-backed queue with inbox/processing semantics."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.inbox_dir = self.root / "inbox"
        self.processing_dir = self.root / "processing"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        self._signal: asyncio.Queue[str] = asyncio.Queue()
        self._processing: dict[str, Path] = {}

    def _next_inbox_file(self) -> Path | None:
        # Filenames are random UUIDs, so lexical order is meaningless. Sort by
        # mtime (insertion order) so streamed deltas are consumed FIFO instead
        # of in a scrambled order.
        files = sorted(self.inbox_dir.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
        for path in files:
            if path.is_file():
                return path
        return None

    def _claim(self, path: Path) -> tuple[str, Path]:
        delivery_id = uuid.uuid4().hex
        processing_path = self.processing_dir / f"{delivery_id}.json"
        os.replace(path, processing_path)
        return delivery_id, processing_path

    async def recover(
        self,
        active_session_keys: set[str] | None = None,
    ) -> int:
        """Move files left in processing back to inbox and return the count.

        When ``active_session_keys`` is provided, only messages whose session
        key is in that set are requeued; the rest are dropped. This prevents
        ``recover()`` from replaying stale messages for sessions that finished
        or were deleted before the restart.
        """
        recovered = 0
        for path in sorted(self.processing_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.exception("Failed to read durable queue file {}", path)
                continue
            key = _durable_session_key(data)
            if active_session_keys is not None and key not in active_session_keys:
                path.unlink(missing_ok=True)
                continue
            target = self.inbox_dir / path.name
            counter = 0
            while target.exists():
                counter += 1
                target = self.inbox_dir / f"{path.stem}-{counter}{path.suffix}"
            try:
                os.replace(path, target)
                recovered += 1
            except OSError:
                logger.exception("Failed to recover durable queue file {}", path)
        if recovered:
            logger.info("Recovered {} durable message(s) from processing", recovered)
            for _ in range(recovered):
                try:
                    self._signal.put_nowait("recovered")
                except asyncio.QueueFull:
                    break
        return recovered

    async def _publish(
        self,
        data: dict[str, Any],
        put_signal: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        delivery_id = uuid.uuid4().hex
        data["_delivery_id"] = delivery_id
        path = self.inbox_dir / f"{delivery_id}.json"
        content = json.dumps(data, ensure_ascii=False, default=_encode_datetime)
        # Atomic fsync is synchronous I/O; run it in a thread so a slow disk
        # cannot block the asyncio event loop for seconds and kill heartbeats
        # for live channels (Discord/WebSocket pings).
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, atomic_write_text, path, content)
        if put_signal is not None:
            result = put_signal()
            if isinstance(result, Awaitable):
                loop.create_task(result)


class DurableInboundQueue(DurableMessageQueue):
    """Durable inbound message queue."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace / "bus" / "inbound")

    def purge_for_session(self, session_key: str) -> int:
        """Delete durable inbound messages bound to *session_key*.

        Messages store ``channel``/``chat_id`` (the session key is derived as
        ``{channel}:{chat_id}``). This removes matching inbox and processing
        files so a deleted session is not recreated by ``recover()`` on the
        next gateway start.
        """
        removed = 0
        chat_id = session_key.split(":", 1)[1] if ":" in session_key else session_key
        for directory in (self.inbox_dir, self.processing_dir):
            for path in directory.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("chat_id") != chat_id:
                    continue
                try:
                    path.unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    logger.exception("Failed to purge durable inbound message {}", path)
        return removed

    async def publish(self, msg: InboundMessage) -> None:
        await self._publish(
            _inbound_to_dict(msg), put_signal=lambda: self._signal.put("published")
        )

    async def consume(self) -> InboundMessage:
        path = self._next_inbox_file()
        while path is None:
            # The signal is only a wake-up hint; the inbox is the source of
            # truth. If a signal fires while the inbox looks empty (e.g. a
            # burst is mid-write) keep waiting for one, but never treat a
            # signal as an obligation to claim a file -- that would orphan it.
            await self._signal.get()
            path = self._next_inbox_file()
        delivery_id, processing_path = self._claim(path)
        data = json.loads(processing_path.read_text(encoding="utf-8"))
        msg = _dict_to_inbound(data)
        msg._delivery_id = delivery_id  # type: ignore[attr-defined]
        self._processing[delivery_id] = processing_path
        return msg

    async def ack(self, msg: InboundMessage) -> None:
        delivery_id = getattr(msg, "_delivery_id", None)
        path = self._processing.pop(delivery_id, None) if isinstance(delivery_id, str) else None
        if path is not None:
            path.unlink(missing_ok=True)

    async def nack(self, msg: InboundMessage) -> None:
        delivery_id = getattr(msg, "_delivery_id", None)
        path = self._processing.pop(delivery_id, None) if isinstance(delivery_id, str) else None
        if path is None or not path.exists():
            return
        target = self.inbox_dir / path.name
        try:
            os.replace(path, target)
            self._signal.put_nowait("requeued")
        except OSError:
            logger.exception("Failed to requeue durable inbound message {}", delivery_id)

    def size(self) -> int:
        return len(list(self.inbox_dir.glob("*.json")))


class DurableOutboundQueue(DurableMessageQueue):
    """Durable outbound message queue."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace / "bus" / "outbound")

    async def publish(self, msg: OutboundMessage) -> None:
        await self._publish(
            _outbound_to_dict(msg), put_signal=lambda: self._signal.put("published")
        )

    async def consume(self) -> OutboundMessage:
        path = self._next_inbox_file()
        while path is None:
            await self._signal.get()
            path = self._next_inbox_file()
        delivery_id, processing_path = self._claim(path)
        data = json.loads(processing_path.read_text(encoding="utf-8"))
        msg = _dict_to_outbound(data)
        msg._delivery_id = delivery_id  # type: ignore[attr-defined]
        self._processing[delivery_id] = processing_path
        return msg

    async def ack(self, msg: OutboundMessage) -> None:
        delivery_id = getattr(msg, "_delivery_id", None)
        path = self._processing.pop(delivery_id, None) if isinstance(delivery_id, str) else None
        if path is not None:
            path.unlink(missing_ok=True)

    async def nack(self, msg: OutboundMessage) -> None:
        delivery_id = getattr(msg, "_delivery_id", None)
        path = self._processing.pop(delivery_id, None) if isinstance(delivery_id, str) else None
        if path is None or not path.exists():
            return
        target = self.inbox_dir / path.name
        try:
            os.replace(path, target)
            self._signal.put_nowait("requeued")
        except OSError:
            logger.exception("Failed to requeue durable outbound message {}", delivery_id)

    def size(self) -> int:
        return len(list(self.inbox_dir.glob("*.json")))
