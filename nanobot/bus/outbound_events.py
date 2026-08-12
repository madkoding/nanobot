"""Typed outbound events carried by :class:`OutboundMessage`.

The message bus still transports :class:`nanobot.bus.events.OutboundMessage`
because channels need chat routing fields. Runtime/UI semantics live on the
message's explicit ``event`` field rather than in reserved metadata flags.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from nanobot.bus.events import OutboundMessage


class OutboundEvent:
    """Marker base for internal outbound runtime events."""


@dataclass(frozen=True)
class ProgressEvent(OutboundEvent):
    content: str = ""
    tool_hint: bool = False
    reasoning: bool = False
    reasoning_delta: bool = False
    reasoning_end: bool = False
    stream_id: str | None = None
    tool_events: list[dict[str, Any]] | None = None
    file_edit_events: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class RetryWaitEvent(OutboundEvent):
    content: str = ""


@dataclass(frozen=True)
class StreamDeltaEvent(OutboundEvent):
    content: str = ""
    stream_id: str | None = None


@dataclass(frozen=True)
class StreamEndEvent(OutboundEvent):
    content: str = ""
    stream_id: str | None = None
    resuming: bool = False


@dataclass(frozen=True)
class StreamedResponseEvent(OutboundEvent):
    pass


@dataclass(frozen=True)
class TurnEndEvent(OutboundEvent):
    latency_ms: int | None = None
    goal_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class GoalStatusEvent(OutboundEvent):
    status: str
    started_at: float | None = None


@dataclass(frozen=True)
class GoalStateSyncEvent(OutboundEvent):
    goal_state: dict[str, Any]


@dataclass(frozen=True)
class SessionUpdatedEvent(OutboundEvent):
    scope: str | None = None


@dataclass(frozen=True)
class RuntimeModelUpdatedEvent(OutboundEvent):
    model: str | None
    model_preset: str | None = None


@dataclass(frozen=True)
class TurnModelUpdatedEvent(OutboundEvent):
    """The fallback model currently handling one chat turn."""

    model: str


@dataclass(frozen=True)
class AutomationUpdateEvent(OutboundEvent):
    """Live status for an automation (cron / local trigger) turn."""

    kind: str
    label: str | None = None
    turn_id: str | None = None
    status: str = "running"
    error: str | None = None


def outbound_message_for_event(
    *,
    channel: str,
    chat_id: str,
    event: OutboundEvent,
    content: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OutboundMessage:
    """Build an :class:`OutboundMessage` for a typed event."""

    return OutboundMessage(
        channel=channel,
        chat_id=chat_id,
        content=_event_content(event) if content is None else content,
        event=event,
        metadata=dict(metadata or {}),
    )


def outbound_event_from_message(msg: OutboundMessage) -> OutboundEvent | None:
    """Return the typed outbound event carried by *msg*, if any."""

    return msg.event


def replace_outbound_event(
    msg: OutboundMessage,
    event: OutboundEvent,
    *,
    content: str | None = None,
) -> OutboundMessage:
    """Return *msg* with a new event and optional content."""

    result = replace(
        msg,
        content=_event_content(event) if content is None else content,
        event=event,
    )
    delivery_id = getattr(msg, "_delivery_id", None)
    if delivery_id is not None:
        result._delivery_id = delivery_id
    return result


def _event_content(event: OutboundEvent) -> str:
    if isinstance(event, ProgressEvent | RetryWaitEvent | StreamDeltaEvent | StreamEndEvent):
        return event.content
    return ""
