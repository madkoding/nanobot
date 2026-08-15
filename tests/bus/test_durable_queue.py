"""Tests for the durable message bus backing store."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


@pytest.fixture
def bus(tmp_path: Path) -> MessageBus:
    return MessageBus(workspace=tmp_path)


@pytest.mark.asyncio
async def test_durable_inbound_roundtrip(bus: MessageBus, tmp_path: Path) -> None:
    msg = InboundMessage(
        channel="telegram",
        sender_id="u1",
        chat_id="c1",
        content="hello",
        metadata={"message_id": "m1"},
    )
    await bus.publish_inbound(msg)
    assert bus.inbound_size == 1

    consumed = await bus.consume_inbound()
    assert consumed.channel == "telegram"
    assert consumed.content == "hello"
    assert bus.inbound_size == 0

    await bus.ack_inbound(consumed)
    assert not any((tmp_path / "bus" / "inbound" / "processing").glob("*.json"))


@pytest.mark.asyncio
async def test_durable_inbound_survives_crash(bus: MessageBus, tmp_path: Path) -> None:
    msg = InboundMessage(channel="discord", sender_id="u2", chat_id="c2", content="hi")
    await bus.publish_inbound(msg)

    consumed = await bus.consume_inbound()
    assert consumed.content == "hi"
    # Simulate a crash before ack: the message is still in processing.

    bus2 = MessageBus(workspace=tmp_path)
    recovered = await bus2.recover()
    assert recovered == 1
    assert bus2.inbound_size == 1

    re_consumed = await bus2.consume_inbound()
    assert re_consumed.content == "hi"

    await bus2.ack_inbound(re_consumed)
    assert bus2.inbound_size == 0


@pytest.mark.asyncio
async def test_durable_inbound_nack_requeues(bus: MessageBus, tmp_path: Path) -> None:
    msg = InboundMessage(channel="slack", sender_id="u3", chat_id="c3", content="retry me")
    await bus.publish_inbound(msg)

    consumed = await bus.consume_inbound()
    await bus.nack_inbound(consumed)
    assert bus.inbound_size == 1

    re_consumed = await bus.consume_inbound()
    assert re_consumed.content == "retry me"
    await bus.ack_inbound(re_consumed)
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_durable_inbound_recover_filters_by_active_session(
    bus: MessageBus, tmp_path: Path
) -> None:
    """recover() only requeues messages for active sessions; others are dropped."""
    active = InboundMessage(channel="discord", sender_id="u1", chat_id="c-active", content="keep")
    stale = InboundMessage(channel="telegram", sender_id="u2", chat_id="c-deleted", content="drop")
    await bus.publish_inbound(active)
    await bus.publish_inbound(stale)

    # Both are claimed but not acked, simulating a crash before completion.
    await bus.consume_inbound()
    await bus.consume_inbound()

    bus2 = MessageBus(workspace=tmp_path)
    recovered = await bus2.recover(active_session_keys={"discord:c-active"})
    assert recovered == 1
    assert bus2.inbound_size == 1

    re_consumed = await bus2.consume_inbound()
    assert re_consumed.content == "keep"
    await bus2.ack_inbound(re_consumed)
    assert bus2.inbound_size == 0



@pytest.mark.asyncio
async def test_durable_outbound_roundtrip(bus: MessageBus, tmp_path: Path) -> None:
    msg = OutboundMessage(channel="telegram", chat_id="c1", content="reply")
    await bus.publish_outbound(msg)
    assert bus.outbound_size == 1

    consumed = await bus.consume_outbound()
    assert consumed.content == "reply"
    await bus.ack_outbound(consumed)
    assert bus.outbound_size == 0


@pytest.mark.asyncio
async def test_durable_outbound_fifo_order(bus: MessageBus) -> None:
    # Streamed deltas must be consumed in publication order; the on-disk
    # filenames are random UUIDs, so ordering must come from mtime.
    for i in range(8):
        await bus.publish_outbound(OutboundMessage(channel="websocket", chat_id="c1", content=f"d{i}"))
    consumed = [await bus.consume_outbound() for _ in range(8)]
    assert [m.content for m in consumed] == [f"d{i}" for i in range(8)]
    for m in consumed:
        await bus.ack_outbound(m)


@pytest.mark.asyncio
async def test_in_memory_bus_is_not_durable(tmp_path: Path) -> None:
    bus = MessageBus()
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="x")
    await bus.publish_inbound(msg)
    assert bus.inbound_size == 1
    consumed = await bus.consume_inbound()
    assert consumed is not None
    await bus.ack_inbound(consumed)
    await bus.nack_inbound(consumed)
    assert not (tmp_path / "bus").exists()


@pytest.mark.asyncio
async def test_durable_queue_preserves_timestamp(bus: MessageBus) -> None:
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).replace(microsecond=0)
    msg = InboundMessage(
        channel="telegram",
        sender_id="u1",
        chat_id="c1",
        content="hello",
        timestamp=ts,
    )
    await bus.publish_inbound(msg)
    consumed = await bus.consume_inbound()
    assert consumed.timestamp.replace(microsecond=0) == ts
    await bus.ack_inbound(consumed)


@pytest.mark.asyncio
async def test_durable_publish_wakes_consumer(bus: MessageBus) -> None:
    consumed = None

    async def consumer() -> None:
        nonlocal consumed
        consumed = await bus.consume_inbound()

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="wake")
    await bus.publish_inbound(msg)
    await asyncio.wait_for(task, timeout=1.0)
    assert consumed is not None
    assert consumed.content == "wake"


@pytest.mark.asyncio
async def test_durable_consume_picks_up_signalless_file(bus: MessageBus, tmp_path: Path) -> None:
    """A file in inbox with no pending signal (signal desync from a crash or
    recovery race) must still be consumed; the inbox is the source of truth."""
    import json as _json

    outbound_dir = tmp_path / "bus" / "outbound" / "inbox"
    outbound_dir.mkdir(parents=True, exist_ok=True)
    (outbound_dir / "orphan.json").write_text(
        _json.dumps({"channel": "websocket", "chat_id": "c1", "content": "orphan"})
    )
    msg = await asyncio.wait_for(bus.consume_outbound(), timeout=2)
    assert msg.content == "orphan"
    await bus.ack_outbound(msg)


@pytest.mark.asyncio
async def test_durable_outbound_tolerates_unknown_fields(bus: MessageBus, tmp_path: Path) -> None:
    """A durable outbound message written by an older nanobot version may carry
    fields the current OutboundMessage no longer accepts (e.g. `rich`). These
    must be dropped instead of crashing the outbound dispatcher."""
    import json as _json

    outbound_dir = tmp_path / "bus" / "outbound" / "inbox"
    outbound_dir.mkdir(parents=True, exist_ok=True)
    (outbound_dir / "stale.json").write_text(
        _json.dumps(
            {
                "channel": "telegram",
                "chat_id": "15710279",
                "content": "stale durable message",
                "totally_unknown_field": "x",
                "another_unknown": None,
            }
        )
    )
    msg = await asyncio.wait_for(bus.consume_outbound(), timeout=2)
    assert msg.content == "stale durable message"
    assert not hasattr(msg, "totally_unknown_field")
    assert not hasattr(msg, "another_unknown")
    await bus.ack_outbound(msg)


@pytest.mark.asyncio
async def test_coalesced_delta_keeps_delivery_id() -> None:
    """replace_outbound_event must not drop the durable-queue ack id, or
    coalesced stream deltas leak their processing file forever."""
    from nanobot.bus.outbound_events import (
        StreamDeltaEvent,
        replace_outbound_event,
    )

    msg = OutboundMessage(channel="websocket", chat_id="c1", content="a")
    msg._delivery_id = "abc123"
    merged = replace_outbound_event(
        msg, StreamDeltaEvent(content="ab", stream_id="s1"), content="ab"
    )
    assert merged._delivery_id == "abc123"
