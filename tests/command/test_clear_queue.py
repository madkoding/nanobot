"""Tests for the /clear-queue command (clear unprocessed inbound messages)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.builtin import cmd_clear_queue
from nanobot.command.router import CommandContext


def _inbound(chat_id: str, content: str, override: str | None = None) -> InboundMessage:
    return InboundMessage(
        channel="telegram",
        sender_id="42",
        chat_id=chat_id,
        content=content,
        session_key_override=override,
    )


def _ctx(key: str, loop: object) -> CommandContext:
    return CommandContext(
        msg=MagicMock(channel="telegram", chat_id="55", metadata={}),
        session=None,
        key=key,
        raw="/clear-queue",
        loop=loop,
    )


@pytest.mark.asyncio
async def test_cmd_clear_queue_drains_pending_and_bus():
    """Pending queue + bus purge are both counted in the response."""
    mock_loop = MagicMock()
    mock_loop._pending_queues = {}
    pending = asyncio.Queue()
    await pending.put("msg1")
    await pending.put("msg2")
    mock_loop._pending_queues["telegram:55"] = pending
    mock_loop.bus = MagicMock()
    mock_loop.bus.purge_inbound_for_session = AsyncMock(return_value=3)

    result = await cmd_clear_queue(_ctx("telegram:55", mock_loop))

    assert isinstance(result, OutboundMessage)
    assert "Cleared 5 queued message(s)." in result.content
    assert "telegram:55" not in mock_loop._pending_queues
    mock_loop.bus.purge_inbound_for_session.assert_awaited_once_with("telegram:55")


@pytest.mark.asyncio
async def test_cmd_clear_queue_empty():
    """No pending queue and nothing on the bus -> 'No queued messages.'"""
    mock_loop = MagicMock()
    mock_loop._pending_queues = {}
    mock_loop.bus = MagicMock()
    mock_loop.bus.purge_inbound_for_session = AsyncMock(return_value=0)

    result = await cmd_clear_queue(_ctx("telegram:55", mock_loop))

    assert "No queued messages." in result.content


@pytest.mark.asyncio
async def test_cmd_clear_queue_without_loop():
    """A missing loop must not crash the handler."""
    result = await cmd_clear_queue(_ctx("telegram:55", None))
    assert "No queued messages." in result.content


@pytest.mark.asyncio
async def test_message_bus_purge_inbound_memory():
    """In-memory bus: only the target session's messages are removed."""
    bus = MessageBus()
    await bus.publish_inbound(_inbound("55", "a"))
    await bus.publish_inbound(_inbound("55", "b"))
    await bus.publish_inbound(_inbound("99", "other"))

    removed = await bus.purge_inbound_for_session("telegram:55")

    assert removed == 2
    assert bus.inbound.qsize() == 1
    remaining = bus.inbound.get_nowait()
    assert remaining.chat_id == "99"


@pytest.mark.asyncio
async def test_durable_purge_matches_topic_override(tmp_path):
    """Durable purge removes messages whose session_key_override matches exactly."""
    from nanobot.bus.durable_queue import DurableInboundQueue

    queue = DurableInboundQueue(tmp_path)
    await queue.publish(_inbound("55", "topic msg", override="telegram:55:topic:7"))
    await queue.publish(_inbound("55", "base msg"))
    await queue.publish(_inbound("99", "other chat"))

    removed = queue.purge_for_session("telegram:55:topic:7")

    assert removed == 1
    assert queue.size() == 2
