"""Tests for the BaseChannel send-failure contract."""

from __future__ import annotations

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel


class _FailingChannel(BaseChannel):
    name = "failing"

    def __init__(self, config, bus):
        super().__init__(config, bus)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, msg: OutboundMessage) -> None:
        self._require_ready()
        if msg.chat_id == "boom":
            raise RuntimeError("delivery failed")


def test_require_ready_raises_when_not_running() -> None:
    channel = _FailingChannel({}, MessageBus())
    assert channel._running is False
    with pytest.raises(RuntimeError, match="channel not started"):
        channel._require_ready()


@pytest.mark.asyncio
async def test_send_raises_when_not_running() -> None:
    channel = _FailingChannel({}, MessageBus())
    with pytest.raises(RuntimeError, match="channel not started"):
        await channel.send(OutboundMessage(channel="failing", chat_id="ok", content="hi"))


@pytest.mark.asyncio
async def test_send_raises_on_delivery_failure() -> None:
    channel = _FailingChannel({}, MessageBus())
    channel._running = True
    with pytest.raises(RuntimeError, match="delivery failed"):
        await channel.send(OutboundMessage(channel="failing", chat_id="boom", content="hi"))


@pytest.mark.asyncio
async def test_send_succeeds_when_ready_and_delivery_ok() -> None:
    channel = _FailingChannel({}, MessageBus())
    channel._running = True
    await channel.send(OutboundMessage(channel="failing", chat_id="ok", content="hi"))
