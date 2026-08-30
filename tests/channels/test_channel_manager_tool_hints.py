"""Tests for ChannelManager gating of tool-hint and tool-finish progress events.

The dispatcher is the single gate deciding whether a ``ProgressEvent`` reaches
a channel's ``send()``: tool hints go through only when the destination
channel has ``send_tool_hints`` enabled. This mirrors the per-channel config
(``sendToolHints``) and keeps tool traces out of chat channels like WhatsApp
while leaving them on for channels that render them as structured traces
(e.g. the WebUI over the websocket channel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


class _MockChannel(BaseChannel):
    name = "mock"
    display_name = "Mock"

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self._send_mock = AsyncMock()

    async def start(self):  # pragma: no cover - not exercised
        pass

    async def stop(self):  # pragma: no cover - not exercised
        pass

    async def send(self, msg):
        return await self._send_mock(msg)


@pytest.fixture
def manager() -> ChannelManager:
    config = Config.model_validate({"channels": {"websocket": {"enabled": False}}})
    mgr = ChannelManager(config, MessageBus())
    mgr.channels["mock"] = _MockChannel({}, mgr.bus)
    return mgr


def test_tool_hint_gate_follows_channel_flag(manager):
    channel = manager.channels["mock"]
    channel.send_progress = True

    channel.send_tool_hints = True
    assert manager._should_send_progress("mock", tool_hint=True) is True

    channel.send_tool_hints = False
    assert manager._should_send_progress("mock", tool_hint=True) is False


def test_tool_hint_gate_rejects_unknown_channel(manager):
    assert manager._should_send_progress("ghost", tool_hint=True) is False


def test_send_progress_follows_separate_flag(manager):
    channel = manager.channels["mock"]
    channel.send_tool_hints = False

    channel.send_progress = True
    assert manager._should_send_progress("mock", tool_hint=False) is True

    channel.send_progress = False
    assert manager._should_send_progress("mock", tool_hint=False) is False
