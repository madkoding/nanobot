"""Regression tests for the channel watchdog (manager).

Covers the two failure modes that left the Telegram channel down for ~2h20m
on 2026-08-19 (see docs/spec-fix-watchdog-canales.md):

1. A healthy but idle channel was restarted every 60s in an infinite loop
   because ``_start_channel`` never reset ``last_activity_at``.
2. A channel whose ``start()`` failed (e.g. network timeout) was never
   retried: the watchdog skipped tasks that were done, leaving the channel
   dead until a manual gateway restart or WebUI toggle.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from types import SimpleNamespace

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels import manager as manager_module
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


def _noop_logger() -> SimpleNamespace:
    """Stub for the module logger so failing-start tests don't spam tracebacks."""
    return SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )


def _make_manager(monkeypatch) -> ChannelManager:
    monkeypatch.setattr(
        "nanobot.channels.registry.discover_plugins",
        lambda enabled_names=None: {},
    )
    config = Config.model_validate({"channels": {"websocket": {"enabled": False}}})
    return ChannelManager(config, MessageBus())


class _SilentChannel(BaseChannel):
    """Healthy channel that stays alive but never touches last_activity_at."""

    name = "silent"
    display_name = "Silent"

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.started = asyncio.Event()
        self.start_count = 0
        self._stop_event = asyncio.Event()

    async def start(self):
        self.start_count += 1
        self._running = True
        self.started.set()
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

    async def stop(self):
        self._running = False
        self._stop_event.set()

    async def send(self, msg):  # pragma: no cover - not used by these tests
        raise AssertionError("send should not be called")


class _FailingChannel(BaseChannel):
    """Channel whose start() always fails (simulated network timeout)."""

    name = "fail"
    display_name = "Fail"

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.start_count = 0

    async def start(self):
        self.start_count += 1
        raise RuntimeError("simulated start failure")

    async def stop(self):
        pass

    async def send(self, msg):  # pragma: no cover - not used by these tests
        raise AssertionError("send should not be called")


def _fast_watchdog(monkeypatch) -> None:
    """Shrink watchdog timings so tests run in milliseconds."""
    monkeypatch.setattr(manager_module, "WATCHDOG_INTERVAL_S", 0.02)
    monkeypatch.setattr(manager_module, "WATCHDOG_IDLE_S", 0.05)
    monkeypatch.setattr(manager_module, "WATCHDOG_GRACE_S", 0.01)
    if hasattr(manager_module, "WATCHDOG_RETRY_INTERVAL_S"):
        monkeypatch.setattr(manager_module, "WATCHDOG_RETRY_INTERVAL_S", 0.02)


async def _run_watchdog(manager: ChannelManager, seconds: float) -> None:
    wd = asyncio.create_task(manager._watchdog_loop())
    await asyncio.sleep(seconds)
    wd.cancel()
    with suppress(asyncio.CancelledError):
        await wd


@pytest.mark.asyncio
async def test_watchdog_restart_resets_liveness_timer(monkeypatch) -> None:
    """A healthy-but-idle channel is restarted once, then left alone.

    Regression: before the fix, _start_channel never reset last_activity_at,
    so the watchdog restarted the channel every cycle (infinite loop, ~13k
    "live but silent" warnings in the user's gateway log).
    """
    _fast_watchdog(monkeypatch)
    monkeypatch.setattr(manager_module, "logger", _noop_logger())
    manager = _make_manager(monkeypatch)

    channel = _SilentChannel({"enabled": True}, manager.bus)
    manager.channels["silent"] = channel
    manager._channel_owners["silent"] = "silent"
    manager._channel_tasks["silent"] = asyncio.create_task(
        manager._start_channel("silent", channel)
    )
    await asyncio.wait_for(channel.started.wait(), timeout=1)

    # Simulate a healthy channel that has been idle for a long time.
    channel.last_activity_at = time.monotonic() - 1000

    await _run_watchdog(manager, 0.2)

    # The watchdog restarted it once...
    assert channel.start_count >= 2
    # ...and the restart reset the liveness timer (fix).
    assert channel.last_activity_at == 0.0

    # No infinite loop: the restart count stabilizes.
    count_after = channel.start_count
    await asyncio.sleep(0.2)
    assert channel.start_count == count_after


@pytest.mark.asyncio
async def test_watchdog_retries_channel_whose_start_failed(monkeypatch) -> None:
    """A channel whose start() failed is retried automatically.

    Regression: before the fix, the watchdog skipped done tasks
    (``if task is None or task.done(): continue``), so a channel that failed
    to start (e.g. TimedOut in getMe) stayed dead until a manual restart.
    """
    _fast_watchdog(monkeypatch)
    monkeypatch.setattr(manager_module, "logger", _noop_logger())
    manager = _make_manager(monkeypatch)

    channel = _FailingChannel({"enabled": True}, manager.bus)
    manager.channels["fail"] = channel
    manager._channel_owners["fail"] = "fail"
    manager._channel_tasks["fail"] = asyncio.create_task(
        manager._start_channel("fail", channel)
    )
    await asyncio.sleep(0.05)

    assert "fail" in manager._channel_errors
    assert manager._channel_tasks["fail"].done()

    await _run_watchdog(manager, 0.2)

    # The watchdog retried the failed channel instead of skipping it.
    assert channel.start_count >= 2
