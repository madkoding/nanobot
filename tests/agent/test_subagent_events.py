"""Tests for the SubagentManager live event hook + HTTP status fetch."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import (
    SUBAGENT_STATUS_TTL_S,
    SubagentManager,
)
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.utils.llm_runtime import LLMRuntime


def _runtime(model: str = "test-model") -> LLMRuntime:
    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings(temperature=0.1, max_tokens=1024)
    return LLMRuntime.capture(provider, model, context_window_tokens=128_000)


def _manager(tmp_path: Path) -> SubagentManager:
    return SubagentManager(
        workspace=tmp_path,
        bus=MessageBus(),
        max_tool_result_chars=16_000,
    )


async def _drain(sm: SubagentManager) -> None:
    tasks = list(sm._running_tasks.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)


class TestEventCallback:
    @pytest.mark.asyncio
    async def test_event_callback_receives_done_payload(self, tmp_path: Path) -> None:
        sm = _manager(tmp_path)
        sm.runner.run = AsyncMock(
            return_value=AgentRunResult(
                final_content="ok",
                messages=[],
                stop_reason="completed",
            )
        )

        captured: list[dict] = []

        async def cb(payload: dict) -> None:
            captured.append(payload)

        sm.set_event_callback(cb)

        await sm.spawn("do the thing", runtime=_runtime(), label="X", origin_chat_id="c1")
        await _drain(sm)

        assert captured, "expected at least one event for the finished subagent"
        done = [c for c in captured if c.get("event") == "done"]
        assert len(done) == 1
        payload = done[0]
        assert payload["label"] == "X"
        assert payload["result"] == "ok"
        assert payload["phase"] == "done"
        assert payload["chat_id"] == "c1"

    @pytest.mark.asyncio
    async def test_event_callback_receives_error_payload(self, tmp_path: Path) -> None:
        sm = _manager(tmp_path)
        sm.runner.run = AsyncMock(side_effect=RuntimeError("boom"))

        captured: list[dict] = []

        async def cb(payload: dict) -> None:
            captured.append(payload)

        sm.set_event_callback(cb)

        await sm.spawn("will fail", runtime=_runtime(), origin_chat_id="c2")
        await _drain(sm)

        errors = [c for c in captured if c.get("event") == "error"]
        assert len(errors) == 1
        assert "boom" in (errors[0].get("error") or "")

    @pytest.mark.asyncio
    async def test_event_callback_fault_does_not_break_subagent(self, tmp_path: Path) -> None:
        sm = _manager(tmp_path)
        sm.runner.run = AsyncMock(
            return_value=AgentRunResult(
                final_content="ok",
                messages=[],
                stop_reason="completed",
            )
        )

        async def bad_cb(_payload: dict) -> None:
            raise RuntimeError("callback crash")

        sm.set_event_callback(bad_cb)
        await sm.spawn("task", runtime=_runtime(), origin_chat_id="c3")
        await _drain(sm)
        # Subagent still completed despite callback raising; snapshot survives
        # the cleanup callback via the TTL window.
        all_ids = list(sm._task_statuses) + list(sm._finished_statuses)
        tid = all_ids[0]
        snapshot = sm.get_status(tid)
        assert snapshot is not None
        assert snapshot.result == "ok"


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_get_status_returns_live_snapshot(self, tmp_path: Path) -> None:
        sm = _manager(tmp_path)
        block = asyncio.Event()

        async def slow_run(_spec):
            await block.wait()
            return AgentRunResult(final_content="ok", messages=[], stop_reason="completed")

        sm.runner.run = slow_run
        await sm.spawn("task", runtime=_runtime(), origin_chat_id="c4")
        tid = next(iter(sm._task_statuses.keys()))
        snapshot = sm.get_status(tid)
        assert snapshot is not None
        assert snapshot.task_id == tid
        assert snapshot.phase == "initializing"
        block.set()
        await _drain(sm)
        # After completion the snapshot is still fetchable inside the TTL window.
        snapshot = sm.get_status(tid)
        assert snapshot is not None
        assert snapshot.phase == "done"
        assert snapshot.result == "ok"

    @pytest.mark.asyncio
    async def test_get_status_evicts_after_ttl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sm = _manager(tmp_path)
        sm.runner.run = AsyncMock(
            return_value=AgentRunResult(
                final_content="ok",
                messages=[],
                stop_reason="completed",
            )
        )
        await sm.spawn("task", runtime=_runtime())
        await _drain(sm)
        tid = next(iter(sm._finished_statuses.keys()))
        # Force the snapshot to look finished long ago.
        sm._finished_statuses[tid].finished_at = time.monotonic() - SUBAGENT_STATUS_TTL_S - 1
        # First call evicts; second call returns None.
        assert sm.get_status(tid) is None
        assert sm.get_status(tid) is None
        assert tid not in sm._finished_statuses
