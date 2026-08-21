"""Tests for the RLAIF observer hook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.hook import AgentRunHookContext
from nanobot.agent.rlaif.observer import (
    RlaifObservation,
    RlaifObserver,
    RlaifObserverHook,
    create_rlaif_observer_hook,
    make_rlaif_observer_factory,
)
from nanobot.agent.rlaif.trajectory import Trajectory
from nanobot.providers.base import LLMResponse, ToolCallRequest


class FakeProvider:
    def __init__(self, args: dict[str, Any]) -> None:
        self.args = args
        self.calls: list[dict[str, Any]] = []

    async def chat_with_retry(self, *, messages, model, **kwargs):
        self.calls.append({"messages": messages, "model": model, "kwargs": kwargs})
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="decide_evaluation",
                    arguments=self.args,
                )
            ],
        )


class TestRlaifObserver:
    @pytest.mark.asyncio
    async def test_decide_returns_observation(self) -> None:
        provider = FakeProvider(
            args={
                "should_evaluate": True,
                "task": "refactor foo",
                "reason": "function is too long",
                "confidence": 0.8,
            }
        )
        observer = RlaifObserver(provider, model="fake")
        traj = Trajectory("do stuff")
        result = await observer.decide(traj)
        assert isinstance(result, RlaifObservation)
        assert result.should_evaluate is True
        assert result.task == "refactor foo"
        assert result.confidence == 0.8


class TestCreateRlaifObserverHook:
    def test_disabled_when_enable_false(self, tmp_path: Path) -> None:
        provider = FakeProvider(args={})
        cfg = type("C", (), {"enable": False})()
        hook = create_rlaif_observer_hook(
            cfg, workspace=tmp_path, provider=provider, model="fake"
        )
        assert hook is None

    def test_disabled_when_observer_false(self, tmp_path: Path) -> None:
        provider = FakeProvider(args={})
        cfg = type("C", (), {"enable": True, "observer": False})()
        hook = create_rlaif_observer_hook(
            cfg, workspace=tmp_path, provider=provider, model="fake"
        )
        assert hook is None

    @pytest.mark.asyncio
    async def test_after_run_schedules_when_confident(self, tmp_path: Path) -> None:
        scheduled = []

        def schedule(coro) -> None:
            scheduled.append(coro)

        _ = type(
            "C",
            (),
            {
                "enable": True,
                "observer": True,
                "candidate_count": 2,
                "observer_critic_model": None,
                "test_command": None,
                "lint_command": None,
                "observer_min_confidence": 0.5,
            },
        )()
        provider = FakeProvider(
            args={
                "should_evaluate": True,
                "task": "fix bug",
                "reason": "bug",
                "confidence": 0.9,
            }
        )
        hook = RlaifObserverHook(
            workspace=tmp_path,
            provider=provider,
            model="fake",
            schedule_background=schedule,
            min_confidence=0.5,
        )
        await hook.before_run(AgentRunHookContext(messages=[{"role": "user", "content": "hi"}]))
        await hook.after_run(AgentRunHookContext(messages=[]))
        assert len(scheduled) == 1


class TestMakeRlaifObserverFactory:
    def test_factory_returns_hook(self, tmp_path: Path) -> None:
        from nanobot.agent.hook import AgentTurnHookContext

        provider = FakeProvider(args={})
        cfg = type("C", (), {"enable": True, "observer": True})()
        factory = make_rlaif_observer_factory(
            cfg,
            workspace=tmp_path,
            provider=provider,
            model="fake",
            schedule_background=lambda c: None,
            publish_outbound=lambda m: None,
        )
        hook = factory(AgentTurnHookContext(channel="webui", chat_id="abc"))
        assert isinstance(hook, RlaifObserverHook)
        assert hook._channel == "webui"
        assert hook._chat_id == "abc"
