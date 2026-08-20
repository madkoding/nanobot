import pytest

from nanobot.agent.hook import AgentRunHookContext
from nanobot.agent.rlaif.observer import (
    RlaifBackgroundEvaluator,
    RlaifObserver,
    RlaifObserverHook,
)
from nanobot.bus.events import OutboundMessage
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class DummyProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self._responses = list(responses)

    async def chat(self, *args, **kwargs) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


def _decide_tool_call(should_evaluate: bool, task: str, confidence: float) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(
                id="decide_1",
                name="decide_evaluation",
                arguments={
                    "should_evaluate": should_evaluate,
                    "task": task,
                    "reason": "test reason",
                    "confidence": confidence,
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_observer_decide_positive(tmp_path) -> None:
    provider = DummyProvider([_decide_tool_call(True, "fix logging", 0.8)])
    obs = RlaifObserver(provider, "m")
    from nanobot.agent.rlaif.trajectory import Trajectory

    traj = Trajectory("say hello")
    result = await obs.decide(traj)
    assert result.should_evaluate is True
    assert result.task == "fix logging"
    assert result.confidence == 0.8


@pytest.mark.asyncio
async def test_observer_decide_negative(tmp_path) -> None:
    provider = DummyProvider([_decide_tool_call(False, "", 0.2)])
    obs = RlaifObserver(provider, "m")
    from nanobot.agent.rlaif.trajectory import Trajectory

    traj = Trajectory("say hello")
    result = await obs.decide(traj)
    assert result.should_evaluate is False


@pytest.mark.asyncio
async def test_observer_hook_disabled_by_config(tmp_path) -> None:
    class Cfg:
        enable = False

    hook = RlaifObserverHook.from_config(Cfg(), tmp_path, DummyProvider([]), "m")
    assert hook is None


@pytest.mark.asyncio
async def test_observer_hook_disabled_when_observer_false(tmp_path) -> None:
    class Cfg:
        enable = True
        observer = False

    hook = RlaifObserverHook.from_config(Cfg(), tmp_path, DummyProvider([]), "m")
    assert hook is None


@pytest.mark.asyncio
async def test_observer_hook_schedules_background_and_announces(tmp_path) -> None:
    class Cfg:
        enable = True
        observer = True
        observer_critic_model = None
        candidate_count = 2
        test_command = None
        lint_command = None
        observer_min_confidence = 0.5

    scheduled: list = []
    outbound: list[OutboundMessage] = []

    def _schedule(coro):
        scheduled.append(coro)

    async def _publish(msg):
        outbound.append(msg)

    hook = RlaifObserverHook.from_config(
        Cfg(),
        tmp_path,
        DummyProvider([_decide_tool_call(True, "refactor", 0.9)]),
        "m",
        schedule_background=_schedule,
        publish_outbound=_publish,
        channel="websocket",
        chat_id="test-chat",
    )
    assert hook is not None
    await hook.before_run(AgentRunHookContext(messages=[]))
    await hook.after_run(AgentRunHookContext(messages=[], final_content="done"))
    assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_observer_hook_skips_low_confidence(tmp_path) -> None:
    class Cfg:
        enable = True
        observer = True
        observer_critic_model = None
        candidate_count = 2
        test_command = None
        lint_command = None
        observer_min_confidence = 0.9

    scheduled: list = []

    def _schedule(coro):
        scheduled.append(coro)

    hook = RlaifObserverHook.from_config(
        Cfg(),
        tmp_path,
        DummyProvider([_decide_tool_call(True, "refactor", 0.5)]),
        "m",
        schedule_background=_schedule,
    )
    assert hook is not None
    await hook.before_run(AgentRunHookContext(messages=[]))
    await hook.after_run(AgentRunHookContext(messages=[], final_content="done"))
    assert len(scheduled) == 0


@pytest.mark.asyncio
async def test_observer_hook_announce_routing(tmp_path) -> None:
    class Cfg:
        enable = True
        observer = True
        observer_min_confidence = 0.0

    outbound: list[OutboundMessage] = []

    async def _publish(msg):
        outbound.append(msg)

    hook = RlaifObserverHook.from_config(
        Cfg(),
        tmp_path,
        DummyProvider([_decide_tool_call(True, "task", 1.0)]),
        "m",
        publish_outbound=_publish,
        channel="websocket",
        chat_id="chat-1",
    )
    assert hook is not None
    await hook.before_run(AgentRunHookContext(messages=[]))
    await hook._announce("hello")
    assert len(outbound) == 1
    assert outbound[0].channel == "websocket"
    assert outbound[0].chat_id == "chat-1"
    assert outbound[0].metadata.get("rlaif_background") is True


@pytest.mark.asyncio
async def test_background_evaluator_no_candidates(tmp_path) -> None:
    provider = DummyProvider([])
    ev = RlaifBackgroundEvaluator(
        workspace=tmp_path,
        provider=provider,
        model="m",
        test_command=["python", "-m", "pytest", "-q"],
        lint_command=["python", "-m", "ruff", "check", "."],
    )
    report = await ev.run("task")
    assert "no valid candidates" in report
