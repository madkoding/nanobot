import pytest

from nanobot.agent.rlaif.critic import RlaifCritic
from nanobot.agent.rlaif.dataset import RlaifDataset, RlaifPreference
from nanobot.agent.rlaif.harness import PatchHarnessResult
from nanobot.agent.rlaif.trajectory import Trajectory, TurnStep
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


def _score_tool_call(score: float, issues: list[str] | None = None, reason: str = "ok") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(
                id="score_1",
                name="rate_solution",
                arguments={"score": score, "issues": issues or [], "reason": reason},
            )
        ],
    )


def _compare_tool_call(
    winner: str, score_a: float, score_b: float, reason: str = "A is better"
) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(
                id="compare_1",
                name="choose_solution",
                arguments={
                    "winner": winner,
                    "score_a": score_a,
                    "score_b": score_b,
                    "reason": reason,
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_critic_score_returns_score_and_issues() -> None:
    provider = DummyProvider([_score_tool_call(4.0, ["minor typo"], "good overall")])
    critic = RlaifCritic(provider, "m")
    result = await critic.score("task", {"patch": "diff"})
    assert result.score == 4.0
    assert result.issues == ["minor typo"]
    assert result.reason == "good overall"


@pytest.mark.asyncio
async def test_critic_score_clamps_out_of_range() -> None:
    provider = DummyProvider([_score_tool_call(10.0, [], "too high")])
    critic = RlaifCritic(provider, "m")
    result = await critic.score("task", {"patch": "diff"})
    assert result.score == 5.0


@pytest.mark.asyncio
async def test_critic_compare_returns_winner() -> None:
    provider = DummyProvider([_compare_tool_call("A", 4.0, 3.0)])
    critic = RlaifCritic(provider, "m")
    result = await critic.compare("task", {"patch": "a"}, {"patch": "b"})
    assert result.winner == "A"
    assert result.score == 4.0
    assert result.reason == "A is better"


@pytest.mark.asyncio
async def test_critic_compare_invalid_winner_becomes_none() -> None:
    provider = DummyProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="x",
                        name="choose_solution",
                        arguments={"winner": "Z", "score_a": 3.0, "score_b": 3.0, "reason": "meh"},
                    )
                ],
            )
        ]
    )
    critic = RlaifCritic(provider, "m")
    result = await critic.compare("task", {"patch": "a"}, {"patch": "b"})
    assert result.winner is None


@pytest.mark.asyncio
async def test_critic_score_falls_back_on_provider_error() -> None:
    class FailingProvider(DummyProvider):
        async def chat(self, *args, **kwargs) -> LLMResponse:
            raise RuntimeError("boom")

    critic = RlaifCritic(FailingProvider([]), "m")
    result = await critic.score("task", {"patch": "diff"})
    assert result.score == 1.0
    assert result.issues == ["critic did not call tool"]


def test_dataset_append_and_read(tmp_path) -> None:
    path = tmp_path / "prefs.jsonl"
    ds = RlaifDataset(path)
    pref = RlaifPreference(
        prompt="fix bug",
        chosen={"patch": "good"},
        rejected={"patch": "bad"},
        score_chosen=4.5,
        score_rejected=2.0,
        reason="better",
        task="fix bug",
    )
    ds.append(pref)
    assert ds.count() == 1
    all_prefs = ds.read_all()
    assert len(all_prefs) == 1
    assert all_prefs[0].score_chosen == 4.5


def test_dataset_empty_count(tmp_path) -> None:
    ds = RlaifDataset(tmp_path / "missing.jsonl")
    assert ds.count() == 0
    assert ds.read_all() == []


def test_trajectory_messages_merges_tool_results() -> None:
    traj = Trajectory("task")
    traj.add_step(TurnStep(role="user", content="do it"))
    traj.add_step(TurnStep(role="assistant", tool_calls=[{"name": "exec"}]))
    traj.add_step(
        TurnStep(role="user", tool_results=[{"name": "exec", "content": "ok"}])
    )
    msgs = traj.messages
    assert msgs[-1]["role"] == "user"
    assert "ok" in msgs[-1]["content"]


def test_harness_result_passed_and_bonus() -> None:
    result = PatchHarnessResult(patch="p", summary="s", test_passed=True, lint_passed=True)
    assert result.passed is True
    assert result.score_bonus == 2.0


def test_harness_result_failed_bonus() -> None:
    result = PatchHarnessResult(patch="p", summary="s", test_passed=False, lint_passed=False)
    assert result.passed is False
    assert result.score_bonus == 0.0
