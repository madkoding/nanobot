"""RLAIF observer: watches agent turns and schedules background self-improvement tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.hook import (
    AgentHook,
    AgentHookContext,
    AgentRunHookContext,
)
from nanobot.agent.rlaif.critic import RlaifCritic
from nanobot.agent.rlaif.dataset import RlaifDataset, RlaifPreference
from nanobot.agent.rlaif.harness import PatchHarness
from nanobot.agent.rlaif.trajectory import Trajectory, TurnStep
from nanobot.bus.events import OutboundMessage

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


@dataclass
class RlaifObservation:
    """Decision made by the observer after a turn."""

    should_evaluate: bool
    task: str = ""
    reason: str = ""
    confidence: float = 0.0


class RlaifObserver:
    """Decides whether a completed turn is worth running RLAIF over."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    _DECIDE_TOOL = [
        {
            "type": "function",
            "function": {
                "name": "decide_evaluation",
                "description": (
                    "Decide whether the last agent turn exposed a concrete "
                    "self-improvement opportunity in its own code or behavior."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "should_evaluate": {
                            "type": "boolean",
                            "description": (
                                "True if there is a concrete, bounded improvement "
                                "the agent could try (e.g. fix a bug, refactor a "
                                "function, improve a prompt)."
                            ),
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "If true, a short task description for the RLAIF "
                                "candidate generator. Empty if false."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": "One-sentence rationale.",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence in the decision.",
                        },
                    },
                    "required": ["should_evaluate", "task", "reason", "confidence"],
                },
            },
        }
    ]

    async def decide(self, trajectory: Trajectory) -> RlaifObservation:
        messages = self._build_messages(trajectory)
        try:
            response = await self._provider.chat_with_retry(
                messages=messages,
                tools=self._DECIDE_TOOL,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                tool_choice={
                    "type": "function",
                    "function": {"name": "decide_evaluation"},
                },
            )
            if not response.has_tool_calls:
                return RlaifObservation(should_evaluate=False, reason="observer did not call tool")
            args = response.tool_calls[0].arguments
            if not isinstance(args, dict):
                return RlaifObservation(should_evaluate=False, reason="observer returned invalid args")
            return RlaifObservation(
                should_evaluate=bool(args.get("should_evaluate", False)),
                task=str(args.get("task", "")),
                reason=str(args.get("reason", "")),
                confidence=float(args.get("confidence", 0.0) or 0.0),
            )
        except Exception:
            logger.exception("RlaifObserver.decide failed")
            return RlaifObservation(should_evaluate=False, reason="observer failed")

    @classmethod
    def _build_messages(cls, trajectory: Trajectory) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a meta-critic watching an AI agent work. After each turn, "
                    "decide if the agent just did something that could be improved in its "
                    "own code, prompts, or tool usage. Only flag concrete, bounded tasks: "
                    "a bug fix, a refactor, a prompt tweak, a better error message, etc. "
                    "Ignore general conversation, chit-chat, or vague advice."
                ),
            },
            {
                "role": "user",
                "content": (
                    "## Agent turn\n\n"
                    f"{cls._trajectory_summary(trajectory)}\n\n"
                    "Should we run an RLAIF evaluation to try to improve something? "
                    "Call decide_evaluation."
                ),
            },
        ]

    @staticmethod
    def _trajectory_summary(trajectory: Trajectory) -> str:
        lines: list[str] = [f"Task: {trajectory.task}"]
        for step in trajectory.steps:
            role = step.role
            text = (step.content or "").strip()
            if step.tool_calls:
                names = [tc.get("name", "?") for tc in step.tool_calls]
                text += f"\n[tools: {', '.join(names)}]"
            if step.tool_results:
                results = [f"{r.get('name', '?')}: {str(r.get('content', ''))[:80]}" for r in step.tool_results]
                text += f"\n[results: {'; '.join(results)}]"
            lines.append(f"{role}: {text[:300]}")
        return "\n".join(lines)


class RlaifBackgroundEvaluator:
    """Runs the full generate-evaluate-score loop in the background."""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        critic_model: str | None = None,
        *,
        candidate_count: int = 2,
        test_command: list[str] | None = None,
        lint_command: list[str] | None = None,
        schedule_background: Callable[[asyncio.coroutine], Any] | None = None,
        dataset: RlaifDataset | None = None,
    ) -> None:
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.critic_model = critic_model or model
        self.candidate_count = candidate_count
        self.test_command = test_command
        self.lint_command = lint_command
        self.schedule_background = schedule_background
        self.dataset = dataset or RlaifDataset()

    async def run(self, task: str) -> str:
        """Generate candidates, evaluate, score, save preferences, return report."""
        harness = PatchHarness(
            repo_root=self.workspace,
            test_command=self.test_command,
            lint_command=self.lint_command,
        )
        candidates = await self._generate_candidates(task)
        if not candidates:
            return f"RLAIF: no valid candidates generated for '{task}'."

        evaluated: list[Any] = []
        for patch, summary in candidates:
            result = await harness.evaluate(patch, summary)
            evaluated.append(result)

        critic = RlaifCritic(self.provider, self.critic_model)
        scored: list[tuple[Any, float]] = []
        for result in evaluated:
            extra = self._build_context(result)
            score_result = await critic.score(
                task=task,
                candidate={"patch": result.patch, "summary": result.summary},
                extra_context=extra,
            )
            scored.append((result, score_result.score + result.score_bonus))

        scored.sort(key=lambda x: x[1], reverse=True)
        winner, winner_score = scored[0]
        for loser, loser_score in scored[1:]:
            self.dataset.append(
                RlaifPreference(
                    prompt=task,
                    chosen={"patch": winner.patch, "summary": winner.summary},
                    rejected={"patch": loser.patch, "summary": loser.summary},
                    score_chosen=winner_score,
                    score_rejected=loser_score,
                    reason=f"winner score {winner_score} > loser score {loser_score}",
                    task=task,
                    metadata={
                        "winner_tests": winner.test_passed,
                        "winner_lint": winner.lint_passed,
                        "winner_backend": winner.backend,
                        "loser_tests": loser.test_passed,
                        "loser_lint": loser.lint_passed,
                        "loser_backend": loser.backend,
                    },
                )
            )

        lines = [
            f"# RLAIF background result: {task}",
            f"Candidates: {len(candidates)}",
            f"Winner score: {winner_score:.1f}",
            f"Winner tests: {winner.test_passed}, lint: {winner.lint_passed}",
            f"Winner summary: {winner.summary}",
            f"Dataset total: {self.dataset.count()}",
            "",
            "## Winning patch",
            "",
            "```diff",
            winner.patch,
            "```",
        ]
        return "\n".join(lines)

    async def _generate_candidates(self, task: str) -> list[tuple[str, str]]:
        from nanobot.agent.tools.rlaif_eval import RlaifEvalTool

        tool = RlaifEvalTool(
            workspace=self.workspace,
            default_candidate_count=self.candidate_count,
            provider=self.provider,
            default_model=self.model,
            test_command=self.test_command,
            lint_command=self.lint_command,
        )
        return await tool._generate_candidates(
            provider=self.provider,
            task=task,
            count=self.candidate_count,
            model=self.model,
            temperature=0.7,
        )

    @staticmethod
    def _build_context(result: Any) -> str:
        parts = [
            f"Backend: {result.backend}",
            f"Duration: {result.duration_s:.1f}s",
            f"Tests passed: {result.test_passed}",
            f"Lint passed: {result.lint_passed}",
        ]
        if not result.test_passed:
            parts.append(f"Test output:\n{result.test_output}")
        if not result.lint_passed:
            parts.append(f"Lint output:\n{result.lint_output}")
        return "\n\n".join(parts)


class RlaifObserverHook(AgentHook):
    """AgentHook that observes turns and triggers background RLAIF tasks."""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        critic_model: str | None = None,
        *,
        candidate_count: int = 2,
        test_command: list[str] | None = None,
        lint_command: list[str] | None = None,
        min_confidence: float = 0.6,
        schedule_background: Callable[[asyncio.coroutine], Any] | None = None,
        publish_outbound: Callable[[OutboundMessage], Any] | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.critic_model = critic_model or model
        self.candidate_count = candidate_count
        self.test_command = test_command
        self.lint_command = lint_command
        self.min_confidence = min_confidence
        self.schedule_background = schedule_background
        self._publish_outbound = publish_outbound
        self._channel = channel
        self._chat_id = chat_id
        self._observer = RlaifObserver(provider, model)
        self._evaluator: RlaifBackgroundEvaluator | None = None
        self._current_trajectory: Trajectory | None = None
        self._session_key: str | None = None

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        schedule_background: Callable[[asyncio.coroutine], Any] | None = None,
        publish_outbound: Callable[[OutboundMessage], Any] | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> "RlaifObserverHook | None":
        if not getattr(cfg, "enable", False):
            return None
        if not getattr(cfg, "observer", False):
            return None
        return cls(
            workspace=workspace,
            provider=provider,
            model=model,
            critic_model=getattr(cfg, "observer_critic_model", None) or model,
            candidate_count=getattr(cfg, "candidate_count", 2),
            test_command=getattr(cfg, "test_command", None),
            lint_command=getattr(cfg, "lint_command", None),
            min_confidence=getattr(cfg, "observer_min_confidence", 0.6),
            schedule_background=schedule_background,
            publish_outbound=publish_outbound,
            channel=channel,
            chat_id=chat_id,
        )

    async def before_run(self, context: AgentRunHookContext) -> None:
        self._current_trajectory = Trajectory("agent turn")
        self._session_key = context.messages[0].get("session_key") if context.messages else None

    async def before_iteration(self, context: AgentHookContext) -> None:
        if self._current_trajectory is None:
            return
        # Record assistant messages from the hook context.
        if context.response is not None and context.messages:
            last = context.messages[-1]
            step = TurnStep(
                role=last.get("role", "unknown"),
                content=last.get("content"),
                tool_calls=[tc.to_openai_tool_call() for tc in context.tool_calls],
            )
            self._current_trajectory.add_step(step)

    async def after_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: Any,
        tool: Any,
        params: Any,
        result: Any,
    ) -> None:
        if self._current_trajectory is None:
            return
        step = TurnStep(
            role="user",
            content=None,
            tool_results=[{
                "name": getattr(tool_call, "name", str(tool)),
                "content": str(result)[:2000],
                "is_error": getattr(result, "is_error", False),
            }],
        )
        self._current_trajectory.add_step(step)

    async def after_run(self, context: AgentRunHookContext) -> None:
        if self._current_trajectory is None:
            return
        if self.schedule_background is None:
            return

        observation = await self._observer.decide(self._current_trajectory)
        logger.info(
            "RLAIF observer decided: should_evaluate={}, confidence={}, task={!r}",
            observation.should_evaluate,
            observation.confidence,
            observation.task,
        )

        if not observation.should_evaluate or observation.confidence < self.min_confidence:
            self._current_trajectory = None
            return

        if self._evaluator is None:
            self._evaluator = RlaifBackgroundEvaluator(
                workspace=self.workspace,
                provider=self.provider,
                model=self.model,
                critic_model=self.critic_model,
                candidate_count=self.candidate_count,
                test_command=self.test_command,
                lint_command=self.lint_command,
                schedule_background=self.schedule_background,
            )

        task = observation.task
        self.schedule_background(self._run_background(task))
        self._current_trajectory = None

    async def _run_background(self, task: str) -> None:
        if self._evaluator is None:
            return
        try:
            report = await self._evaluator.run(task)
            logger.info("RLAIF background report:\n{}", report)
            await self._announce(
                f"RLAIF encontró una mejora para: {task}\n\n{report[:1200]}",
            )
        except Exception:
            logger.exception("RLAIF background evaluation failed")

    async def _announce(self, content: str) -> None:
        if self._publish_outbound is None:
            return
        try:
            await self._publish_outbound(
                OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content=content,
                    metadata={"rlaif_background": True},
                )
            )
        except Exception:
            logger.exception("RLAIF failed to announce result")


def create_rlaif_observer_hook(
    cfg: Any,
    workspace: Path,
    provider: LLMProvider,
    model: str,
    schedule_background: Callable[[asyncio.coroutine], Any] | None = None,
) -> AgentHook | None:
    return RlaifObserverHook.from_config(
        cfg,
        workspace,
        provider,
        model,
        schedule_background=schedule_background,
    )
