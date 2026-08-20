"""LLM-as-critic for scoring and pairwise comparison of agent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.utils.helpers import sanitize_surrogates_deep

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


@dataclass
class RlaifCriticResult:
    """Outcome of critic evaluation."""

    score: float
    issues: list[str]
    winner: str | None = None
    reason: str = ""
    raw_response: str | None = None


class RlaifCritic:
    """Evaluate trajectories or candidate patches with an LLM critic."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    _SCORE_TOOL = [
        {
            "type": "function",
            "function": {
                "name": "rate_solution",
                "description": "Rate a candidate solution to a coding task from 1 to 5.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "Quality score from 1 (bad) to 5 (excellent).",
                            "minimum": 1,
                            "maximum": 5,
                        },
                        "issues": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of concrete issues or weaknesses.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "One-sentence summary of the score.",
                        },
                    },
                    "required": ["score", "reason"],
                },
            },
        }
    ]

    _COMPARE_TOOL = [
        {
            "type": "function",
            "function": {
                "name": "choose_solution",
                "description": "Choose the better of two candidate solutions to a coding task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "winner": {
                            "type": "string",
                            "enum": ["A", "B", "tie"],
                            "description": "Which solution is better, or 'tie' if equal.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "One-sentence reason for the choice.",
                        },
                        "score_a": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 5,
                            "description": "Quality score for solution A.",
                        },
                        "score_b": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 5,
                            "description": "Quality score for solution B.",
                        },
                    },
                    "required": ["winner", "reason", "score_a", "score_b"],
                },
            },
        }
    ]

    async def score(
        self,
        task: str,
        candidate: dict[str, Any],
        *,
        extra_context: str = "",
    ) -> RlaifCriticResult:
        """Return a scalar score and issue list for a single candidate."""
        messages = self._build_score_messages(task, candidate, extra_context)
        try:
            response = await self._provider.chat_with_retry(
                messages=messages,
                tools=self._SCORE_TOOL,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                tool_choice={"type": "function", "function": {"name": "rate_solution"}},
            )
            return self._parse_score_response(response)
        except Exception:
            logger.exception("RlaifCritic.score failed")
            return RlaifCriticResult(score=1.0, issues=["critic failed"])

    async def compare(
        self,
        task: str,
        candidate_a: dict[str, Any],
        candidate_b: dict[str, Any],
        *,
        extra_context: str = "",
    ) -> RlaifCriticResult:
        """Compare two candidates and return which one wins (A/B/tie)."""
        messages = self._build_compare_messages(task, candidate_a, candidate_b, extra_context)
        try:
            response = await self._provider.chat_with_retry(
                messages=messages,
                tools=self._COMPARE_TOOL,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                tool_choice={"type": "function", "function": {"name": "choose_solution"}},
            )
            return self._parse_compare_response(response)
        except Exception:
            logger.exception("RlaifCritic.compare failed")
            return RlaifCriticResult(score=1.0, issues=["critic comparison failed"])

    def _build_score_messages(
        self,
        task: str,
        candidate: dict[str, Any],
        extra_context: str,
    ) -> list[dict[str, Any]]:
        candidate_text = self._candidate_to_text(candidate)
        content = (
            f"## Task\n{task}\n\n"
            f"## Candidate solution\n{candidate_text}\n"
        )
        if extra_context:
            content += f"\n## Extra context\n{extra_context}\n"
        content += (
            "\nRate this solution from 1 (broken/very poor) to 5 (excellent). "
            "Use the rate_solution tool. Be strict: a solution that fails tests, "
            "introduces bugs, ignores requirements, or is unnecessarily complex "
            "should get a low score."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a rigorous code-quality critic. You evaluate "
                    "candidate code changes produced by an AI agent. Score 1-5 "
                    "and list concrete issues."
                ),
            },
            {"role": "user", "content": content},
        ]

    def _build_compare_messages(
        self,
        task: str,
        candidate_a: dict[str, Any],
        candidate_b: dict[str, Any],
        extra_context: str,
    ) -> list[dict[str, Any]]:
        text_a = self._candidate_to_text(candidate_a)
        text_b = self._candidate_to_text(candidate_b)
        content = (
            f"## Task\n{task}\n\n"
            f"## Candidate A\n{text_a}\n\n"
            f"## Candidate B\n{text_b}\n"
        )
        if extra_context:
            content += f"\n## Extra context\n{extra_context}\n"
        content += (
            "\nChoose the better solution using choose_solution. "
            "Consider correctness, test passing, simplicity, and adherence to the task. "
            "Select 'tie' only if they are truly equivalent."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a rigorous code-quality critic choosing between two "
                    "candidate code changes produced by an AI agent."
                ),
            },
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _candidate_to_text(candidate: dict[str, Any]) -> str:
        if "patch" in candidate:
            return f"Patch summary: {candidate.get('summary', '')}\n\n```diff\n{candidate['patch']}\n```"
        if "content" in candidate:
            return str(candidate["content"])
        if "trajectory" in candidate:
            traj = candidate["trajectory"]
            if isinstance(traj, dict):
                traj = traj.get("steps", [])
            parts: list[str] = []
            for step in traj:
                role = step.get("role", "unknown")
                text = step.get("content") or ""
                if step.get("tool_calls"):
                    text += f"\n[tool_calls: {step['tool_calls']}]"
                if step.get("tool_results"):
                    text += f"\n[tool_results: {step['tool_results']}]"
                parts.append(f"{role}: {text}")
            return "\n".join(parts)
        return str(candidate)

    @classmethod
    def _parse_score_response(cls, response: Any) -> RlaifCriticResult:
        if not response.has_tool_calls:
            return RlaifCriticResult(score=1.0, issues=["critic did not call tool"])
        args = sanitize_surrogates_deep(response.tool_calls[0].arguments)
        if not isinstance(args, dict):
            return RlaifCriticResult(score=1.0, issues=["critic returned invalid args"])
        score = cls._clamp_score(args.get("score"))
        issues = args.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        return RlaifCriticResult(
            score=score,
            issues=[str(i) for i in issues],
            reason=str(args.get("reason", "")),
            raw_response=response.content,
        )

    @classmethod
    def _parse_compare_response(cls, response: Any) -> RlaifCriticResult:
        if not response.has_tool_calls:
            return RlaifCriticResult(score=1.0, issues=["critic did not call tool"], winner=None)
        args = sanitize_surrogates_deep(response.tool_calls[0].arguments)
        if not isinstance(args, dict):
            return RlaifCriticResult(
                score=1.0,
                issues=["critic returned invalid args"],
                winner=None,
            )
        winner = args.get("winner")
        if winner not in ("A", "B", "tie"):
            winner = None
        return RlaifCriticResult(
            score=cls._clamp_score(args.get("score_a")),
            issues=[],
            winner=winner,
            reason=str(args.get("reason", "")),
            raw_response=response.content,
        )

    @staticmethod
    def _clamp_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 1.0
        return max(1.0, min(5.0, score))
