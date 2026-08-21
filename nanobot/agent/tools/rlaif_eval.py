"""RLAIF evaluation tool: generate candidate patches, score them, and save preferences."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.rlaif.critic import RlaifCritic
from nanobot.agent.rlaif.dataset import RlaifDataset, RlaifPreference
from nanobot.agent.rlaif.diff_utils import (
    extract_unified_diff,
    is_valid_unified_diff,
    summarize_unified_diff,
)
from nanobot.agent.rlaif.harness import PatchHarness, PatchHarnessResult
from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.schema import (
    BooleanSchema,
    IntegerSchema,
    NumberSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.config_base import Base

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class RlaifToolConfig(Base):
    """RLAIF self-improvement tool configuration."""

    enable: bool = False
    candidate_count: int = 2
    # If true, the tool will actually write the chosen patch to the real repo.
    auto_apply: bool = False
    # If true, a per-turn observer schedules background RLAIF evaluations.
    observer: bool = False
    observer_critic_model: str | None = None
    observer_min_confidence: float = 0.6
    test_command: list[str] | None = None
    lint_command: list[str] | None = None


@tool_parameters(
    tool_parameters_schema(
        required=["task"],
        task=StringSchema(
            "Coding task to solve. The agent will generate candidate patches and pick the best one."
        ),
        candidate_count=IntegerSchema(
            description="Number of candidate patches to generate (2-4).",
            minimum=2,
            maximum=4,
            nullable=True,
        ),
        auto_apply=BooleanSchema(
            description=(
                "If true, apply the winning patch to the workspace immediately. "
                "Default false: returns the patch for approval."
            ),
            default=False,
            nullable=True,
        ),
        temperature=NumberSchema(
            description="Sampling temperature for generating candidates.",
            minimum=0.0,
            maximum=1.5,
            nullable=True,
        ),
        model=StringSchema(
            "Model override for the candidate generator. Omit to use the active runtime model.",
            nullable=True,
        ),
        critic_model=StringSchema(
            "Model override for the critic. Omit to use the active runtime model.",
            nullable=True,
        ),
    )
)
class RlaifEvalTool(Tool):
    """Generate candidate code patches for a task, score them with a critic, and save preferences."""

    config_key = "rlaif"
    _scopes = {"core", "subagent"}

    def __init__(
        self,
        workspace: Path,
        *,
        default_candidate_count: int = 2,
        auto_apply: bool = False,
        test_command: list[str] | None = None,
        lint_command: list[str] | None = None,
        default_model: str | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.workspace = workspace
        self.default_candidate_count = default_candidate_count
        self.auto_apply = auto_apply
        self.test_command = test_command
        self.lint_command = lint_command
        self.default_model = default_model
        self._provider = provider
        self.dataset = RlaifDataset()

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        return getattr(ctx.config, "rlaif", RlaifToolConfig()).enable

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        cfg = getattr(ctx.config, "rlaif", RlaifToolConfig())
        runtime = getattr(ctx, "provider_snapshot_loader", None)
        provider = None
        default_model = None
        if runtime is not None:
            try:
                snapshot = runtime()
                provider = getattr(snapshot, "provider", None)
                default_model = getattr(snapshot, "model", None)
            except Exception:
                logger.exception("Failed to load provider snapshot for rlaif_eval")
        return cls(
            workspace=Path(ctx.workspace),
            default_candidate_count=cfg.candidate_count,
            auto_apply=cfg.auto_apply,
            test_command=cfg.test_command,
            lint_command=cfg.lint_command,
            default_model=default_model,
            provider=provider,
        )

    @property
    def name(self) -> str:
        return "rlaif_eval"

    @property
    def description(self) -> str:
        return (
            "RLAIF self-improvement: generate candidate patches for a coding task, "
            "run tests and lint on each, score with an LLM critic, save preference "
            "pairs, and optionally apply the best patch."
        )

    async def execute(
        self,
        task: str,
        candidate_count: int | None = None,
        auto_apply: bool | None = None,
        temperature: float | None = None,
        model: str | None = None,
        critic_model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        request_ctx = current_request_context()
        provider = self._resolve_provider(request_ctx)
        if provider is None:
            return ToolResult.error(
                "Error: rlaif_eval requires an active LLM provider. "
                "Set a model preset or pass model= and critic_model=."
            )

        candidate_count = max(2, min(4, candidate_count or self.default_candidate_count))
        auto_apply = self.auto_apply if auto_apply is None else bool(auto_apply)
        generator_model = model or self.default_model
        critic_model_name = critic_model or model or self.default_model
        if generator_model is None or critic_model_name is None:
            return ToolResult.error(
                "Error: rlaif_eval requires an explicit model= or critic_model=; "
                "no active runtime model was detected."
            )

        harness = PatchHarness(
            repo_root=self.workspace,
            test_command=self.test_command,
            lint_command=self.lint_command,
        )

        candidates = await self._generate_candidates(
            provider=provider,
            task=task,
            count=candidate_count,
            model=generator_model,
            temperature=temperature if temperature is not None else 0.7,
        )
        if not candidates:
            return ToolResult.error("Error: rlaif_eval failed to generate any valid candidate patches.")

        evaluated: list[PatchHarnessResult] = []
        for patch, summary in candidates:
            result = await harness.evaluate(patch, summary)
            evaluated.append(result)

        critic = RlaifCritic(provider, critic_model_name)
        scored: list[tuple[PatchHarnessResult, float]] = []
        for result in evaluated:
            extra_context = self._build_evaluation_context(result)
            critic_result = await critic.score(
                task=task,
                candidate={"patch": result.patch, "summary": result.summary},
                extra_context=extra_context,
            )
            total_score = critic_result.score + result.score_bonus
            scored.append((result, total_score))

        if not scored:
            return ToolResult.error("Error: no candidate could be scored.")

        scored.sort(key=lambda item: item[1], reverse=True)
        winner, winner_score = scored[0]
        self._save_preferences(task, winner, winner_score, scored)

        report = self._build_report(task, candidates, evaluated, scored, winner, winner_score)

        if auto_apply:
            apply_result = await self._apply_diff(winner.patch)
            report += f"\n\n## Applied patch\n\n{apply_result}"

        return report

    @staticmethod
    def _build_evaluation_context(result: PatchHarnessResult) -> str:
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

    def _save_preferences(
        self,
        task: str,
        winner: PatchHarnessResult,
        winner_score: float,
        scored: list[tuple[PatchHarnessResult, float]],
    ) -> None:
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

    def _build_report(
        self,
        task: str,
        candidates: list[tuple[str, str]],
        evaluated: list[PatchHarnessResult],
        scored: list[tuple[PatchHarnessResult, float]],
        winner: PatchHarnessResult,
        winner_score: float,
    ) -> str:
        lines = [
            f"# RLAIF evaluation for: {task}",
            "",
            f"Candidates generated: {len(candidates)}",
            f"Evaluated: {len(evaluated)}",
            f"Preferences saved: {len(scored) - 1}",
            f"Dataset total: {self.dataset.count()}",
            "",
            "## Candidate summary",
            "",
            "| candidate | score | tests | lint | backend | summary |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for i, (candidate, score) in enumerate(scored, start=1):
            marker = "🏆" if candidate is winner else ""
            lines.append(
                f"| {i} {marker} | {score:.1f} | {candidate.test_passed} | "
                f"{candidate.lint_passed} | {candidate.backend} | {candidate.summary} |"
            )
        lines.extend([
            "",
            f"## Winner ({winner_score:.1f})",
            f"- summary: {winner.summary}",
            f"- tests passed: {winner.test_passed}",
            f"- lint passed: {winner.lint_passed}",
            f"- backend: {winner.backend}",
            "",
            "## Winning patch",
            "",
            "```diff",
            winner.patch,
            "```",
        ])
        return "\n".join(lines)

    async def _generate_candidates(
        self,
        provider: LLMProvider,
        task: str,
        count: int,
        model: str,
        temperature: float,
    ) -> list[tuple[str, str]]:
        """Ask the LLM to generate N unified-diff patches for the task."""
        system = (
            "You are a precise coding assistant. Your job is to output a single, "
            "well-formed unified diff patch (git diff -u format) that solves the task. "
            "Rules:\n"
            "1. Start every patch with `--- ` and `+++ ` headers.\n"
            "2. Include hunk headers like `@@ -line,count +line,count @@`.\n"
            "3. Only modify files that need changing.\n"
            "4. Do not explain the patch outside the diff.\n"
            "5. Do not wrap the patch in markdown fences.\n"
            "6. Keep the context lines minimal (3 is enough)."
        )
        user = (
            f"## Task\n{task}\n\n"
            "Return a single unified diff patch in git diff -u format. "
            "No markdown, no commentary, only the patch."
        )
        candidates: list[tuple[str, str]] = []
        for i in range(count):
            try:
                response = await provider.chat_with_retry(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    model=model,
                    max_tokens=8192,
                    temperature=temperature,
                )
                text = response.content or ""
                patch = extract_unified_diff(text)
                if not is_valid_unified_diff(patch):
                    logger.warning("RlaifEval candidate %s is not a valid unified diff; skipping", i + 1)
                    continue
                summary = summarize_unified_diff(patch)
                candidates.append((patch, summary))
            except Exception:
                logger.exception("Failed to generate rlaif candidate %s", i + 1)
        return candidates

    def _resolve_provider(self, request_ctx: Any) -> LLMProvider | None:
        if self._provider is not None:
            return self._provider
        if request_ctx is not None and request_ctx.runtime is not None:
            return getattr(request_ctx.runtime, "provider", None)
        return None

    async def _apply_diff(self, patch: str) -> str:
        import subprocess
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as f:
                f.write(patch)
                patch_path = f.name

            proc = subprocess.run(
                ["git", "apply", "--check", patch_path],
                cwd=self.workspace,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                return f"Patch apply check failed: {proc.stderr}"

            proc = subprocess.run(
                ["git", "apply", patch_path],
                cwd=self.workspace,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                return f"Patch applied to {self.workspace}"
            return f"Patch apply failed: {proc.stderr}"
        except Exception as exc:
            return f"Patch apply error: {exc}"
