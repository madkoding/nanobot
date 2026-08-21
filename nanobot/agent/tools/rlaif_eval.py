"""RLAIF evaluation tool: generate candidate patches, score them, and save preferences."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.rlaif.critic import RlaifCritic
from nanobot.agent.rlaif.dataset import RlaifDataset, RlaifPreference
from nanobot.agent.rlaif.diff_utils import (
    _normalize_unified_diff,
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
    # If true, the background observer applies the winning patch to the real
    # repo when it passes tests + lint (instead of only announcing it).
    observer_auto_apply: bool = False
    # If true, the observer also stages and commits the applied patch to the
    # repo's current branch after a successful auto-apply. No push.
    observer_auto_commit: bool = False
    # If true, the observer also pushes the auto-commit to the remote
    # (``origin``, current branch). Requires SSH auth available in the
    # gateway process (default ~/.ssh/id_ed25519 works).
    observer_auto_push: bool = False
    # ---- Proactive scanner (runs independently of agent turns) -----------
    # If true, a background task periodically picks a Python file in the
    # workspace, asks the critic LLM to propose a single bounded
    # improvement, evaluates it with the configured tests + lint, and
    # (optionally) auto-applies / commits / pushes it. Independent of the
    # observer, which only fires after a turn.
    scanner_enable: bool = False
    scanner_critic_model: str | None = None
    scanner_min_confidence: float = 0.7
    scanner_interval_s: float = 3600.0
    scanner_auto_apply: bool = True
    scanner_auto_commit: bool = True
    scanner_auto_push: bool = True
    # Workspace root where candidate patches are evaluated. Falls back to the
    # agent's default workspace when omitted.
    workspace: str | None = None
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
        workspace = Path(cfg.workspace).expanduser().resolve() if cfg.workspace else Path(ctx.workspace)
        return cls(
            workspace=workspace,
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
            apply_result = await self._apply_diff(winner.patch, workspace=self.workspace)
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
        """Ask the LLM to generate N unified-diff patches for the task.

        If the task references existing files under the workspace, the file
        contents are included in the prompt so the generated patch has real
        context lines to apply against.
        """
        file_context = self._build_file_context(task)
        system = (
            "You are a precise coding assistant. Your job is to output a single, "
            "well-formed unified diff patch (git diff -u format) that solves the task. "
            "Rules:\n"
            "1. Start every patch with `--- ` and `+++ ` headers.\n"
            "2. Include hunk headers like `@@ -line,count +line,count @@`.\n"
            "3. Only modify files that need changing.\n"
            "4. Do not explain the patch outside the diff.\n"
            "5. Do not wrap the patch in markdown fences.\n"
            "6. Keep the context lines minimal but enough to apply: at least 3 "
            "   unchanged lines above and below every changed block.\n"
            "7. Patches must apply cleanly with `git apply`; use the exact existing "
            "   lines shown in the file context.\n"
            "8. Every hunk must have matching line counts: if the hunk header says "
            "   `@@ -oldline,oldcount +newline,newcount @@`, include exactly "
            "   oldcount context lines plus removed lines and exactly newcount "
            "   context lines plus added lines."
        )
        user_parts = [f"## Task\n{task}\n\n"]
        if file_context:
            user_parts.append("## Current files\n\n")
            user_parts.append(file_context)
            user_parts.append("\n\n")
        user_parts.append(
            "Return a single unified diff patch in git diff -u format. "
            "No markdown, no commentary, only the patch."
        )
        user = "".join(user_parts)
        candidates: list[tuple[str, str]] = []
        for i in range(count):
            try:
                response = await provider.chat_with_retry(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    model=model,
                    max_tokens=8192,
                    temperature=temperature,
                )
                patch = extract_unified_diff(response.content or "")
                if not is_valid_unified_diff(patch):
                    # Some reasoning-heavy models return the reasoning trace as the
                    # assistant message content. Try to extract the diff from it.
                    reasoning = getattr(response, "reasoning_content", None) or ""
                    patch = extract_unified_diff(reasoning)
                if not is_valid_unified_diff(patch):
                    logger.warning("RlaifEval candidate %s is not a valid unified diff; skipping", i + 1)
                    continue
                patch = _normalize_unified_diff(patch)
                summary = summarize_unified_diff(patch)
                candidates.append((patch, summary))
            except Exception:
                logger.exception("Failed to generate rlaif candidate %s", i + 1)
        return candidates

    def _build_file_context(self, task: str) -> str:
        """Extract likely file paths from the task and return their contents."""
        import re

        paths: list[Path] = []
        # Match absolute paths and paths starting with a known project prefix.
        for match in re.finditer(r"(?:^|\s)(/[a-zA-Z0-9_/.\-]+|nanobot/[a-zA-Z0-9_/.\-]+|tests/[a-zA-Z0-9_/.\-]+)", task):
            raw = match.group(1).strip()
            if not raw:
                continue
            candidate = Path(raw)
            if candidate.is_absolute():
                paths.append(candidate)
            else:
                paths.append(self.workspace / candidate)
        seen: set[Path] = set()
        parts: list[str] = []
        for path in paths:
            try:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                if not resolved.is_file():
                    continue
                # Stay within the workspace for safety.
                try:
                    resolved.relative_to(self.workspace.resolve())
                except ValueError:
                    logger.warning("RLAIF task references file outside workspace: %s", resolved)
                    continue
                text = resolved.read_text(encoding="utf-8", errors="replace")
                parts.append(f"### {resolved}\n\n```\n{text}\n```")
            except Exception:
                logger.exception("Failed to read file context for RLAIF task: %s", path)
        return "\n\n".join(parts)

    def _resolve_provider(self, request_ctx: Any) -> LLMProvider | None:
        if self._provider is not None:
            return self._provider
        if request_ctx is not None and request_ctx.runtime is not None:
            return getattr(request_ctx.runtime, "provider", None)
        return None

    @staticmethod
    async def _apply_diff(patch: str, *, workspace: Path | None = None) -> str:
        import subprocess
        import tempfile

        target = workspace or getattr(
            RlaifEvalTool, "_default_apply_workspace", None
        )
        if target is None:
            return "Patch apply error: no workspace available"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as f:
                f.write(patch)
                patch_path = f.name

            for cmd in (["git", "apply", patch_path], ["patch", "-p1", "-i", patch_path]):
                if cmd[0] == "patch" and not shutil.which("patch"):
                    continue
                proc = subprocess.run(
                    cmd,
                    cwd=str(target),
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0:
                    return f"Patch applied to {target}"
            return f"Patch apply failed: {proc.stderr}"
        except Exception as exc:
            return f"Patch apply error: {exc}"
