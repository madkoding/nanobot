"""RLAIF proactive scanner: periodically reviews the repo and proposes
concrete improvements (bug fixes, refactors, prompt tweaks) that pass
the configured tests + lint, then applies, commits, and pushes them.

Unlike the observer hook (which fires after an agent turn), this scanner
runs on its own asyncio loop while the gateway is up. It samples a
small Python file at random from the configured workspace, asks the
critic LLM for a unified diff of one concrete improvement, and runs
the diff through ``PatchHarness`` like the observer does. Successful
patches reuse the same auto-apply / auto-commit / auto-push helpers.

ponytail: file selection is random with a "skip if recently touched"
memory file so the scanner doesn't keep re-evaluating the same files
after a busy edit session. Critic is a single LLM call (no multi-shot
debate) because the budget is hourly, not per-turn.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from nanobot.agent.rlaif.dataset import RlaifDataset, RlaifPreference
from nanobot.agent.rlaif.harness import PatchHarness
from nanobot.agent.rlaif.observer import _git_commit, _git_push
from nanobot.providers.base import LLMProvider


PROPOSE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "propose_improvement",
            "description": (
                "Propose ONE concrete, bounded improvement to the file as a "
                "unified diff. The patch must apply cleanly to the current "
                "contents. Only propose changes you can justify in one sentence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": "One-sentence justification for the change.",
                    },
                    "patch": {
                        "type": "string",
                        "description": (
                            "Unified diff (--- a/path/to/file +++ b/path/to/file) "
                            "that applies cleanly with 'git apply'. Empty string if "
                            "the file is already clean and no improvement is worth making."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in the change being correct and useful.",
                    },
                },
                "required": ["rationale", "patch", "confidence"],
            },
        },
    }
]


@dataclass
class ScannerState:
    """In-memory state of the proactive scanner, persisted across runs."""

    last_run_at: float = 0.0
    files_seen: dict[str, float] = field(default_factory=dict)
    last_report: str = ""


class RlaifProactiveScanner:
    """Periodically propose and apply self-improvements to the workspace."""

    STATE_FILE = ".rlaif_scanner_state.json"

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        *,
        critic_model: str | None = None,
        interval_s: float = 3600.0,
        min_confidence: float = 0.7,
        test_command: list[str] | None = None,
        lint_command: list[str] | None = None,
        auto_apply: bool = True,
        auto_commit: bool = True,
        auto_push: bool = True,
        max_file_size_kb: int = 80,
        sample_pool: int = 30,
        on_report: Callable[[str], Any] | None = None,
    ) -> None:
        self.workspace = workspace.resolve(strict=False)
        self.provider = provider
        self.model = model
        self.critic_model = critic_model or model
        self.interval_s = max(60.0, interval_s)
        self.min_confidence = min_confidence
        self.test_command = test_command
        self.lint_command = lint_command
        self.auto_apply = auto_apply
        self.auto_commit = auto_commit
        self.auto_push = auto_push
        self.max_file_size_kb = max_file_size_kb
        self.sample_pool = max(1, sample_pool)
        self.on_report = on_report
        self._state = self._load_state()
        self._stop = asyncio.Event()

    def _state_path(self) -> Path:
        return self.workspace / self.STATE_FILE

    def _load_state(self) -> ScannerState:
        path = self._state_path()
        if not path.exists():
            return ScannerState()
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            return ScannerState(
                last_run_at=float(data.get("last_run_at", 0.0)),
                files_seen={k: float(v) for k, v in (data.get("files_seen") or {}).items()},
                last_report=str(data.get("last_report", "")),
            )
        except Exception as exc:  # ponytail: never let a corrupt state file brick the scanner
            logger.warning("RLAIF scanner: cannot read state file: {}", exc)
            return ScannerState()

    def _save_state(self) -> None:
        import json

        data = {
            "last_run_at": self._state.last_run_at,
            "files_seen": self._state.files_seen,
            "last_report": self._state.last_report,
        }
        try:
            self._state_path().write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            logger.warning("RLAIF scanner: cannot write state file: {}", exc)

    def stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        """Entry point: run ``tick`` every ``interval_s`` until ``stop()``."""
        # ponytail: stagger the first run by up to 60s so multiple gateways don't
        # all hammer the same file at the same time.
        initial_delay = random.uniform(0.0, min(60.0, self.interval_s / 2))
        await self._sleep_or_stop(initial_delay)
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception:  # ponytail: keep the loop alive no matter what
                logger.exception("RLAIF scanner: tick failed")
            self._save_state()
            await self._sleep_or_stop(self.interval_s)

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def tick(self) -> str:
        """Run one pass: pick a file, propose, evaluate, maybe apply. Returns the report."""
        self._state.last_run_at = time.time()
        target = self._pick_file()
        if target is None:
            report = "RLAIF scanner: no candidate files found in workspace."
            self._state.last_report = report
            return report

        rel = str(target.relative_to(self.workspace))
        self._state.files_seen[rel] = time.time()
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            report = f"RLAIF scanner: cannot read {rel}: {exc}"
            self._state.last_report = report
            return report

        logger.info("RLAIF scanner: reviewing {}", rel)
        proposal = await self._propose(rel, text)
        if not proposal:
            report = f"RLAIF scanner: no proposal for {rel}."
            self._state.last_report = report
            return report

        if proposal.get("confidence", 0.0) < self.min_confidence:
            report = (
                f"RLAIF scanner: low confidence ({proposal['confidence']:.2f}) "
                f"for {rel}, skipping."
            )
            self._state.last_report = report
            return report

        patch = (proposal.get("patch") or "").strip()
        if not patch:
            report = f"RLAIF scanner: empty patch for {rel}, skipping."
            self._state.last_report = report
            return report

        # Normalize the patch header so it targets the file we just read.
        patch = self._normalize_patch_paths(patch, rel)

        harness = PatchHarness(
            repo_root=self.workspace,
            test_command=self.test_command,
            lint_command=self.lint_command,
        )
        result = await harness.evaluate(patch, patch_summary=proposal.get("rationale", ""))
        if not result.passed:
            why = []
            if not result.test_passed:
                why.append("tests")
            if not result.lint_passed:
                why.append("lint")
            report = (
                f"RLAIF scanner: candidate for {rel} failed {','.join(why) or 'checks'}."
            )
            self._state.last_report = report
            logger.info(report)
            return report

        applied_status = "skipped (auto_apply off)"
        if self.auto_apply:
            from nanobot.agent.tools.rlaif_eval import RlaifEvalTool

            applied = await RlaifEvalTool._apply_diff(patch, workspace=self.workspace)
            applied_status = str(applied)
            if self.auto_commit and applied.startswith("Patch applied"):
                commit_msg = (
                    f"rlaif(scanner): {rel}: {proposal.get('rationale', '')[:200]}\n\n"
                    f"Auto-proposed by proactive scanner. Tests+lint passed."
                )
                commit_result = await asyncio.to_thread(
                    _git_commit, self.workspace, commit_msg
                )
                applied_status = f"{applied_status}; {commit_result or '(commit failed)'}"
                if self.auto_push and commit_result and commit_result != "no changes to commit":
                    push_result = await asyncio.to_thread(_git_push, self.workspace)
                    applied_status = f"{applied_status}; {push_result or '(push failed)'}"

        # Record the preference so the dataset keeps growing even without an agent turn.
        try:
            RlaifDataset().append(
                RlaifPreference(
                    prompt=f"Proactive scan: {rel}",
                    chosen={"patch": patch, "summary": proposal.get("rationale", "")},
                    rejected={"patch": "", "summary": "no challenger (proactive)"},
                    score_chosen=1.0,
                    score_rejected=0.0,
                    reason=f"tests+lint passed; {proposal.get('rationale', '')}",
                    task=rel,
                    metadata={
                        "auto_apply": applied_status,
                        "winner_tests": result.test_passed,
                        "winner_lint": result.lint_passed,
                        "winner_backend": result.backend,
                        "scanner_proactive": True,
                    },
                )
            )
        except Exception:  # ponytail: dataset append is best-effort
            logger.exception("RLAIF scanner: failed to record preference")

        report = (
            f"RLAIF scanner: applied improvement to {rel}\n"
            f"Rationale: {proposal.get('rationale', '')}\n"
            f"Status: {applied_status}\n"
            f"Confidence: {proposal['confidence']:.2f}"
        )
        self._state.last_report = report
        logger.info(report)
        if self.on_report is not None:
            try:
                maybe = self.on_report(report)
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception:
                logger.exception("RLAIF scanner: on_report callback failed")
        return report

    def _pick_file(self) -> Path | None:
        """Pick a Python file under self.workspace, avoiding ones seen recently."""
        candidates: list[Path] = []
        max_bytes = self.max_file_size_kb * 1024
        skip_dirs = {
            ".git", "__pycache__", "node_modules", "dist", "build",
            ".venv", "venv", ".mypy_cache", ".pytest_cache",
        }
        try:
            for path in self.workspace.rglob("*.py"):
                if any(part in skip_dirs for part in path.parts):
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        continue
                except OSError:
                    continue
                rel = str(path.relative_to(self.workspace))
                # De-prioritize files seen in the last 6 hours.
                last_seen = self._state.files_seen.get(rel, 0.0)
                if time.time() - last_seen < 6 * 3600:
                    continue
                candidates.append(path)
        except OSError as exc:
            logger.warning("RLAIF scanner: cannot enumerate workspace: {}", exc)
            return None
        if not candidates:
            return None
        if len(candidates) > self.sample_pool:
            candidates = random.sample(candidates, self.sample_pool)
        return random.choice(candidates)

    async def _propose(self, rel_path: str, text: str) -> dict[str, Any] | None:
        """Ask the critic LLM for a unified diff of one improvement."""
        # Truncate the file content so the prompt stays bounded.
        max_chars = 12_000
        truncated = text if len(text) <= max_chars else text[:max_chars] + "\n... (truncated)"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior Python reviewer working on the nanobot agent framework. "
                    "Read the file and propose ONE concrete, bounded improvement as a unified "
                    "diff. Allowed: bug fix, refactor, better error message, prompt tweak, "
                    "dead-code removal, type-hint fix. Disallowed: new features, big rewrites, "
                    "anything speculative. If the file is already clean, return an empty patch. "
                    "The diff must apply with `git apply` against the file as shown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## File: {rel_path}\n\n```python\n{truncated}\n```\n\n"
                    "Propose a unified diff (--- a/<rel> +++ b/<rel>) using the "
                    "propose_improvement tool."
                ),
            },
        ]
        try:
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=PROPOSE_TOOL,
                model=self.critic_model,
                temperature=0.0,
                tool_choice={
                    "type": "function",
                    "function": {"name": "propose_improvement"},
                },
            )
        except Exception:
            logger.exception("RLAIF scanner: critic call failed for {}", rel_path)
            return None
        if not response.has_tool_calls:
            return None
        args = response.tool_calls[0].arguments
        if not isinstance(args, dict):
            return None
        return args

    @staticmethod
    def _normalize_patch_paths(patch: str, rel_path: str) -> str:
        """Rewrite a/ and b/ headers to point at the actual file we read."""
        lines = patch.splitlines()
        out: list[str] = []
        for line in lines:
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                tail = line[6:]  # strip "--- a/" or "+++ b/"
                out.append(line[:6] + rel_path)
            else:
                out.append(line)
        return "\n".join(out) + ("\n" if patch.endswith("\n") else "")


def build_scanner_from_config(
    cfg: Any,
    workspace: Path,
    provider: LLMProvider,
    model: str,
    *,
    on_report: Callable[[str], Any] | None = None,
) -> RlaifProactiveScanner | None:
    """Construct a scanner from the rlaif config block, or None if disabled."""
    if not getattr(cfg, "enable", False):
        return None
    if not getattr(cfg, "scanner_enable", False):
        return None
    return RlaifProactiveScanner(
        workspace=Path(getattr(cfg, "workspace", None) or workspace).expanduser().resolve(),
        provider=provider,
        model=model,
        critic_model=getattr(cfg, "scanner_critic_model", None) or model,
        interval_s=float(getattr(cfg, "scanner_interval_s", 3600.0)),
        min_confidence=float(getattr(cfg, "scanner_min_confidence", 0.7)),
        test_command=getattr(cfg, "test_command", None),
        lint_command=getattr(cfg, "lint_command", None),
        auto_apply=getattr(cfg, "scanner_auto_apply", True),
        auto_commit=getattr(cfg, "scanner_auto_commit", True),
        auto_push=getattr(cfg, "scanner_auto_push", True),
        on_report=on_report,
    )
