"""RLAIF proactive scanner: periodically reviews the repo and proposes
concrete improvements (bug fixes, refactors, prompt tweaks) that pass
the configured tests + lint, then queues them for the user to review.

Unlike the observer hook (which fires after an agent turn), this scanner
runs on its own asyncio loop while the gateway is up. It samples a
small Python file at random from the configured workspace, asks the
critic LLM for a unified diff of one concrete improvement, runs a
preflight (lint + git apply --check with a few recovery strategies),
and if everything looks good, queues the patch as a pending proposal
in the WebUI. The user reviews and approves/rejects; approval runs
the full test suite and commits+pushes the change.

Architecture note: we re-use RlaifEvalTool._generate_candidates() for
the actual diff generation. That helper builds file context from the
absolute path, asks the LLM for a unified diff in plain text (no
tool call), and extracts/validates the diff. The previous find+replace
approach was hallucinating because small models can't copy exact
text from long prompts; asking for a unified diff description works
because it's a "describe the change" task without copy-fidelity
requirements.
"""

from __future__ import annotations

import asyncio
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from nanobot.agent.rlaif.dataset import RlaifDataset, RlaifPreference
from nanobot.agent.rlaif.harness import PatchHarness
from nanobot.agent.rlaif.observer import _git_commit, _git_push
from nanobot.providers.base import LLMProvider


@dataclass
class ScannerState:
    """In-memory state of the proactive scanner, persisted across runs."""

    last_run_at: float = 0.0
    files_seen: dict[str, float] = field(default_factory=dict)
    last_report: str = ""
    pending_proposals: list[dict[str, Any]] = field(default_factory=list)
    next_proposal_id: int = 1


class RlaifProactiveScanner:
    """Periodically propose and queue self-improvements for user review."""

    STATE_FILE = ".rlaif_scanner_state.json"

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        *,
        critic_model: str | None = None,
        interval_s: float = 3600.0,
        min_confidence: float = 0.0,
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

    # --- state persistence -------------------------------------------------

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
                pending_proposals=list(data.get("pending_proposals") or []),
                next_proposal_id=int(data.get("next_proposal_id", 1)),
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
            "pending_proposals": self._state.pending_proposals,
            "next_proposal_id": self._state.next_proposal_id,
        }
        try:
            self._state_path().write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            logger.warning("RLAIF scanner: cannot write state file: {}", exc)

    def stop(self) -> None:
        self._stop.set()

    # --- main loop ---------------------------------------------------------

    async def run_forever(self) -> None:
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
        """Run one pass: pick a file, propose, evaluate, queue if good."""
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
        if not proposal or not proposal.get("patch"):
            report = proposal.get("rationale") if proposal else "no proposal"
            report = f"RLAIF scanner: no patch for {rel} ({report[:80]})"
            self._state.last_report = report
            logger.info(report)
            return report

        patch = proposal["patch"]
        if not self._patch_applies_strict(self.workspace, patch):
            report = f"RLAIF scanner: candidate for {rel} has unapplyable diff (skipped)."
            self._state.last_report = report
            logger.info(report)
            return report

        # Preflight lint: only reject if the patch introduced NEW errors.
        # Preexisting errors in the file are not our problem.
        lint_pre = self._run_lint(self.workspace, self.lint_command)
        harness = PatchHarness(
            repo_root=self.workspace,
            test_command=["true"],  # skip pytest on the preflight
            lint_command=self.lint_command,
            timeout=120.0,
        )
        preflight = await harness.evaluate(patch, patch_summary=proposal["rationale"])
        if preflight.lint_passed or not self._lint_introduced_new(
            lint_pre, preflight.lint_output, rel
        ):
            # Queue as a pending proposal.
            proposal_id = self._state.next_proposal_id
            self._state.next_proposal_id += 1
            self._state.pending_proposals.append(
                {
                    "id": proposal_id,
                    "created_at": time.time(),
                    "file": rel,
                    "rationale": proposal.get("rationale", ""),
                    "patch": patch,
                    "confidence": float(proposal.get("confidence", 0.0)),
                    "test_command": self.test_command,
                    "lint_command": self.lint_command,
                    "auto_commit": self.auto_commit,
                    "auto_push": self.auto_push,
                }
            )
            if len(self._state.pending_proposals) > 50:
                self._state.pending_proposals = self._state.pending_proposals[-50:]
            self._save_state()
            report = (
                f"RLAIF scanner: queued proposal #{proposal_id} for {rel} "
                f"(confidence {proposal.get('confidence', 0):.2f}); awaiting approval."
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

        # Patch introduced new lint errors. Discard.
        logger.warning(
            "RLAIF scanner: {} introduced new lint errors:\n{}",
            rel, preflight.lint_output[-300:],
        )
        report = f"RLAIF scanner: candidate for {rel} introduced new lint errors (skipped)."
        self._state.last_report = report
        logger.info(report)
        return report

    # --- proposal API ------------------------------------------------------

    def list_proposals(self) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in p.items() if k != "patch"}
            | {"preview": p.get("patch", "")[:200]}
            for p in self._state.pending_proposals
        ]

    def get_proposal(self, proposal_id: int) -> dict[str, Any] | None:
        for p in self._state.pending_proposals:
            if p.get("id") == proposal_id:
                return p
        return None

    async def approve_proposal(self, proposal_id: int) -> str:
        prop = self.get_proposal(proposal_id)
        if prop is None:
            return f"proposal #{proposal_id} not found"
        rel = prop["file"]
        patch = prop["patch"]

        harness = PatchHarness(
            repo_root=self.workspace,
            test_command=prop.get("test_command") or self.test_command,
            lint_command=prop.get("lint_command") or self.lint_command,
            timeout=600.0,
        )
        result = await harness.evaluate(patch, patch_summary=prop.get("rationale", ""))
        if not result.passed:
            why = []
            if not result.test_passed:
                why.append("tests")
            if not result.lint_passed:
                why.append("lint")
            return f"approval aborted: {','.join(why) or 'checks'} failed for {rel}"

        from nanobot.agent.tools.rlaif_eval import RlaifEvalTool

        applied = await RlaifEvalTool._apply_diff(patch, workspace=self.workspace)
        status = str(applied)
        if prop.get("auto_commit", True) and applied.startswith("Patch applied"):
            commit_msg = (
                f"rlaif(scanner): {rel}: {prop.get('rationale', '')[:200]}\n\n"
                f"Approved by user (proposal #{proposal_id})."
            )
            commit_result = await asyncio.to_thread(_git_commit, self.workspace, commit_msg)
            status = f"{status}; {commit_result or '(commit failed)'}"
            if (
                prop.get("auto_push", True)
                and commit_result
                and commit_result != "no changes to commit"
            ):
                push_result = await asyncio.to_thread(_git_push, self.workspace)
                status = f"{status}; {push_result or '(push failed)'}"

        try:
            RlaifDataset().append(
                RlaifPreference(
                    prompt=f"Proactive scan approved: {rel}",
                    chosen={"patch": patch, "summary": prop.get("rationale", "")},
                    rejected={"patch": "", "summary": "no challenger (proactive)"},
                    score_chosen=1.0,
                    score_rejected=0.0,
                    reason=f"user-approved proposal #{proposal_id}",
                    task=rel,
                    metadata={
                        "auto_apply": status,
                        "winner_tests": result.test_passed,
                        "winner_lint": result.lint_passed,
                        "winner_backend": result.backend,
                        "scanner_proactive": True,
                        "proposal_id": proposal_id,
                    },
                )
            )
        except Exception:
            logger.exception("RLAIF scanner: failed to record preference")

        self._state.pending_proposals = [
            p for p in self._state.pending_proposals if p.get("id") != proposal_id
        ]
        self._save_state()
        return f"applied proposal #{proposal_id} for {rel}: {status}"

    def reject_proposal(self, proposal_id: int) -> str:
        before = len(self._state.pending_proposals)
        self._state.pending_proposals = [
            p for p in self._state.pending_proposals if p.get("id") != proposal_id
        ]
        if len(self._state.pending_proposals) == before:
            return f"proposal #{proposal_id} not found"
        self._save_state()
        return f"rejected proposal #{proposal_id}"

    # --- file selection ----------------------------------------------------

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

    # --- LLM proposal -----------------------------------------------------

    async def _propose(self, rel_path: str, text: str) -> dict[str, Any] | None:
        """Ask the critic LLM for a unified-diff patch via RlaifEvalTool.

        Returns a dict with 'patch', 'rationale', and 'confidence'.
        """
        from nanobot.agent.rlaif.diff_utils import (
            extract_unified_diff,
            is_valid_unified_diff,
        )
        from nanobot.agent.tools.rlaif_eval import RlaifEvalTool

        abs_path = (self.workspace / rel_path).resolve()
        task = (
            f"Improve {rel_path}. "
            f"File path: {abs_path}. "
            "Pick ONE small, concrete improvement. Allowed categories: "
            "bug fix, dead code removal, better error message, type-hint "
            "fix, docstring fix, simplification. The diff must apply "
            "cleanly with `git apply` to the file as shown."
        )
        tool = RlaifEvalTool(workspace=self.workspace)
        try:
            candidates = await tool._generate_candidates(
                provider=self.provider,
                task=task,
                count=2,
                model=self.critic_model,
                temperature=0.7,
            )
        except Exception:
            logger.exception("RLAIF scanner: critic call failed for {}", rel_path)
            return None

        if not candidates:
            logger.info(
                "RLAIF scanner: critic for {} returned no candidate patch", rel_path,
            )
            return None

        for idx, (summary, raw) in enumerate(candidates):
            patch = extract_unified_diff(raw)
            if not is_valid_unified_diff(patch):
                logger.info(
                    "RLAIF scanner: candidate #{} for {} failed validation. raw content:\n{}",
                    idx + 1, rel_path, raw[:600],
                )
                continue
            # Rewrite the absolute path back to the relative one so
            # git apply produces a tidy diff with the repo-relative
            # a/ and b/ prefixes.
            patch = patch.replace(str(abs_path), rel_path)
            return {
                "rationale": summary or "improvement",
                "patch": patch,
                "confidence": 0.5,
            }

        logger.info(
            "RLAIF scanner: critic for {} produced {} candidate(s) but none were valid unified diffs",
            rel_path, len(candidates),
        )
        return None

    # --- patch recovery / lint helpers ------------------------------------

    @staticmethod
    def _patch_applies_strict(workspace: Path, patch: str) -> bool:
        """Check if `git apply --check` accepts the patch. We do the
        strict check here; the recovery strategies live in the worker
        thread at approve time.
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as f:
                f.write(patch)
                pp = f.name
            r = subprocess.run(
                ["git", "apply", "--check", pp],
                cwd=str(workspace),
                capture_output=True, text=True, timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False
        finally:
            try:
                import os
                os.unlink(pp)
            except (OSError, UnboundLocalError):
                pass

    @staticmethod
    def _run_lint(workspace: Path, lint_command: list[str]) -> str:
        try:
            proc = subprocess.run(
                lint_command,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("RLAIF scanner: pre-patch lint run failed: {}", exc)
            return ""
        return (proc.stdout or "") + (proc.stderr or "")

    @staticmethod
    def _lint_introduced_new(lint_pre: str, lint_post: str, rel_path: str) -> bool:
        """True if every (code, message) in lint_post is also in lint_pre.

        We ignore line numbers (the patch may shift them).
        """
        import re

        def _extract(s: str) -> set[str]:
            out: set[str] = set()
            for raw in s.splitlines():
                line = raw.replace("\r", "").strip()
                if not line:
                    continue
                m = re.match(r"^.*?:(\d+):(\d+):\s*([A-Z]+\d+)\s+(.*)$", line)
                if not m:
                    continue
                out.add(f"{m.group(3)}:{m.group(4)}")
            return out

        pre = _extract(lint_pre)
        post = _extract(lint_post)
        if not post:
            return False
        new = post - pre
        if new:
            logger.info(
                "RLAIF scanner: patch for {} introduced {} new lint errors: {}",
                rel_path, len(new), sorted(new)[:5],
            )
        return bool(new)


def build_scanner_from_config(
    cfg: Any,
    workspace: Path,
    provider: LLMProvider,
    model: str,
    *,
    on_report: Callable[[str], Any] | None = None,
    critic_model: str | None = None,
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
        critic_model=critic_model or getattr(cfg, "scanner_critic_model", None) or model,
        interval_s=float(getattr(cfg, "scanner_interval_s", 3600.0)),
        min_confidence=float(getattr(cfg, "scanner_min_confidence", 0.0)),
        test_command=getattr(cfg, "test_command", None),
        lint_command=getattr(cfg, "lint_command", None),
        auto_apply=getattr(cfg, "scanner_auto_apply", True),
        auto_commit=getattr(cfg, "scanner_auto_commit", True),
        auto_push=getattr(cfg, "scanner_auto_push", True),
        on_report=on_report,
    )
