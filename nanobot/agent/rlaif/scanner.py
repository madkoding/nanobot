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
    pending_proposals: list[dict[str, Any]] = field(default_factory=list)
    next_proposal_id: int = 1


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

        # ponytail: by default we run a quick lint-only preflight so obvious
        # garbage (e.g. malformed patches from a non-diff model) is rejected
        # before we burn 30+ seconds on pytest. The full tests are only run
        # once the patch is approved.
        preflight_harness = PatchHarness(
            repo_root=self.workspace,
            test_command=["true"],  # skip pytest on the preflight
            lint_command=self.lint_command,
            timeout=120.0,
        )
        preflight = await preflight_harness.evaluate(
            patch, patch_summary=proposal.get("rationale", "")
        )
        if not preflight.lint_passed:
            logger.warning(
                "RLAIF scanner: {} failed preflight lint:\n{}\n\nlint tail:\n{}",
                rel, patch, preflight.lint_output[-300:],
            )
            report = f"RLAIF scanner: candidate for {rel} failed lint preflight."
            self._state.last_report = report
            logger.info(report)
            return report

        # Quick syntactic check: can `git apply` actually find where the diff
        # goes? If it can't, the patch is malformed — don't bother the user.
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as f:
                f.write(patch)
                pp = f.name
            r = subprocess.run(
                ["git", "apply", "--check", pp],
                cwd=str(self.workspace),
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                logger.warning(
                    "RLAIF scanner: {} has unapplyable diff (will be hidden):\n{}\n\ngit apply said: {}",
                    rel, patch, r.stderr.strip()[:200],
                )
                report = f"RLAIF scanner: candidate for {rel} has unapplyable diff (skipped)."
                self._state.last_report = report
                logger.info(report)
                return report
        except Exception as exc:
            logger.warning("RLAIF scanner: preflight git apply check failed: {}", exc)

        # Patch is lint-clean and syntactically applyable. Save as a pending
        # proposal. The user reviews it in the WebUI and either approves
        # (which then runs the full tests + commit + push) or rejects.
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
        # Cap to the most recent 50 to keep the state file small.
        if len(self._state.pending_proposals) > 50:
            self._state.pending_proposals = self._state.pending_proposals[-50:]

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

    def list_proposals(self) -> list[dict[str, Any]]:
        """Return a copy of the pending proposals (without the patch body)."""
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
        """Apply a pending proposal: run full tests, apply, commit, push."""
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

        # Record the preference in the dataset so it counts for offline DPO/GRPO.
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

        # Remove the proposal from the pending list.
        self._state.pending_proposals = [
            p for p in self._state.pending_proposals if p.get("id") != proposal_id
        ]
        self._save_state()
        return f"applied proposal #{proposal_id} for {rel}: {status}"

    def reject_proposal(self, proposal_id: int) -> str:
        """Drop a pending proposal without applying it."""
        before = len(self._state.pending_proposals)
        self._state.pending_proposals = [
            p for p in self._state.pending_proposals if p.get("id") != proposal_id
        ]
        if len(self._state.pending_proposals) == before:
            return f"proposal #{proposal_id} not found"
        self._save_state()
        return f"rejected proposal #{proposal_id}"

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
                    "You are a senior Python reviewer on the nanobot agent framework. "
                    "Your job: propose ONE small, concrete improvement to the file as a "
                    "unified diff.\n\n"
                    "Allowed categories (pick whichever fits):\n"
                    "  - Bug fix (off-by-one, missing None check, wrong exception type, "
                    "    swallowed error, race condition, etc.)\n"
                    "  - Dead code removal (unused import, unreachable branch, redundant "
                    "    None default, leftover TODO comment, etc.)\n"
                    "  - Better error message (e.g. include the offending value, suggest "
                    "    a fix, log the context).\n"
                    "  - Type-hint fix (missing return type, Any that should be specific, "
                    "    Optional that is always None, etc.)\n"
                    "  - Docstring / comment fix (factual error, missing Args/Returns, "
                    "    typo).\n"
                    "  - Simplification (collapse nested if, use a guard, replace repeated "
                    "    literal with constant).\n\n"
                    "Disallowed: new features, big rewrites, renaming across the file, "
                    "speculative changes. If you truly cannot find anything, return an "
                    "empty patch and confidence 0.3 — but try hard first.\n\n"
                    "CRITICAL FORMAT for the diff (otherwise the patch will be rejected):\n"
                    "  1. The hunk header (e.g. @@ -10,6 +10,12 @@) must list the EXACT "
                    "number of lines that follow in old and new context. Count them.\n"
                    "  2. Every old line (with leading space) must appear verbatim in the "
                    "file at the hunk's location.\n"
                    "  3. Include 3 lines of unchanged context before and after the change.\n"
                    "  4. The hunk must be COMPLETE: every open paren / bracket / quote "
                    "must be closed. Never end a hunk in the middle of a line.\n"
                    "  5. If you're not 100% sure the diff will apply, return empty patch."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## File: {rel_path}\n\n```python\n{truncated}\n```\n\n"
                    "Look hard for ONE small improvement. The kinds of changes I want "
                    "are tiny — fix a typo in a docstring, swap a bare `except:` for a "
                    "specific exception, add a missing return type, remove a dead import, "
                    "tighten an error message. Be conservative but DO propose something. "
                    "Use the propose_improvement tool with a unified diff."
                ),
            },
        ]
        try:
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=PROPOSE_TOOL,
                model=self.critic_model,
                max_tokens=8192,
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
            # Fallback: try to extract a unified diff from the raw content.
            # Some code-tuned models write diffs in plain text instead of
            # using the tool-calling API.
            content = response.content or ""
            extracted = self._extract_diff_from_text(content)
            if extracted:
                logger.info(
                    "RLAIF scanner: critic for {} returned no tool call, but "
                    "extracted diff from raw content ({} chars).",
                    rel_path, len(extracted),
                )
                return {
                    "rationale": "extracted from raw content",
                    "patch": extracted,
                    "confidence": 0.5,
                }
            logger.info(
                "RLAIF scanner: critic for {} returned no tool call. content: {}",
                rel_path,
                content[:200],
            )
            return None
        args = response.tool_calls[0].arguments
        if not isinstance(args, dict):
            logger.warning("RLAIF scanner: critic returned non-dict args: {}", args)
            return None
        return args

    @staticmethod
    def _extract_diff_from_text(text: str) -> str:
        """Pull a unified diff out of a free-form LLM response. Returns '' if none.

        Supports three formats seen in the wild:
          1. Standard unified diff (--- a/... +++ b/...)
          2. OpenAI "*** Begin Patch / *** Update File: / *** End Patch" style
          3. Raw `@@ -N,M +N,K @@` hunks without a header
        """
        if not text:
            return ""
        lines = text.splitlines()

        # 1. Standard unified diff
        for i, line in enumerate(lines):
            if line.startswith("--- a/") and i + 1 < len(lines) and lines[i + 1].startswith("+++ b/"):
                return "\n".join(lines[i:]).rstrip() + "\n"

        # 2. OpenAI custom patch format
        if any(l.startswith("*** Begin Patch") for l in lines):
            return RlaifProactiveScanner._translate_openai_patch(lines)

        # 3. Raw hunks — find first @@
        for i, line in enumerate(lines):
            if line.startswith("@@"):
                return RlaifProactiveScanner._wrap_hunks_with_headers(lines[i:])

        return ""

    @staticmethod
    def _translate_openai_patch(lines: list[str]) -> str:
        """Convert OpenAI's *** Begin Patch format to a unified diff."""
        out: list[str] = []
        current_file: str | None = None
        hunks: list[list[str]] = []
        hunk: list[str] = []
        for raw in lines:
            line = raw.rstrip("\n")
            if line.startswith("*** Begin Patch"):
                continue
            if line.startswith("*** End Patch"):
                if hunk:
                    hunks.append(hunk)
                    hunk = []
                break
            if line.startswith("*** Update File:"):
                if hunk:
                    hunks.append(hunk)
                    hunk = []
                current_file = line.split(":", 1)[1].strip()
                continue
            if line.startswith("*** Add File:") or line.startswith("*** Delete File:"):
                # Not supported yet
                return ""
            if line.startswith("@@"):
                if hunk:
                    hunks.append(hunk)
                hunk = [line]
                continue
            if hunk is not None:
                hunk.append(line)
        if hunk:
            hunks.append(hunk)
        if not current_file or not hunks:
            return ""
        for h in hunks:
            out.append(f"--- a/{current_file}")
            out.append(f"+++ b/{current_file}")
            for ln in h:
                out.append(ln)
        return "\n".join(out) + "\n"

    @staticmethod
    def _wrap_hunks_with_headers(hunk_lines: list[str]) -> str:
        """If the model emitted hunks without `--- a/` headers, add a placeholder.

        We don't know the file here; the caller (RlaifProactiveScanner) sets the
        file path later via _normalize_patch_paths, so we use a placeholder.
        """
        out = ["--- a/_", "+++ b/_"]
        out.extend(hunk_lines)
        return "\n".join(out) + "\n"

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
        min_confidence=float(getattr(cfg, "scanner_min_confidence", 0.7)),
        test_command=getattr(cfg, "test_command", None),
        lint_command=getattr(cfg, "lint_command", None),
        auto_apply=getattr(cfg, "scanner_auto_apply", True),
        auto_commit=getattr(cfg, "scanner_auto_commit", True),
        auto_push=getattr(cfg, "scanner_auto_push", True),
        on_report=on_report,
    )
