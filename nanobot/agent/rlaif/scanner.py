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
                "Propose ONE small improvement to the file using an anchor-based "
                "find-and-replace. Specify a small block of EXISTING text (the "
                "'find' field) and the new text that should replace it (the "
                "'replace' field). The system will turn this into a unified "
                "diff that applies with 'git apply'. DO NOT generate a unified "
                "diff yourself — just give the find/replace pair."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": "One-sentence justification for the change.",
                    },
                    "find": {
                        "type": "string",
                        "description": (
                            "A small block of text that appears EXACTLY in the "
                            "file as shown. Include 1-3 lines of context around "
                            "the change to make the match unique. The text "
                            "must match verbatim including indentation."
                        ),
                    },
                    "replace": {
                        "type": "string",
                        "description": (
                            "What the 'find' block should become after the "
                            "change. Same length plus or minus a few lines. "
                            "Keep the same indentation."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in the change being correct and useful.",
                    },
                },
                "required": ["rationale", "find", "replace", "confidence"],
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
        """Ask the critic LLM for a small find/replace improvement.

        Returns a dict with 'patch' (a unified diff the system generated from
        the anchor-based find/replace), 'rationale', and 'confidence'. The
        model never has to produce a unified diff directly; it just gives
        a small block of text that already exists and what it should become.
        """
        # Truncate the file content so the prompt stays bounded.
        max_chars = 12_000
        truncated = text if len(text) <= max_chars else text[:max_chars] + "\n... (truncated)"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior Python reviewer on the nanobot agent framework. "
                    "Your job: propose ONE small, concrete improvement to the file.\n\n"
                    "Use the propose_improvement tool with `find` and `replace` fields. "
                    "The `find` field is a small block of text that appears EXACTLY in "
                    "the file (verbatim, including indentation and trailing whitespace). "
                    "The `replace` field is what that block should become.\n\n"
                    "Allowed categories (pick whichever fits):\n"
                    "  - Bug fix (off-by-one, missing None check, wrong exception type).\n"
                    "  - Dead code removal (unused import, unreachable branch, redundant "
                    "    default, leftover TODO comment).\n"
                    "  - Better error message (include the offending value, log context).\n"
                    "  - Type-hint fix (missing return type, Any that should be specific).\n"
                    "  - Docstring / comment fix (factual error, missing Args/Returns, typo).\n"
                    "  - Simplification (collapse nested if, use a guard, hoist literal).\n\n"
                    "Disallowed: new features, big rewrites, renaming across the file, "
                    "speculative changes. If you truly cannot find anything, return an "
                    "empty 'find' field and confidence 0.2 — but try hard first.\n\n"
                    "CRITICAL FORMAT RULES — the find text will be matched with "
                    "string.find() against the file:\n"
                    "  1. The 'find' text must be a literal copy from the file. "
                    "Do NOT paraphrase, summarize, or 'improve' the existing text.\n"
                    "  2. Each line of the 'find' text must appear in the file exactly "
                    "as written, including leading whitespace, trailing whitespace, "
                    "and line endings.\n"
                    "  3. Copy lines verbatim from the file shown in the user message. "
                    "If you can't copy them exactly, return an empty find.\n"
                    "  4. Include 1-3 lines of surrounding context so the block is "
                    "unique in the file. 3-8 lines total is ideal.\n"
                    "  5. Indentation is 4 spaces (Python standard). Tabs are not used."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## File: {rel_path}\n\n```python\n{truncated}\n```\n\n"
                    "Find ONE small change. The 'find' field MUST be a verbatim copy "
                    "of a small block from the file above. Copy the lines exactly as "
                    "they appear, character for character. Then write the 'replace' "
                    "field as what those lines should become. Call propose_improvement."
                ),
            },
        ]
        try:
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=PROPOSE_TOOL,
                model=self.critic_model,
                max_tokens=4096,
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
            logger.info(
                "RLAIF scanner: critic for {} returned no tool call. content: {}",
                rel_path,
                (response.content or "")[:200],
            )
            return None
        args = response.tool_calls[0].arguments
        if not isinstance(args, dict):
            logger.warning("RLAIF scanner: critic returned non-dict args: {}", args)
            return None

        # Build a unified diff from the find/replace pair (if both present).
        find_text = (args.get("find") or "").strip()
        replace_text = (args.get("replace") or "").strip()
        if not find_text:
            return {
                "rationale": args.get("rationale", ""),
                "patch": "",
                "confidence": float(args.get("confidence", 0.2)),
            }
        patch = self._anchor_to_unified_diff(
            file_text=text,
            rel_path=rel_path,
            find_text=find_text,
            replace_text=replace_text,
        )
        if not patch:
            # ponytail: the critic often hallucinates plausible-looking text
            # instead of copying from the file. Try a fuzzy match first
            # (small substring anchored to a unique fragment) before
            # asking the model to retry with explicit line numbers.
            patch = self._fuzzy_anchor_to_unified_diff(
                file_text=text,
                rel_path=rel_path,
                find_text=find_text,
                replace_text=replace_text,
            )
        if not patch:
            logger.info(
                "RLAIF scanner: critic's find text did not match {} "
                "(rationale: {}, find was: {!r}); retrying with line numbers",
                rel_path, args.get("rationale", "")[:80], find_text[:200],
            )
            retry = await self._retry_with_line_numbers(
                rel_path=rel_path,
                text=text,
                rationale=args.get("rationale", ""),
                find_text=find_text,
                replace_text=replace_text,
            )
            if retry:
                return retry
            logger.warning(
                "RLAIF scanner: giving up on {} after retry; find was: {!r}",
                rel_path, find_text[:200],
            )
            return None
        return {
            "rationale": args.get("rationale", ""),
            "patch": patch,
            "confidence": float(args.get("confidence", 0.5)),
        }

    @staticmethod
    def _anchor_to_unified_diff(
        *,
        file_text: str,
        rel_path: str,
        find_text: str,
        replace_text: str,
    ) -> str:
        """Build a unified diff from a find/replace pair.

        The find text must appear exactly once in the file. The diff has
        enough context for `git apply` to find the location without fuzz.
        """
        if not find_text:
            return ""

        # Normalize line endings.
        find_norm = find_text.replace("\r\n", "\n").rstrip("\n") + "\n"
        replace_norm = replace_text.replace("\r\n", "\n").rstrip("\n") + "\n"
        file_norm = file_text.replace("\r\n", "\n")

        # Look for an exact match.
        if find_norm not in file_norm:
            # Try matching with stripped trailing whitespace.
            stripped = "\n".join(line.rstrip() for line in find_norm.splitlines()) + "\n"
            if stripped not in file_norm:
                return ""
            find_norm = stripped
            replace_norm = (
                "\n".join(line.rstrip() for line in replace_norm.splitlines()) + "\n"
            )

        file_lines = file_norm.splitlines()
        find_lines = find_norm.rstrip("\n").splitlines()
        replace_lines = replace_norm.rstrip("\n").splitlines()

        # Find the start line (0-indexed).
        match_start = None
        for i in range(len(file_lines) - len(find_lines) + 1):
            if file_lines[i : i + len(find_lines)] == find_lines:
                if match_start is not None:
                    # Multiple matches — ambiguous.
                    return ""
                match_start = i
        if match_start is None:
            return ""

        # Build the unified diff with 3 lines of context before and after.
        ctx_before = 3
        ctx_after = 3
        hunk_start_old = max(0, match_start - ctx_before)
        hunk_start_new = hunk_start_old  # context lines preserve alignment
        old_lines = file_lines[hunk_start_old : match_start + len(find_lines) + ctx_after]
        new_lines = (
            file_lines[hunk_start_old : match_start]
            + replace_lines
            + file_lines[match_start + len(find_lines) : match_start + len(find_lines) + ctx_after]
        )

        # Trim context to actual file boundaries.
        # We just truncate the lists; the hunk header counts what we actually
        # emit.
        old_count = len(old_lines)
        new_count = len(new_lines)

        out: list[str] = [
            f"--- a/{rel_path}",
            f"+++ b/{rel_path}",
            f"@@ -{hunk_start_old + 1},{old_count} +{hunk_start_new + 1},{new_count} @@",
        ]
        for line in file_lines[hunk_start_old : match_start]:
            out.append(" " + line)
        for line in find_lines:
            out.append("-" + line)
        for line in replace_lines:
            out.append("+" + line)
        for line in file_lines[match_start + len(find_lines) : match_start + len(find_lines) + ctx_after]:
            out.append(" " + line)
        return "\n".join(out) + "\n"

    @staticmethod
    def _fuzzy_anchor_to_unified_diff(
        *,
        file_text: str,
        rel_path: str,
        find_text: str,
        replace_text: str,
    ) -> str:
        """Try to recover from a hallucinated find by anchoring on a unique line.

        The critic often produces text that LOOKS like the file but has
        small drifts (different var name, wrong whitespace, missing
        context). We try a few cheap fuzzy strategies before giving up:

          1. Find the longest line in the find text; search for it in the
             file; if it appears exactly once, anchor on it and use
             the lines around it as the real `find`.
          2. Same as 1 but ignoring leading whitespace.
          3. Same as 1 but case-insensitive.

        If anchoring succeeds, we still need a sensible `replace_text`
        anchored to the same place. We assume the critic's intent is:
        replace the matched span with the same span but with whatever
        the replace_text says (line-by-line substitution when lengths
        differ, else full swap).
        """
        if not find_text or not file_text:
            return ""

        find_lines = find_text.replace("\r\n", "\n").rstrip("\n").splitlines()
        if not find_lines:
            return ""

        # Pick the longest non-blank line as the anchor.
        anchor_candidates = [ln for ln in find_lines if ln.strip()]
        if not anchor_candidates:
            return ""
        anchor = max(anchor_candidates, key=len)

        file_norm = file_text.replace("\r\n", "\n")
        file_lines = file_norm.splitlines()

        match_idx = RlaifProactiveScanner._find_anchor_in_file(file_lines, anchor)
        if match_idx is None:
            return ""

        # ponytail: figure out the indentation level at the match site
        # so we can re-indent the replace_text if the critic left it
        # unindented (which it usually does). The rule: for each line in
        # replace_text, if its stripped form already exists in the file
        # around the match site, keep its current indent (don't touch);
        # otherwise add the match site's indent.
        match_indent = len(file_lines[match_idx]) - len(file_lines[match_idx].lstrip())
        replace_norm = replace_text.replace("\r\n", "\n").rstrip("\n")
        replace_lines_raw = replace_norm.splitlines()
        # Local file lines around the match (used to detect which replace
        # lines already exist verbatim).
        local_window = file_lines[max(0, match_idx - 8) : min(len(file_lines), match_idx + 8)]
        local_stripped = {ln.strip() for ln in local_window if ln.strip()}
        if match_indent > 0:
            replace_lines = []
            for ln in replace_lines_raw:
                stripped = ln.strip()
                if not stripped:
                    replace_lines.append(ln)
                elif stripped in local_stripped:
                    # Already exists in the file; keep the line but
                    # re-indent it to match the match site (the critic
                    # probably forgot the indent when copying from the
                    # file).
                    replace_lines.append(" " * match_indent + stripped)
                else:
                    # Genuinely new line from the critic — re-indent
                    # so it lines up with the surrounding code.
                    replace_lines.append(" " * match_indent + stripped)
        else:
            replace_lines = replace_lines_raw

        # Build a find block using len(find_lines) lines around the anchor.
        ctx = len(find_lines)
        start = max(0, match_idx - (ctx - 1) // 2)
        end = min(len(file_lines), start + ctx)
        actual_find_lines = file_lines[start:end]
        actual_find = "\n".join(actual_find_lines)

        # Build the replace block: line-by-line substitution where
        # possible (preserving length), full swap otherwise.
        if len(replace_lines) == len(actual_find_lines):
            # Pair them up. The critic usually wants the same context
            # lines plus a small change. Substitute the find_lines
            # that don't match into the actual file lines, keep
            # matching ones as-is. This is a heuristic but works for
            # small edits.
            new_replace: list[str] = []
            for i, actual_line in enumerate(actual_find_lines):
                if i < len(find_lines) and actual_line != find_lines[i]:
                    new_replace.append(replace_lines[i] if i < len(replace_lines) else actual_line)
                else:
                    new_replace.append(actual_line)
            actual_replace = new_replace
        else:
            # Different shape: do a positional replacement centered
            # on the anchor.
            anchor_offset = anchor_candidates.index(anchor)
            before = replace_lines[:anchor_offset]
            after = replace_lines[anchor_offset + 1 :]
            actual_replace = list(file_lines[start:match_idx]) + before + after + list(
                file_lines[match_idx + 1 : end]
            )

        return RlaifProactiveScanner._build_diff_from_lines(
            rel_path=rel_path,
            file_lines=file_lines,
            start=start,
            find_lines=actual_find_lines,
            replace_lines=actual_replace,
        )

    @staticmethod
    def _find_anchor_in_file(file_lines: list[str], anchor: str) -> int | None:
        """Find the line index of `anchor` in file_lines. Tries exact,
        whitespace-stripped, and case-insensitive matches. Returns the
        index of the unique match, or None if there isn't one."""
        # Exact
        matches = [i for i, ln in enumerate(file_lines) if ln == anchor]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None  # ambiguous; bail.
        # Whitespace-stripped
        stripped = anchor.strip()
        matches = [i for i, ln in enumerate(file_lines) if ln.strip() == stripped]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
        # Case-insensitive on stripped
        lowered = stripped.lower()
        matches = [i for i, ln in enumerate(file_lines) if ln.strip().lower() == lowered]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _build_diff_from_lines(
        *,
        rel_path: str,
        file_lines: list[str],
        start: int,
        find_lines: list[str],
        replace_lines: list[str],
        ctx_before: int = 3,
        ctx_after: int = 3,
    ) -> str:
        match_start = start
        hunk_start_old = max(0, match_start - ctx_before)
        old_lines = file_lines[hunk_start_old : match_start + len(find_lines) + ctx_after]
        new_lines = (
            file_lines[hunk_start_old : match_start]
            + replace_lines
            + file_lines[match_start + len(find_lines) : match_start + len(find_lines) + ctx_after]
        )
        old_count = len(old_lines)
        new_count = len(new_lines)
        out: list[str] = [
            f"--- a/{rel_path}",
            f"+++ b/{rel_path}",
            f"@@ -{hunk_start_old + 1},{old_count} +{hunk_start_old + 1},{new_count} @@",
        ]
        for line in file_lines[hunk_start_old : match_start]:
            out.append(" " + line)
        for line in find_lines:
            out.append("-" + line)
        for line in replace_lines:
            out.append("+" + line)
        for line in file_lines[match_start + len(find_lines) : match_start + len(find_lines) + ctx_after]:
            out.append(" " + line)
        return "\n".join(out) + "\n"

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

    async def _retry_with_line_numbers(
        self,
        *,
        rel_path: str,
        text: str,
        rationale: str,
        find_text: str,
        replace_text: str,
    ) -> dict[str, Any] | None:
        """Give the critic a second chance with a numbered version of the file.

        The previous attempt's `find` didn't match the file (hallucinated).
        We send the file with line numbers and ask for `start_line` and
        `end_line` (1-indexed, inclusive) instead of a free-form find.
        Then we look up the actual lines in the file ourselves and build
        the diff. This sidesteps the hallucination problem.
        """
        # Cap the file at ~12k chars so the prompt stays small.
        max_chars = 12_000
        truncated = text if len(text) <= max_chars else text[:max_chars] + "\n... (truncated)"
        numbered = "\n".join(
            f"{i+1:5d}  {ln}" for i, ln in enumerate(truncated.splitlines())
        )
        retry_tool = [
            {
                "type": "function",
                "function": {
                    "name": "specify_line_range",
                    "description": (
                        "Specify the EXACT line range in the file that you want to change. "
                        "Return the start_line and end_line (both 1-indexed, inclusive) of the "
                        "lines to be replaced, and the new text that should replace them."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rationale": {
                                "type": "string",
                                "description": "One-sentence justification.",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": (
                                    "The 1-indexed line number of the first line to be replaced. "
                                    "Must match a line number shown in the user message."
                                ),
                            },
                            "end_line": {
                                "type": "integer",
                                "description": (
                                    "The 1-indexed line number of the last line to be replaced "
                                    "(inclusive). Must be >= start_line."
                                ),
                            },
                            "replace": {
                                "type": "string",
                                "description": (
                                    "The new text that should replace lines [start_line, end_line]. "
                                    "Use exact indentation (4 spaces)."
                                ),
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": ["rationale", "start_line", "end_line", "replace", "confidence"],
                    },
                },
            }
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "Your previous 'find' text didn't match the file. You probably hallucinated. "
                    "Try again, but this time return EXACT line numbers from the numbered file "
                    "below. Pick a small range (1-8 lines) and provide the new text for those "
                    "lines. Do NOT invent code that isn't there — read the numbered lines carefully."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## File: {rel_path}\n\n```\n{numbered}\n```\n\n"
                    f"Previous rationale: {rationale}\n"
                    f"Previous find (didn't match): {find_text[:200]}\n\n"
                    "Call specify_line_range with start_line, end_line, replace, confidence."
                ),
            },
        ]
        try:
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=retry_tool,
                model=self.critic_model,
                max_tokens=4096,
                temperature=0.0,
                tool_choice={
                    "type": "function",
                    "function": {"name": "specify_line_range"},
                },
            )
        except Exception:
            logger.exception("RLAIF scanner: retry-with-lines call failed for {}", rel_path)
            return None
        if not response.has_tool_calls:
            return None
        args = response.tool_calls[0].arguments
        if not isinstance(args, dict):
            return None
        try:
            start = int(args.get("start_line", 0))
            end = int(args.get("end_line", 0))
        except (TypeError, ValueError):
            return None
        replace_text_retry = (args.get("replace") or "").rstrip("\n")
        if not start or not end or end < start:
            return None
        if start < 1 or end > len(text.splitlines()):
            logger.info(
                "RLAIF scanner: retry line range out of bounds: {}-{} (file has {} lines)",
                start, end, len(text.splitlines()),
            )
            return None

        file_lines = text.replace("\r\n", "\n").splitlines()
        # 1-indexed to 0-indexed.
        find_lines = file_lines[start - 1 : end]
        replace_lines = replace_text_retry.splitlines()
        if not find_lines:
            return None
        patch = self._build_diff_from_lines(
            rel_path=rel_path,
            file_lines=file_lines,
            start=start - 1,
            find_lines=find_lines,
            replace_lines=replace_lines,
        )
        # Sanity: the patch must apply with git apply --check.
        import subprocess, tempfile
        try:
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
                    "RLAIF scanner: retry patch unapplyable for {} ({}-{}): {}",
                    rel_path, start, end, r.stderr.strip()[:200],
                )
                return None
        except Exception as exc:
            logger.warning("RLAIF scanner: retry git apply check failed: {}", exc)
            return None

        return {
            "rationale": args.get("rationale") or rationale,
            "patch": patch,
            "confidence": float(args.get("confidence", 0.5)),
        }
