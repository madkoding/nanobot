"""Patch-evaluation harness: apply candidate edits and run tests/lint.

Supports two backends:
- ``git worktree`` (default): clean, fast, preserves the original repo state.
- ``tempfile`` fallback when the repo is not a git checkout or git is unavailable.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# ponytail: explicit source->test mapping for the RLAIF package. A patch that
# touches one of these files only needs its own test file to be judged. Files
# not listed here fall back to the full suite. Add rows as new modules appear.
_SOURCE_TO_TEST = {
    "harness.py": "test_rlaif_harness.py",
    "dataset.py": "test_rlaif_dataset.py",
    "critic.py": "test_rlaif_critic.py",
    "diff_utils.py": "test_rlaif_diff_utils.py",
    "observer.py": "test_rlaif_observer.py",
}


@dataclass
class PatchHarnessResult:
    """Outcome of evaluating one candidate patch."""

    patch: str
    summary: str
    test_passed: bool
    lint_passed: bool
    test_output: str = ""
    lint_output: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    baseline_failed: bool = False
    patch_apply_failed: bool = False

    @property
    def passed(self) -> bool:
        return self.test_passed and self.lint_passed

    @property
    def not_evaluable(self) -> bool:
        """Baseline (clean worktree) already fails; the patch can't be judged."""
        return self.baseline_failed

    @property
    def score_bonus(self) -> float:
        """Simple objective bonus on top of critic score for passing checks."""
        if self.test_passed and self.lint_passed:
            return 2.0
        if self.test_passed:
            return 1.0
        return 0.0


class PatchHarness:
    """Evaluate a candidate patch in a temporary copy of the repository."""

    def __init__(
        self,
        repo_root: Path,
        *,
        test_command: list[str] | None = None,
        lint_command: list[str] | None = None,
        timeout: float = 300.0,
        keep_temp: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve(strict=False)
        self.test_command = test_command or ["python", "-m", "pytest", "-q"]
        self.lint_command = lint_command or ["python", "-m", "ruff", "check", "."]
        self.timeout = timeout
        self.keep_temp = keep_temp
        self._use_git = self._detect_git_worktree_support()
        self._baseline_cache: dict[str, bool] = {}

    async def evaluate(
        self,
        patch_text: str,
        patch_summary: str = "",
    ) -> PatchHarnessResult:
        """Apply the patch to a temp copy of the repo and run test + lint."""
        start = asyncio.get_event_loop().time()
        worktree_dir: Path | None = None
        tmp_dir: Path | None = None
        backend = "unknown"
        try:
            if self._use_git:
                worktree_dir = await self._create_git_worktree()
                target = worktree_dir
                backend = "git-worktree"
            else:
                tmp_dir = Path(tempfile.mkdtemp(prefix="nanobot_rlaif_"))
                await self._prepare_copy(tmp_dir)
                target = tmp_dir
                backend = "temp-copy"

            patch_file = target / "candidate.patch"
            patch_file.write_text(patch_text, encoding="utf-8")

            test_cmd = self._scoped_test_command(patch_text)

            baseline_ok = await self._baseline_check(target, test_cmd)
            if not baseline_ok:
                duration = asyncio.get_event_loop().time() - start
                return PatchHarnessResult(
                    patch=patch_text,
                    summary=patch_summary,
                    test_passed=False,
                    lint_passed=False,
                    test_output="Baseline (clean worktree) failed; patch not evaluable.",
                    baseline_failed=True,
                    duration_s=duration,
                    backend=backend,
                )

            apply = await self._run(
                ["git", "apply", str(patch_file)],
                cwd=target,
            )
            apply_ok = apply.returncode == 0
            if not apply_ok and shutil.which("patch"):
                apply = await self._run(
                    ["patch", "-p1", "-i", str(patch_file)],
                    cwd=target,
                )
                apply_ok = apply.returncode == 0
            if not apply_ok:
                duration = asyncio.get_event_loop().time() - start
                return PatchHarnessResult(
                    patch=patch_text,
                    summary=patch_summary,
                    test_passed=False,
                    lint_passed=False,
                    test_output=f"Patch apply failed:\n{apply.stdout}\n{apply.stderr}",
                    exit_code=apply.returncode,
                    duration_s=duration,
                    backend=backend,
                    patch_apply_failed=True,
                )

            test_proc = await self._run(test_cmd, cwd=target)
            lint_proc = await self._run(self.lint_command, cwd=target)
            # ponytail: ruff is a standalone binary; on some runners `python -m ruff`
            # is not importable. Fall back to the `ruff` executable when that happens.
            if lint_proc.returncode != 0 and self.lint_command[:2] == ["python", "-m"] and self.lint_command[2] == "ruff" and "No module named" in lint_proc.stderr:
                lint_proc = await self._run(["ruff", *self.lint_command[3:]], cwd=target)

            duration = asyncio.get_event_loop().time() - start
            return PatchHarnessResult(
                patch=patch_text,
                summary=patch_summary,
                test_passed=test_proc.returncode == 0,
                lint_passed=lint_proc.returncode == 0,
                test_output=f"{test_proc.stdout}\n{test_proc.stderr}".strip(),
                lint_output=f"{lint_proc.stdout}\n{lint_proc.stderr}".strip(),
                exit_code=test_proc.returncode,
                duration_s=duration,
                backend=backend,
            )
        except Exception as exc:
            logger.exception("PatchHarness.evaluate failed")
            duration = asyncio.get_event_loop().time() - start
            return PatchHarnessResult(
                patch=patch_text,
                summary=patch_summary,
                test_passed=False,
                lint_passed=False,
                test_output=f"Harness error: {exc}",
                duration_s=duration,
                backend=backend,
            )
        finally:
            if worktree_dir is not None and not self.keep_temp:
                await self._remove_git_worktree(worktree_dir)
            if tmp_dir is not None and not self.keep_temp:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _scoped_test_command(self, patch_text: str) -> list[str]:
        """Narrow the test command to the test file for the touched source file."""
        touched = self._touched_files(patch_text)
        test_files = [
            _SOURCE_TO_TEST[name]
            for name in touched
            if name in _SOURCE_TO_TEST
        ]
        if not test_files:
            return self.test_command
        return [*self.test_command, *test_files]

    @staticmethod
    def _touched_files(patch_text: str) -> list[str]:
        files: list[str] = []
        for line in patch_text.splitlines():
            if line.startswith("+++ b/"):
                files.append(line[6:].split("/")[-1])
        return files

    async def _baseline_check(self, target: Path, test_cmd: list[str]) -> bool:
        """Run tests+lint on the clean worktree; cache per target dir."""
        key = str(target)
        if key in self._baseline_cache:
            return self._baseline_cache[key]
        test_proc = await self._run(test_cmd, cwd=target)
        lint_proc = await self._run(self.lint_command, cwd=target)
        ok = test_proc.returncode == 0 and lint_proc.returncode == 0
        self._baseline_cache[key] = ok
        return ok

    def _detect_git_worktree_support(self) -> bool:
        if not shutil.which("git"):
            return False
        git_dir = self.repo_root / ".git"
        try:
            if not (git_dir.is_dir() or git_dir.is_file()):
                return False
        except OSError:
            return False
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except Exception:
            return False

    async def _create_git_worktree(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="nanobot_rlaif_wt_"))
        result = await self._run(
            ["git", "worktree", "add", "-f", str(path), "HEAD"],
            cwd=self.repo_root,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {result.stderr}")
        return path

    async def _remove_git_worktree(self, path: Path) -> None:
        await self._run(
            ["git", "worktree", "remove", "-f", str(path)],
            cwd=self.repo_root,
        )

    async def _prepare_copy(self, target: Path) -> None:
        if shutil.which("rsync"):
            process = await self._run(
                [
                    "rsync",
                    "-a",
                    "--exclude=.git",
                    "--exclude=__pycache__",
                    "--exclude=.venv",
                    "--exclude=venv",
                    "--exclude=node_modules",
                    "--exclude=.pytest_cache",
                    "--exclude=build",
                    "--exclude=dist",
                    str(self.repo_root) + "/",
                    str(target) + "/",
                ],
            )
            if process.returncode != 0:
                raise RuntimeError(f"rsync failed: {process.stderr}")
        else:
            shutil.copytree(
                self.repo_root,
                target,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "node_modules",
                    ".pytest_cache",
                    "build",
                    "dist",
                ),
                dirs_exist_ok=True,
            )

    async def _run(
        self,
        command: list[str],
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        # ponytail: `python` on the PATH may not be the interpreter that has
        # the tools installed (esp. on Windows runners). Resolve module-style
        # commands through the current interpreter so pytest/ruff actually run.
        command = self._resolve_interpreter(command)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                command,
                cwd=cwd or self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            ),
        )

    @staticmethod
    def _resolve_interpreter(command: list[str]) -> list[str]:
        if command and command[0] == "python" and command[1:2] == ["-m"]:
            return [sys.executable, *command[1:]]
        return command
