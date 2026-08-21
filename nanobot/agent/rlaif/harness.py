"""Patch-evaluation harness: apply candidate edits and run tests/lint.

Supports two backends:
- ``git worktree`` (default): clean, fast, preserves the original repo state.
- ``tempfile`` fallback when the repo is not a git checkout or git is unavailable.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


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

    @property
    def passed(self) -> bool:
        return self.test_passed and self.lint_passed

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
        timeout: float = 120.0,
        keep_temp: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve(strict=False)
        self.test_command = test_command or ["python", "-m", "pytest", "-q"]
        self.lint_command = lint_command or ["python", "-m", "ruff", "check", "."]
        self.timeout = timeout
        self.keep_temp = keep_temp
        self._use_git = self._detect_git_worktree_support()

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
                )

            test_proc = await self._run(self.test_command, cwd=target)
            lint_proc = await self._run(self.lint_command, cwd=target)

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
