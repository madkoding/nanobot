"""Tests for the RLAIF patch-evaluation harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.rlaif.harness import PatchHarness


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.ruff]\nline-length = 100\n", encoding="utf-8"
    )
    (tmp_path / "test_sample.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


class TestPatchHarness:
    @pytest.mark.asyncio
    async def test_evaluate_passing_patch(self, repo: Path) -> None:
        patch = (
            "--- a/src/__init__.py\n"
            "+++ b/src/__init__.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        harness = PatchHarness(
            repo_root=repo,
            test_command=["python", "-m", "pytest", "-q"],
            lint_command=["python", "-m", "ruff", "check", "."],
        )
        result = await harness.evaluate(patch, "change x")
        assert result.test_passed is True
        assert result.lint_passed is True
        assert result.backend == "temp-copy"

    @pytest.mark.asyncio
    async def test_evaluate_failing_test(self, repo: Path) -> None:
        patch = (
            "--- a/test_sample.py\n"
            "+++ b/test_sample.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def test_ok():\n"
            "-    assert True\n"
            "+    assert False\n"
        )
        harness = PatchHarness(
            repo_root=repo,
            test_command=["python", "-m", "pytest", "-q"],
            lint_command=["python", "-m", "ruff", "check", "."],
        )
        result = await harness.evaluate(patch, "break test")
        assert result.test_passed is False
        assert result.lint_passed is True

    @pytest.mark.asyncio
    async def test_evaluate_invalid_patch(self, repo: Path) -> None:
        harness = PatchHarness(repo_root=repo)
        result = await harness.evaluate("not a patch", "bad patch")
        assert result.test_passed is False
        assert "Patch apply failed" in result.test_output
