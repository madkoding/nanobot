"""Tests for search.py walk caps and indexer integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.indexer import WorkspaceIndexer
from nanobot.agent.tools.search import FindFilesTool, GrepTool


@pytest.fixture
def find_tool(tmp_path: Path) -> FindFilesTool:
    tool = FindFilesTool.__new__(FindFilesTool)
    tool.__init__(workspace=tmp_path)
    return tool


@pytest.fixture
def grep_tool(tmp_path: Path) -> GrepTool:
    tool = GrepTool.__new__(GrepTool)
    tool.__init__(workspace=tmp_path)
    return tool


class TestFindFilesTool:
    async def test_head_limit_zero_uses_default(
        self, find_tool: FindFilesTool, tmp_path: Path
    ) -> None:
        for i in range(250):
            (tmp_path / f"f{i:03d}.py").write_text("x", encoding="utf-8")
        result = await find_tool.execute(path=str(tmp_path), head_limit=0)
        assert "(scanned up to 20000 files)" not in result
        # Should not return all 250 files; default cap is 200.
        assert len(result.splitlines()) <= 210

    async def test_walk_file_cap_note(
        self, find_tool: FindFilesTool, tmp_path: Path
    ) -> None:
        # Create enough files to trip the walked-files cap quickly.
        for i in range(20010):
            (tmp_path / f"f{i:05d}.txt").write_text("x", encoding="utf-8")
        result = await find_tool.execute(path=str(tmp_path))
        assert "(scanned up to 20000 files)" in result


class TestGrepTool:
    async def test_grep_uses_index_when_available(
        self, grep_tool: GrepTool, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text("hello world = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("goodbye = 1\n", encoding="utf-8")
        WorkspaceIndexer(tmp_path).index_workspace()

        result = await grep_tool.execute(
            pattern="hello",
            path=str(tmp_path),
            output_mode="files_with_matches",
        )
        assert "a.py" in result
        assert "b.py" not in result
        assert "(used workspace index)" in result

    async def test_grep_without_index_does_not_claim_index(
        self, grep_tool: GrepTool, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text("hello world = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("hello again = 1\n", encoding="utf-8")

        result = await grep_tool.execute(
            pattern="hello",
            path=str(tmp_path),
            output_mode="files_with_matches",
        )
        assert "a.py" in result
        assert "b.py" in result
        assert "(used workspace index)" not in result

    async def test_grep_head_limit_zero_clamped(
        self, grep_tool: GrepTool, tmp_path: Path
    ) -> None:
        for i in range(300):
            (tmp_path / f"f{i:03d}.py").write_text("hello\n", encoding="utf-8")
        result = await grep_tool.execute(
            pattern="hello",
            path=str(tmp_path),
            output_mode="files_with_matches",
            head_limit=0,
        )
        # Default limit is 250, head_limit=0 is treated as default.
        assert len(result.splitlines()) <= 260
