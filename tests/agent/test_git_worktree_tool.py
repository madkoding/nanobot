"""Tests for the git_worktree inspection tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.agent.tools.git_worktree import GitWorktreeTool


@pytest.fixture
def tool(tmp_path: Path) -> GitWorktreeTool:
    return GitWorktreeTool(workspace=str(tmp_path))


@pytest.mark.asyncio
async def test_status_reports_no_repo(tool: GitWorktreeTool, tmp_path: Path) -> None:
    result = await tool.execute("status")
    assert "cwd:" in result
    assert "in_worktree: False" in result
    assert "branch: (detached/none)" in result
    assert "dirty: False" in result


@pytest.mark.asyncio
async def test_status_reports_inside_mock_repo(tool: GitWorktreeTool, tmp_path: Path) -> None:
    git_file = tmp_path / ".git"
    git_file.write_text("gitdir: /fake/main/.git/worktrees/wt")

    with (
        patch("nanobot.agent.tools.git_worktree.current_tool_workspace") as mock_access,
        patch("nanobot.webui.worktrees.worktree_is_dirty", return_value=True) as mock_dirty,
        patch.object(tool, "_current_branch", return_value="feature-x"),
    ):
        mock_access.return_value = MagicMock(project_path=str(tmp_path))
        result = await tool.execute("status")

    assert "in_worktree: True" in result
    assert "branch: feature-x" in result
    assert "dirty: True" in result
    mock_dirty.assert_called_once()


@pytest.mark.asyncio
async def test_list_reports_no_repo(tool: GitWorktreeTool, tmp_path: Path) -> None:
    result = await tool.execute("list")
    assert "not inside a git repository" in result


@pytest.mark.asyncio
async def test_list_reports_worktrees(tool: GitWorktreeTool, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    fake_worktrees = [
        {"branch": "main", "path": str(tmp_path)},
        {"branch": "feature-x", "path": str(tmp_path / "wt")},
    ]

    with patch("nanobot.webui.worktrees.list_worktrees", return_value=fake_worktrees):
        result = await tool.execute("list")

    assert "repo:" in result
    assert "main" in result
    assert "feature-x" in result


@pytest.mark.asyncio
async def test_list_handles_exception(tool: GitWorktreeTool, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    with patch("nanobot.webui.worktrees.list_worktrees", side_effect=RuntimeError("boom")):
        result = await tool.execute("list")

    assert "failed to list worktrees" in result


@pytest.mark.asyncio
async def test_unknown_action(tool: GitWorktreeTool) -> None:
    result = await tool.execute("nope")
    assert "unknown action: nope" == result
