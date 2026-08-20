"""Tests for nanobot.agent.indexer."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.indexer import WorkspaceIndexer


@pytest.fixture
def indexer(tmp_path: Path) -> WorkspaceIndexer:
    return WorkspaceIndexer(tmp_path)


class TestWorkspaceIndexer:
    def test_needs_reindex_before_build(self, indexer: WorkspaceIndexer) -> None:
        assert indexer.needs_reindex() is True

    def test_index_and_search(self, indexer: WorkspaceIndexer, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def hello_world(): pass\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("def goodbye_world(): pass\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("# hello project\n", encoding="utf-8")

        indexed, removed = indexer.index_workspace()
        assert indexed == 3
        assert removed == 0

        # Tokenizer keeps underscores, so "hello" alone only matches README.md.
        result = indexer.search("hello")
        assert result["files"] == ["README.md"]

        result = indexer.search("hello_world")
        assert sorted(result["files"]) == ["a.py"]

        result = indexer.search("hello_world", output_mode="count")
        assert result["counts"]["a.py"] == 1

    def test_incremental_reindex(self, indexer: WorkspaceIndexer, tmp_path: Path) -> None:
        target = tmp_path / "a.py"
        target.write_text("def foo(): pass\n", encoding="utf-8")
        indexer.index_workspace()

        assert indexer.search("bar")["files"] == []

        target.write_text("def bar(): pass\n", encoding="utf-8")
        indexed, _ = indexer.index_workspace()
        assert indexed == 1

        assert indexer.search("bar")["files"] == ["a.py"]

    def test_skips_ignored_dirs(self, indexer: WorkspaceIndexer, tmp_path: Path) -> None:
        ignored = tmp_path / "node_modules" / "pkg" / "deep.py"
        ignored.parent.mkdir(parents=True)
        ignored.write_text("secret_token\n", encoding="utf-8")
        (tmp_path / "main.py").write_text("secret_token\n", encoding="utf-8")

        indexer.index_workspace()
        result = indexer.search("secret_token")
        assert result["files"] == ["main.py"]

    def test_search_filters_by_file_type(self, indexer: WorkspaceIndexer, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("hello\n", encoding="utf-8")
        (tmp_path / "b.ts").write_text("hello\n", encoding="utf-8")
        indexer.index_workspace()

        assert indexer.search("hello", file_type="py")["files"] == ["a.py"]

    def test_stats(self, indexer: WorkspaceIndexer, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("hello world\n", encoding="utf-8")
        indexer.index_workspace()
        stats = indexer.stats()
        assert stats["files"] == 1
        assert stats["tokens"] == 2
