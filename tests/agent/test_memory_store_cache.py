"""Tests for MemoryStore history.jsonl mtime cache."""

from __future__ import annotations

from pathlib import Path

from nanobot.agent.memory_store import MemoryStore


class TestMemoryStoreHistoryCache:
    def test_history_cache_uses_mtime(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.append_history(
            "first summary",
            session_key="cli:direct",
            sender_id="user",
        )

        # First read parses the file and caches.
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert store._history_entries_cache is not None

        # Same mtime and size: cache hit, no re-parse.
        cached_entries = store.read_unprocessed_history(since_cursor=0)
        assert cached_entries is entries

    def test_history_cache_invalidates_on_write(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.append_history(
            "first summary",
            session_key="cli:direct",
            sender_id="user",
        )
        store.read_unprocessed_history(since_cursor=0)
        assert store._history_entries_cache is not None
        old_cache = store._history_entries_cache

        store.append_history(
            "second summary",
            session_key="cli:direct",
            sender_id="user",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        # Cache key changed because file mtime/size changed.
        assert store._history_entries_cache[1] == entries
        assert store._history_entries_cache is not old_cache  # rebuilt
