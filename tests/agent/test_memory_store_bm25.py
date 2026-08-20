"""Tests for MemoryStore BM25 integration."""

from __future__ import annotations

from pathlib import Path

from nanobot.agent.memory_store import MemoryStore


class TestMemoryStoreBM25:
    def test_append_history_indexes_bm25(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.append_history(
            "remember to buy pizza ingredients",
            session_key="cli:direct",
            sender_id="user",
        )

        results = store.search_memory("pizza ingredients")
        assert len(results) == 1
        assert "pizza" in results[0]["content"]

    def test_search_memory_session_filter(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.append_history("work project deadline tomorrow", session_key="work")
        store.append_history("buy milk", session_key="personal")

        results = store.search_memory("deadline", session_key="work")
        assert len(results) == 1
        assert "work" in results[0]["session_key"]
