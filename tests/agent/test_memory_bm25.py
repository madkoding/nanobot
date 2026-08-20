"""Tests for nanobot.agent.memory_bm25."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.memory_bm25 import BM25MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> BM25MemoryStore:
    return BM25MemoryStore(tmp_path)


class TestBM25MemoryStore:
    def test_add_and_search(self, store: BM25MemoryStore) -> None:
        store.add_chunk("I like pizza and pasta")
        store.add_chunk("The weather is sunny today")
        store.add_chunk("My favorite food is pizza")

        results = store.search("pizza")
        assert len(results) == 2
        assert all("pizza" in r["content"].lower() for r in results)

    def test_search_filters_by_session(self, store: BM25MemoryStore) -> None:
        store.add_chunk("project alpha status is green", session_key="work")
        store.add_chunk("buy milk", session_key="personal")

        results = store.search("project", session_key="work")
        assert len(results) == 1
        assert results[0]["session_key"] == "work"

    def test_add_text_splits(self, store: BM25MemoryStore) -> None:
        text = "word " * 1000
        ids = store.add_text(text, chunk_size=50)
        assert len(ids) == 20

    def test_compact(self, store: BM25MemoryStore) -> None:
        for i in range(10):
            store.add_chunk(f"chunk {i}")
        deleted = store.compact(keep_last_n=5)
        assert deleted == 5
        assert store.stats()["chunks"] == 5

    def test_stats(self, store: BM25MemoryStore) -> None:
        store.add_chunk("hello world hello again")
        stats = store.stats()
        assert stats["chunks"] == 1
        assert stats["tokens"] == 3  # hello, world, again
