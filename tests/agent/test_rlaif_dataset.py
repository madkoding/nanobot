"""Tests for RLAIF dataset persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.rlaif.dataset import RlaifDataset, RlaifPreference


@pytest.fixture
def dataset(tmp_path: Path) -> RlaifDataset:
    return RlaifDataset(path=tmp_path / "prefs.jsonl")


class TestRlaifDataset:
    def test_append_and_count(self, dataset: RlaifDataset) -> None:
        dataset.append(
            RlaifPreference(
                prompt="task",
                chosen={"patch": "p1"},
                rejected={"patch": "p2"},
                score_chosen=4.0,
                score_rejected=2.0,
                reason="better",
            )
        )
        assert dataset.count() == 1
        all_items = dataset.read_all()
        assert len(all_items) == 1
        assert all_items[0].prompt == "task"

    def test_read_corrupt_lines_are_ignored(self, dataset: RlaifDataset) -> None:
        dataset._path.write_text('{"prompt":"ok"}\nnot-json\n', encoding="utf-8")
        items = dataset.read_all()
        assert len(items) == 0
