"""Dataset persistence for RLAIF preference pairs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nanobot.utils.helpers import ensure_dir, timestamp


@dataclass
class RlaifPreference:
    """One preference pair: chosen trajectory beats rejected trajectory."""

    prompt: str
    chosen: dict[str, Any]
    rejected: dict[str, Any]
    score_chosen: float
    score_rejected: float
    reason: str
    task: str = ""
    timestamp: str = field(default_factory=timestamp)
    metadata: dict[str, Any] = field(default_factory=dict)


class RlaifDataset:
    """Append-only JSONL store of preference pairs."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from nanobot.config.paths import get_runtime_subdir

            path = get_runtime_subdir("rlaif") / "preferences.jsonl"
        self._path = ensure_dir(path.parent) / path.name

    @property
    def path(self) -> Path:
        return self._path

    def append(self, preference: RlaifPreference) -> None:
        record = asdict(preference)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()

    def read_all(self) -> list[RlaifPreference]:
        results: list[RlaifPreference] = []
        if not self._path.exists():
            return results
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    results.append(RlaifPreference(**data))
                except Exception:
                    continue
        return results

    def count(self) -> int:
        if not self._path.exists():
            return 0
        count = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
