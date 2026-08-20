"""BM25 memory retrieval for history.jsonl (stdlib only).

No sentence-transformers, no numpy, no disk-heavy vector index. This keeps
memory search working on constrained/home directories with quota limits and
spinning disks.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-ish tokenization; CJK chars count as tokens."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _avgdl(docs: list[list[str]]) -> float:
    if not docs:
        return 0.0
    return sum(len(d) for d in docs) / len(docs)


class BM25Index:
    """Tiny in-memory BM25 index."""

    def __init__(self, documents: list[tuple[str, Any]]):
        """*documents* is [(text, payload), ...]; payloads are returned as-is."""
        self.payloads = [doc[1] for doc in documents]
        self.doc_tokens = [_tokenize(doc[0]) for doc in documents]
        self.doc_freqs: dict[str, int] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freqs.update(set(tokens))
        self.n = len(documents)
        self.avgdl = _avgdl(self.doc_tokens)
        self.k1 = 1.2
        self.b = 0.75

    def search(self, query: str, top_k: int = 10) -> list[tuple[Any, float]]:
        """Return (payload, score) ranked descending by BM25 score."""
        q_tokens = _tokenize(query)
        if not q_tokens or not self.doc_tokens:
            return []

        idfs: dict[str, float] = {}
        for token in set(q_tokens):
            df = self.doc_freqs.get(token, 0)
            # standard BM25 IDF with +0.5 smoothing
            idfs[token] = math.log(1 + (self.n - df + 0.5) / (df + 0.5))

        q_counts = Counter(q_tokens)
        scored: list[tuple[int, float]] = []
        for idx, tokens in enumerate(self.doc_tokens):
            if not tokens:
                continue
            dl = len(tokens)
            denom = dl / self.avgdl if self.avgdl else 1.0
            score = 0.0
            for token, qf in q_counts.items():
                tf = tokens.count(token)
                if tf == 0:
                    continue
                score += idfs.get(token, 0.0) * (
                    tf * (self.k1 + 1)
                ) / (tf + self.k1 * (1 - self.b + self.b * denom))
            if score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [(self.payloads[idx], score) for idx, _ in scored[:top_k]]


def build_history_index(history_file: Path) -> BM25Index | None:
    """Build a BM25 index from history.jsonl lines."""
    if not history_file.exists():
        return None
    documents: list[tuple[str, Any]] = []
    try:
        with history_file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json

                    entry = json.loads(line)
                except Exception:
                    continue
                if not isinstance(entry, dict):
                    continue
                text_parts = [
                    entry.get("content", ""),
                    entry.get("session_key", ""),
                    entry.get("timestamp", ""),
                ]
                text = " ".join(str(p) for p in text_parts if p)
                documents.append((text, entry))
    except OSError:
        return None
    return BM25Index(documents)
