"""Lightweight BM25 memory store backed by SQLite.

This is a stdlib-only alternative to a full vector database. It stores chunks of
text (conversation turns, memory facts) in SQLite and retrieves the most
relevant chunks for the current query using a simple BM25-style score over
lowercased word tokens.

The store lives at ``<workspace>/memory/memory_bm25.db``.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)

# BM25 parameters — conservative defaults tuned for small personal workspaces.
_K1 = 1.2
_B = 0.75
_EPSILON = 0.25

# Default cap: chunk size and number of chunks returned per turn.
_DEFAULT_CHUNK_SIZE = 512
_DEFAULT_MAX_CHUNKS_PER_TURN = 10


class BM25MemoryStore:
    """SQLite-backed BM25 memory for agent context retrieval."""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace).expanduser().resolve()
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.memory_dir / "memory_bm25.db"
        self._local = threading.local()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _ensure_schema(self) -> None:
        conn = self._connect()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    session_key TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tokens (
                    token TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (token, chunk_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tokens_token ON tokens(token)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at)"
            )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]

    def add_chunk(
        self,
        content: str,
        *,
        source: str = "history",
        session_key: str | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Add one memory chunk and return its id."""
        if not content or not content.strip():
            raise ValueError("content must be non-empty")
        tokens = self._tokenize(content)
        if not tokens:
            raise ValueError("content contains no indexable tokens")
        counts = Counter(tokens)
        created_at = timestamp or time.time()
        conn = self._connect()
        with conn:
            cursor = conn.execute(
                "INSERT INTO chunks (content, source, session_key, created_at) VALUES (?, ?, ?, ?)",
                (content, source, session_key, created_at),
            )
            chunk_id = cursor.lastrowid
            conn.executemany(
                "INSERT INTO tokens (token, chunk_id, count) VALUES (?, ?, ?)",
                [(tok, chunk_id, c) for tok, c in counts.items()],
            )
        return chunk_id

    def add_text(
        self,
        text: str,
        *,
        source: str = "history",
        session_key: str | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        timestamp: float | None = None,
    ) -> list[int]:
        """Split *text* into chunks and index them. Returns chunk ids."""
        ids: list[int] = []
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                ids.append(
                    self.add_chunk(
                        chunk,
                        source=source,
                        session_key=session_key,
                        timestamp=timestamp,
                    )
                )
        return ids

    def search(
        self,
        query: str,
        *,
        session_key: str | None = None,
        source: str | None = None,
        limit: int = _DEFAULT_MAX_CHUNKS_PER_TURN,
    ) -> list[dict[str, Any]]:
        """Return the most relevant chunks for *query* ordered by BM25 score."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        conn = self._connect()
        placeholders = ", ".join("?" for _ in query_tokens)

        # Build optional filters.
        filters: list[str] = []
        params: list[Any] = []
        if session_key is not None:
            filters.append("c.session_key = ?")
            params.append(session_key)
        if source is not None:
            filters.append("c.source = ?")
            params.append(source)
        where = "WHERE " + " AND ".join(filters) if filters else ""

        # Total number of chunks and average length.
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total == 0:
            return []
        avg_len = (
            conn.execute("SELECT AVG(LENGTH(content)) FROM chunks").fetchone()[0]
            or 1.0
        )

        # Document frequency for query tokens.
        df_sql = f"""
            SELECT token, COUNT(DISTINCT chunk_id) AS df
            FROM tokens
            WHERE token IN ({placeholders})
            GROUP BY token
        """
        df_by_token = {
            row[0]: row[1] for row in conn.execute(df_sql, query_tokens).fetchall()
        }

        # Token counts per chunk, restricted by filters.
        token_sql = f"""
            SELECT c.id, c.content, c.source, c.session_key, c.created_at,
                   t.token, t.count
            FROM chunks c
            JOIN tokens t ON c.id = t.chunk_id
            WHERE t.token IN ({placeholders})
            {where.replace("WHERE", "AND") if where else ""}
        """
        rows = conn.execute(token_sql, query_tokens + params).fetchall()

        chunks: dict[int, dict[str, Any]] = {}
        token_counts: dict[int, dict[str, int]] = {}
        for chunk_id, content, src, sk, created_at, token, count in rows:
            if chunk_id not in chunks:
                chunks[chunk_id] = {
                    "id": chunk_id,
                    "content": content,
                    "source": src,
                    "session_key": sk,
                    "created_at": created_at,
                }
                token_counts[chunk_id] = {}
            token_counts[chunk_id][token] = count

        def score(chunk_id: int) -> float:
            chunk_len = len(chunks[chunk_id]["content"])
            doc_norm = _K1 * (1 - _B + _B * (chunk_len / avg_len))
            s = 0.0
            for token in query_tokens:
                df = df_by_token.get(token, 1)
                idf = math.log((total - df + 0.5) / (df + 0.5) + 1)
                tf = token_counts[chunk_id].get(token, 0)
                s += idf * ((tf * (_K1 + 1)) / (tf + doc_norm))
            return s

        results = sorted(
            (chunks[chunk_id] for chunk_id in chunks),
            key=lambda c: score(c["id"]),
            reverse=True,
        )[:limit]
        return results

    def compact(self, keep_last_n: int = 1000) -> int:
        """Drop oldest chunks, keeping *keep_last_n*. Returns deleted count."""
        conn = self._connect()
        with conn:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE id NOT IN ("
                "SELECT id FROM chunks ORDER BY created_at DESC LIMIT ?"
                ")",
                (keep_last_n,),
            )
            return cursor.rowcount

    def stats(self) -> dict[str, int]:
        conn = self._connect()
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        tokens = conn.execute("SELECT COUNT(DISTINCT token) FROM tokens").fetchone()[0]
        return {"chunks": chunks, "tokens": tokens}
