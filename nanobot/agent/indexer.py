"""Lightweight SQLite inverted index for workspace file content search.

The index lives next to the workspace it indexes:

    <workspace>/.nanobot_index/index.db

It stores one row per file (path, mtime_ns, size, sha256) and one row per
lowercased word token per file. Reindexing is incremental: only files whose
mtime or size changed are re-read and re-tokenized.

This is deliberately simple. It trades off a small amount of startup time and
disk space for avoiding repeated full-tree `os.walk` + `read()` scans on every
`grep` call, which is painful on spinning disks.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nanobot.agent.tools.filesystem import ListDirTool

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*", re.ASCII)
# Shared ignore list with filesystem/search tools.
_IGNORE_DIRS = set(ListDirTool._IGNORE_DIRS)


class WorkspaceIndexer:
    """Incremental inverted index backed by SQLite."""

    # ponytail: keep tokens <= 64 chars to avoid abuse / bloated index rows.
    _MAX_TOKEN_LEN = 64
    # ponytail: skip files larger than 10 MB for indexing; still tracked as stale.
    _MAX_INDEX_BYTES = 10 * 1024 * 1024

    # ponytail: after this many seconds without a full reindex, treat the index
    # as stale so callers fall back to a live walk instead of serving old data.
    STALE_THRESHOLD_S = 300

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace).expanduser().resolve()
        self._index_dir = self.workspace / ".nanobot_index"
        self._db_path = self._index_dir / "index.db"
        self._lock = threading.RLock()
        self._local = threading.local()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        # Each thread gets its own connection.
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self._index_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tokens (
                    token TEXT NOT NULL,
                    path TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (token, path)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tokens_token ON tokens(token)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                )
                """
            )

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def index_workspace(self, *, progress_every: int = 500) -> tuple[int, int]:
        """Incrementally index the workspace. Returns (indexed_files, removed_files)."""
        indexed = 0
        removed = 0
        seen: set[str] = set()
        conn = self._connect()

        # Walk the tree, collecting candidate files and pruning ignored dirs.
        for dirpath, dirnames, filenames in os.walk(self.workspace):
            dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
            if ".nanobot_index" in dirnames:
                dirnames.remove(".nanobot_index")
            current = Path(dirpath)
            for filename in filenames:
                file_path = current / filename
                rel = file_path.relative_to(self.workspace).as_posix()
                seen.add(rel)
                if self._index_file(conn, file_path, rel):
                    indexed += 1
                    if indexed % progress_every == 0:
                        conn.commit()

        # Remove entries for files that no longer exist.
        cursor = conn.execute("SELECT path FROM files")
        existing = {row[0] for row in cursor.fetchall()}
        to_remove = existing - seen
        if to_remove:
            with conn:
                for rel in to_remove:
                    conn.execute("DELETE FROM tokens WHERE path = ?", (rel,))
                    conn.execute("DELETE FROM files WHERE path = ?", (rel,))
                    removed += 1

        # Record completion metadata so callers can detect staleness cheaply.
        try:
            workspace_mtime = int(self.workspace.stat().st_mtime_ns)
        except OSError:
            workspace_mtime = 0
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("last_scan_at", int(time.time())),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("workspace_mtime_ns", workspace_mtime),
            )

        conn.commit()
        return indexed, removed

    def _index_file(self, conn: sqlite3.Connection, file_path: Path, rel: str) -> bool:
        try:
            stat = file_path.stat()
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
        except OSError:
            return False

        # Skip binaries and huge files.
        if size > self._MAX_INDEX_BYTES:
            return False

        row = conn.execute(
            "SELECT mtime_ns, size, sha256 FROM files WHERE path = ?", (rel,)
        ).fetchone()

        if row is not None:
            stored_mtime, stored_size, stored_sha = row
            if stored_mtime == mtime_ns and stored_size == size:
                # Already up to date.
                return False

        try:
            with file_path.open("rb") as f:
                data = f.read()
        except OSError:
            return False

        if b"\x00" in data:
            return False  # binary
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False

        sha256 = hashlib.sha256(data).hexdigest()
        if row is not None and row[2] == sha256:
            # Same content; just update metadata without re-tokenizing.
            with conn:
                conn.execute(
                    "UPDATE files SET mtime_ns = ?, size = ? WHERE path = ?",
                    (mtime_ns, size, rel),
                )
            return False

        tokens = _tokenize(text)
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO files (path, mtime_ns, size, sha256) VALUES (?, ?, ?, ?)",
                (rel, mtime_ns, size, sha256),
            )
            conn.execute("DELETE FROM tokens WHERE path = ?", (rel,))
            conn.executemany(
                "INSERT INTO tokens (token, path, count) VALUES (?, ?, ?)",
                [(tok, rel, count) for tok, count in token_counts.items()],
            )
        return True

    def search(
        self,
        pattern: str,
        *,
        output_mode: str = "files_with_matches",
        case_insensitive: bool = False,
        fixed_strings: bool = False,
        glob: str | None = None,
        file_type: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Query the index.

        Returns a dict with:
            files: list of relative paths (sorted by mtime desc)
            counts: dict path -> match count (for output_mode == "count")
            snippets: dict path -> list of (line_no, line_text) (for content mode)
            stale_index: bool (True if index is missing or older than workspace)
            limit_hit: bool
        """
        import fnmatch

        if fixed_strings:
            terms = _tokenize(pattern) or [pattern.lower()]
        else:
            terms = _tokenize(pattern) or [pattern.lower()]

        if not terms:
            return {
                "files": [],
                "counts": {},
                "snippets": {},
                "stale_index": self.needs_reindex(),
                "limit_hit": False,
            }

        conn = self._connect()
        placeholders = ", ".join("?" for _ in terms)
        query = f"""
            SELECT t.path, SUM(t.count) as score
            FROM tokens t
            WHERE t.token IN ({placeholders})
            GROUP BY t.path
            ORDER BY score DESC
        """
        cursor = conn.execute(query, terms)
        rows = cursor.fetchall()

        # Apply filters and load snippets for content mode.
        files: list[str] = []
        counts: dict[str, int] = {}
        snippets: dict[str, list[tuple[int, str]]] = {}

        if glob:
            glob_pat = glob

        for rel, score in rows:
            if glob and not fnmatch.fnmatch(rel, glob_pat):
                continue
            if file_type and not _matches_file_type(rel, file_type):
                continue
            files.append(rel)
            if output_mode == "count":
                counts[rel] = score
            if output_mode == "content":
                snippets[rel] = self._snippets_for(rel, pattern, case_insensitive, fixed_strings)
            if limit is not None and len(files) >= limit:
                break

        # Sort by file mtime desc, then path.
        mtimes = self._mtimes(files)
        files.sort(key=lambda p: (-mtimes.get(p, 0), p))

        return {
            "files": files,
            "counts": counts,
            "snippets": snippets,
            "stale_index": self.needs_reindex(),
            "limit_hit": limit is not None and len(files) >= limit,
        }

    def _snippets_for(
        self,
        rel: str,
        pattern: str,
        case_insensitive: bool,
        fixed_strings: bool,
    ) -> list[tuple[int, str]]:
        import re as _re

        file_path = self.workspace / rel
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        flags = _re.IGNORECASE if case_insensitive else 0
        try:
            needle = _re.escape(pattern) if fixed_strings else pattern
            regex = _re.compile(needle, flags)
        except _re.error:
            return []

        results: list[tuple[int, str]] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                results.append((idx, line))
                if len(results) >= 20:
                    break
        return results

    def _mtimes(self, paths: Iterable[str]) -> dict[str, int]:
        if not paths:
            return {}
        conn = self._connect()
        placeholders = ", ".join("?" for _ in paths)
        cursor = conn.execute(
            f"SELECT path, mtime_ns FROM files WHERE path IN ({placeholders})",
            list(paths),
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def needs_reindex(self) -> bool:
        """Return True if the index is empty, old, or workspace changed."""
        if not self._db_path.exists():
            return True
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
        if row is None or row[0] == 0:
            return True
        meta = {
            key: value
            for key, value in conn.execute("SELECT key, value FROM meta").fetchall()
        }
        if "last_scan_at" not in meta or "workspace_mtime_ns" not in meta:
            return True
        if int(time.time()) - int(meta["last_scan_at"]) > self.STALE_THRESHOLD_S:
            return True
        try:
            current_mtime = int(self.workspace.stat().st_mtime_ns)
        except OSError:
            return True
        return current_mtime != int(meta["workspace_mtime_ns"])

    def stats(self) -> dict[str, int]:
        conn = self._connect()
        file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        token_count = conn.execute(
            "SELECT COUNT(DISTINCT token) FROM tokens"
        ).fetchone()[0]
        return {"files": file_count, "tokens": token_count}


async def index_workspace_async(workspace: Path | str, *, progress_every: int = 500) -> tuple[int, int]:
    """Async wrapper around WorkspaceIndexer.index_workspace."""
    indexer = WorkspaceIndexer(workspace)
    try:
        return await asyncio.to_thread(indexer.index_workspace, progress_every=progress_every)
    finally:
        await asyncio.to_thread(indexer.close)


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = []
    for match in _TOKEN_RE.finditer(lowered):
        token = match.group(0)
        if len(token) <= WorkspaceIndexer._MAX_TOKEN_LEN:
            tokens.append(token)
    return tokens


def _matches_file_type(name: str, file_type: str) -> bool:
    from nanobot.agent.tools.search import _TYPE_GLOB_MAP

    lowered = file_type.strip().lower()
    patterns = _TYPE_GLOB_MAP.get(lowered, (f"*.{lowered}",))
    import fnmatch

    return any(fnmatch.fnmatch(name.lower(), pattern.lower()) for pattern in patterns)
