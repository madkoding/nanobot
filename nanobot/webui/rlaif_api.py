"""WebUI RLAIF watch API: read preference pairs and gateway log lines.

Read-only. Polled by the WebUI every couple of seconds; no streaming.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


DEFAULT_LOG_TAIL = 200
DEFAULT_LOG_MAX_BYTES = 256 * 1024
RLAIF_LOG_PATTERN = "rlaif"


def _preferences_path() -> Path:
    from nanobot.config.paths import get_runtime_subdir

    return get_runtime_subdir("rlaif") / "preferences.jsonl"


def _gateway_log_path() -> Path:
    from nanobot.gateway.runtime import GatewayRuntimePaths

    return GatewayRuntimePaths.for_instance().log_path


def read_preferences(
    *,
    offset: int = 0,
    limit: int | None = None,
    since_index: int | None = None,
) -> dict[str, Any]:
    """Return preference rows from ``preferences.jsonl``.

    The file is append-only JSONL. ``offset`` is a line index (0-based) used
    for pagination from the start; ``since_index`` returns only rows whose
    index is strictly greater than the cursor (for polling incremental diffs).
    ``limit`` caps the returned rows when set.
    """
    path = _preferences_path()
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return {"path": str(path), "total": 0, "items": [], "next_index": 0}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if since_index is not None and index <= since_index:
                continue
            if offset and index < offset:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("rlaif_api: malformed preference line at {}", index)
                continue
            record["_index"] = index
            rows.append(record)
            if limit is not None and len(rows) >= limit:
                break

    total = _count_nonempty_lines(path)
    next_index = rows[-1]["_index"] + 1 if rows else (since_index or 0)
    return {"path": str(path), "total": total, "items": rows, "next_index": next_index}


def read_log(
    *,
    since_line: int | None = None,
    max_lines: int = DEFAULT_LOG_TAIL,
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    pattern: str = RLAIF_LOG_PATTERN,
) -> dict[str, Any]:
    """Return gateway log lines containing ``pattern`` (case-insensitive).

    ``since_line`` is a 0-based line index into the matched (filtered) lines;
    omitted returns the tail. Caller is expected to keep the cursor and
    re-send it on the next poll.
    """
    path = _gateway_log_path()
    if not path.exists():
        return {"path": str(path), "total": 0, "items": [], "next_line": 0}

    needle = pattern.lower() if pattern else ""

    # ponytail: full file scan + filter; dataset is small (gateway.log ~MB) and
    # polled at low rate. Index by filtered line number for cheap incremental
    # reads. Upgrade path: index file with byte offsets if log outgrows MB.
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_from = max(0, size - max_bytes)
            f.seek(read_from)
            raw = f.read()
            if read_from > 0:
                # drop partial first line
                nl = raw.find(b"\n")
                if nl != -1:
                    raw = raw[nl + 1 :]
        text = raw.decode("utf-8", errors="replace")
    except OSError as exc:
        logger.warning("rlaif_api: cannot read gateway log {}: {}", path, exc)
        return {"path": str(path), "total": 0, "items": [], "next_line": 0, "error": str(exc)}

    matched: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines()):
        if needle and needle not in line.lower():
            continue
        matched.append({"line_no": line_no, "text": line})

    total = len(matched)
    if since_line is not None and since_line >= 0:
        items = matched[since_line + 1 :]
    else:
        items = matched[-max_lines:]

    next_line = (items[-1]["line_no"] if items else (since_line if since_line is not None else -1))
    return {"path": str(path), "total": total, "items": items, "next_line": next_line}


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


@dataclass(frozen=True)
class RlaifApiPaths:
    preferences_path: Path
    log_path: Path


# ponytail: the running scanner instance is stashed on the cli.commands
# module by the gateway; we read it lazily on each call. This avoids
# threading the instance through the WebUI HTTP plumbing.
def _get_scanner() -> Any | None:
    try:
        from nanobot.cli import commands as _cmd_mod
    except Exception:
        return None
    return getattr(_cmd_mod, "_rlaif_scanner", None)


def list_proposals() -> dict[str, Any]:
    """Return pending RLAIF scanner proposals awaiting user approval."""
    scanner = _get_scanner()
    if scanner is None:
        return {"items": [], "scanner_active": False}
    return {
        "scanner_active": True,
        "items": scanner.list_proposals(),
    }


def get_proposal(proposal_id: int) -> dict[str, Any] | None:
    scanner = _get_scanner()
    if scanner is None:
        return None
    return scanner.get_proposal(proposal_id)


async def approve_proposal(proposal_id: int) -> str:
    scanner = _get_scanner()
    if scanner is None:
        return "scanner not running"
    return await scanner.approve_proposal(proposal_id)


def reject_proposal(proposal_id: int) -> str:
    scanner = _get_scanner()
    if scanner is None:
        return "scanner not running"
    return scanner.reject_proposal(proposal_id)
