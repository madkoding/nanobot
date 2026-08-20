"""Subagent status/pending persistence (extracted from subagent.py)."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.security.workspace_access import WorkspaceScope


@dataclass(slots=True)
class SubagentStatus:
    """Real-time status of a running subagent."""

    task_id: str
    label: str
    task_description: str
    started_at: float          # time.monotonic()
    phase: str = "initializing"  # initializing | awaiting_tools | tools_completed | final_response | done | error
    iteration: int = 0
    tool_events: list = field(default_factory=list)   # [{name, status, detail}, ...]
    usage: dict = field(default_factory=dict)          # token usage
    stop_reason: str | None = None
    error: str | None = None
    result: str | None = None
    finished_at: float | None = None
    chat_id: str | None = None
    persisted_at: float | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize for WS / HTTP transport."""
        return {
            "task_id": self.task_id,
            "label": self.label,
            "task_description": self.task_description,
            "phase": self.phase,
            "iteration": self.iteration,
            "tool_events": list(self.tool_events),
            "usage": dict(self.usage),
            "stop_reason": self.stop_reason,
            "error": self.error,
            "result": self.result,
            "chat_id": self.chat_id,
            "persisted_at": self.persisted_at,
        }


#: How long a finished subagent keeps its status snapshot for HTTP fetch.
#: ponytail: 24h window — enough to reopen the panel after the user comes back,
#: without keeping finished snapshots forever.
SUBAGENT_STATUS_TTL_S = 86400.0


def _storage_key(key: str) -> str:
    """Collision-resistant encoding for subagent snapshot subdirectories."""
    return base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")


def _subagent_snapshot_dir(workspace: Path, session_key: str | None) -> Path:
    """Return the directory where a session's subagent snapshots live."""
    base = workspace / "subagents"
    if session_key:
        return base / _storage_key(session_key)
    return base / "_unknown_"


def _persist_subagent_status(
    workspace: Path,
    session_key: str | None,
    status: SubagentStatus,
) -> None:
    """Persist a subagent snapshot so it survives gateway restarts."""
    if status.phase not in ("done", "error"):
        # ponytail: only finished snapshots are persisted. Running subagents
        # would need task-reconstruction; that is left for a future phase.
        return
    directory = _subagent_snapshot_dir(workspace, session_key)
    directory.mkdir(parents=True, exist_ok=True)
    status.persisted_at = time.time()
    path = directory / f"{status.task_id}.json"
    try:
        path.write_text(json.dumps(status.to_payload(), ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("Failed to persist subagent status for {}", status.task_id)


def _subagent_status_from_payload(payload: dict[str, Any]) -> SubagentStatus | None:
    """Reconstruct a SubagentStatus from a persisted payload."""
    try:
        return SubagentStatus(
            task_id=payload["task_id"],
            label=payload.get("label", ""),
            task_description=payload.get("task_description", ""),
            started_at=payload.get("started_at", 0.0),
            phase=payload.get("phase", "done"),
            iteration=payload.get("iteration", 0),
            tool_events=list(payload.get("tool_events", [])),
            usage=dict(payload.get("usage", {})),
            stop_reason=payload.get("stop_reason"),
            error=payload.get("error"),
            result=payload.get("result"),
            finished_at=payload.get("finished_at"),
            chat_id=payload.get("chat_id"),
        )
    except Exception:
        logger.warning("Skipping malformed subagent snapshot: {}", payload)
        return None


def _load_persisted_subagent_statuses(
    workspace: Path,
    ttl_s: float = SUBAGENT_STATUS_TTL_S,
) -> dict[str, SubagentStatus]:
    """Load non-expired finished subagent snapshots from disk.

    Removes expired snapshot files while scanning.
    """
    base = workspace / "subagents"
    if not base.exists():
        return {}
    now = time.time()
    loaded: dict[str, SubagentStatus] = {}
    for session_dir in base.iterdir():
        if not session_dir.is_dir():
            continue
        for path in session_dir.glob("*.json"):
            if path.suffixes == [".pending", ".json"]:
                # Pending records are handled by resume_pending, not here.
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not read subagent snapshot {}", path)
                continue
            persisted_at = payload.get("persisted_at")
            if not isinstance(persisted_at, (int, float)) or now - persisted_at > ttl_s:
                try:
                    path.unlink()
                except Exception:
                    pass
                continue
            status = _subagent_status_from_payload(payload)
            if status is not None:
                status.persisted_at = persisted_at
                loaded[status.task_id] = status
        # Remove empty session directories to keep the workspace tidy.
        try:
            if not any(session_dir.iterdir()):
                session_dir.rmdir()
        except Exception:
            pass
    return loaded


def _pending_path(workspace: Path, session_key: str | None, task_id: str) -> Path:
    """Path for a subagent pending record."""
    return _subagent_snapshot_dir(workspace, session_key) / f"{task_id}.pending.json"


def _persist_subagent_pending(
    workspace: Path,
    task_id: str,
    task: str,
    label: str | None,
    origin_channel: str,
    origin_chat_id: str,
    session_key: str | None,
    origin_message_id: str | None,
    temperature: float | None,
    workspace_scope: WorkspaceScope | None,
    model_preset: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> None:
    """Persist a pending subagent record so it can be relaunched after restart."""
    directory = _subagent_snapshot_dir(workspace, session_key)
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "task": task,
        "label": label,
        "origin_channel": origin_channel,
        "origin_chat_id": origin_chat_id,
        "session_key": session_key,
        "origin_message_id": origin_message_id,
        "temperature": temperature,
        "workspace_scope": workspace_scope.to_dict() if workspace_scope is not None else None,
        "model_preset": model_preset,
        "persisted_at": time.time(),
    }
    if checkpoint is not None:
        payload["checkpoint"] = checkpoint
    path = _pending_path(workspace, session_key, task_id)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("Failed to persist subagent pending record for {}", task_id)


def _delete_subagent_pending(
    workspace: Path,
    session_key: str | None,
    task_id: str,
) -> None:
    """Remove a pending subagent record once it finishes."""
    path = _pending_path(workspace, session_key, task_id)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _load_subagent_pendings(workspace: Path) -> list[dict[str, Any]]:
    """Load all pending subagent records from disk.

    Removes expired pending records while scanning.
    """
    base = workspace / "subagents"
    if not base.exists():
        return []
    now = time.time()
    loaded: list[dict[str, Any]] = []
    for session_dir in base.iterdir():
        if not session_dir.is_dir():
            continue
        for path in session_dir.glob("*.pending.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not read subagent pending record {}", path)
                continue
            persisted_at = payload.get("persisted_at")
            if not isinstance(persisted_at, (int, float)) or now - persisted_at > SUBAGENT_STATUS_TTL_S:
                try:
                    path.unlink()
                except Exception:
                    pass
                continue
            loaded.append(payload)
    return loaded
