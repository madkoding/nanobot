"""WebUI Todos API.

CRUD over per-list JSON files under ``<workspace>/todo/``. Each list is a
single file ``todo/<slug>.json`` with this shape::

    {
      "id": "uuid",
      "slug": "compras",
      "name": "Compras",
      "created_at": "ISO",
      "updated_at": "ISO",
      "items": [
        {
          "id": "uuid", "text": "...", "done": false, "created": "ISO",
          "done_at": null, "due_date": null, "link": null, "price_clp": null,
          "assignee": "madkoding", "notes": null
        }
      ]
    }

A shared roster lives in ``todo/_users.json``::

    { "users": { "<id>": { "name": "...", "phone": "...", "authorized": true } } }

All mutations are atomic (write to a temp file then ``os.replace`` + fsync).
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot.security.workspace_access import WorkspaceScope
from nanobot.security.workspace_policy import (
    WorkspaceBoundaryError,
    resolve_allowed_path,
)
from nanobot.utils.atomic_write import atomic_write_text

TODO_DIR_NAME = "todo"
USERS_FILENAME = "_users.json"
LEGACY_FILENAME = "todos.json"
TODO_LIST_METADATA_KEY = "todo_list"

_SLUG_MAX_LEN = 64
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ITEM_MUTABLE_FIELDS = (
    "text",
    "done",
    "done_at",
    "due_date",
    "link",
    "price_clp",
    "assignee",
    "notes",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _workspace_root(scope: WorkspaceScope) -> Path:
    return scope.project_path


def _todo_dir(scope: WorkspaceScope) -> Path:
    return _workspace_root(scope) / TODO_DIR_NAME


def _resolve_todo_dir(scope: WorkspaceScope, *, create: bool = False) -> Path:
    """Return the todo dir path, optionally creating it. Jails to workspace."""
    workspace = _workspace_root(scope)
    allowed_root = workspace if scope.restrict_to_workspace else None
    try:
        resolved = resolve_allowed_path(
            TODO_DIR_NAME,
            workspace=workspace,
            allowed_root=allowed_root,
            strict=False,
        )
    except (WorkspaceBoundaryError, OSError, ValueError) as exc:
        raise ValueError(f"todo dir is outside workspace boundary: {exc}") from exc
    if create and not resolved.exists():
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _slugify(name: str) -> str:
    """Turn a free-form name into a filesystem-safe slug."""
    base = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not base:
        base = "lista"
    return base[:_SLUG_MAX_LEN]


def _ensure_unique_slug(todo_dir: Path, slug: str) -> str:
    """Append -2, -3, ... until the slug has no collision."""
    candidate = slug
    n = 2
    while (todo_dir / f"{candidate}.json").exists():
        suffix = f"-{n}"
        candidate = f"{slug[: _SLUG_MAX_LEN - len(suffix)]}{suffix}"
        n += 1
    return candidate


def _list_file(todo_dir: Path, slug: str) -> Path:
    if not _is_valid_slug(slug):
        raise ValueError("invalid slug")
    return todo_dir / f"{slug}.json"


def _is_valid_slug(slug: str) -> bool:
    if not slug or len(slug) > _SLUG_MAX_LEN:
        return False
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug))


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, encoded)


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _normalize_item(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    notes = raw.get("notes")
    transfer = raw.get("transfer_info")
    if notes is None and isinstance(transfer, dict):
        # Legacy: turn structured transfer_info into readable plain-text notes.
        lines = [f"{k}: {v}" for k, v in transfer.items() if v is not None]
        notes = "\n".join(lines) if lines else None
    item = {
        "id": str(raw.get("id") or uuid.uuid4()),
        "text": str(raw.get("text") or ""),
        "done": bool(raw.get("done") or False),
        "created": str(raw.get("created") or _now_iso()),
        "done_at": raw.get("done_at"),
        "due_date": raw.get("due_date"),
        "link": raw.get("link"),
        "price_clp": raw.get("price_clp"),
        "assignee": raw.get("assignee"),
        "notes": notes,
    }
    return item


# Kept for legacy migration compatibility.
_legacy_transfer_keys = ("transfer_info",)


def _normalize_list(raw: dict[str, Any], slug: str) -> dict[str, Any]:
    items_raw = raw.get("items")
    items = [_normalize_item(x) for x in items_raw] if isinstance(items_raw, list) else []
    return {
        "id": str(raw.get("id") or uuid.uuid4()),
        "slug": slug,
        "name": str(raw.get("name") or slug),
        "created_at": str(raw.get("created_at") or _now_iso()),
        "updated_at": str(raw.get("updated_at") or _now_iso()),
        "items": items,
    }


def _summary(list_data: dict[str, Any]) -> dict[str, Any]:
    items = list_data.get("items") or []
    done = sum(1 for x in items if x.get("done"))
    return {
        "id": list_data.get("id"),
        "slug": list_data.get("slug"),
        "name": list_data.get("name"),
        "item_count": len(items),
        "done_count": done,
        "updated_at": list_data.get("updated_at"),
    }


def _users_path(todo_dir: Path) -> Path:
    return todo_dir / USERS_FILENAME


def _normalize_users(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    users_raw = raw.get("users")
    if not isinstance(users_raw, dict):
        users_raw = {}
    users: dict[str, Any] = {}
    for key, val in users_raw.items():
        if not isinstance(val, dict):
            continue
        users[key] = {
            "name": str(val.get("name") or key),
            "phone": val.get("phone"),
            "authorized": bool(val.get("authorized") or False),
        }
    return {"users": users}


def _read_users(todo_dir: Path) -> dict[str, Any]:
    raw = _read_json_file(_users_path(todo_dir))
    if raw is None:
        return {"users": {}}
    return _normalize_users(raw)


def _write_users(todo_dir: Path, users: dict[str, Any]) -> None:
    todo_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(_users_path(todo_dir), _normalize_users(users))


def _touch_list(list_data: dict[str, Any]) -> dict[str, Any]:
    list_data["updated_at"] = _now_iso()
    return list_data


# -- Public API ----------------------------------------------------------------


def list_todo_lists(scope: WorkspaceScope) -> dict[str, Any]:
    """Return all lists under ``todo/`` (excluding ``_users.json``)."""
    try:
        todo_dir = _resolve_todo_dir(scope)
    except ValueError as e:
        return {"error": str(e), "lists": []}
    if not todo_dir.is_dir():
        return {"lists": []}
    lists: list[dict[str, Any]] = []
    for entry in sorted(todo_dir.iterdir(), key=lambda x: x.name.lower()):
        if entry.name == USERS_FILENAME or not entry.name.endswith(".json"):
            continue
        if entry.name.endswith(".migrated"):
            continue
        raw = _read_json_file(entry)
        if raw is None:
            continue
        slug = entry.stem
        lists.append(_summary(_normalize_list(raw, slug)))
    return {"lists": lists}


def fetch_todo_list(slug: str, scope: WorkspaceScope) -> dict[str, Any]:
    """Return one list + the roster."""
    try:
        todo_dir = _resolve_todo_dir(scope)
    except ValueError as e:
        return {"error": str(e)}
    if not todo_dir.is_dir():
        return {"error": "todo directory does not exist"}
    path = _list_file(todo_dir, slug)
    raw = _read_json_file(path)
    if raw is None:
        return {"error": f"list '{slug}' not found"}
    return {
        "list": _normalize_list(raw, slug),
        "users": _read_users(todo_dir).get("users", {}),
    }


def fetch_users(scope: WorkspaceScope) -> dict[str, Any]:
    try:
        todo_dir = _resolve_todo_dir(scope)
    except ValueError as e:
        return {"error": str(e)}
    if not todo_dir.is_dir():
        return {"users": {}}
    return {"users": _read_users(todo_dir).get("users", {})}


def create_todo_list(name: str, scope: WorkspaceScope, slug: str | None = None) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        return {"error": "name is required"}
    try:
        todo_dir = _resolve_todo_dir(scope, create=True)
    except ValueError as e:
        return {"error": str(e)}
    final_slug = _ensure_unique_slug(
        todo_dir, slug.strip() if slug and slug.strip() else _slugify(name)
    )
    if not _is_valid_slug(final_slug):
        return {"error": "invalid slug"}
    path = _list_file(todo_dir, final_slug)
    if path.exists():
        return {"error": f"list '{final_slug}' already exists"}
    now = _now_iso()
    data = {
        "id": str(uuid.uuid4()),
        "slug": final_slug,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "items": [],
    }
    _atomic_write(path, data)
    return _summary(data)


def delete_todo_list(slug: str, scope: WorkspaceScope) -> dict[str, Any]:
    try:
        todo_dir = _resolve_todo_dir(scope)
    except ValueError as e:
        return {"error": str(e)}
    if not todo_dir.is_dir():
        return {"error": "todo directory does not exist"}
    path = _list_file(todo_dir, slug)
    if not path.is_file():
        return {"error": f"list '{slug}' not found"}
    try:
        path.unlink()
    except OSError as e:
        return {"error": f"cannot delete list: {e}"}
    return {"ok": True, "slug": slug}


def create_item(slug: str, item: dict[str, Any], scope: WorkspaceScope) -> dict[str, Any]:
    try:
        todo_dir = _resolve_todo_dir(scope)
    except ValueError as e:
        return {"error": str(e)}
    if not todo_dir.is_dir():
        return {"error": "todo directory does not exist"}
    path = _list_file(todo_dir, slug)
    raw = _read_json_file(path)
    if raw is None:
        return {"error": f"list '{slug}' not found"}
    list_data = _normalize_list(raw, slug)
    new_item = _normalize_item(item)
    if not str(new_item.get("text") or "").strip():
        return {"error": "text is required"}
    list_data["items"].append(new_item)
    _touch_list(list_data)
    _atomic_write(path, list_data)
    return {"item": new_item, "list": _summary(list_data)}


def update_item(
    slug: str,
    item_id: str,
    changes: dict[str, Any],
    scope: WorkspaceScope,
) -> dict[str, Any]:
    try:
        todo_dir = _resolve_todo_dir(scope)
    except ValueError as e:
        return {"error": str(e)}
    if not todo_dir.is_dir():
        return {"error": "todo directory does not exist"}
    path = _list_file(todo_dir, slug)
    raw = _read_json_file(path)
    if raw is None:
        return {"error": f"list '{slug}' not found"}
    list_data = _normalize_list(raw, slug)
    target = None
    for it in list_data["items"]:
        if it.get("id") == item_id:
            target = it
            break
    if target is None:
        return {"error": f"item '{item_id}' not found"}
    changed = False
    for field in _ITEM_MUTABLE_FIELDS:
        if field in changes:
            value = changes[field]
            if field == "done":
                new_done = bool(value)
                if target.get("done") != new_done:
                    target["done_at"] = _now_iso() if new_done else None
                    target["done"] = new_done
                    changed = True
            else:
                target[field] = value
                changed = True
    if changed:
        _touch_list(list_data)
        _atomic_write(path, list_data)
    return {"item": target, "list": _summary(list_data)}


def delete_item(slug: str, item_id: str, scope: WorkspaceScope) -> dict[str, Any]:
    try:
        todo_dir = _resolve_todo_dir(scope)
    except ValueError as e:
        return {"error": str(e)}
    if not todo_dir.is_dir():
        return {"error": "todo directory does not exist"}
    path = _list_file(todo_dir, slug)
    raw = _read_json_file(path)
    if raw is None:
        return {"error": f"list '{slug}' not found"}
    list_data = _normalize_list(raw, slug)
    before = len(list_data["items"])
    list_data["items"] = [x for x in list_data["items"] if x.get("id") != item_id]
    if len(list_data["items"]) == before:
        return {"error": f"item '{item_id}' not found"}
    _touch_list(list_data)
    _atomic_write(path, list_data)
    return {"ok": True, "item_id": item_id, "list": _summary(list_data)}


def update_users(users: dict[str, Any], scope: WorkspaceScope) -> dict[str, Any]:
    try:
        todo_dir = _resolve_todo_dir(scope, create=True)
    except ValueError as e:
        return {"error": str(e)}
    _write_users(todo_dir, users)
    return {"users": _read_users(todo_dir).get("users", {})}


def migrate_legacy(scope: WorkspaceScope) -> dict[str, Any]:
    """Migrate the legacy ``todo/todos.json`` (users + lists keyed by user) into
    one file per user (``todo/<user>.json``) plus ``todo/_users.json``.

    Idempotent: if the legacy file does not exist, no-op. If per-user files
    already exist, they are left untouched; only missing lists are created.
    The legacy file is renamed to ``todos.json.migrated`` on success.
    """
    try:
        todo_dir = _resolve_todo_dir(scope, create=True)
    except ValueError as e:
        return {"error": str(e)}
    legacy_path = todo_dir / LEGACY_FILENAME
    raw = _read_json_file(legacy_path)
    if raw is None:
        return {"ok": True, "migrated": False, "lists": [], "users": {}}
    users_raw = raw.get("users")
    if not isinstance(users_raw, dict):
        users_raw = {}
    users = _normalize_users({"users": users_raw})
    lists_raw = raw.get("lists")
    if not isinstance(lists_raw, dict):
        lists_raw = {}
    migrated: list[str] = []
    for user_id, items_raw in lists_raw.items():
        if not isinstance(items_raw, list):
            continue
        slug = _ensure_unique_slug(todo_dir, _slugify(user_id))
        path = _list_file(todo_dir, slug)
        if path.exists():
            continue
        items = [
            _normalize_item({**x, "assignee": x.get("assignee") or user_id}) for x in items_raw
        ]
        now = _now_iso()
        data = {
            "id": str(uuid.uuid4()),
            "slug": slug,
            "name": (users.get("users", {}).get(user_id, {}) or {}).get("name") or user_id,
            "created_at": now,
            "updated_at": now,
            "items": items,
        }
        _atomic_write(path, data)
        migrated.append(slug)
    _write_users(todo_dir, users)
    # Mark the legacy file as migrated so it isn't picked up as a list anymore.
    migrated_name = LEGACY_FILENAME + ".migrated"
    try:
        legacy_path.rename(legacy_path.parent / migrated_name)
    except OSError:
        pass
    return {
        "ok": True,
        "migrated": bool(migrated),
        "lists": migrated,
        "users": users.get("users", {}),
    }
