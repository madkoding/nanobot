"""WebUI Research API helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from nanobot.security.workspace_access import WorkspaceScope
from nanobot.security.workspace_policy import (
    WorkspaceBoundaryError,
    resolve_allowed_path,
)


def _resolve_scope_path(
    raw_path: str,
    scope: WorkspaceScope,
    *,
    allow_missing: bool = False,
) -> Path:
    """Resolve a path against the workspace scope."""
    if not isinstance(raw_path, str):
        raise ValueError("path is required")
    if len(raw_path) > 4096:
        raise ValueError("path is too long")

    if not raw_path.strip():
        return scope.project_path

    workspace = scope.project_path
    allowed_root = workspace if scope.restrict_to_workspace else None
    try:
        resolved = resolve_allowed_path(
            raw_path,
            workspace=workspace,
            allowed_root=allowed_root,
            strict=False,
        )
    except WorkspaceBoundaryError as exc:
        raise ValueError(f"Path '{raw_path}' is outside workspace boundary") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid path '{raw_path}'") from exc

    if not allow_missing and not resolved.exists():
        raise ValueError(f"Path '{raw_path}' does not exist")

    return resolved


def share_research_article(path: str, scope: WorkspaceScope) -> dict[str, Any]:
    """Share a research article via sharemd and persist the returned URL.

    Returns a dict with ``ok``, ``url``, and optionally ``error``.
    """
    try:
        target = _resolve_scope_path(path, scope)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    if not target.is_file():
        return {"ok": False, "error": "Path is not a file"}
    if target.suffix.lower() != ".md":
        return {"ok": False, "error": "Only .md files can be shared"}

    sharemd = shutil.which("sharemd")
    if not sharemd:
        return {"ok": False, "error": "sharemd CLI not found. Install with: npm install -g sharemd"}

    # Check if we already have a cached URL for this exact file content.
    sharemd_json = target.with_suffix(".sharemd.json")
    try:
        if sharemd_json.exists():
            data = json.loads(sharemd_json.read_text(encoding="utf-8"))
            cached_url = data.get("url")
            if isinstance(cached_url, str) and cached_url.startswith("http"):
                return {"ok": True, "url": cached_url}
    except Exception:
        pass

    try:
        result = subprocess.run(
            [sharemd, str(target)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "sharemd timed out"}
    except Exception as e:
        return {"ok": False, "error": f"Failed to run sharemd: {e}"}

    if result.returncode != 0:
        return {
            "ok": False,
            "error": result.stderr.strip() or result.stdout.strip() or "sharemd failed",
        }

    # Parse URL from sharemd output. Typical output: https://sharemd.sh/<id>
    url = None
    for line in result.stdout.splitlines():
        for word in line.split():
            if word.startswith("https://sharemd.sh/"):
                url = word
                break
        if url:
            break

    if not url:
        return {"ok": False, "error": "Could not parse sharemd URL from output"}

    try:
        sharemd_json.write_text(
            json.dumps({"url": url}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Non-fatal: we still have the URL.
        pass

    return {"ok": True, "url": url}
