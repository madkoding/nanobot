"""WebUI Workspace File Browser API.

Provides HTTP endpoints for browsing and manipulating files within the workspace,
respecting workspace security policies and protecting sensitive files.
"""

from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path
from typing import Any

from nanobot.security.workspace_access import WorkspaceScope
from nanobot.security.workspace_policy import (
    WorkspaceBoundaryError,
    resolve_allowed_path,
)

# Sensitive file patterns that should never be exposed or modified through the UI
SENSITIVE_PATTERNS = {
    ".env",
    ".env.local",
    ".env.production",
    ".git",
    ".gitignore",
    ".nanobot",
    "node_modules",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
}

# Nanobot runtime paths that must remain intact for the agent to keep working.
# These are shown in the workspace browser marked in red, but any mutating
# operation (write, rename, move, delete, create) is blocked. They are only
# treated as protected when they live at the root of the workspace.
NANOBOT_RUNTIME_PROTECTED = {
    "sessions",
    "memory",
    "cron",
    "bus",
    "logs",
    "subagents",
    "skills",
    "triggers",
    "users.json",
    "AGENTS.md",
    "SOUL.md",
    "USER.md",
    "HEARTBEAT.md",
}


def is_sensitive_path(path: Path, workspace_root: Path) -> bool:
    """Check if a path is sensitive and should be protected."""
    try:
        rel_path = path.resolve(strict=False).relative_to(workspace_root.resolve(strict=False))
        parts = rel_path.parts

        # Check each part of the path against sensitive patterns
        for part in parts:
            if part in SENSITIVE_PATTERNS:
                return True
            # Block hidden files/directories (starting with .)
            if part.startswith("."):
                return True

        # Check filename patterns
        if parts and any(
            parts[-1].endswith(pattern.lstrip("*"))
            for pattern in SENSITIVE_PATTERNS
            if pattern.startswith("*")
        ):
            return True

    except (ValueError, OSError):
        return True

    return False


def _is_workspace_root_path(path: Path, workspace_root: Path) -> bool:
    """Return True if path is a direct child of the workspace root."""
    try:
        rel_path = path.resolve(strict=False).relative_to(workspace_root.resolve(strict=False))
    except (ValueError, OSError):
        return False
    return len(rel_path.parts) == 1


def is_protected_runtime_path(path: Path, workspace_root: Path) -> bool:
    """Check if a path is a nanobot runtime-critical entry (only at workspace root)."""
    if not _is_workspace_root_path(path, workspace_root):
        return False
    try:
        rel_path = path.resolve(strict=False).relative_to(workspace_root.resolve(strict=False))
        name = rel_path.parts[0]
    except (ValueError, OSError):
        return False
    return name in NANOBOT_RUNTIME_PROTECTED


def _workspace_root(scope: WorkspaceScope) -> Path:
    """Return the effective workspace root for a scope."""
    return scope.project_path


def _protected_runtime_message(name: str, action: str) -> str:
    """Return a user-friendly error message for a blocked mutating action."""
    return f"Cannot {action} '{name}': required for nanobot to function"


def _resolve_scope_path(
    raw_path: str,
    scope: WorkspaceScope,
    *,
    allow_missing: bool = False,
) -> Path:
    """Resolve a path against the scope, enforcing workspace boundaries.

    Uses ``resolve_allowed_path`` (the canonical security mechanism) so that
    ``restrict_to_workspace`` and the session scope are respected, not just the
    global workspace root.
    """
    if not isinstance(raw_path, str):
        raise ValueError("path is required")
    if len(raw_path) > 4096:
        raise ValueError("path is too long")

    # Empty path means the workspace root itself.
    if not raw_path.strip():
        resolved = scope.project_path
        if not allow_missing and not resolved.exists():
            raise ValueError("Workspace root does not exist")
        return resolved

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

    if is_sensitive_path(resolved, workspace):
        raise ValueError(f"Access to '{raw_path}' is restricted")

    return resolved


def workspace_list_files(
    scope: WorkspaceScope,
    subpath: str = "",
) -> dict[str, Any]:
    """List files and directories in the workspace."""
    workspace_root = _workspace_root(scope)

    try:
        target_path = _resolve_scope_path(subpath, scope, allow_missing=True)
    except ValueError as e:
        return {"error": str(e), "files": []}

    if not target_path.exists():
        return {"error": "Path does not exist", "files": []}

    if not target_path.is_dir():
        return {"error": "Path is not a directory", "files": []}

    files = []
    try:
        for entry in sorted(target_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            # Skip sensitive files/directories entirely; they remain hidden.
            if is_sensitive_path(entry, workspace_root):
                continue

            try:
                stat = entry.stat()
                files.append(
                    {
                        "name": entry.name,
                        "path": str(entry.relative_to(workspace_root)),
                        "is_directory": entry.is_dir(),
                        "size": stat.st_size if entry.is_file() else 0,
                        "modified_at": stat.st_mtime,
                        "created_at": stat.st_ctime,
                        "protected": is_protected_runtime_path(entry, workspace_root),
                    }
                )
            except (OSError, PermissionError):
                continue
    except (OSError, PermissionError) as e:
        return {"error": f"Cannot access directory: {e}", "files": []}

    return {
        "current_path": str(target_path.relative_to(workspace_root)),
        "parent_path": str(target_path.parent.relative_to(workspace_root))
        if target_path != workspace_root
        else None,
        "files": files,
    }


def workspace_read_file(
    path: str,
    scope: WorkspaceScope,
) -> dict[str, Any]:
    """Read the contents of a file in the workspace."""
    try:
        full_path = _resolve_scope_path(path, scope)
    except ValueError as e:
        return {"error": str(e)}

    if not full_path.is_file():
        return {"error": "Path is not a file"}

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "path": path,
            "content": content,
            "encoding": "utf-8",
            "size": len(content.encode("utf-8")),
        }
    except UnicodeDecodeError:
        try:
            size = full_path.stat().st_size
            return {
                "path": path,
                "is_binary": True,
                "size": size,
                "message": "This is a binary file and cannot be displayed as text",
            }
        except OSError as e:
            return {"error": f"Cannot read file: {e}"}
    except (OSError, PermissionError) as e:
        return {"error": f"Cannot read file: {e}"}


def workspace_write_file(
    path: str,
    content: str,
    scope: WorkspaceScope,
) -> dict[str, Any]:
    """Write content to a file in the workspace."""
    workspace_root = _workspace_root(scope)

    try:
        full_path = _resolve_scope_path(path, scope, allow_missing=True)
    except ValueError as e:
        return {"error": str(e)}

    if full_path.exists():
        if is_sensitive_path(full_path, workspace_root):
            return {"error": f"Cannot modify restricted file: {path}"}
        if is_protected_runtime_path(full_path, workspace_root):
            return {"error": _protected_runtime_message(full_path.name, "modify")}

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True,
            "path": path,
            "size": len(content.encode("utf-8")),
        }
    except (OSError, PermissionError) as e:
        return {"error": f"Cannot write file: {e}"}


def workspace_rename(
    old_path: str,
    new_name: str,
    scope: WorkspaceScope,
) -> dict[str, Any]:
    """Rename a file or directory in the workspace."""
    workspace_root = _workspace_root(scope)

    try:
        old_full_path = _resolve_scope_path(old_path, scope)
    except ValueError as e:
        return {"error": str(e)}

    if ".." in new_name or "/" in new_name or "\\" in new_name:
        return {"error": "Invalid new name"}

    if is_protected_runtime_path(old_full_path, workspace_root):
        return {"error": _protected_runtime_message(old_full_path.name, "rename")}

    new_full_path = old_full_path.parent / new_name

    if is_sensitive_path(new_full_path, workspace_root):
        return {"error": "Cannot rename to restricted name"}

    if is_protected_runtime_path(new_full_path, workspace_root):
        return {"error": _protected_runtime_message(new_full_path.name, "rename to")}

    if new_full_path.exists():
        return {"error": "Destination already exists"}

    try:
        old_full_path.rename(new_full_path)
        return {
            "success": True,
            "old_path": old_path,
            "new_path": str(new_full_path.relative_to(workspace_root)),
        }
    except (OSError, PermissionError) as e:
        return {"error": f"Cannot rename: {e}"}


def workspace_move(
    source_path: str,
    dest_path: str,
    scope: WorkspaceScope,
) -> dict[str, Any]:
    """Move a file or directory within the workspace."""
    workspace_root = _workspace_root(scope)

    try:
        source_full = _resolve_scope_path(source_path, scope)
    except ValueError as e:
        return {"error": f"Source: {e}"}

    try:
        dest_full = _resolve_scope_path(dest_path, scope, allow_missing=True)
    except ValueError as e:
        return {"error": f"Destination: {e}"}

    if is_protected_runtime_path(source_full, workspace_root):
        return {"error": _protected_runtime_message(source_full.name, "move")}

    if is_sensitive_path(dest_full, workspace_root):
        return {"error": "Cannot move to restricted location"}

    if is_protected_runtime_path(dest_full, workspace_root):
        return {"error": _protected_runtime_message(dest_full.name, "move to")}

    if dest_full.is_dir():
        dest_full = dest_full / source_full.name

    try:
        shutil.move(str(source_full), str(dest_full))
        return {
            "success": True,
            "source": source_path,
            "destination": str(dest_full.relative_to(workspace_root)),
        }
    except (OSError, PermissionError) as e:
        return {"error": f"Cannot move: {e}"}


def workspace_delete(
    path: str,
    scope: WorkspaceScope,
) -> dict[str, Any]:
    """Delete a file or directory from the workspace."""
    workspace_root = _workspace_root(scope)

    try:
        full_path = _resolve_scope_path(path, scope)
    except ValueError as e:
        return {"error": str(e)}

    if is_sensitive_path(full_path, workspace_root):
        return {"error": "Cannot delete restricted path"}

    if is_protected_runtime_path(full_path, workspace_root):
        return {"error": _protected_runtime_message(full_path.name, "delete")}

    try:
        if full_path.is_dir():
            shutil.rmtree(full_path)
        else:
            full_path.unlink()
        return {
            "success": True,
            "path": path,
        }
    except (OSError, PermissionError) as e:
        return {"error": f"Cannot delete: {e}"}


def workspace_create_directory(
    path: str,
    scope: WorkspaceScope,
) -> dict[str, Any]:
    """Create a new directory in the workspace."""
    workspace_root = _workspace_root(scope)

    try:
        full_path = _resolve_scope_path(path, scope, allow_missing=True)
    except ValueError as e:
        return {"error": str(e)}

    if full_path.exists():
        return {"error": "Path already exists"}

    if is_sensitive_path(full_path, workspace_root):
        return {"error": "Cannot create restricted directory"}

    if is_protected_runtime_path(full_path, workspace_root):
        return {"error": _protected_runtime_message(full_path.name, "create")}

    try:
        full_path.mkdir(parents=True, exist_ok=True)
        return {
            "success": True,
            "path": path,
        }
    except (OSError, PermissionError) as e:
        return {"error": f"Cannot create directory: {e}"}


def workspace_copy(
    source_path: str,
    dest_path: str,
    scope: WorkspaceScope,
) -> dict[str, Any]:
    """Copy a file or directory within the workspace."""
    workspace_root = _workspace_root(scope)

    try:
        source_full = _resolve_scope_path(source_path, scope)
    except ValueError as e:
        return {"error": f"Source: {e}"}

    try:
        dest_full = _resolve_scope_path(dest_path, scope, allow_missing=True)
    except ValueError as e:
        return {"error": f"Destination: {e}"}

    if is_protected_runtime_path(source_full, workspace_root):
        return {"error": _protected_runtime_message(source_full.name, "copy")}

    if is_sensitive_path(dest_full, workspace_root):
        return {"error": "Cannot copy to restricted location"}

    if is_protected_runtime_path(dest_full, workspace_root):
        return {"error": _protected_runtime_message(dest_full.name, "copy to")}

    if dest_full.is_dir():
        dest_full = dest_full / source_full.name

    try:
        if source_full.is_dir():
            shutil.copytree(str(source_full), str(dest_full))
        else:
            shutil.copy2(str(source_full), str(dest_full))
        return {
            "success": True,
            "source": source_path,
            "destination": str(dest_full.relative_to(workspace_root)),
        }
    except (OSError, PermissionError) as e:
        return {"error": f"Cannot copy: {e}"}


def workspace_file_bytes(
    path: str,
    scope: WorkspaceScope,
) -> dict[str, Any]:
    """Return raw file contents with guessed MIME type for the workspace browser.

    This is used for image thumbnails/preview for files that are not text.
    Respects the same workspace scope and sensitive-path protections as read."""
    try:
        full_path = _resolve_scope_path(path, scope)
    except ValueError as e:
        return {"error": str(e)}

    if not full_path.is_file():
        return {"error": "Path is not a file"}

    content_type, _ = mimetypes.guess_type(full_path.name)
    if not content_type:
        content_type = "application/octet-stream"

    try:
        data = full_path.read_bytes()
    except (OSError, PermissionError) as e:
        return {"error": f"Cannot read file: {e}"}

    return {
        "data": data,
        "mime_type": content_type,
        "name": full_path.name,
    }
