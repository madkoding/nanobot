"""WebUI Projects: per-project capsules with instructions and uploaded files.

Each project lives at ``<data_dir>/webui/projects/<id>/``::

    project.json   # {id, name, instructions_md, created_at, updated_at}
    files/         # uploaded files, one entry per file id

Files are stored on disk as ``<file_id>.bin`` with a sibling
``<file_id>.meta.json`` describing the original name and mime type.
This module is intentionally I/O-only and duck-types the
``SessionManager`` so it never pulls in ``nanobot.command`` or
``nanobot.agent`` (the modules that historically caused circular
imports when the WebUI was wired into the gateway startup).
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ProjectError(Exception):
    """Raised when a project operation fails (404, validation, IO)."""


@dataclass(frozen=True)
class ProjectSummary:
    id: str
    name: str
    instructions_md: str
    created_at_ms: int
    updated_at_ms: int
    file_count: int
    byte_count: int
    folder_count: int = 0


@dataclass(frozen=True)
class ProjectFile:
    id: str
    project_id: str
    name: str
    mime_type: str
    size: int
    created_at_ms: int


@dataclass(frozen=True)
class ProjectFolder:
    path: str
    created_at_ms: int


_PROJECTS_DIRNAME = "projects"
_FILES_DIRNAME = "files"
_BOARD_FILENAME = "board.json"

_DEFAULT_COLUMNS = ["Backlog", "In Progress", "Review", "Done"]


def _slugify_id(raw: str) -> str:
    """Return a filesystem-safe id (UUID if input is empty/unsafe)."""
    candidate = "".join(
        c if c.isalnum() or c in ("-", "_") else "-" for c in (raw or "").strip()
    ).strip("-")
    if not candidate:
        candidate = uuid.uuid4().hex
    return candidate[:64] or uuid.uuid4().hex


def _now_ms() -> int:
    return int(time.time() * 1000)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"corrupt json at {path}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply that may include surrounding text.

    Tries ``json.loads`` on the raw text first, then extracts the first ``{...}``
    block. Returns {} on failure.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    """Parse ``data:<mime>;base64,<payload>`` into (bytes, mime_type)."""
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ProjectError("file must be a base64 data URL")
    head, _, payload = data_url.partition(",")
    if not payload:
        raise ProjectError("empty data URL payload")
    mime = "application/octet-stream"
    if ";" in head:
        mime = head[5:].split(";", 1)[0] or mime
    try:
        return base64.b64decode(payload, validate=True), mime
    except (ValueError, TypeError) as exc:
        raise ProjectError(f"invalid base64 payload: {exc}") from exc


class WebUIProjectsController:
    """CRUD for project capsules + file storage under ``<data_dir>/webui/projects/``."""

    def __init__(self, data_dir: Path, worktree_root: Path | None = None) -> None:
        self._root = (data_dir / _PROJECTS_DIRNAME).resolve(strict=False)
        self._root.mkdir(parents=True, exist_ok=True)
        self._worktree_root = (
            worktree_root.resolve(strict=False) if worktree_root is not None else None
        )

    def _project_dir(self, project_id: str) -> Path:
        return self._root / project_id

    def _files_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / _FILES_DIRNAME

    def files_dir_for(self, project_id: str) -> Path:
        """Public accessor for a project's uploaded-files directory."""
        return self._files_dir(project_id)

    def extra_read_dirs_for(self, project_id: str) -> tuple[Path, ...]:
        """Read-only roots for a project: uploaded-files dir + associated folders.

        These are exposed to the agent (via ``read_file`` / ``exec``) so it can
        reach project files while still being confined to the workspace.
        """
        roots: list[Path] = []
        fdir = self._files_dir(project_id)
        if fdir.is_dir():
            roots.append(fdir.resolve(strict=False))
        for folder in self.list_folders(project_id):
            try:
                p = Path(folder.path).expanduser().resolve(strict=False)
            except (OSError, ValueError, RuntimeError):
                continue
            if p.is_dir():
                roots.append(p)
        return tuple(roots)

    def migrate_worktrees(self) -> dict[str, int]:
        """Move legacy global-scoped worktrees into per-project subdirs.

        Legacy cards used ``<worktree_root>/<card_id>``; the new layout is
        ``<worktree_root>/<project_id>/<card_id>``. Rewrites each card's
        ``worktree_path`` in ``board.json`` and moves the worktree on disk.
        Skips cards with a running subagent (would break its cwd). Returns
        ``{"moved": n, "skipped": n}``.
        """
        from nanobot.webui.worktrees import move_worktree

        root = self._worktree_root or Path("~/.nanobot/worktrees").expanduser()
        moved = 0
        skipped = 0
        for child in sorted(self._root.iterdir()):
            if not child.is_dir() or not (child / "project.json").is_file():
                continue
            project_id = child.name
            board = self.get_board(project_id)
            repo_path = board.get("repo_path")
            if not repo_path:
                continue
            new_parent = root / project_id
            changed = False
            for card in board.get("cards", []):
                wt = card.get("worktree_path")
                if not wt:
                    continue
                old = Path(wt)
                # Already scoped per-project? Skip.
                if old.parent == new_parent:
                    continue
                if not old.is_dir():
                    continue
                if card.get("subagent_task_id"):
                    skipped += 1
                    continue
                new_path = new_parent / old.name
                if new_path.exists():
                    skipped += 1
                    continue
                try:
                    move_worktree(Path(repo_path).expanduser(), old, new_path)
                    card["worktree_path"] = str(new_path)
                    changed = True
                    moved += 1
                except Exception:
                    skipped += 1
            if changed:
                self._write_board(project_id, board)
                self._touch_project(project_id)
        return {"moved": moved, "skipped": skipped}

    def _meta_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _file_meta_path(self, project_id: str, file_id: str) -> Path:
        return self._files_dir(project_id) / f"{file_id}.meta.json"
        return self._files_dir(project_id) / f"{file_id}.meta.json"

    def _file_data_path(self, project_id: str, file_id: str) -> Path:
        return self._files_dir(project_id) / f"{file_id}.bin"

    def _folders_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "folders.json"

    def _board_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / _BOARD_FILENAME

    def list_projects(self) -> list[ProjectSummary]:
        out: list[ProjectSummary] = []
        for child in sorted(self._root.iterdir()):
            if not child.is_dir() or not (child / "project.json").is_file():
                continue
            try:
                out.append(self._summary(child))
            except ProjectError:
                continue
        return out

    def get_project(self, project_id: str) -> ProjectSummary:
        pdir = self._project_dir(project_id)
        if not (pdir / "project.json").is_file():
            raise ProjectError(f"project not found: {project_id}")
        return self._summary(pdir)

    def create_project(self, name: str, instructions_md: str) -> ProjectSummary:
        clean_name = (name or "").strip()
        if not clean_name:
            raise ProjectError("project name is required")
        clean_instructions = (instructions_md or "").strip()
        project_id = self._unique_id(clean_name)
        meta = {
            "id": project_id,
            "name": clean_name,
            "instructions_md": clean_instructions,
            "created_at_ms": _now_ms(),
            "updated_at_ms": _now_ms(),
        }
        pdir = self._project_dir(project_id)
        pdir.mkdir(parents=True, exist_ok=False)
        (pdir / _FILES_DIRNAME).mkdir(parents=True, exist_ok=True)
        _write_json(self._meta_path(project_id), meta)
        return self._summary(pdir)

    def update_project(
        self,
        project_id: str,
        name: str,
        instructions_md: str,
    ) -> ProjectSummary:
        meta_path = self._meta_path(project_id)
        if not meta_path.is_file():
            raise ProjectError(f"project not found: {project_id}")
        meta = _read_json(meta_path)
        clean_name = (name or "").strip()
        if not clean_name:
            raise ProjectError("project name is required")
        meta["name"] = clean_name
        meta["instructions_md"] = (instructions_md or "").strip()
        meta["updated_at_ms"] = _now_ms()
        _write_json(meta_path, meta)
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> None:
        pdir = self._project_dir(project_id)
        if not pdir.is_dir():
            raise ProjectError(f"project not found: {project_id}")
        for child in pdir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                for sub in child.rglob("*"):
                    if sub.is_file() or sub.is_symlink():
                        sub.unlink()
                child.rmdir()
        pdir.rmdir()

    def list_files(self, project_id: str) -> list[ProjectFile]:
        fdir = self._files_dir(project_id)
        if not fdir.is_dir():
            raise ProjectError(f"project not found: {project_id}")
        out: list[ProjectFile] = []
        for meta in sorted(fdir.glob("*.meta.json")):
            try:
                data = _read_json(meta)
            except ProjectError:
                continue
            out.append(
                ProjectFile(
                    id=data.get("id", meta.stem),
                    project_id=project_id,
                    name=data.get("name", meta.stem),
                    mime_type=data.get("mime_type", "application/octet-stream"),
                    size=int(data.get("size", 0)),
                    created_at_ms=int(data.get("created_at_ms", 0)),
                )
            )
        return out

    def add_file(
        self,
        project_id: str,
        name: str,
        data_url: str,
    ) -> ProjectFile:
        if not self._meta_path(project_id).is_file():
            raise ProjectError(f"project not found: {project_id}")
        clean_name = (name or "").strip()
        if not clean_name:
            raise ProjectError("file name is required")
        payload, mime = _decode_data_url(data_url)
        fdir = self._files_dir(project_id)
        fdir.mkdir(parents=True, exist_ok=True)
        file_id = uuid.uuid4().hex
        data_path = self._file_data_path(project_id, file_id)
        meta_path = self._file_meta_path(project_id, file_id)
        with data_path.open("wb") as fh:
            fh.write(payload)
        meta = {
            "id": file_id,
            "name": clean_name,
            "mime_type": mime,
            "size": len(payload),
            "created_at_ms": _now_ms(),
        }
        _write_json(meta_path, meta)
        self._touch_project(project_id)
        return ProjectFile(
            id=file_id,
            project_id=project_id,
            name=clean_name,
            mime_type=mime,
            size=len(payload),
            created_at_ms=meta["created_at_ms"],
        )

    def read_file(self, project_id: str, file_id: str) -> tuple[bytes, ProjectFile]:
        meta_path = self._file_meta_path(project_id, file_id)
        if not meta_path.is_file():
            raise ProjectError(f"file not found: {file_id}")
        data_path = self._file_data_path(project_id, file_id)
        if not data_path.is_file():
            raise ProjectError(f"file payload missing: {file_id}")
        meta = _read_json(meta_path)
        with data_path.open("rb") as fh:
            return fh.read(), ProjectFile(
                id=meta.get("id", file_id),
                project_id=project_id,
                name=meta.get("name", file_id),
                mime_type=meta.get("mime_type", "application/octet-stream"),
                size=int(meta.get("size", 0)),
                created_at_ms=int(meta.get("created_at_ms", 0)),
            )

    def delete_file(self, project_id: str, file_id: str) -> None:
        meta_path = self._file_meta_path(project_id, file_id)
        if not meta_path.is_file():
            raise ProjectError(f"file not found: {file_id}")
        meta_path.unlink()
        data_path = self._file_data_path(project_id, file_id)
        if data_path.is_file():
            data_path.unlink()
        self._touch_project(project_id)

    def list_folders(self, project_id: str) -> list[ProjectFolder]:
        if not self._meta_path(project_id).is_file():
            raise ProjectError(f"project not found: {project_id}")
        data = _read_json(self._folders_path(project_id))
        raw = data.get("folders", [])
        out: list[ProjectFolder] = []
        for entry in raw if isinstance(raw, list) else []:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            out.append(
                ProjectFolder(
                    path=path.strip(),
                    created_at_ms=int(entry.get("created_at_ms", 0)),
                )
            )
        return out

    def add_folder(self, project_id: str, path: str) -> ProjectFolder:
        if not self._meta_path(project_id).is_file():
            raise ProjectError(f"project not found: {project_id}")
        clean = (path or "").strip()
        if not clean:
            raise ProjectError("folder path is required")
        if "\0" in clean:
            raise ProjectError("folder path contains invalid characters")
        folders = self.list_folders(project_id)
        if any(f.path == clean for f in folders):
            raise ProjectError("folder already associated")
        folder = ProjectFolder(path=clean, created_at_ms=_now_ms())
        self._write_folders(project_id, [*folders, folder])
        self._touch_project(project_id)
        return folder

    def remove_folder(self, project_id: str, path: str) -> None:
        if not self._meta_path(project_id).is_file():
            raise ProjectError(f"project not found: {project_id}")
        clean = (path or "").strip()
        folders = self.list_folders(project_id)
        remaining = [f for f in folders if f.path != clean]
        if len(remaining) == len(folders):
            raise ProjectError(f"folder not found: {clean}")
        self._write_folders(project_id, remaining)
        self._touch_project(project_id)

    def _write_folders(self, project_id: str, folders: list[ProjectFolder]) -> None:
        _write_json(
            self._folders_path(project_id),
            {"folders": [{"path": f.path, "created_at_ms": f.created_at_ms} for f in folders]},
        )

    # ---- Board (kanban of worktrees) ----

    def get_board(self, project_id: str) -> dict[str, Any]:
        """Return the board for a project, or None-ish empty dict if unset."""
        if not self._meta_path(project_id).is_file():
            raise ProjectError(f"project not found: {project_id}")
        return _read_json(self._board_path(project_id))

    def setup_board(self, project_id: str, repo_path: str) -> dict[str, Any]:
        """Initialize a board for a project pointing at a git repo."""
        if not self._meta_path(project_id).is_file():
            raise ProjectError(f"project not found: {project_id}")
        clean = (repo_path or "").strip()
        if not clean:
            raise ProjectError("repo path is required")
        board = self.get_board(project_id)
        if board.get("repo_path"):
            raise ProjectError("board already configured")
        board["repo_path"] = clean
        board["columns"] = [{"id": _slugify_id(name), "name": name} for name in _DEFAULT_COLUMNS]
        board["cards"] = []
        self._write_board(project_id, board)
        self._touch_project(project_id)
        return board

    def add_column(self, project_id: str, name: str) -> dict[str, Any]:
        board = self.get_board(project_id)
        if not board.get("repo_path"):
            raise ProjectError("board not configured")
        clean = (name or "").strip()
        if not clean:
            raise ProjectError("column name is required")
        col = {"id": _slugify_id(clean), "name": clean}
        board.setdefault("columns", []).append(col)
        self._write_board(project_id, board)
        self._touch_project(project_id)
        return col

    def remove_column(self, project_id: str, column_id: str) -> None:
        board = self.get_board(project_id)
        cols = board.get("columns", [])
        remaining = [c for c in cols if c.get("id") != column_id]
        if len(remaining) == len(cols):
            raise ProjectError(f"column not found: {column_id}")
        board["columns"] = remaining
        board["cards"] = [c for c in board.get("cards", []) if c.get("column_id") != column_id]
        self._write_board(project_id, board)
        self._touch_project(project_id)

    def rename_column(self, project_id: str, column_id: str, name: str) -> dict[str, Any]:
        board = self.get_board(project_id)
        clean = (name or "").strip()
        if not clean:
            raise ProjectError("column name is required")
        for col in board.get("columns", []):
            if col.get("id") == column_id:
                col["name"] = clean
                self._write_board(project_id, board)
                self._touch_project(project_id)
                return col
        raise ProjectError(f"column not found: {column_id}")

    def create_card(
        self,
        project_id: str,
        brief: str,
        column_id: str,
        title: str = "",
    ) -> dict[str, Any]:
        """Create a card and its git worktree. ``brief`` is the user's detailed
        task description; ``title`` is optional and normally set by the planner."""
        board = self.get_board(project_id)
        repo_path = board.get("repo_path")
        if not repo_path:
            raise ProjectError("board not configured")
        clean_brief = (brief or "").strip()
        if not clean_brief:
            raise ProjectError("card brief is required")
        if not any(c.get("id") == column_id for c in board.get("columns", [])):
            raise ProjectError(f"column not found: {column_id}")
        card_id = uuid.uuid4().hex[:12]
        branch = f"card-{card_id}"
        wt_root = self._worktree_root or Path("~/.nanobot/worktrees").expanduser()
        wt_path = wt_root / _slugify_id(project_id) / card_id
        from nanobot.webui.worktrees import create_worktree

        create_worktree(Path(repo_path).expanduser(), branch, wt_path)
        card = {
            "id": card_id,
            "column_id": column_id,
            "title": (title or "").strip(),
            "brief": clean_brief,
            "branch": branch,
            "worktree_path": str(wt_path),
            "chat_session_key": None,
            "subagent_task_id": None,
            "plan": "",
            "build_result": "",
            "review_summary": "",
            "current_phase": None,
            "phase_history": [],
            "created_at_ms": _now_ms(),
            "updated_at_ms": _now_ms(),
        }
        board.setdefault("cards", []).append(card)
        self._write_board(project_id, board)
        self._touch_project(project_id)
        return card

    def move_card(self, project_id: str, card_id: str, column_id: str) -> dict[str, Any]:
        board = self.get_board(project_id)
        if not any(c.get("id") == column_id for c in board.get("columns", [])):
            raise ProjectError(f"column not found: {column_id}")
        for card in board.get("cards", []):
            if card.get("id") == card_id:
                card["column_id"] = column_id
                card["updated_at_ms"] = _now_ms()
                self._write_board(project_id, board)
                self._touch_project(project_id)
                return card
        raise ProjectError(f"card not found: {card_id}")

    def set_card_chat(self, project_id: str, card_id: str, session_key: str) -> dict[str, Any]:
        board = self.get_board(project_id)
        for card in board.get("cards", []):
            if card.get("id") == card_id:
                card["chat_session_key"] = session_key
                card["updated_at_ms"] = _now_ms()
                self._write_board(project_id, board)
                self._touch_project(project_id)
                return card
        raise ProjectError(f"card not found: {card_id}")

    def delete_card(self, project_id: str, card_id: str) -> None:
        board = self.get_board(project_id)
        cards = board.get("cards", [])
        target = next((c for c in cards if c.get("id") == card_id), None)
        if target is None:
            raise ProjectError(f"card not found: {card_id}")
        from nanobot.webui.worktrees import remove_worktree

        wt = target.get("worktree_path")
        if wt:
            try:
                remove_worktree(
                    Path(board["repo_path"]).expanduser(),
                    Path(wt),
                    force=True,
                )
            except ProjectError:
                pass
        board["cards"] = [c for c in cards if c.get("id") != card_id]
        self._write_board(project_id, board)
        self._touch_project(project_id)

    def merge_card(self, project_id: str, card_id: str, into: str = "main") -> str:
        board = self.get_board(project_id)
        target = next((c for c in board.get("cards", []) if c.get("id") == card_id), None)
        if target is None:
            raise ProjectError(f"card not found: {card_id}")
        from nanobot.webui.worktrees import merge_branch

        return merge_branch(Path(board["repo_path"]).expanduser(), target["branch"], into)

    def spawn_card(
        self,
        project_id: str,
        card_id: str,
        *,
        subagent_manager: Any,
        runtime_resolver: Any,
    ) -> dict[str, Any]:
        """Backwards-compatible alias for the build phase."""
        return self.run_card_phase(
            project_id,
            card_id,
            "build",
            subagent_manager=subagent_manager,
            runtime_resolver=runtime_resolver,
        )

    def run_card_phase(
        self,
        project_id: str,
        card_id: str,
        phase: str,
        *,
        subagent_manager: Any,
        runtime_resolver: Any,
    ) -> dict[str, Any]:
        """Spawn a background subagent to run a card phase (plan/build/validate).

        Returns the card dict with the phase task_id stored. The subagent runs
        in the card's worktree and announces a structured JSON result which is
        parsed into the card's ``plan`` / ``build_result`` / ``review_summary``.
        """
        board = self.get_board(project_id)
        target = next((c for c in board.get("cards", []) if c.get("id") == card_id), None)
        if target is None:
            raise ProjectError(f"card not found: {card_id}")
        if target.get("subagent_task_id"):
            raise ProjectError("card already has a running subagent")
        if not target.get("chat_session_key"):
            raise ProjectError("card has no chat session; open the card's chat first")

        from nanobot.security.workspace_access import build_workspace_scope

        scope = build_workspace_scope(
            target["worktree_path"],
            "restricted",
            source_channel="websocket",
        )
        extra = self.extra_read_dirs_for(project_id)
        if extra:
            from dataclasses import replace

            scope = replace(scope, extra_read_dirs=tuple(extra))

        meta = self._meta_path(project_id)
        instructions = _read_json(meta).get("instructions_md", "")
        session_key = target["chat_session_key"]
        runtime = runtime_resolver(session_key) if callable(runtime_resolver) else None
        project_id_slug = _slugify_id(project_id)

        if phase == "plan":
            tool_scope = "plan"
            label = "planner"
            task = (
                "You are the PLANNER for a task card. Read the project files and "
                f"the worktree at {target['worktree_path']} to understand what is being asked.\n"
                f"Project instructions:\n{instructions}\n"
                f"Task brief:\n{target.get('brief', '')}\n"
                "Produce a complete, step-by-step plan to execute the whole task. "
                "You may only READ files and think; do not modify or write anything.\n"
                "Reply with JSON ONLY, no prose around it, in this exact shape:\n"
                '{"title": "<short card title>", "plan": "<markdown plan>"}'
            )
        elif phase == "build":
            tool_scope = "subagent"
            label = "builder"
            task = (
                "You are the BUILDER for a task card. Follow the plan below exactly "
                f"and implement the task in the worktree at {target['worktree_path']}.\n"
                f"Project instructions:\n{instructions}\n"
                f"Card brief:\n{target.get('brief', '')}\n"
                f"Plan:\n{target.get('plan', '')}\n"
                f"You are on branch {target['branch']}. Commit your work on this branch. "
                "Do not merge or push unless asked.\n"
                "When done, reply with JSON ONLY in this exact shape:\n"
                '{"build_result": "<markdown summary of what was implemented and any tests run>"}'
            )
        elif phase == "validate":
            tool_scope = "validator"
            label = "validator"
            task = (
                "You are the VALIDATOR for a task card. Run the project's tests and "
                "verify the build actually implemented the plan.\n"
                f"Worktree: {target['worktree_path']}\n"
                f"Project instructions:\n{instructions}\n"
                f"Plan:\n{target.get('plan', '')}\n"
                f"Build result:\n{target.get('build_result', '')}\n"
                "You may run commands to execute tests (e.g. pytest / npm test) but must NOT "
                "modify any source files. Write a clear review of whether everything the plan "
                "promised was done, including test results.\n"
                "Reply with JSON ONLY in this exact shape:\n"
                '{"review_summary": "<markdown review>"}'
            )
        else:
            raise ProjectError(f"unknown phase: {phase}")

        task_id = subagent_manager.spawn(
            task=task,
            label=f"card-{label}:{project_id_slug}:{card_id[:8]}",
            origin_channel="websocket",
            origin_chat_id=session_key,
            session_key=session_key,
            workspace_scope=scope,
            runtime=runtime,
            tool_scope=tool_scope,
            extra_metadata={"card_phase": phase, "card_id": card_id, "project_id": project_id},
            on_announce=self._on_card_phase_announce,
        )
        target["subagent_task_id"] = task_id
        target["current_phase"] = phase
        history = target.setdefault("phase_history", [])
        history.append(
            {
                "phase": phase,
                "task_id": task_id,
                "started_at_ms": _now_ms(),
                "finished_at_ms": None,
                "status": "running",
            }
        )
        target["updated_at_ms"] = _now_ms()
        self._write_board(project_id, board)
        self._touch_project(project_id)
        return target

    async def _on_card_phase_announce(self, result: str, metadata: dict[str, Any]) -> None:
        """Parse a subagent's structured JSON result and write it to board.json."""
        project_id = metadata.get("project_id")
        card_id = metadata.get("card_id")
        phase = metadata.get("card_phase")
        if not project_id or not card_id or not phase:
            return
        parsed: dict[str, Any] = {}
        try:
            parsed = _extract_json_object(result)
        except Exception:
            parsed = {}
        try:
            board = self.get_board(project_id)
        except ProjectError:
            return
        target = next((c for c in board.get("cards", []) if c.get("id") == card_id), None)
        if target is None:
            return
        if phase == "plan":
            title = str(parsed.get("title") or "").strip()
            if title:
                target["title"] = title
            target["plan"] = str(parsed.get("plan") or result)
        elif phase == "build":
            target["build_result"] = str(parsed.get("build_result") or result)
        elif phase == "validate":
            target["review_summary"] = str(parsed.get("review_summary") or result)
        target["subagent_task_id"] = None
        target["current_phase"] = None
        for entry in target.setdefault("phase_history", []):
            if entry.get("task_id") == metadata.get("subagent_task_id"):
                entry["finished_at_ms"] = _now_ms()
                entry["status"] = "ok" if parsed else "error"
        target["updated_at_ms"] = _now_ms()
        self._write_board(project_id, board)
        self._touch_project(project_id)

    def card_subagent_status(
        self, project_id: str, card_id: str, subagent_manager: Any
    ) -> dict[str, Any] | None:
        """Return the subagent status payload for a card, or None if none."""
        board = self.get_board(project_id)
        target = next((c for c in board.get("cards", []) if c.get("id") == card_id), None)
        if target is None:
            raise ProjectError(f"card not found: {card_id}")
        task_id = target.get("subagent_task_id")
        if not task_id:
            return None
        status = subagent_manager.get_status(task_id)
        if status is None:
            return None
        return status.to_payload()

    def _write_board(self, project_id: str, board: dict[str, Any]) -> None:
        _write_json(self._board_path(project_id), board)

    def _unique_id(self, name: str) -> str:
        base = _slugify_id(name.lower().replace(" ", "-"))
        candidate = base
        suffix = 1
        while (self._project_dir(candidate) / "project.json").exists():
            suffix += 1
            candidate = f"{base}-{suffix}"
        return candidate

    def _touch_project(self, project_id: str) -> None:
        meta_path = self._meta_path(project_id)
        if not meta_path.is_file():
            return
        meta = _read_json(meta_path)
        meta["updated_at_ms"] = _now_ms()
        _write_json(meta_path, meta)

    def _summary(self, pdir: Path) -> ProjectSummary:
        meta = _read_json(pdir / "project.json")
        if not meta:
            raise ProjectError(f"project meta missing in {pdir}")
        fdir = pdir / _FILES_DIRNAME
        file_count = 0
        byte_count = 0
        if fdir.is_dir():
            for meta_path in fdir.glob("*.meta.json"):
                file_count += 1
                try:
                    byte_count += int(_read_json(meta_path).get("size", 0))
                except ProjectError:
                    continue
        return ProjectSummary(
            id=meta.get("id", pdir.name),
            name=meta.get("name", pdir.name),
            instructions_md=meta.get("instructions_md", ""),
            created_at_ms=int(meta.get("created_at_ms", 0)),
            updated_at_ms=int(meta.get("updated_at_ms", 0)),
            file_count=file_count,
            byte_count=byte_count,
            folder_count=len(self.list_folders(meta.get("id", pdir.name))),
        )


# ---- payload builders (no IO) ----


def projects_list_payload(controller: WebUIProjectsController) -> dict[str, Any]:
    return {
        "projects": [
            {
                "id": s.id,
                "name": s.name,
                "instructions_md": s.instructions_md,
                "created_at_ms": s.created_at_ms,
                "updated_at_ms": s.updated_at_ms,
                "file_count": s.file_count,
                "byte_count": s.byte_count,
                "folder_count": s.folder_count,
            }
            for s in controller.list_projects()
        ]
    }


def project_detail_payload(
    controller: WebUIProjectsController,
    project_id: str,
) -> dict[str, Any]:
    s = controller.get_project(project_id)
    return {
        "id": s.id,
        "name": s.name,
        "instructions_md": s.instructions_md,
        "created_at_ms": s.created_at_ms,
        "updated_at_ms": s.updated_at_ms,
        "file_count": s.file_count,
        "byte_count": s.byte_count,
        "folders": [
            {"path": f.path, "created_at_ms": f.created_at_ms}
            for f in controller.list_folders(project_id)
        ],
        "files": [
            {
                "id": f.id,
                "name": f.name,
                "mime_type": f.mime_type,
                "size": f.size,
                "created_at_ms": f.created_at_ms,
            }
            for f in controller.list_files(project_id)
        ],
    }


def project_file_payload(
    controller: WebUIProjectsController,
    project_id: str,
    file_id: str,
) -> dict[str, Any]:
    payload, file = controller.read_file(project_id, file_id)
    return {
        "id": file.id,
        "project_id": file.project_id,
        "name": file.name,
        "mime_type": file.mime_type,
        "size": file.size,
        "created_at_ms": file.created_at_ms,
        "data_url": "data:"
        + file.mime_type
        + ";base64,"
        + base64.b64encode(payload).decode("ascii"),
    }


def board_payload(controller: WebUIProjectsController, project_id: str) -> dict[str, Any]:
    """Board payload with a ``configured`` flag so the UI can prompt for setup."""
    board = controller.get_board(project_id)
    return {
        "configured": bool(board.get("repo_path")),
        "repo_path": board.get("repo_path", ""),
        "columns": board.get("columns", []),
        "cards": board.get("cards", []),
    }
