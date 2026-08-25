"""Persisted WebUI project workspace state."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_webui_dir
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceScope,
    WorkspaceScopeError,
    build_workspace_scope,
    default_workspace_scope,
    validate_workspace_scope_payload,
)
from nanobot.utils.atomic_write import atomic_write_text

WEBUI_WORKSPACE_STATE_SCHEMA_VERSION = 1
_MAX_STATE_FILE_BYTES = 128 * 1024
_DEFAULT_ACCESS_MODES = {"default", "full"}
_LEGACY_RESTRICTED_DEFAULT_ACCESS_MODE = "restricted"
_WEBUI_SCOPE_CHANNEL = "websocket"


def _scope_change_is_non_escalating(current: WorkspaceScope, requested: WorkspaceScope) -> bool:
    """Allow a remote request only when it keeps the project and does not add access."""
    return (
        requested.project_path == current.project_path
        and (not current.restrict_to_workspace or requested.restrict_to_workspace)
    )


def webui_workspace_state_path() -> Path:
    return get_webui_dir() / "workspace-state.json"


def default_webui_workspace_state() -> dict[str, Any]:
    return {
        "schema_version": WEBUI_WORKSPACE_STATE_SCHEMA_VERSION,
        "default_access_mode": "default",
        "updated_at": None,
    }


def normalize_webui_workspace_state(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    state = default_webui_workspace_state()
    updated_at = raw.get("updated_at")
    state["updated_at"] = updated_at if isinstance(updated_at, str) else None
    default_access_mode = raw.get("default_access_mode")
    if default_access_mode in _DEFAULT_ACCESS_MODES:
        state["default_access_mode"] = default_access_mode
    return state


def read_webui_workspace_state() -> dict[str, Any]:
    path = webui_workspace_state_path()
    if not path.is_file():
        return default_webui_workspace_state()
    try:
        if path.stat().st_size > _MAX_STATE_FILE_BYTES:
            logger.warning("webui workspace state too large, ignoring: {}", path)
            return default_webui_workspace_state()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("read webui workspace state failed {}: {}", path, e)
        return default_webui_workspace_state()
    return normalize_webui_workspace_state(raw)


def write_webui_workspace_state(raw: dict[str, Any]) -> dict[str, Any]:
    state = normalize_webui_workspace_state(raw)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    encoded = json.dumps(
        state,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > _MAX_STATE_FILE_BYTES:
        raise ValueError("workspace state is too large")

    path = webui_workspace_state_path()
    atomic_write_text(path, encoded.decode("utf-8") + "\n")
    return state


def read_webui_default_access_mode() -> str:
    state = read_webui_workspace_state()
    mode = state.get("default_access_mode")
    return mode if mode in _DEFAULT_ACCESS_MODES else "default"


def write_webui_default_access_mode(mode: str) -> bool:
    if mode == _LEGACY_RESTRICTED_DEFAULT_ACCESS_MODE:
        mode = "default"
    if mode not in _DEFAULT_ACCESS_MODES:
        raise ValueError("default access mode must be default or full")
    state = read_webui_workspace_state()
    changed = state.get("default_access_mode") != mode
    if changed:
        state["default_access_mode"] = mode
        write_webui_workspace_state(state)
    return changed


def default_scope_for_webui(
    default_workspace: Path,
    default_restrict_to_workspace: bool,
) -> WorkspaceScope:
    # ponytail: tools.restrictToWorkspace is the single source of truth.
    # The webui/workspace-state.json cache is a derived snapshot. If the
    # cache disagrees with the config, prefer the config and sync the
    # cache so subsequent reads are consistent.
    cache_mode = read_webui_default_access_mode()
    config_mode = _to_access_mode(default_restrict_to_workspace)
    if cache_mode != config_mode:
        logger.info(
            "webui default_access_mode cache ({}) differs from config ({}); using config",
            cache_mode, config_mode,
        )
        write_webui_default_access_mode(config_mode)
        cache_mode = config_mode
    if cache_mode == "default":
        return default_workspace_scope(
            default_workspace,
            default_restrict_to_workspace,
            source_channel=_WEBUI_SCOPE_CHANNEL,
        )
    return build_workspace_scope(default_workspace, cache_mode, source_channel=_WEBUI_SCOPE_CHANNEL)


def _to_access_mode(restrict_to_workspace: bool) -> str:
    # True (restrict) → "default" (use the restrictive default scope).
    # False (allow full fs) → "full".
    return "default" if restrict_to_workspace else "full"


def workspaces_payload(
    *,
    default_workspace: Path,
    default_restrict_to_workspace: bool,
    controls_available: bool,
) -> dict[str, Any]:
    default_access_mode = read_webui_default_access_mode()
    default_scope = (
        default_workspace_scope(
            default_workspace,
            default_restrict_to_workspace,
            source_channel=_WEBUI_SCOPE_CHANNEL,
        )
        if default_access_mode == "default"
        else build_workspace_scope(default_workspace, default_access_mode, source_channel=_WEBUI_SCOPE_CHANNEL)
    )
    return {
        "schema_version": WEBUI_WORKSPACE_STATE_SCHEMA_VERSION,
        "default_access_mode": default_access_mode,
        "default_scope": default_scope.payload(),
        "controls": {
            "can_change_project": controls_available,
            "can_use_full_access": controls_available,
        },
    }


class WebUIWorkspaceController:
    """Own WebUI project scope persistence and validation."""

    def __init__(
        self,
        *,
        session_manager: Any | None,
        default_workspace: Path,
        default_restrict_to_workspace: bool,
    ) -> None:
        self._sessions = session_manager
        self._default_workspace = default_workspace
        self._default_restrict_to_workspace = default_restrict_to_workspace

    def default_scope(self) -> WorkspaceScope:
        return default_scope_for_webui(
            self._default_workspace,
            self._default_restrict_to_workspace,
        )

    def scope_for_session_key(self, session_key: str) -> WorkspaceScope:
        if self._sessions is None:
            return self.default_scope()
        metadata_reader = getattr(self._sessions, "read_session_metadata", None)
        if callable(metadata_reader):
            data = metadata_reader(session_key)
        else:
            data = self._sessions.read_session_file(session_key)
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        if not isinstance(metadata, dict) or WORKSPACE_SCOPE_METADATA_KEY not in metadata:
            return self.default_scope()
        try:
            return validate_workspace_scope_payload(
                metadata.get(WORKSPACE_SCOPE_METADATA_KEY),
                default_workspace=self._default_workspace,
                default_restrict_to_workspace=self._default_restrict_to_workspace,
                source_channel=_WEBUI_SCOPE_CHANNEL,
            )
        except WorkspaceScopeError:
            return self.default_scope()

    def payload(self, *, controls_available: bool) -> dict[str, Any]:
        return workspaces_payload(
            default_workspace=self._default_workspace,
            default_restrict_to_workspace=self._default_restrict_to_workspace,
            controls_available=controls_available,
        )

    def scope_from_envelope(
        self,
        envelope: dict[str, Any],
        *,
        session_key: str | None,
        controls_available: bool,
    ) -> WorkspaceScope:
        current = self.scope_for_session_key(session_key) if session_key else self.default_scope()
        raw = envelope.get(WORKSPACE_SCOPE_METADATA_KEY)
        if raw is None:
            scope = current
        else:
            scope = validate_workspace_scope_payload(
                raw,
                default_workspace=self._default_workspace,
                default_restrict_to_workspace=self._default_restrict_to_workspace,
                source_channel=_WEBUI_SCOPE_CHANNEL,
            )
        if not controls_available and not _scope_change_is_non_escalating(current, scope):
            raise WorkspaceScopeError("workspace controls are localhost-only", status=403)
        return scope

    def scope_for_new_chat(
        self,
        envelope: dict[str, Any],
        *,
        controls_available: bool,
    ) -> WorkspaceScope:
        return self.scope_from_envelope(
            envelope,
            session_key=None,
            controls_available=controls_available,
        )

    def scope_for_set_request(
        self,
        envelope: dict[str, Any],
        *,
        chat_id: str,
        chat_running: bool,
        controls_available: bool,
    ) -> WorkspaceScope:
        if chat_running:
            raise WorkspaceScopeError("chat_running", status=409)
        return self.scope_from_envelope(
            envelope,
            session_key=f"websocket:{chat_id}",
            controls_available=controls_available,
        )

    def scope_for_message(
        self,
        envelope: dict[str, Any],
        *,
        chat_id: str,
        chat_running: bool,
        controls_available: bool,
    ) -> WorkspaceScope:
        scope = self.scope_from_envelope(
            envelope,
            session_key=f"websocket:{chat_id}",
            controls_available=controls_available,
        )
        if (
            WORKSPACE_SCOPE_METADATA_KEY in envelope
            and chat_running
            and scope.metadata() != self.scope_for_session_key(f"websocket:{chat_id}").metadata()
        ):
            raise WorkspaceScopeError("chat_running", status=409)
        return scope

    def persist_scope(self, chat_id: str, scope: WorkspaceScope) -> None:
        if self._sessions is not None:
            session = self._sessions.get_or_create(f"websocket:{chat_id}")
            session.metadata["webui"] = True
            session.metadata[WORKSPACE_SCOPE_METADATA_KEY] = scope.metadata()
            self._sessions.save(session)

    def persist_todo_list(self, chat_id: str, slug: str) -> None:
        """Bind a chat to a todo list slug via ``session.metadata['todo_list']``."""
        if self._sessions is None:
            return
        session = self._sessions.get_or_create(f"websocket:{chat_id}")
        session.metadata["todo_list"] = slug
        self._sessions.save(session)

    def persist_agenda_appointment(self, chat_id: str, appointment_id: str) -> None:
        """Bind a chat to an agenda appointment id via ``session.metadata['agenda_appointment']``."""
        if self._sessions is None:
            return
        session = self._sessions.get_or_create(f"websocket:{chat_id}")
        session.metadata["agenda_appointment"] = appointment_id
        self._sessions.save(session)

    def persist_research(self, chat_id: str, marker: str) -> None:
        """Mark a chat as an ephemeral research surface via ``session.metadata['research']``."""
        if self._sessions is None:
            return
        session = self._sessions.get_or_create(f"websocket:{chat_id}")
        session.metadata["research"] = marker
        self._sessions.save(session)
