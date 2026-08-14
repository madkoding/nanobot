"""Per-chat workspace registry for WhatsApp.

Maps WhatsApp group JIDs (e.g. ``120363000@g.us``) and DM chat IDs/sender IDs
to a workspace directory whose ``AGENTS.md``/``SOUL.md`` override the
channel-level workspace for turns originating in that chat. Backed by literal
``dict[str, str]`` from ``WhatsAppConfig`` — no filesystem walks, no auto-creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from nanobot.utils.helpers import truncate_text


def is_group_jid(chat_id: str) -> bool:
    """Return True if *chat_id* looks like a WhatsApp group JID."""
    return "@g.us" in chat_id


def _is_dm_chat_id(chat_id: str) -> bool:
    """Return True if *chat_id* looks like a WhatsApp DM chat ID."""
    return "@s.whatsapp.net" in chat_id


class ChatWorkspaceRegistry:
    """Resolve a workspace directory for a given WhatsApp chat."""

    _MAX_RULESET_CHARS = 8_000

    def __init__(
        self,
        *,
        group_workspaces: Mapping[str, str] | None = None,
        dm_workspace: str = "",
        dm_workspaces: Mapping[str, str] | None = None,
        log: Any | None = None,
    ) -> None:
        self._log = log or logger
        self._group_paths: dict[str, Path] = {}
        self._dm_paths: dict[str, Path] = {}
        self._default_dm_path: Path | None = None
        for raw_jid, raw_path in (group_workspaces or {}).items():
            jid = str(raw_jid).strip()
            if not jid or not is_group_jid(jid):
                self._log.warning("WhatsApp group_workspaces: ignoring non-group key {!r}", jid)
                continue
            path = self._resolve_path(raw_path, f"group_workspaces[{jid}]")
            if path is not None:
                self._group_paths[jid] = path
        for raw_sender, raw_path in (dm_workspaces or {}).items():
            sender = str(raw_sender).strip()
            if not sender:
                continue
            path = self._resolve_path(raw_path, f"dm_workspaces[{sender}]")
            if path is not None:
                self._dm_paths[sender] = path
        if dm_workspace:
            self._default_dm_path = self._resolve_path(dm_workspace, "dm_workspace")

    def _resolve_path(self, raw_path: str, context: str) -> Path | None:
        expanded = Path(str(raw_path)).expanduser()
        if not expanded.is_absolute():
            self._log.warning("WhatsApp {}: path {} is not absolute, skipping", context, expanded)
            return None
        resolved = expanded.resolve(strict=False)
        if not resolved.is_dir():
            self._log.warning("WhatsApp {}: {} is not a directory, skipping", context, resolved)
            return None
        return resolved

    def resolve(self, chat_id: str, sender_id: str | None = None) -> Path | None:
        """Return the configured workspace for *chat_id*, or ``None``."""
        if is_group_jid(chat_id):
            return self._group_paths.get(chat_id)
        if _is_dm_chat_id(chat_id) or sender_id is not None:
            target = sender_id if sender_id is not None else chat_id
            target = str(target).strip()
            if target in self._dm_paths:
                return self._dm_paths[target]
            return self._default_dm_path
        return None

    def load_ruleset(self, chat_id: str, sender_id: str | None = None) -> str | None:
        """Load and cap ``AGENTS.md``/``SOUL.md`` from the chat's workspace.

        Returns ``None`` when the chat has no workspace, when both files are
        missing or empty, or when reading fails. The format mirrors
        :class:`nanobot.agent.context.ContextBuilder` so the resulting block
        slots into the system prompt without surprising the model.
        """
        root = self.resolve(chat_id, sender_id=sender_id)
        if root is None:
            return None
        parts: list[str] = []
        for filename in ("AGENTS.md", "SOUL.md"):
            path = root / filename
            try:
                text = path.read_text(encoding="utf-8").rstrip()
            except (OSError, UnicodeDecodeError):
                continue
            if text:
                parts.append(f"{filename}:\n{text}")
        if not parts:
            return None
        joined = "\n\n".join(parts)
        return truncate_text(joined, self._MAX_RULESET_CHARS)

    def known_group_jids(self) -> tuple[str, ...]:
        return tuple(self._group_paths.keys())

    def known_jids(self) -> tuple[str, ...]:
        """Deprecated alias for :meth:`known_group_jids`."""
        return self.known_group_jids()

    def known_dm_targets(self) -> tuple[str, ...]:
        return tuple(self._dm_paths.keys())


class GroupWorkspaceRegistry(ChatWorkspaceRegistry):
    """Deprecated alias kept for compatibility with callers that import it.

    Use :class:`ChatWorkspaceRegistry` directly for new code.
    """

    def __init__(
        self,
        mapping: Mapping[str, str] | None = None,
        log: Any | None = None,
    ) -> None:
        super().__init__(group_workspaces=mapping, log=log)
