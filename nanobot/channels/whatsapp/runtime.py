"""WhatsApp channel implementation using neonize."""

from __future__ import annotations

import asyncio
import io
import json
import mimetypes
import os
import re
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel, BoundedSet
from nanobot.channels.whatsapp.group_workspace import ChatWorkspaceRegistry
from nanobot.config.paths import get_media_dir, get_runtime_subdir
from nanobot.config.schema import Base
from nanobot.runtime_context import RUNTIME_CONTEXT_INPUT_META, RuntimeContextBlock
from nanobot.utils.helpers import is_owner_match


class WhatsAppConfig(Base):
    """WhatsApp channel configuration.

    All non-hot-reloadable fields require a channel rebuild (WebUI
    disable → enable) to take effect. ``lidMappings`` is the only
    hot-reloadable runtime field — the channel writes newly observed
    LID→phone pairs back to the canonical config and reads them on
    every inbound without a restart.
    """

    enabled: bool = False
    # Hot-reload: no. Sender IDs allowed to send messages to the bot.
    # Star ("*") allows everyone; pairing store approves the rest.
    allow_from: list[str] = Field(default_factory=list)
    # Hot-reload: no. Whether group messages trigger the bot without a
    # mention.
    group_policy: Literal["open", "mention"] = "open"
    # Hot-reload: no. Path to the neonize sqlite session file.
    database_path: str = ""
    # Hot-reload: yes. Single source of truth for LID→phone resolution.
    # Runtime observations of new (lid, phone) pairs are written back
    # here automatically by the channel. Static entries (declared by
    # the user) are authoritative — runtime observations that conflict
    # with a declared entry are logged and dropped.
    lid_mappings: dict[str, str] = Field(default_factory=dict)
    # Outbound allowlist: if non-empty, only messages to these chat IDs
    # (bare phone numbers or group JIDs) are sent. Prevents the bot from
    # messaging unknown numbers and getting banned by WhatsApp.
    # Empty list = allow all (backward compatible).
    allow_send_to: list[str] = Field(default_factory=list)
    # ponytail: login timeout cap. The neonize client internally relies on
    # whatsmeow's reconnect loop; without an external cap, an unscanned QR
    # keeps the channel alive indefinitely spamming 515/EOF logs. After
    # this many seconds the channel calls client.stop() and exits cleanly.
    # Set to 0 to disable (login runs forever, original behavior).
    login_timeout_s: int = Field(default=300, ge=0, le=3600)
    # ponytail: 463 cooldown. WhatsApp replies with error 463 when it
    # throttles a number (whatsmeow issue tulir/whatsmeow#1197). Hammering
    # through the throttle hardens it into a real ban. After
    # ``throttle_threshold`` consecutive 463s, the channel stops sending
    # for ``throttle_cooldown_s`` seconds (state persisted to disk so a
    # restart cannot bring a cooling number back early). The default
    # 2-hour cooldown is conservative; the murpati.com reference suggests
    # 24h once a number has been throttled, but starting shorter lets the
    # user probe whether the block has lifted. Set threshold=0 to disable.
    throttle_threshold: int = Field(default=3, ge=0, le=100)
    throttle_cooldown_s: int = Field(default=7200, ge=0, le=604800)
    # Per-group workspace override. Maps a WhatsApp group JID (e.g.
    # "120363000@g.us") to an absolute path whose AGENTS.md / SOUL.md are
    # loaded into the system prompt for turns originating in that group.
    # Keys that aren't group JIDs, paths that aren't absolute, or paths
    # that don't exist as directories are skipped (with a warning) so a
    # stale entry never breaks a turn.
    group_workspaces: dict[str, str] = Field(default_factory=dict)
    # Per-DM workspace override. Single path used for all DMs when no
    # per-sender dm_workspaces entry matches.
    dm_workspace: str = ""
    # Per-sender DM workspace override. The key can be a bare phone number,
    # a WhatsApp "lid", or a full "56912345678@s.whatsapp.net" JID.
    dm_workspaces: dict[str, str] = Field(default_factory=dict)
    # ponytail: per-workspace model preset overrides. Sessions whose first
    # inbound arrives in a configured workspace get their session metadata
    # preset stamped with this name so the loop runs that turn (and only
    # that turn, until the user changes it with /model) on the chosen
    # model. Empty string / missing key = no override, falls back to the
    # global default preset. Keys must match a configured workspace path.
    dm_workspace_model_preset: str = ""
    group_workspace_presets: dict[str, str] = Field(default_factory=dict)
    dm_workspace_presets: dict[str, str] = Field(default_factory=dict)
    # Number of recent messages to keep per group as conversation context.
    group_context_buffer_size: int = 20
    # DM policy for messages from numbers not in known_contacts.
    # "open": process immediately (legacy behavior).
    # "screen": ask the sender to identify themselves, accumulate messages,
    #           and send the owner a summary after ``unknown_dm_summary_after``
    #           messages so the owner can decide whether to authorize the chat.
    unknown_dm_policy: Literal["open", "screen"] = "open"
    # Number of buffered messages from an unknown DM sender before the owner
    # receives a summary request. 0 disables automatic summary.
    unknown_dm_summary_after: int = 3
    # Known contacts (bare phone numbers or full JIDs) that bypass the screen.
    # The owner is always considered known. Empty list = only owner is known.
    known_contacts: list[str] = Field(default_factory=list)


class _NeonizeAPI(NamedTuple):
    NewAClient: Any
    ConnectedEv: Any
    ConnectFailureEv: Any
    DisconnectedEv: Any
    LoggedOutEv: Any
    MessageEv: Any
    PairStatusEv: Any
    StreamErrorEv: Any
    build_jid: Any
    Message: Any
    ExtendedTextMessage: Any


class _MediaInfo(NamedTuple):
    kind: str
    message: Any
    mimetype: str
    filename: str
    is_voice: bool = False


_NEONIZE_API: _NeonizeAPI | None = None
_LEGACY_BRIDGE_CONFIG_FIELDS = ("bridgeUrl", "bridgeToken", "bridge_url", "bridge_token")
# Names that look like generated WhatsApp IDs rather than human push names.
_NAME_LIKE_ID_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)


def _default_database_path() -> Path:
    return get_runtime_subdir("whatsapp-auth") / "neonize.db"


def _legacy_bridge_config_fields(config: dict[str, Any]) -> list[str]:
    return [field for field in _LEGACY_BRIDGE_CONFIG_FIELDS if field in config]


def _load_neonize() -> _NeonizeAPI:
    global _NEONIZE_API
    if _NEONIZE_API is not None:
        return _NEONIZE_API

    try:
        from neonize.aioze.client import NewAClient
        from neonize.aioze.events import (
            ConnectedEv,
            ConnectFailureEv,
            DisconnectedEv,
            LoggedOutEv,
            MessageEv,
            PairStatusEv,
            StreamErrorEv,
        )
        from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import ExtendedTextMessage, Message
        from neonize.utils.jid import build_jid
    except ImportError as exc:
        raise RuntimeError(
            "WhatsApp dependencies not installed. Run: nanobot plugins enable whatsapp"
        ) from exc

    _ensure_ffmpeg_in_path()

    _NEONIZE_API = _NeonizeAPI(
        NewAClient=NewAClient,
        ConnectedEv=ConnectedEv,
        ConnectFailureEv=ConnectFailureEv,
        DisconnectedEv=DisconnectedEv,
        LoggedOutEv=LoggedOutEv,
        MessageEv=MessageEv,
        PairStatusEv=PairStatusEv,
        StreamErrorEv=StreamErrorEv,
        build_jid=build_jid,
        Message=Message,
        ExtendedTextMessage=ExtendedTextMessage,
    )
    return _NEONIZE_API


def _ensure_ffmpeg_in_path() -> None:
    """Make sure ``ffmpeg``/``ffprobe`` are resolvable on PATH.

    neonize shells out to ``ffprobe`` when sending audio (to read the
    duration). Without these binaries every audio send fails with
    ``ffprobe: not found``. We prefer the host install when present; otherwise
    we fall back to the bundled binaries shipped by ``static-ffmpeg`` (downloaded
    once on first use, then cached in the venv).
    """
    import shutil

    path = os.environ.get("PATH", "")
    for binary in ("ffprobe", "ffmpeg"):
        if shutil.which(binary):
            continue
        try:
            from static_ffmpeg import run as static_ffmpeg_run  # type: ignore[import-not-found]
        except ImportError:
            return
        try:
            ffmpeg_path, ffprobe_path = static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
        except Exception:  # noqa: BLE001 - offline / locked / partial download, fall through
            return
        bin_dir = str(Path(ffmpeg_path).parent)
        if bin_dir not in path.split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + path


# JID and message-field helpers live in whatsapp/jid.py and
# whatsapp/message_helpers.py; re-export so existing import sites keep working.
from nanobot.channels.whatsapp.jid import (  # noqa: E402
    _JID_RE,
    _bare_jid,
    _classify_sender_ids,
    _jid_to_string,
    _normalize_jid,
    _typing_task_key,
)
from nanobot.channels.whatsapp.message_helpers import (  # noqa: E402
    _context_infos,
    _media_message,
    _MediaInfo,
    _message_text,
    _safe_attr,
)


class WhatsAppChannel(BaseChannel):
    """WhatsApp channel using neonize's async WhatsApp client."""

    name = "whatsapp"
    display_name = "WhatsApp"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WhatsAppConfig().model_dump(by_alias=True)

    @classmethod
    def build_kwargs(cls, manager: Any) -> dict[str, Any]:
        # Pass the manager so the channel can persist runtime config changes
        # (learned LID->phone pairs) back to the canonical config.json.
        return {"manager": manager}

    def __init__(
        self,
        config: Any,
        bus: MessageBus,
        *,
        owner_id: str | list[str] | None = None,
        manager: Any | None = None,
    ):
        legacy_bridge_fields = _legacy_bridge_config_fields(config) if isinstance(config, dict) else []
        # ponytail: if the caller handed us a raw dict from the channel
        # manager (which holds the live config), capture a reference to its
        # ``lidMappings`` dict so any mutation we make flows straight back
        # to the manager's config and gets persisted on
        # ``manager.persist_config_change``. Without this the WhatsAppConfig
        # below would be a fresh object with its own dict, disconnected
        # from the manager's view.
        raw_lid_mappings: dict | None = None
        if isinstance(config, dict):
            raw = config.get("lidMappings")
            if isinstance(raw, dict):
                raw_lid_mappings = raw
            config = WhatsAppConfig.model_validate(config)
        if raw_lid_mappings is not None:
            # Re-point our config view at the manager's dict.
            config.lid_mappings = raw_lid_mappings
        super().__init__(config, bus, owner_id=owner_id)
        self._mgr = manager
        if legacy_bridge_fields:
            self.logger.warning(
                "Ignoring deprecated WhatsApp bridge config fields: {}. "
                "Run 'nanobot channels login whatsapp' to create a neonize session.",
                ", ".join(legacy_bridge_fields),
            )
        self._client: Any | None = None
        self._connected = False
        self._processed_message_ids = self._bounded_set(1000)
        # ponytail: single source of truth for LID->phone is now
        # self.config.lid_mappings (see the _lid_to_phone property). The
        # legacy message_state.json::lid_to_phone migration runs at
        # start() (when the display-name path is initialized); we just
        # initialize the property here for direct-construction callers
        # (tests, headless scenarios). NOTE: we do NOT assign to
        # ``self._lid_to_phone`` here — that triggers the legacy setter
        # which would replace ``self.config.lid_mappings`` with a clone,
        # breaking the reference we just wired to the manager's dict.
        # The property getter reads from ``self.config.lid_mappings`` so
        # no eager assignment is needed.
        # ponytail: debounced persistence for _processed_message_ids.
        # LID->phone pairs are no longer written here; they go straight
        # to config.json via _persist_lid_mapping. Saves happen on a
        # background task so the hot path (every inbound message) stays
        # sync-cheap. The cap below is the last-seen monotonic count;
        # we write when it doubles OR every 5 minutes, whichever comes
        # first.
        self._state_dirty_count: int = 0
        self._state_last_save_at: float = 0.0
        self._self_jids: set[str] = set()
        # ponytail: holds the live neonize client so mention detection can
        # retry _remember_self_jids lazily if it ran before me/JID was
        # populated (some accounts expose only LID/PN at first).
        self._current_client: Any = None
        self._started_at = 0.0
        self._typing_tasks: dict[str, asyncio.Task[None]] = {}
        # Per-group turn queue: one worker per chat processes messages sequentially.
        self._group_queues: dict[str, asyncio.Queue[tuple[Any, str | None]]] = {}
        self._group_queue_tasks: dict[str, asyncio.Task[None]] = {}
        # Lifecycle hook fired by the ConnectedEv handler. start() installs
        # a callback here to signal that login completed, so the
        # login_timeout_s cap doesn't tear down a healthy session.
        self._on_connected_for_lifecycle: Callable[[], None] | None = None
        # Persisted display-name mappings per chat (phone/sender_id -> push_name).
        self._display_names: dict[str, dict[str, str]] = {}
        self._display_names_path = get_runtime_subdir("whatsapp-auth") / "display_names.json"
        # Per-chat workspace registry. Channels.manager hands this to
        # AgentLoop so turns originating in a mapped group or DM pick up
        # the chat's AGENTS.md/SOUL.md instead of the channel-level workspace.
        self.group_workspace_registry = ChatWorkspaceRegistry(
            group_workspaces=self.config.group_workspaces,
            dm_workspace=self.config.dm_workspace,
            dm_workspaces=self.config.dm_workspaces,
            dm_workspace_model_preset=self.config.dm_workspace_model_preset,
            group_workspace_presets=self.config.group_workspace_presets,
            dm_workspace_presets=self.config.dm_workspace_presets,
            log=self.logger,
        )
        # Rolling buffer of recent messages per WhatsApp group (chat_jid -> deque).
        # Each entry is (sender_id, display_name, text, timestamp).
        self._group_context_buffers: dict[str, deque[tuple[str, str | None, str, float]]] = {}
        # Buffered messages from unknown DM senders while they are being screened.
        # Key is sender_id, value is a list of (display_name, text, timestamp).
        self._unknown_dm_buffers: dict[str, list[tuple[str | None, str, float]]] = {}

    def _database_path(self) -> Path:
        configured = self.config.database_path.strip()
        return Path(configured).expanduser() if configured else _default_database_path()

    def _load_display_names(self) -> dict[str, dict[str, str]]:
        """Load persisted {chat_id: {sender_key: push_name}} map."""
        if not self._display_names_path.exists():
            return {}
        try:
            data = json.loads(self._display_names_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    str(k): {str(sk): str(sv) for sk, sv in v.items() if sv}
                    for k, v in data.items()
                    if isinstance(v, dict)
                }
        except Exception:
            self.logger.exception("Failed to load WhatsApp display names")
        return {}

    def _save_display_names(self) -> None:
        """Persist the display-name map atomically."""
        self._display_names_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._display_names_path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(self._display_names, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            tmp.replace(self._display_names_path)
        except Exception:
            self.logger.exception("Failed to save WhatsApp display names")

    def _message_state_path(self) -> Path:
        return self._display_names_path.parent / "message_state.json"

    def _load_message_state(self) -> BoundedSet:
        """Load persisted dedup cache. Missing file → empty.

        LID->phone pairs are loaded from ``config.lidMappings``; legacy
        entries in ``message_state.json::lid_to_phone`` are migrated to
        ``config.lidMappings`` at __init__ and dropped from this file.
        """
        path = self._message_state_path()
        ids = self._bounded_set(1000)
        if not path.exists():
            return ids
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.logger.exception("Failed to load WhatsApp message state; resetting")
            return ids
        ids_raw = data.get("processed_ids") if isinstance(data, dict) else None
        if isinstance(ids_raw, list):
            for mid in ids_raw[-1000:]:
                if isinstance(mid, str) and mid:
                    ids.add(mid)
        return ids

    def _save_message_state(self) -> None:
        """Persist the dedup LRU atomically.

        LID->phone pairs are no longer written here; they live in
        ``config.json::lidMappings`` as the single source of truth and are
        flushed via ``_persist_lid_mapping``.
        """
        path = self._message_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(
                    {"processed_ids": self._processed_message_ids.keys()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:
            self.logger.exception("Failed to save WhatsApp message state")

    def _touch_message_state(self) -> None:
        """Mark state dirty and persist on the debounced cadence.

        Called on every inbound message. Keeps the hot path cheap: we
        only hit the disk when the dirty count doubles or 5 minutes
        have passed.
        """
        import time as _time
        self._state_dirty_count += 1
        now = _time.monotonic()
        if self._state_last_save_at == 0.0:
            self._state_last_save_at = now
        if (now - self._state_last_save_at) >= 300.0 or self._state_dirty_count >= 500:
            self._save_message_state()
            self._state_dirty_count = 0
            self._state_last_save_at = now

    def _throttle_state_path(self) -> Path:
        return get_runtime_subdir("whatsapp-auth") / "throttle.json"

    def _load_throttle_state(self) -> dict[str, Any]:
        """Load persisted 463-throttle state. Missing file → empty state."""
        path = self._throttle_state_path()
        if not path.exists():
            return {"consecutive_463": 0, "cooldown_until": 0.0}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            consecutive = int(data.get("consecutive_463", 0) or 0)
            cooldown_until = float(data.get("cooldown_until", 0.0) or 0.0)
        except Exception:
            self.logger.exception("Failed to load WhatsApp throttle state; resetting")
            return {"consecutive_463": 0, "cooldown_until": 0.0}
        return {"consecutive_463": max(consecutive, 0), "cooldown_until": max(cooldown_until, 0.0)}

    def _save_throttle_state(self, consecutive: int, cooldown_until: float) -> None:
        """Persist throttle state atomically."""
        path = self._throttle_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(
                    {"consecutive_463": consecutive, "cooldown_until": cooldown_until},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:
            self.logger.exception("Failed to save WhatsApp throttle state")

    def _check_throttle(self) -> float | None:
        """Return remaining cooldown seconds if the channel is throttled, else None.

        Expired cooldowns auto-clear so the next send can resume normally.
        """
        state = self._load_throttle_state()
        cooldown_until = state["cooldown_until"]
        now = time.time()
        if cooldown_until and now >= cooldown_until:
            if state["consecutive_463"] or cooldown_until:
                self._save_throttle_state(0, 0.0)
                self.logger.info(
                    "WhatsApp 463 cooldown expired; sending resumed."
                )
            return None
        if cooldown_until and now < cooldown_until:
            return cooldown_until - now
        return None

    def _record_send_success(self) -> None:
        """Reset the 463 counter on a successful send."""
        state = self._load_throttle_state()
        if state["consecutive_463"] or state["cooldown_until"]:
            self._save_throttle_state(0, 0.0)

    def _record_send_throttled(self) -> float | None:
        """Bump the 463 counter and trip the cooldown if the threshold is reached.

        Returns the cooldown duration in seconds if a new cooldown was just
        tripped, otherwise None.
        """
        threshold = int(getattr(self.config, "throttle_threshold", 0) or 0)
        if threshold <= 0:
            return None
        cooldown_s = int(getattr(self.config, "throttle_cooldown_s", 0) or 0)
        if cooldown_s <= 0:
            return None
        state = self._load_throttle_state()
        consecutive = state["consecutive_463"] + 1
        now = time.time()
        if state["cooldown_until"] and now < state["cooldown_until"]:
            # Already cooling down; just persist the bumped counter.
            self._save_throttle_state(consecutive, state["cooldown_until"])
            return None
        if consecutive >= threshold:
            cooldown_until = now + cooldown_s
            self._save_throttle_state(consecutive, cooldown_until)
            self.logger.warning(
                "WhatsApp 463 throttle tripped after {} consecutive errors; "
                "pausing sends for {}s (~{}h) until {}. "
                "After that, sends resume automatically on the next success.",
                consecutive, cooldown_s, cooldown_s // 3600,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cooldown_until)),
            )
            return float(cooldown_s)
        self._save_throttle_state(consecutive, state["cooldown_until"])
        self.logger.warning(
            "WhatsApp 463 throttled ({}/{} before cooldown).",
            consecutive, threshold,
        )
        return None

    def _remember_display_name(
        self, chat_id: str, sender_key: str, push_name: str | None
    ) -> str | None:
        """Store a display name for a sender in a chat and return it if usable."""
        if not push_name or not self._looks_like_name(push_name):
            return None
        chat_map = self._display_names.setdefault(chat_id, {})
        if chat_map.get(sender_key) != push_name:
            chat_map[sender_key] = push_name
            self._save_display_names()
        return push_name

    def _display_name_for(self, chat_id: str, sender_key: str) -> str | None:
        """Return a previously seen display name for a sender, if any."""
        return self._display_names.get(chat_id, {}).get(sender_key)

    def _load_lid_mappings(self) -> dict[str, str]:
        # Deprecated: lidMappings now live directly in config and are read via
        # the _lid_to_phone derived property. Kept as a no-op for callers
        # that still construct the channel the old way (tests, direct
        # instantiation without a manager).
        return dict(self.config.lid_mappings)

    @property
    def _lid_to_phone(self) -> dict[str, str]:
        """Single source of truth for LID->phone resolution.

        Returns the live ``config.lid_mappings`` dict so every reader sees
        the same values. Mutating callers (the inbound handler when it
        learns a new pair) write to ``self.config.lid_mappings`` and
        trigger ``self._persist_lid_mapping`` to flush to disk.
        """
        return self.config.lid_mappings

    @_lid_to_phone.setter
    def _lid_to_phone(self, value: dict[str, str]) -> None:
        # Compatibility shim: legacy code that assigned ``self._lid_to_phone
        # = {lid: phone}`` now redirects into the canonical config dict.
        self.config.lid_mappings = dict(value)

    async def _persist_lid_mapping(self, lid: str, phone: str) -> bool:
        """Add ``lid -> phone`` to the canonical config and flush to disk.

        Returns True if a new mapping was actually persisted (caller can
        skip logging when False). The runtime is the only writer here so
        concurrent channels don't race; the manager's
        ``persist_config_change`` re-serializes the whole config object.
        """
        if not lid or not phone:
            return False
        existing = self.config.lid_mappings.get(lid, "")
        if existing == phone:
            return False
        self.config.lid_mappings[lid] = phone
        if self._mgr is not None and hasattr(self._mgr, "persist_config_change"):
            return await self._mgr.persist_config_change()
        self.logger.debug(
            "LID mapping {} -> {} updated in-memory only (no manager wired)",
            lid, phone,
        )
        return True

    def _learned_lid_phone_pair(self, lid: str, phone: str) -> None:
        """In-memory + scheduled-on-event-loop persistence for a new LID pair.

        Used from the sync inbound handler. Updates the config dict
        immediately so subsequent resolution sees it, and schedules the
        async disk flush so the caller doesn't block on I/O. If a loop
        is not running, falls back to a sync persist (best-effort).
        """
        if not lid or not phone:
            return
        if self.config.lid_mappings.get(lid) == phone:
            return
        self.config.lid_mappings[lid] = phone
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            # No event loop (tests, sync callers). Persist synchronously.
            if self._mgr is not None and hasattr(self._mgr, "persist_config_change"):
                try:
                    import asyncio as _asyncio
                    _asyncio.run(self._mgr.persist_config_change())
                except Exception:
                    self.logger.exception("Sync LID->phone persist failed for {} -> {}", lid, phone)
            return
        loop.create_task(self._persist_lid_mapping(lid, phone))

    def _migrate_legacy_lid_cache(self) -> bool:
        """One-shot migration of legacy ``message_state.json::lid_to_phone``.

        Pre-single-source-of-truth builds persisted learned LID->phone
        pairs to ``message_state.json``. Migrate those into the canonical
        ``config.lid_mappings`` and drop the legacy field. Idempotent.
        """
        path = self._message_state_path()
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(data, dict):
            return False
        lid_raw = data.get("lid_to_phone")
        if not isinstance(lid_raw, dict) or not lid_raw:
            return False
        added = 0
        for lid, phone in lid_raw.items():
            if not isinstance(lid, str) or not isinstance(phone, str):
                continue
            lid = lid.strip()
            phone = phone.strip()
            if not lid or not phone:
                continue
            if self.config.lid_mappings.get(lid) == phone:
                continue
            self.config.lid_mappings[lid] = phone
            added += 1
        if added == 0:
            return False
        self.logger.info(
            "Migrated {} legacy LID->phone mappings from message_state.json to lidMappings",
            added,
        )
        return True

    def _new_client(self) -> Any:
        api = _load_neonize()
        db_path = self._database_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return api.NewAClient(str(db_path))

    async def login(self, force: bool = False) -> bool:
        db_path = self._database_path()
        if force:
            self._reset_database(db_path)

        client = self._new_client()
        login_result = asyncio.get_running_loop().create_future()
        self._register_handlers(client, login_result=login_result, handle_messages=False)

        try:
            self.logger.info("Starting WhatsApp login with neonize...")
            connect_task = await client.connect()
            self._fail_login_on_connect_task_done(connect_task, login_result)
            await login_result
            self.logger.info("WhatsApp login complete")
            return True
        except Exception as exc:
            self.logger.error("WhatsApp login failed: {}", exc)
            return False
        finally:
            with suppress(Exception):
                await client.stop()

    async def start(self) -> None:
        """Run the channel: connect once and block on idle().

        No auto-reconnect. If the session is lost (LoggedOut, disconnect,
        stream error), the channel stops. The user re-links manually from
        a shell (``nanobot channels login whatsapp``) and restarts the
        gateway.
        """
        await self._run_session()

    async def _run_session(self) -> None:
        """Run connect→idle cycles, reconnecting when the websocket dies.

        neonize's ``idle()`` blocks on a Go call that never returns when the
        websocket drops (EOF / "stream replaced"), so the channel would sit
        "connected" but dead until the manager watchdog restarts it ~10 min
        later. A health-check task polls ``is_connected()`` and stops the
        client when the socket is actually dead, which unblocks ``idle()`` and
        lets this loop reconnect immediately.
        """
        self._running = True
        self._started_at = time.time()
        self._display_names = self._load_display_names()
        # One-shot migration of any legacy LID->phone cache left in
        # message_state.json by an earlier build. Migrates into
        # config.lidMappings (the single source of truth) and flushes
        # to disk if anything was added.
        migrated = self._migrate_legacy_lid_cache()
        if migrated and self._mgr is not None and hasattr(self._mgr, "persist_config_change"):
            await self._mgr.persist_config_change()
        # ponytail: restore dedup LRU so a restart doesn't re-process
        # buffered messages. LID->phone routing now lives in
        # config.lidMappings and is the single source of truth.
        loaded_ids = self._load_message_state()
        if loaded_ids:
            self._processed_message_ids = loaded_ids
            self.logger.info(
                "Restored {} processed message ids from message_state.json",
                len(loaded_ids),
            )
        self._state_dirty_count = 0
        self._state_last_save_at = 0.0

        login_timeout_s = int(getattr(self.config, "login_timeout_s", 0) or 0)
        first = True
        try:
            while self._running:
                self._reconnect_triggered = False
                client = self._new_client()
                self._client = client
                self._register_handlers(client, handle_messages=True)
                try:
                    self.logger.info("Connecting WhatsApp channel with neonize...")
                    await client.connect()
                    if first and login_timeout_s > 0:
                        self.logger.info(
                            "WhatsApp login timeout: {} seconds; channel will stop if no login completes by then.",
                            login_timeout_s,
                        )
                        # The cap only matters before login completes. Once
                        # we're connected, idle() running past the cap must
                        # NOT tear down the channel — that would kill a
                        # healthy session every 300s. Race the login-completion
                        # signal against the cap, then fall through to the
                        # common idle path below.
                        connected_event = asyncio.Event()
                        self._on_connected_for_lifecycle = connected_event.set
                        try:
                            try:
                                await asyncio.wait_for(
                                    connected_event.wait(), timeout=login_timeout_s
                                )
                            except asyncio.TimeoutError:
                                if not self._connected:
                                    self.logger.warning(
                                        "WhatsApp login not completed in {} seconds; stopping channel. "
                                        "Scan the QR and restart the gateway to retry.",
                                        login_timeout_s,
                                    )
                                    return
                        finally:
                            self._on_connected_for_lifecycle = None
                    # Block on idle() so the channel keeps the websocket
                    # alive and processes events. A dead socket is detected
                    # by the health-check task below, which stops the client
                    # and unblocks this await.
                    health = asyncio.create_task(self._health_check(client))
                    try:
                        await client.idle()
                    finally:
                        health.cancel()
                        with suppress(asyncio.CancelledError):
                            await health
                finally:
                    self._connected = False
                    if self._client is client:
                        self._client = None
                    with suppress(Exception):
                        await client.stop()
                first = False
                if not self._running:
                    break
                if not self._reconnect_triggered:
                    # Clean stop or terminal auth error: exit without retry.
                    break
                self.logger.warning("WhatsApp connection lost; reconnecting...")
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False
            self._connected = False
            # ponytail: flush dedup + LID map to disk on every session end
            # so the next start() picks up where we left off.
            self._save_message_state()

    async def _health_check(self, client: Any) -> None:
        """Stop the client when the websocket is dead so idle() unblocks.

        Only acts once we previously believed we were connected (so it never
        kills an in-progress QR login). ``client.stop()`` cancels the Go-side
        context, which makes ``idle()`` return and the reconnect loop run.
        """
        while True:
            await asyncio.sleep(15)
            if not self._connected:
                continue
            try:
                # ponytail: bump activity every healthy poll so the manager
                # watchdog (WATCHDOG_IDLE_S=600s) doesn't kill idle sessions.
                if not client.is_connected():
                    self._reconnect_triggered = True
                    self.logger.warning(
                        "WhatsApp websocket lost; stopping client to reconnect"
                    )
                    await client.stop()
                    return
                self._touch_activity()
            except Exception:
                return


    async def stop(self) -> None:
        self._running = False
        self._connected = False
        client = self._client
        self._client = None
        self._group_context_buffers.clear()
        if client is not None:
            await client.stop()

    def _check_outbound_allowed(self, chat_id: str) -> None:
        """Reject sends to chat IDs not in the outbound allowlist.

        If ``allow_send_to`` is empty, all destinations are allowed
        (backward compatible). Otherwise the bare chat ID (phone number
        or group JID without server suffix) must appear in the list.
        Groups are allowed if the bare group ID is listed.
        """
        allow = getattr(self.config, "allow_send_to", None) or []
        if not allow:
            return
        bare = _bare_jid(chat_id)
        allow_set = {str(item).strip() for item in allow if str(item).strip()}
        if bare in allow_set or chat_id in allow_set:
            return
        if "@lid" in chat_id:
            lid_bare = chat_id.split("@", 1)[0]
            phone = self._lid_to_phone.get(lid_bare)
            if phone and phone in allow_set:
                return
        self.logger.warning(
            "WhatsApp outbound allowlist blocked send to {} "
            "(not in allow_send_to: {}). "
            "Add the number to channels.whatsapp.allowSendTo in config.",
            chat_id, ", ".join(sorted(allow_set)),
        )
        raise RuntimeError(
            f"WhatsApp outbound allowlist blocked send to {chat_id}"
        )

    @staticmethod
    def _fail_login_on_connect_task_done(
        connect_task: asyncio.Task[Any] | None,
        login_result: asyncio.Future[None],
    ) -> None:
        if connect_task is None:
            return

        def _on_done(task: asyncio.Task[Any]) -> None:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                return
            if login_result.done():
                return
            if exc is not None:
                login_result.set_exception(exc)
            else:
                login_result.set_exception(
                    RuntimeError("WhatsApp connection ended before login completed")
                )

        connect_task.add_done_callback(_on_done)

    async def send(self, msg: OutboundMessage) -> None:
        client = self._client
        if client is None or not self._connected:
            raise RuntimeError("WhatsApp channel is not connected")

        remaining = self._check_throttle()
        if remaining is not None:
            hours = remaining / 3600
            self.logger.warning(
                "WhatsApp 463 cooldown active; skipping send to {} "
                "(~{}h remaining).",
                msg.chat_id, f"{hours:.1f}",
            )
            raise RuntimeError(
                f"WhatsApp 463 cooldown active; {remaining:.0f}s remaining"
            )

        self._check_outbound_allowed(msg.chat_id)

        to = self._resolve_send_target(msg.chat_id)
        typing_jid = self._build_typing_jid(msg.chat_id)
        await self._stop_typing(typing_jid)
        content = msg.content
        mention = (msg.metadata or {}).get("whatsapp_mention")
        if mention and self._is_group_chat(msg.chat_id):
            content = f"{mention} {content}"
        try:
            if content:
                # Send text as a typed extendedTextMessage instead of the plain
                # `conversation` field. WhatsApp silently drops `conversation`
                # in group chats, while typed fields (extendedTextMessage,
                # audioMessage, ...) are delivered reliably — the same path the
                # media senders use. Build it here so the mention prefix above
                # still lands in the text.
                api = _load_neonize()
                msg_obj = api.Message(
                    extendedTextMessage=api.ExtendedTextMessage(text=content)
                )
                await client.send_message(to, msg_obj)
        except Exception as exc:
            cls_name = type(exc).__name__
            if cls_name == "SendMessageError" and "463" in str(exc):
                self._record_send_throttled()
            raise
        else:
            self._record_send_success()

        ptt_override = (msg.metadata or {}).get("ptt")
        for media_path in msg.media or []:
            ptt = self._resolve_ptt(media_path, ptt_override)
            await self._send_media(client, to, media_path, ptt=ptt)

    @staticmethod
    def _resolve_ptt(media_path: str, ptt_override: Any) -> bool:
        """Decide whether a media attachment should be sent as a WhatsApp voice note.

        ``ptt_override`` wins when explicitly set to a bool. Otherwise, audio
        attachments that WhatsApp treats as voice notes (audio/ogg Opus or
        audio/mpeg from tts) are auto-marked so callers don't need to know
        about the WhatsApp-specific flag.
        """
        if isinstance(ptt_override, bool):
            return ptt_override
        mime, _ = mimetypes.guess_type(str(Path(media_path).expanduser()))
        # Only native OGG Opus files are safe to auto-mark as PTT. MP3 must be
        # sent as regular audio or WhatsApp mobile clients often refuse it.
        return (mime or "").lower() in {"audio/ogg", "audio/opus"}

    async def _start_typing(self, jid: Any) -> None:
        """Start periodic 'composing' presence updates for a chat.

        WhatsApp requires the bot to refresh the presence every ~5s; otherwise
        the client shows nothing. The task is cancelled automatically when
        :meth:`send` (or :meth:`_stop_typing`) runs.
        """
        key = _typing_task_key(jid)
        await self._stop_typing(jid)
        client = self._client
        if client is None or not self._connected:
            return

        try:
            from neonize.utils.enum import ChatPresence, ChatPresenceMedia
        except ImportError:
            return

        async def _loop() -> None:
            try:
                # Send the first presence immediately so the indicator appears
                # without waiting for the first sleep interval.
                try:
                    await client.send_chat_presence(
                        jid, ChatPresence.CHAT_PRESENCE_COMPOSING, ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT
                    )
                    self.logger.debug("typing started for {}", key)
                except Exception as exc:  # noqa: BLE001 - presence is best-effort
                    self.logger.debug("typing presence failed for {}: {}", key, exc)
                while True:
                    try:
                        await asyncio.sleep(5)
                    except asyncio.CancelledError:
                        return
                    try:
                        await client.send_chat_presence(
                            jid, ChatPresence.CHAT_PRESENCE_COMPOSING, ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT
                        )
                    except Exception as exc:  # noqa: BLE001 - presence is best-effort
                        self.logger.debug("typing presence failed for {}: {}", key, exc)
            finally:
                self._typing_tasks.pop(key, None)

        self._typing_tasks[key] = asyncio.create_task(_loop())

    async def _stop_typing(self, jid: Any) -> None:
        """Stop typing for a chat and notify the client we paused."""
        key = _typing_task_key(jid)
        task = self._typing_tasks.pop(key, None)
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        client = self._client
        if client is None or not self._connected:
            return
        try:
            from neonize.utils.enum import ChatPresence, ChatPresenceMedia
        except ImportError:
            return
        try:
            await client.send_chat_presence(
                jid, ChatPresence.CHAT_PRESENCE_PAUSED, ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("typing pause failed for {}: {}", key, exc)

    def _build_jid(self, raw: str) -> Any:
        api = _load_neonize()
        target = raw.strip()
        match = _JID_RE.match(_normalize_jid(target))
        if not match:
            return api.build_jid(target)

        user = match.group("user").split(":", 1)[0]
        server = match.group("server")
        return api.build_jid(user, server)

    def owner_chat_id(self) -> str | None:
        # ponytail: route error notifications to the operator's WhatsApp DM
        # (phone + "@s.whatsapp.net") instead of spamming the originating
        # group when a send fails. The global owner_id may hold multiple
        # identities (Discord, Telegram, ...); pick the phone-shaped one.
        # Phones are 8-15 digits; Discord snowflakes are 17+ so they're
        # excluded by the length cap.
        owner_ids = self._owner_id if isinstance(self._owner_id, (list, tuple, set)) else [self._owner_id]
        for ident in owner_ids:
            digits = "".join(ch for ch in str(ident) if ch.isdigit())
            if 8 <= len(digits) <= 15:
                return self._build_jid(f"{digits}@s.whatsapp.net")
        return None


    def _resolve_send_target(self, chat_id: str) -> Any:
        """Pick a routable JID for sending.

        WhatsApp now exposes inbound messages with a ``@lid`` chat_jid even when
        the user is a regular DM. Sending to that LID JID is silently dropped
        by the server for some sessions. We translate it to the verified phone
        number when we have a mapping (filled by ``_classify_sender_ids`` on
        inbound), and fall back to the LID otherwise.
        """
        normalized = _normalize_jid(chat_id)
        if "@lid" in normalized:
            bare = normalized.split("@", 1)[0]
            phone = self._lid_to_phone.get(bare)
            if phone:
                return self._build_jid(f"{phone}@s.whatsapp.net")
        return self._build_jid(chat_id)

    def _is_known_sender(self, sender_id: str) -> bool:
        """Return True when the sender is the owner or in the known contacts list."""
        if not sender_id:
            return False
        if self._owner_id and is_owner_match(sender_id, self._owner_id):
            return True
        normalized = _normalize_jid(sender_id)
        bare = _bare_jid(sender_id)
        for contact in self.config.known_contacts:
            contact_text = str(contact or "").strip()
            if not contact_text:
                continue
            if contact_text == sender_id or contact_text == normalized or contact_text == bare:
                return True
            contact_bare = _bare_jid(contact_text)
            if contact_bare and (contact_bare == bare or contact_bare == normalized):
                return True
        return False

    def _unknown_dm_summary(self, sender_id: str) -> tuple[str, str] | None:
        """Return (summary_text, sender_label) for buffered unknown DM messages."""
        buffer = self._unknown_dm_buffers.get(sender_id)
        if not buffer:
            return None
        lines = []
        label = sender_id
        for display_name, text, _timestamp in buffer:
            if display_name:
                label = display_name
            lines.append(f"- [{label}]: {text[:200]}")
        summary = "\n".join(lines)
        return summary, label

    async def _screen_unknown_dm(
        self,
        client: Any,
        sender_id: str,
        chat_jid: str,
        display_name: str | None,
        text: str,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Buffer an unknown DM and ask the sender to identify themselves.

        Once ``unknown_dm_summary_after`` messages have been buffered, send the
        owner a summary so they can decide whether to authorize the chat.
        """
        buffer = self._unknown_dm_buffers.setdefault(sender_id, [])
        buffer.append((display_name, text, timestamp or time.monotonic()))
        self.logger.info(
            "Screening unknown WhatsApp DM from {} ({} messages buffered)",
            sender_id, len(buffer),
        )

        # Reply once, politely, asking the sender to identify themselves.
        if len(buffer) == 1:
            greeting = display_name or "Hola"
            reply = (
                f"Hola {greeting}, soy el asistente de {self._owner_name()}. "
                "No reconozco tu número. Por favor cuéntame quién eres "
                "y cómo puedo ayudarte. Tu mensaje será revisado."
            )
            await self._send_whatsapp_text(client, chat_jid, reply)

        summary_after = self.config.unknown_dm_summary_after
        if summary_after > 0 and len(buffer) >= summary_after:
            summary, label = self._unknown_dm_summary(sender_id) or ("", sender_id)
            owner_chat = self.owner_chat_id()
            if owner_chat and summary:
                header = (
                    f"Resumen de mensajes de un número desconocido ({sender_id} / {label}):\n\n"
                )
                footer = (
                    "\n\nEste remitente no está en mis contactos conocidos. "
                    "Responde en este chat si quieres que autorice la conversación."
                )
                await self._send_whatsapp_text(client, owner_chat, header + summary + footer)
            # Keep buffering; the owner can reply to authorize later.

    def _owner_name(self) -> str:
        """Return a display name for the owner in the greeting message."""
        owner_ids = self._owner_id if isinstance(self._owner_id, (list, tuple, set)) else [self._owner_id]
        for ident in owner_ids:
            text = str(ident or "").strip()
            if text:
                return text
        return "mi operador"

    async def _send_whatsapp_text(self, client: Any, to: Any, text: str) -> None:
        """Send a plain text WhatsApp message, translating LID JIDs when possible."""
        if not hasattr(client, "send_message"):
            self.logger.debug("client has no send_message; dropping text to {}", to)
            return
        target = self._resolve_send_target(to) if isinstance(to, str) else to
        await client.send_message(target, text)

    def _whatsapp_session_key(self, chat_id: str, sender_id: str, is_group: bool) -> str:
        """Return an isolated session key per sender inside group chats.

        In DMs the default ``whatsapp:<chat_id>`` is fine. In groups, every
        sender gets ``whatsapp:<chat_id>:<sender_id>`` so that parallel
        conversations about unrelated topics do not contaminate each other.
        """
        base = f"{self.name}:{chat_id}"
        if is_group and sender_id:
            return f"{base}:{sender_id}"
        return base

    async def _send_media(self, client: Any, to: Any, media_path: str, *, ptt: bool = False) -> None:
        path = str(Path(media_path).expanduser())
        mime, _ = mimetypes.guess_type(path)
        mimetype = mime or "application/octet-stream"
        self.logger.info(
            "whatsapp _send_media to={} path={} mimetype={} ptt={}",
            _jid_to_string(to),
            path,
            mimetype,
            ptt,
        )
        if mimetype.startswith("image/"):
            await client.send_image(to, path)
        elif mimetype.startswith("video/"):
            await client.send_video(to, path)
        elif mimetype.startswith("audio/"):
            await client.send_audio(to, path, ptt=ptt)
        else:
            await client.send_document(
                to,
                path,
                filename=Path(path).name,
                mimetype=mimetype,
            )

    def _register_handlers(
        self,
        client: Any,
        *,
        login_result: asyncio.Future[None] | None = None,
        handle_messages: bool,
    ) -> None:
        api = _load_neonize()

        @client.qr
        async def _on_qr(_: Any, qr_data: bytes) -> None:
            import segno

            self.logger.info("Scan the WhatsApp QR code with Linked Devices")
            qr_text = io.StringIO()
            segno.make_qr(qr_data).terminal(compact=True, out=qr_text)
            # ponytail: render the QR inside a marked banner so it stays findable
            # even when interleaved with other loguru lines, and so it always
            # lands in the managed log file (not just stdout). When the gateway
            # runs under the background manager, stdout is the log file —
            # these banner lines survive grep/less jumps and don't get lost.
            banner = "WHATSAPP_QR_BEGIN"
            self.logger.info(banner)
            for line in qr_text.getvalue().splitlines():
                self.logger.info(line)
            self.logger.info("WHATSAPP_QR_END")

        @client.event(api.ConnectedEv)
        async def _on_connected(current_client: Any, _: Any) -> None:
            self._connected = True
            try:
                await self._remember_self_jids(current_client)
            except Exception as exc:
                if login_result is not None and not login_result.done():
                    login_result.set_exception(exc)
                raise
            if login_result is not None and not login_result.done():
                login_result.set_result(None)
            self.logger.info("WhatsApp connected")
            lifecycle_cb = getattr(self, "_on_connected_for_lifecycle", None)
            if lifecycle_cb is not None:
                lifecycle_cb()
            self._touch_activity()

        @client.event(api.DisconnectedEv)
        async def _on_disconnected(_: Any, event: Any) -> None:
            self._connected = False
            self._touch_activity()
            if login_result is not None and not login_result.done():
                login_result.set_exception(
                    RuntimeError(f"WhatsApp disconnected before login completed: {event}")
                )
            self.logger.warning("WhatsApp disconnected: {}", event)

        @client.event(api.PairStatusEv)
        async def _on_pair_status(_: Any, event: Any) -> None:
            error = str(_safe_attr(event, "Error", "") or "")
            if error:
                exc = RuntimeError(f"WhatsApp pair status error: {error}")
                if login_result is not None and not login_result.done():
                    login_result.set_exception(exc)
                raise exc
            self.logger.info("WhatsApp pair status: {}", event)

        @client.event(api.LoggedOutEv)
        async def _on_logged_out(_: Any, event: Any) -> None:
            reason = _safe_attr(event, "Reason", 0)
            on_connect = _safe_attr(event, "OnConnect", False)
            self.logger.warning(
                "WhatsApp logged out: reason={} on_connect={}", reason, on_connect
            )
            if login_result is not None and not login_result.done():
                login_result.set_exception(
                    RuntimeError(
                        "WhatsApp logged out (reason={} on_connect={}); "
                        "this usually means the session was opened on another device. "
                        "Run `nanobot channels login whatsapp --force` to re-authenticate.".format(
                            reason, on_connect
                        )
                    )
                )

        @client.event(api.ConnectFailureEv)
        async def _on_connect_failure(_: Any, event: Any) -> None:
            reason = _safe_attr(event, "Reason", 0)
            message = _safe_attr(event, "Message", "")
            self.logger.warning("WhatsApp connect failure: reason={} message={}", reason, message)
            if login_result is not None and not login_result.done():
                login_result.set_exception(
                    RuntimeError(f"WhatsApp connect failure: {message or reason}")
                )

        @client.event(api.StreamErrorEv)
        async def _on_stream_error(_: Any, event: Any) -> None:
            code = _safe_attr(event, "Code", "")
            self.logger.warning("WhatsApp stream error: code={}", code)
            if login_result is not None and not login_result.done():
                login_result.set_exception(
                    RuntimeError(f"WhatsApp stream error: {code}")
                )

        if not handle_messages:
            return

        @client.event(api.MessageEv)
        async def _on_message(current_client: Any, event: Any) -> None:
            self._current_client = current_client
            try:
                await self._handle_neonize_message(current_client, event)
            except Exception:
                self.logger.exception("Error handling WhatsApp message")
                raise
            self._touch_activity()

    async def _remember_self_jids(self, client: Any) -> None:
        device = _safe_attr(client, "me")
        if device is None:
            device = await client.get_me()

        for attr in ("JID", "LID", "PN"):
            jid = _normalize_jid(_safe_attr(device, attr))
            if jid:
                self._self_jids.add(jid)
                self._self_jids.add(_bare_jid(jid))

    async def _send_read_receipt(self, client: Any, source: Any, message_id: str) -> None:
        """Send a read receipt (blue double-check) for an incoming message.

        Best-effort: any failure is logged at debug level and swallowed so it
        never blocks message processing.
        """
        if not message_id:
            return
        try:
            from neonize.utils.enum import ReceiptType

            chat = _safe_attr(source, "Chat")
            sender = _safe_attr(source, "Sender")
            if chat is None or sender is None:
                return
            await client.mark_read(
                message_id,
                chat=chat,
                sender=sender,
                receipt=ReceiptType.READ,
            )
        except Exception as exc:  # noqa: BLE001 - read receipt is best-effort
            self.logger.debug("Failed to send WhatsApp read receipt: {}", exc)

    def _build_typing_jid(self, chat_jid: str) -> Any:
        """Return a JID suitable for typing presence on this chat.

        For LID-based DMs we must send presence to the LID JID (the same
        `chat_jid` we received). Sending typing to the resolved phone-number
        JID does not render on the sender's screen. Sending messages still uses
        `_resolve_send_target` for delivery.
        """
        return self._build_jid(chat_jid)

    async def _handle_neonize_message(self, client: Any, event: Any) -> None:
        info = _safe_attr(event, "Info")
        message = _safe_attr(event, "Message")
        source = _safe_attr(info, "MessageSource")
        if info is None or message is None or source is None:
            raise ValueError("WhatsApp MessageEv is missing Info, Message, or MessageSource")

        if bool(_safe_attr(source, "IsFromMe", False)):
            return

        chat_jid = _normalize_jid(_safe_attr(source, "Chat"))
        if not chat_jid:
            raise ValueError("WhatsApp message has no chat JID")
        if chat_jid == "status@broadcast":
            self.logger.debug("Ignoring WhatsApp status broadcast message")
            return

        timestamp = float(_safe_attr(info, "Timestamp", 0) or 0)
        if self._started_at and timestamp and timestamp < self._started_at:
            self.logger.debug(
                "Ignoring pre-start WhatsApp message id={} ts={} started_at={}",
                str(_safe_attr(info, "ID", "") or ""),
                int(timestamp),
                int(self._started_at),
            )
            return

        is_group = bool(_safe_attr(source, "IsGroup", False))
        message_id = str(_safe_attr(info, "ID", "") or "")
        if message_id:
            if message_id in self._processed_message_ids:
                self.logger.debug("Ignoring duplicate WhatsApp message id={}", message_id)
                return
            self._processed_message_ids.add(message_id)
            self._touch_message_state()

        # Mark the incoming message as read (blue double-check). Best-effort.
        await self._send_read_receipt(client, source, message_id)

        participant_jid = _normalize_jid(_safe_attr(source, "Sender"))
        sender_alt_jid = _normalize_jid(_safe_attr(source, "SenderAlt"))
        sender_candidates = [sender_alt_jid, participant_jid]
        if not is_group:
            sender_candidates.append(chat_jid)

        phone_id, lid_id = _classify_sender_ids(sender_candidates)
        if phone_id and lid_id:
            static_phone = self.config.lid_mappings.get(lid_id, "")
            if static_phone:
                if static_phone != phone_id:
                    self.logger.warning(
                        "Runtime LID {} -> {} differs from static lidMappings {}; keeping static",
                        lid_id, phone_id, static_phone,
                    )
            else:
                self._learned_lid_phone_pair(lid_id, phone_id)

        if not phone_id and lid_id:
            mapped = self.config.lid_mappings.get(lid_id, "")
            if mapped:
                phone_id = mapped

        sender_id = phone_id or self.config.lid_mappings.get(lid_id, "") or lid_id
        if not sender_id:
            raise ValueError("WhatsApp message has no resolvable sender ID")

        push_name = str(_safe_attr(info, "Pushname", "") or "").strip()
        self._remember_display_name(chat_jid, sender_id, push_name)
        display_name = push_name or self._display_name_for(chat_jid, sender_id)

        # Retry self-JID discovery if the first attempt (on connect) ran
        # before me/JID was populated; otherwise mention detection fails.
        if not self._self_jids:
            await self._refresh_self_jids()

        is_addressed = self._is_addressed_to_bot(message)
        if is_group and self.config.group_policy == "mention" and not is_addressed:
            # Still buffer the message for later context, but do not respond now.
            text_for_buffer = _message_text(message)
            self._add_group_context(
                chat_jid,
                sender_id,
                display_name,
                text_for_buffer,
                timestamp=timestamp,
            )
            return

        # Always buffer group messages for context when the bot is addressed too.
        text_for_buffer = _message_text(message)
        if is_group:
            self._add_group_context(
                chat_jid,
                sender_id,
                display_name,
                text_for_buffer,
                timestamp=timestamp,
            )

        mention = self._resolve_mention(sender_id, display_name) if is_group else None
        identity_block = self._sender_identity_block(
            sender_id, display_name, self._is_reply_to_bot(message), phone_id
        )
        contact_block = self._known_contacts_block(chat_jid, sender_id)
        runtime_blocks: list[RuntimeContextBlock] = [identity_block]
        if contact_block is not None:
            runtime_blocks.append(contact_block)
        if is_group and self.config.group_context_buffer_size > 0:
            recent_context = self._get_group_context(chat_jid)
            if recent_context:
                runtime_blocks.append(
                    RuntimeContextBlock(source="whatsapp_recent_context", content=recent_context)
                )
        metadata = {
            "message_id": message_id or None,
            "timestamp": int(timestamp) if timestamp else None,
            "is_group": is_group,
            "is_forwarded": self._is_forwarded(message),
            "participant": participant_jid or None,
            "sender_alt": sender_alt_jid or None,
            "lid": lid_id or None,
            "phone": phone_id or None,
            "is_reply_to_bot": self._is_reply_to_bot(message),
            "whatsapp_mention": mention,
            RUNTIME_CONTEXT_INPUT_META: runtime_blocks,
        }
        sender_allowed = self.is_allowed(sender_id)
        group_allow_id = self._group_allow_id(chat_jid) if is_group else None
        authorization_id = sender_id if sender_allowed else group_allow_id

        # The typing indicator must be sent to the *chat* JID we received
        # (this is what the sender's WhatsApp client watches). For delivery we
        # still use `_resolve_send_target` so LID-based DMs reach a phone JID.
        typing_jid = self._build_typing_jid(chat_jid)
        session_key = self._whatsapp_session_key(chat_jid, sender_id, is_group)

        # Screen DMs from unknown senders before handing them to the agent.
        if not is_group and self.config.unknown_dm_policy == "screen":
            if not self._is_known_sender(sender_id):
                text_for_buffer = _message_text(message)
                await self._screen_unknown_dm(
                    client,
                    sender_id,
                    chat_jid,
                    display_name,
                    text_for_buffer,
                    timestamp=timestamp,
                )
                return

        async def _process(mention_tag: str | None) -> None:
            if authorization_id is None:
                self.logger.info(
                    "Passing unauthorized WhatsApp sender {} to pairing flow "
                    "(phone={}, lid={}, chat={})",
                    sender_id,
                    phone_id or "",
                    lid_id or "",
                    chat_jid,
                )
                await self._start_typing(typing_jid)
                try:
                    await self._handle_message(
                        sender_id=sender_id,
                        chat_id=chat_jid,
                        content=_message_text(message),
                        media=[],
                        metadata={**metadata, "whatsapp_mention": mention_tag},
                        is_dm=not is_group,
                        session_key=session_key,
                    )
                finally:
                    await self._stop_typing(typing_jid)
                return

            text = _message_text(message)
            media_paths: list[str] = []
            media = _media_message(message)
            if media is not None:
                path = await self._download_media(client, event, media)
                if media.kind == "audio" and media.is_voice:
                    transcription = await self.transcribe_audio(path)
                    if transcription:
                        text = transcription
                    else:
                        media_paths.append(path)
                        text = self._append_media_tag(text, "audio", path)
                else:
                    media_paths.append(path)
                    text = self._append_media_tag(text, media.kind, path)

            if not text and not media_paths:
                self.logger.debug(
                    "WhatsApp message id={} from {} has no parseable text/media; dropping",
                    message_id, sender_id,
                )
                return

            await self._start_typing(typing_jid)
            try:
                await self._handle_message(
                    sender_id=sender_id,
                    chat_id=chat_jid,
                    content=text,
                    media=media_paths,
                    metadata={**metadata, "whatsapp_mention": mention_tag},
                    is_dm=not is_group,
                    authorization_id=authorization_id,
                    session_key=session_key,
                )
            finally:
                await self._stop_typing(typing_jid)

        if is_group:
            await self._enqueue_group_turn(chat_jid, _process, mention)
        else:
            await _process(mention)

    def _group_allow_id(self, chat_jid: str) -> str | None:
        if self.is_allowed(chat_jid):
            return chat_jid
        bare_chat_id = _bare_jid(chat_jid)
        if bare_chat_id and bare_chat_id != chat_jid and self.is_allowed(bare_chat_id):
            return bare_chat_id
        return None

    def _resolve_mention(self, sender_id: str, push_name: str) -> str | None:
        """Return a safe WhatsApp mention tag for a group reply.

        Prefer the sender's push name when it looks like a real display name.
        Fall back to an E.164-style phone number derived from a phone JID.
        Otherwise omit the mention entirely.
        """
        if push_name and not _NAME_LIKE_ID_RE.match(push_name) and self._looks_like_name(push_name):
            return f"@{push_name}"

        phone_candidate = _bare_jid(sender_id)
        if phone_candidate and phone_candidate.isdigit() and 7 <= len(phone_candidate) <= 15:
            return f"@+{phone_candidate}"
        return None

    @staticmethod
    def _looks_like_name(value: str) -> bool:
        """A display name should contain at least one letter and not be only digits/hex IDs."""
        value = value.strip()
        if not value or len(value) > 50:
            return False
        if value.isdigit():
            return False
        if not any(c.isalpha() for c in value):
            return False
        # Reject single repeated punctuation or strings like "..." or "++++".
        if not any(c.isalnum() for c in value):
            return False
        return True

    @staticmethod
    def _sender_identity_block(
        sender_id: str, display_name: str | None, is_reply_to_bot: bool, phone_id: str | None = None
    ) -> RuntimeContextBlock:
        """Return a runtime-context block telling the agent who sent this message."""
        lines = [
            f"phone: +{phone_id}" if phone_id else None,
            f"display_name: {display_name}" if display_name else None,
            f"sender_id: {sender_id}",
            f"reply_to_bot: {'yes' if is_reply_to_bot else 'no'}",
        ]
        identity = "\n".join(line for line in lines if line)
        return RuntimeContextBlock(
            source="whatsapp_sender_identity",
            content=(
                "[WhatsApp sender identity for this turn]\n"
                f"{identity}\n\n"
                "This is the human who sent the current message. "
                "Do not assume it is the operator unless these values match."
            ),
        )

    def _known_contacts_block(self, chat_id: str, current_sender_id: str) -> RuntimeContextBlock | None:
        """Return a block listing known contacts in this chat, minus the current sender."""
        chat_map = self._display_names.get(chat_id, {})
        entries = [
            f"- {name} (sender_id: {sid})"
            for sid, name in sorted(chat_map.items(), key=lambda kv: kv[1].lower())
            if sid != current_sender_id and name
        ]
        if not entries:
            return None
        return RuntimeContextBlock(
            source="whatsapp_known_contacts",
            content=(
                "[WhatsApp contacts seen in this chat]\n"
                + "\n".join(entries)
                + "\n\nUse these names when referring to other people in the group."
            ),
        )

    def _get_group_context(self, chat_jid: str) -> str:
        """Format the recent buffered group messages as prompt context."""
        buffer = self._group_context_buffers.get(chat_jid)
        if not buffer or len(buffer) == 0:
            return ""
        # The last entry is the current message; exclude it from the context.
        context_entries = list(buffer)[:-1]
        if not context_entries:
            return ""
        lines: list[str] = []
        for sender_id, display_name, content, _timestamp in context_entries:
            label = display_name or sender_id
            snippet = content[:200]
            lines.append(f"[{label}]: {snippet}")
        return "\n".join(
            [
                f"Recent messages in this WhatsApp group (last {len(context_entries)}):",
                *lines,
            ]
        )

    def _add_group_context(
        self,
        chat_jid: str,
        sender_id: str,
        display_name: str | None,
        content: str,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Add one message to the rolling per-group context buffer."""
        if self.config.group_context_buffer_size <= 0:
            return
        if chat_jid not in self._group_context_buffers:
            self._group_context_buffers[chat_jid] = deque(
                maxlen=self.config.group_context_buffer_size
            )
        self._group_context_buffers[chat_jid].append(
            (sender_id, display_name, content, timestamp or time.monotonic())
        )

    def _is_group_chat(self, chat_id: str) -> bool:
        return str(chat_id).endswith("@g.us")

    async def _enqueue_group_turn(
        self,
        chat_jid: str,
        handler: Callable[[str | None], Awaitable[None]],
        mention: str | None,
    ) -> None:
        """Queue a turn for a group chat so responses go out one by one."""
        queue = self._group_queues.setdefault(chat_jid, asyncio.Queue())
        await queue.put((handler, mention))
        if chat_jid not in self._group_queue_tasks or self._group_queue_tasks[chat_jid].done():
            self._group_queue_tasks[chat_jid] = asyncio.create_task(
                self._process_group_queue(chat_jid)
            )

    async def _process_group_queue(self, chat_jid: str) -> None:
        """Consume turns for one group sequentially."""
        queue = self._group_queues.get(chat_jid)
        if queue is None:
            return
        try:
            while True:
                try:
                    handler, mention = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    await handler(mention)
                except Exception:
                    self.logger.exception("Unhandled error processing WhatsApp group turn")
                # Small pause so multiple rapid messages feel like separate turns.
                await asyncio.sleep(0.6)
        finally:
            self._group_queue_tasks.pop(chat_jid, None)
            if queue.empty():
                self._group_queues.pop(chat_jid, None)

    async def _drain_group_queue(self, chat_jid: str) -> None:
        """Wait until the group queue has been fully processed. (Tests only.)"""
        while True:
            task = self._group_queue_tasks.get(chat_jid)
            if task is None and (chat_jid not in self._group_queues or self._group_queues[chat_jid].empty()):
                return
            if task is not None and not task.done():
                await task
            else:
                await asyncio.sleep(0.005)

    def _is_addressed_to_bot(self, message: Any) -> bool:
        return self._was_mentioned(message) or self._is_reply_to_bot(message)

    async def _refresh_self_jids(self) -> None:
        """Best-effort repopulate _self_jids from the live client.

        Retried lazily on mention detection in case the first population ran
        before me/JID was available (some accounts expose only LID/PN).
        """
        client = self._current_client
        if client is None:
            return
        try:
            await self._remember_self_jids(client)
        except Exception:
            self.logger.debug("Failed to refresh WhatsApp self JIDs", exc_info=True)

    def _was_mentioned(self, message: Any) -> bool:
        for context in _context_infos(message):
            mentioned = (
                _safe_attr(context, "mentionedJID")
                or _safe_attr(context, "mentionedJid")
                or _safe_attr(context, "mentioned_jid")
                or []
            )
            for jid in mentioned:
                normalized = _normalize_jid(jid)
                if normalized in self._self_jids or _bare_jid(normalized) in self._self_jids:
                    return True
        if self._mentioned_by_text(message):
            return True
        return False

    def _mentioned_by_text(self, message: Any) -> bool:
        """Detect a plain-text @mention (e.g. "@motoko") even when WhatsApp
        omits contextInfo.mentionedJID."""
        text = _message_text(message)
        if not text or "@" not in text:
            return False
        bot_names = {self._bot_name()}
        if self._self_jids:
            for jid in self._self_jids:
                bare = _bare_jid(jid)
                if bare and bare.isdigit():
                    bot_names.add(bare)
        lowered = text.lower()
        for name in bot_names:
            if name and ("@" + name.lower()) in lowered:
                return True
        return False

    @staticmethod
    def _bot_name() -> str:
        try:
            from nanobot.config.loader import load_config

            return load_config().agents.defaults.bot_name
        except Exception:
            return "nanobot"

    def _is_reply_to_bot(self, message: Any) -> bool:
        if not self._self_jids:
            return False
        for context in _context_infos(message):
            participant = _normalize_jid(
                _safe_attr(context, "participant")
                or _safe_attr(context, "Participant")
                or ""
            )
            if participant in self._self_jids or _bare_jid(participant) in self._self_jids:
                return True
        return False

    @staticmethod
    def _is_forwarded(message: Any) -> bool:
        for context in _context_infos(message):
            if bool(_safe_attr(context, "isForwarded", False)):
                return True
            if int(_safe_attr(context, "forwardingScore", 0) or 0) > 0:
                return True
        return False

    async def _download_media(self, client: Any, event: Any, media: _MediaInfo) -> str:
        info = _safe_attr(event, "Info")
        message_id = str(_safe_attr(info, "ID", "") or "")
        path = self._media_path(message_id, media)
        await client.download_any(_safe_attr(event, "Message"), str(path))
        return str(path)

    def _media_path(self, message_id: str, media: _MediaInfo) -> Path:
        media_dir = get_media_dir("whatsapp")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", message_id or str(int(time.time())))
        filename = Path(media.filename).name if media.filename else ""
        suffix = Path(filename).suffix if filename else ""
        if not suffix:
            suffix = mimetypes.guess_extension(media.mimetype) or {
                "image": ".jpg",
                "video": ".mp4",
                "audio": ".ogg",
                "sticker": ".webp",
            }.get(media.kind, ".bin")
        return media_dir / f"wa_{safe_id}_{secrets.token_hex(4)}{suffix}"

    @staticmethod
    def _append_media_tag(text: str, kind: str, path: str) -> str:
        label = kind if kind in {"image", "video", "audio", "sticker"} else "file"
        tag = f"[{label}: {path}]"
        return f"{text}\n{tag}" if text else tag

    @staticmethod
    def _reset_database(path: Path) -> None:
        for candidate in (
            path,
            path.with_suffix(path.suffix + "-shm"),
            path.with_suffix(path.suffix + "-wal"),
        ):
            if candidate.exists():
                candidate.unlink()
