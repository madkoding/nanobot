"""WhatsApp JID normalization helpers (extracted from whatsapp/runtime.py)."""

from __future__ import annotations

import re
from typing import Any

from nanobot.channels.whatsapp.message_helpers import _safe_attr

_JID_RE = re.compile(r"^(?P<user>[^@]+)@(?P<server>[^@]+)$")


def _jid_to_string(jid: Any) -> str:
    if jid is None:
        return ""
    if isinstance(jid, str):
        return jid.strip()
    if bool(_safe_attr(jid, "IsEmpty", False)):
        return ""

    user = str(_safe_attr(jid, "User", "") or "").strip()
    server = str(_safe_attr(jid, "Server", "") or "").strip()
    if user and server:
        return f"{user}@{server}"
    return server or user


def _typing_task_key(jid: Any) -> str:
    """Stable key for the per-chat typing task.

    Falls back to ``(user, server)`` tuples and finally to ``id(jid)`` so the
    helper works for both real neonize JID protos and test doubles.
    """
    as_string = _jid_to_string(jid)
    if as_string:
        return as_string
    if isinstance(jid, tuple) and len(jid) >= 2:
        return f"{jid[0]}@{jid[1]}"
    return f"jid:{id(jid)}"


def _normalize_jid(raw: Any) -> str:
    jid = _jid_to_string(raw).strip()
    if not jid:
        return ""
    if jid.endswith("@lid.whatsapp.net"):
        return jid[: -len(".whatsapp.net")]
    return jid


def _bare_jid(raw: Any) -> str:
    jid = _normalize_jid(raw)
    if "@" not in jid:
        return jid
    return jid.split("@", 1)[0].split(":", 1)[0]


def _classify_sender_ids(jids: list[Any]) -> tuple[str, str]:
    phone_id = ""
    lid_id = ""

    for raw in jids:
        jid = _normalize_jid(raw)
        if not jid:
            continue
        match = _JID_RE.match(jid)
        if match:
            user = match.group("user").split(":", 1)[0]
            server = match.group("server")
            if server in {"s.whatsapp.net", "c.us"}:
                phone_id = phone_id or user
            elif server in {"lid", "lid.whatsapp.net"}:
                lid_id = lid_id or user
            continue

        if not phone_id:
            phone_id = jid

    return phone_id, lid_id
