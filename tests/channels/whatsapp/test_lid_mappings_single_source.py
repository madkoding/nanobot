"""Tests for the single-source-of-truth LID->phone resolution.

Before this refactor, the WhatsApp channel kept a runtime-only
``_lid_to_phone`` dict (persisted to ``message_state.json``) that
silently overwrote the static ``config.lidMappings`` on every observed
inbound. That violated the single-source-of-truth principle and
required manual reconciliation when a user declared a mapping in
config.

The new contract:

* ``config.channels.whatsapp.lidMappings`` is the **only** place LID->phone
  pairs live on disk.
* The runtime learns new pairs in the background and persists them back
  to the config.
* Static entries (declared by the user) win over runtime observations;
  conflicting runtime pairs are dropped with a warning.
* ``message_state.json`` no longer carries LID->phone. Legacy entries
  are migrated to ``lidMappings`` on first ``start()`` and dropped.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.channels.whatsapp import runtime as whatsapp_module
from nanobot.channels.whatsapp.runtime import WhatsAppChannel, WhatsAppConfig


class _Proto:
    """Minimal attribute-bag double mirroring the neonize protos."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __eq__(self, other):
        return isinstance(other, _Proto) and self.__dict__ == other.__dict__


def _jid(user: str, server: str) -> _Proto:
    return _Proto(User=user, Server=server, IsEmpty=False)


def _event(*, message, sender=None, sender_alt=None, chat=None, is_group=False):
    source = _Proto(
        Chat=chat or _jid("120363422292889459", "g.us"),
        Sender=sender,
        SenderAlt=sender_alt,
        IsGroup=is_group,
        IsFromMe=False,
    )
    return _Proto(
        Info=_Proto(ID="m1", Timestamp=1, MessageSource=source),
        Message=message,
    )


def _make_channel(config=None) -> WhatsAppChannel:
    merged = {"enabled": True, "allowFrom": ["*"]}
    if config:
        merged.update(config)
    return WhatsAppChannel(merged, MagicMock())


@pytest.mark.asyncio
async def test_runtime_learned_lid_persists_to_config() -> None:
    """A (phone, lid) pair observed in an inbound must land in
    ``config.lidMappings`` (single source of truth), not in a separate
    runtime cache. The async persist helper does both: mutate in-memory
    and flush to disk via the manager.
    """
    mgr = MagicMock()
    mgr.persist_config_change = AsyncMock(return_value=True)
    ch = WhatsAppChannel({"enabled": True, "allowFrom": ["*"]}, MagicMock(), manager=mgr)

    persisted = await ch._persist_lid_mapping("230343776985329", "56975746099")
    assert persisted is True
    assert ch.config.lid_mappings["230343776985329"] == "56975746099"
    mgr.persist_config_change.assert_awaited_once()

    # Idempotent: re-persisting the same pair is a no-op (no flush).
    mgr.persist_config_change.reset_mock()
    again = await ch._persist_lid_mapping("230343776985329", "56975746099")
    assert again is False
    mgr.persist_config_change.assert_not_awaited()


@pytest.mark.asyncio
async def test_lid_only_inbound_resolves_via_config() -> None:
    """Group messages where SenderAlt is absent must still resolve the
    sender_id via ``config.lidMappings`` — the runtime cache no longer
    holds authoritative LID pairs.
    """
    ch = WhatsAppChannel(
        {
            "enabled": True,
            "allowFrom": ["*"],
            "lidMappings": {"230343776985329": "56975746099"},
        },
        MagicMock(),
    )
    ch._started_at = 0
    ch._handle_message = AsyncMock()

    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=_Proto(conversation="@bot hola"),
            chat=_jid("120363422292889459", "g.us"),
            sender=_jid("230343776985329", "lid"),
            is_group=True,
        ),
    )
    await ch._drain_group_queue("120363422292889459@g.us")

    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["sender_id"] == "56975746099"


@pytest.mark.asyncio
async def test_static_lid_wins_over_runtime_observation() -> None:
    """If the user declared a LID mapping in config and the runtime
    observes a different phone for that LID, the static mapping wins
    and no persist happens. The user-declared entry is authoritative.
    """
    mgr = MagicMock()
    mgr.persist_config_change = AsyncMock(return_value=True)
    ch = WhatsAppChannel(
        {
            "enabled": True,
            "allowFrom": ["*"],
            "lidMappings": {"230343776985329": "11111111111"},
        },
        MagicMock(),
        manager=mgr,
    )
    ch._started_at = 0
    ch._handle_message = AsyncMock()

    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=_Proto(conversation="hola"),
            chat=_jid("11111111111", "s.whatsapp.net"),
            sender=_jid("11111111111", "s.whatsapp.net"),
            sender_alt=_jid("230343776985329", "lid"),
            is_group=False,
        ),
    )

    # Static mapping kept; runtime pair rejected.
    assert ch.config.lid_mappings["230343776985329"] == "11111111111"
    # No flush — nothing changed.
    assert mgr.persist_config_change.await_count == 0


@pytest.mark.asyncio
async def test_legacy_message_state_lid_migrates_to_config(tmp_path, monkeypatch) -> None:
    """A pre-refactor ``message_state.json::lid_to_phone`` must be
    migrated into ``config.lidMappings`` on the first ``start()`` and
    dropped from the on-disk state file. Idempotent.
    """
    state_path = tmp_path / "message_state.json"
    monkeypatch.setattr(
        whatsapp_module.WhatsAppChannel,
        "_message_state_path",
        lambda self, _f=state_path: _f,
    )
    state_path.write_text(
        json.dumps(
            {
                "processed_ids": ["m1"],
                "lid_to_phone": {"230343776985329": "56975746099"},
            }
        ),
        encoding="utf-8",
    )

    mgr = MagicMock()
    mgr.persist_config_change = AsyncMock(return_value=True)

    ch = WhatsAppChannel({"enabled": True, "allowFrom": ["*"]}, MagicMock(), manager=mgr)
    ch._started_at = 0
    ch._running = False  # don't enter the neonize loop

    # Simulate the relevant portion of start() that performs the migration.
    ch._display_names = ch._load_display_names()
    migrated = ch._migrate_legacy_lid_cache()
    if migrated:
        await mgr.persist_config_change()

    assert ch.config.lid_mappings == {"230343776985329": "56975746099"}
    mgr.persist_config_change.assert_awaited_once()

    # Second call is a no-op (already migrated).
    assert ch._migrate_legacy_lid_cache() is False


@pytest.mark.asyncio
async def test_lid_mapping_survives_channel_rebuild() -> None:
    """Rebuilding the channel from config must read LID mappings back
    from ``config.lidMappings`` — proving config is the durable store.
    """
    cfg = WhatsAppConfig.model_validate(
        {"enabled": True, "allowFrom": ["*"], "lidMappings": {"111": "222"}}
    )
    ch1 = WhatsAppChannel(cfg, MagicMock())
    assert ch1._lid_to_phone == {"111": "222"}

    # Build a second channel from the same config snapshot; mapping is there.
    ch2 = WhatsAppChannel(
        WhatsAppConfig.model_validate(cfg.model_dump(by_alias=True)),
        MagicMock(),
    )
    assert ch2._lid_to_phone == {"111": "222"}
