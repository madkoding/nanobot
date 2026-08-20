from types import SimpleNamespace

import pytest

from nanobot.security.workspace_access import WorkspaceScopeResolver


@pytest.fixture
def monkeypatched_config(monkeypatch):
    def _patch(owner_id, owner_name="operator"):
        def _fake_load_config():
            return SimpleNamespace(
                owner_id=owner_id,
                owner_name=owner_name,
                owner_display_name=owner_name,
                owner_identifiers=lambda: {
                    "websocket": {"owner-1"},
                    "discord": {"owner-1"},
                },
                is_owner=lambda ch, sid: sid == "owner-1",
            )

        monkeypatch.setattr(
            "nanobot.config.loader.load_config", _fake_load_config
        )

    return _patch


def test_owner_webui_turn_gets_full_access(monkeypatched_config, tmp_path: str) -> None:
    monkeypatched_config(["websocket:owner-1"])
    resolver = WorkspaceScopeResolver(
        default_workspace=tmp_path,
        default_restrict_to_workspace=True,
    )
    scope = resolver.for_turn(
        channel="websocket",
        message_metadata={"workspace_scope": {"project_path": str(tmp_path), "access_mode": "restricted"}},
        session_metadata={},
        sender_id="owner-1",
    )
    assert scope.access_mode == "full"
    assert scope.restrict_to_workspace is False


def test_non_owner_webui_turn_respects_restricted_scope(monkeypatched_config, tmp_path: str) -> None:
    monkeypatched_config(["websocket:owner-1"])
    resolver = WorkspaceScopeResolver(
        default_workspace=tmp_path,
        default_restrict_to_workspace=True,
    )
    scope = resolver.for_turn(
        channel="websocket",
        message_metadata={"workspace_scope": {"project_path": str(tmp_path), "access_mode": "restricted"}},
        session_metadata={},
        sender_id="stranger",
    )
    assert scope.access_mode == "restricted"
    assert scope.restrict_to_workspace is True


def test_owner_non_webui_channel_uses_default_scope(monkeypatched_config, tmp_path: str) -> None:
    monkeypatched_config(["discord:owner-1"])
    resolver = WorkspaceScopeResolver(
        default_workspace=tmp_path,
        default_restrict_to_workspace=True,
    )
    scope = resolver.for_turn(
        channel="discord",
        message_metadata={},
        session_metadata={},
        sender_id="owner-1",
    )
    # Non-webui channels fall back to the loop default (still restricted here).
    assert scope.access_mode == "restricted"
    assert scope.restrict_to_workspace is True
