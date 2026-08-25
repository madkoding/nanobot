"""Tests that ``tools.restrictToWorkspace`` is the single source of truth
for default workspace access mode.

The WebUI historically kept its own ``workspace-state.json::default_access_mode``
cache that drifted from the canonical ``config.tools.restrictToWorkspace``.
The new contract: config wins, cache is derived and auto-synced on read.
"""
from __future__ import annotations

import json

from nanobot.webui.workspaces import (
    _to_access_mode,
    default_scope_for_webui,
    read_webui_default_access_mode,
    write_webui_default_access_mode,
)


def test_to_access_mode_mapping() -> None:
    assert _to_access_mode(True) == "default"
    assert _to_access_mode(False) == "full"


def test_config_true_overrides_cache_full(tmp_path, monkeypatch) -> None:
    """If config says ``restrict_to_workspace=True`` (use restrictive
    default scope) and the cache says ``full``, the resolver returns
    the config-side value and syncs the cache.
    """
    monkeypatch.setattr(
        "nanobot.webui.workspaces.webui_workspace_state_path",
        lambda: tmp_path / "workspace-state.json",
    )
    write_webui_default_access_mode("full")
    scope = default_scope_for_webui(tmp_path, default_restrict_to_workspace=True)
    assert scope.restrict_to_workspace is True
    # Cache was synced.
    assert read_webui_default_access_mode() == "default"


def test_config_false_overrides_cache_default(tmp_path, monkeypatch) -> None:
    """Inverse: config says full, cache says default → resolver uses full."""
    monkeypatch.setattr(
        "nanobot.webui.workspaces.webui_workspace_state_path",
        lambda: tmp_path / "workspace-state.json",
    )
    write_webui_default_access_mode("default")
    scope = default_scope_for_webui(tmp_path, default_restrict_to_workspace=False)
    assert scope.restrict_to_workspace is False
    assert read_webui_default_access_mode() == "full"


def test_aligned_cache_and_config_no_op(tmp_path, monkeypatch) -> None:
    """Cache and config already agree → no rewrite, scope matches."""
    target = tmp_path / "workspace-state.json"
    monkeypatch.setattr(
        "nanobot.webui.workspaces.webui_workspace_state_path",
        lambda: target,
    )
    # Seed the cache explicitly so the file exists and has an aligned value.
    target.write_text(
        json.dumps({"schema_version": 1, "default_access_mode": "default", "updated_at": None}),
        encoding="utf-8",
    )
    initial_content = json.loads(target.read_text(encoding="utf-8"))

    scope = default_scope_for_webui(tmp_path, default_restrict_to_workspace=True)
    assert scope.restrict_to_workspace is True

    # updated_at must not have been refreshed — proves no rewrite happened.
    after_content = json.loads(target.read_text(encoding="utf-8"))
    assert after_content["updated_at"] == initial_content["updated_at"]
