"""Tests for the WhatsApp group-workspace registry."""

from __future__ import annotations

from pathlib import Path

from nanobot.channels.whatsapp.group_workspace import (
    ChatWorkspaceRegistry,
    GroupWorkspaceRegistry,
    is_group_jid,
)


class TestIsGroupJid:
    def test_group_jid(self):
        assert is_group_jid("120363000@g.us")

    def test_bare_group_id_is_not_group_jid(self):
        assert not is_group_jid("120363000")

    def test_dm_jid(self):
        assert not is_group_jid("5491155555555@s.whatsapp.net")

    def test_empty_string(self):
        assert not is_group_jid("")


class TestRegistryResolve:
    def test_resolves_configured_group(self, tmp_path):
        ws = tmp_path / "grupo_dev"
        ws.mkdir()
        registry = GroupWorkspaceRegistry({"120363000@g.us": str(ws)})
        assert registry.resolve("120363000@g.us") == ws.resolve()

    def test_returns_none_for_unmapped_group(self, tmp_path):
        ws = tmp_path / "grupo_dev"
        ws.mkdir()
        registry = GroupWorkspaceRegistry({"120363000@g.us": str(ws)})
        assert registry.resolve("999999@g.us") is None

    def test_returns_none_for_dm(self, tmp_path):
        ws = tmp_path / "grupo_dev"
        ws.mkdir()
        registry = GroupWorkspaceRegistry({"120363000@g.us": str(ws)})
        assert registry.resolve("5491155555555@s.whatsapp.net") is None

    def test_returns_none_for_bare_group_id(self, tmp_path):
        ws = tmp_path / "grupo_dev"
        ws.mkdir()
        registry = GroupWorkspaceRegistry({"120363000@g.us": str(ws)})
        # Bare numeric ID is not a JID — must not match the @g.us key.
        assert registry.resolve("120363000") is None

    def test_ignores_non_group_keys(self, tmp_path):
        ws = tmp_path / "x"
        ws.mkdir()
        registry = GroupWorkspaceRegistry(
            {"5491155555555@s.whatsapp.net": str(ws)}
        )
        assert registry.known_jids() == ()

    def test_ignores_non_absolute_paths(self, tmp_path):
        registry = GroupWorkspaceRegistry({"120363000@g.us": "relative/path"})
        assert registry.resolve("120363000@g.us") is None

    def test_ignores_missing_directories(self, tmp_path):
        registry = GroupWorkspaceRegistry(
            {"120363000@g.us": str(tmp_path / "nope")}
        )
        assert registry.resolve("120363000@g.us") is None

    def test_empty_mapping_is_safe(self):
        registry = GroupWorkspaceRegistry({})
        assert registry.resolve("120363000@g.us") is None
        assert registry.load_ruleset("120363000@g.us") is None

    def test_none_mapping_is_safe(self):
        registry = GroupWorkspaceRegistry(None)
        assert registry.resolve("120363000@g.us") is None


class TestRegistryLoadRuleset:
    def _setup_group(self, root: Path, *, agents: str = "", soul: str = "") -> None:
        root.mkdir(parents=True, exist_ok=True)
        if agents:
            (root / "AGENTS.md").write_text(agents, encoding="utf-8")
        if soul:
            (root / "SOUL.md").write_text(soul, encoding="utf-8")

    def test_loads_agents_md(self, tmp_path):
        ws = tmp_path / "g"
        self._setup_group(ws, agents="Only programming questions")
        registry = GroupWorkspaceRegistry({"120363000@g.us": str(ws)})
        ruleset = registry.load_ruleset("120363000@g.us")
        assert ruleset is not None
        assert "Only programming questions" in ruleset
        assert "AGENTS.md:" in ruleset

    def test_loads_soul_md(self, tmp_path):
        ws = tmp_path / "g"
        self._setup_group(ws, soul="Professional tone")
        registry = GroupWorkspaceRegistry({"120363000@g.us": str(ws)})
        ruleset = registry.load_ruleset("120363000@g.us")
        assert ruleset is not None
        assert "Professional tone" in ruleset

    def test_returns_none_when_no_files(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        registry = GroupWorkspaceRegistry({"120363000@g.us": str(ws)})
        assert registry.load_ruleset("120363000@g.us") is None

    def test_returns_none_for_unmapped_chat(self, tmp_path):
        ws = tmp_path / "g"
        self._setup_group(ws, agents="rules")
        registry = GroupWorkspaceRegistry({"120363000@g.us": str(ws)})
        assert registry.load_ruleset("999999@g.us") is None

    def test_truncates_oversized_ruleset(self, tmp_path):
        ws = tmp_path / "huge"
        self._setup_group(ws, agents="x" * 20_000)
        registry = GroupWorkspaceRegistry({"120363000@g.us": str(ws)})
        ruleset = registry.load_ruleset("120363000@g.us")
        assert ruleset is not None
        # truncate_text may append a small suffix marker, so allow a small margin
        # above the nominal cap. Anything materially above 8k means truncation
        # didn't happen.
        assert len(ruleset) <= 8_100
        assert len(ruleset) < 20_000


class TestDmWorkspaceRegistry:
    def _setup_dm(self, root: Path, *, agents: str = "", soul: str = "") -> None:
        root.mkdir(parents=True, exist_ok=True)
        if agents:
            (root / "AGENTS.md").write_text(agents, encoding="utf-8")
        if soul:
            (root / "SOUL.md").write_text(soul, encoding="utf-8")

    def test_default_dm_workspace(self, tmp_path):
        ws = tmp_path / "dms"
        self._setup_dm(ws, agents="DM rules")
        registry = ChatWorkspaceRegistry(dm_workspace=str(ws))
        assert registry.resolve("5491155555555@s.whatsapp.net") == ws.resolve()

    def test_per_sender_dm_workspace(self, tmp_path):
        ws = tmp_path / "dms"
        self._setup_dm(ws, agents="Sender rules")
        registry = ChatWorkspaceRegistry(dm_workspaces={"56912345678": str(ws)})
        assert registry.resolve("5491155555555@s.whatsapp.net", sender_id="56912345678") == ws.resolve()

    def test_sender_override_beats_default(self, tmp_path):
        default = tmp_path / "default"
        specific = tmp_path / "specific"
        self._setup_dm(default, agents="default")
        self._setup_dm(specific, agents="specific")
        registry = ChatWorkspaceRegistry(
            dm_workspace=str(default),
            dm_workspaces={"56912345678": str(specific)},
        )
        assert registry.resolve("5491155555555@s.whatsapp.net", sender_id="56912345678") == specific.resolve()

    def test_dm_ignores_non_dm_chat(self, tmp_path):
        ws = tmp_path / "dms"
        self._setup_dm(ws, agents="DM rules")
        registry = ChatWorkspaceRegistry(dm_workspace=str(ws))
        assert registry.resolve("120363000@g.us") is None


class TestResolveModelPreset:
    def _setup(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)

    def test_default_dm_preset(self, tmp_path):
        dm = tmp_path / "dms"
        self._setup(dm)
        registry = ChatWorkspaceRegistry(
            dm_workspace=str(dm),
            dm_workspace_model_preset="nemotron3",
        )
        assert registry.resolve_model_preset("56912345678@s.whatsapp.net") == "nemotron3"

    def test_group_preset_per_jid(self, tmp_path):
        g1 = tmp_path / "g1"
        self._setup(g1)
        g2 = tmp_path / "g2"
        self._setup(g2)
        registry = ChatWorkspaceRegistry(
            group_workspaces={"120363000@g.us": str(g1), "120363111@g.us": str(g2)},
            group_workspace_presets={
                "120363000@g.us": "nemotron3",
                "120363111@g.us": "ejecutor-deepseek",
            },
        )
        assert registry.resolve_model_preset("120363000@g.us") == "nemotron3"
        assert registry.resolve_model_preset("120363111@g.us") == "ejecutor-deepseek"

    def test_unmapped_chat_returns_none(self, tmp_path):
        dm = tmp_path / "dms"
        self._setup(dm)
        g = tmp_path / "g"
        self._setup(g)
        registry = ChatWorkspaceRegistry(
            dm_workspace=str(dm),
            dm_workspace_model_preset="nemotron3",
            group_workspaces={"120363000@g.us": str(g)},
            group_workspace_presets={"120363000@g.us": "nemotron3"},
        )
        assert registry.resolve_model_preset("999999@g.us") is None

    def test_empty_preset_map_returns_none(self, tmp_path):
        dm = tmp_path / "dms"
        self._setup(dm)
        registry = ChatWorkspaceRegistry(dm_workspace=str(dm))
        assert registry.resolve_model_preset("56912345678@s.whatsapp.net") is None

    def test_star_key_rebinds_default_dm_preset(self, tmp_path):
        dm = tmp_path / "dms"
        self._setup(dm)
        registry = ChatWorkspaceRegistry(
            dm_workspace=str(dm),
            dm_workspace_model_preset="nemotron3",
            dm_workspace_presets={"*": "ejecutor-deepseek"},
        )
        assert registry.resolve_model_preset("56912345678@s.whatsapp.net") == "ejecutor-deepseek"

    def test_skips_preset_for_unconfigured_key(self, tmp_path):
        dm = tmp_path / "dms"
        self._setup(dm)
        registry = ChatWorkspaceRegistry(
            dm_workspace=str(dm),
            dm_workspace_presets={"56912345678": "ejecutor-deepseek"},
        )
        # No dm_workspaces entry for that sender, so the override is dropped.
        assert registry.resolve_model_preset("56912345678@s.whatsapp.net", sender_id="56912345678") is None
