"""Quick integration test for WhatsApp WebUI advanced fields save + snapshot."""
from nanobot.channels.whatsapp.manifest import PLUGIN as WHATSAPP_PLUGIN
from nanobot.channels.whatsapp.runtime import WhatsAppConfig
from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import Config
from nanobot.optional_features import _channel_config_snapshot
from nanobot.webui.settings_routes import WebUISettingsRouter


def test_whatsapp_webui_advanced_roundtrip(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)

    def discover(enabled_names=None):
        plugins = {"whatsapp": WHATSAPP_PLUGIN}
        if enabled_names is None:
            return plugins
        return {name: p for name, p in plugins.items() if name in enabled_names}

    monkeypatch.setattr("nanobot.channels.registry.discover_plugins", discover)

    router = object.__new__(WebUISettingsRouter)
    saved = router._save_channel_config_values(
        "whatsapp",
        {
            "channels.whatsapp.allowFrom": "56912345678, 56987654321",
            "channels.whatsapp.groupPolicy": "mention",
            "channels.whatsapp.allowSendTo": "56912345678",
            "channels.whatsapp.loginTimeoutS": "300",
            "channels.whatsapp.throttleThreshold": "3",
            "channels.whatsapp.throttleCooldownS": "7200",
            "channels.whatsapp.lidMappings": "lid123=56912345678\nlid456=56987654321",
            "channels.whatsapp.groupWorkspaces": "120363000@g.us=/tmp/group-ws",
            "channels.whatsapp.dmWorkspace": "/tmp/dm-ws",
            "channels.whatsapp.dmWorkspaces": "56912345678=/tmp/sender-ws",
        },
    )

    assert set(saved) == {
        "channels.whatsapp.allowFrom",
        "channels.whatsapp.groupPolicy",
        "channels.whatsapp.allowSendTo",
        "channels.whatsapp.loginTimeoutS",
        "channels.whatsapp.throttleThreshold",
        "channels.whatsapp.throttleCooldownS",
        "channels.whatsapp.lidMappings",
        "channels.whatsapp.groupWorkspaces",
        "channels.whatsapp.dmWorkspace",
        "channels.whatsapp.dmWorkspaces",
    }

    config = load_config(config_path)
    raw = config.channels.whatsapp
    whatsapp = WhatsAppConfig.model_validate(raw)
    assert whatsapp.allow_from == ["56912345678", "56987654321"]
    assert whatsapp.group_policy == "mention"
    assert whatsapp.allow_send_to == ["56912345678"]
    assert whatsapp.login_timeout_s == 300
    assert whatsapp.throttle_threshold == 3
    assert whatsapp.throttle_cooldown_s == 7200
    assert whatsapp.lid_mappings == {"lid123": "56912345678", "lid456": "56987654321"}
    assert whatsapp.group_workspaces == {"120363000@g.us": "/tmp/group-ws"}
    assert whatsapp.dm_workspace == "/tmp/dm-ws"
    assert whatsapp.dm_workspaces == {"56912345678": "/tmp/sender-ws"}

    values, configured = _channel_config_snapshot(
        whatsapp.model_dump(mode="json", by_alias=True),
        "whatsapp",
        WHATSAPP_PLUGIN.setup,
    )
    assert values["channels.whatsapp.lidMappings"] == "lid123=56912345678\nlid456=56987654321"
    assert values["channels.whatsapp.groupWorkspaces"] == "120363000@g.us=/tmp/group-ws"
    assert values["channels.whatsapp.dmWorkspace"] == "/tmp/dm-ws"
    assert values["channels.whatsapp.dmWorkspaces"] == "56912345678=/tmp/sender-ws"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
