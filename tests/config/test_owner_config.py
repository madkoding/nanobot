from nanobot.config.schema import Config


class TestOwnerConfig:
    def test_owner_identifiers_group_by_channel(self) -> None:
        cfg = Config(
            ownerId=[
                "discord:940323605223444601",
                "whatsapp:+56975746099",
                "webui:*",
            ]
        )
        ids = cfg.owner_identifiers()
        assert ids["discord"] == {"940323605223444601"}
        assert ids["whatsapp"] == {"56975746099"}
        assert ids["webui"] == {"*"}

    def test_bare_owner_id_register_on_all_channels(self) -> None:
        cfg = Config(ownerId=["shared-id"])
        ids = cfg.owner_identifiers()
        assert all("shared-id" in v for v in ids.values())

    def test_prefixed_id_does_not_duplicate_as_bare(self) -> None:
        cfg = Config(ownerId=["discord:shared-id"])
        ids = cfg.owner_identifiers()
        assert ids["discord"] == {"shared-id"}
        assert ids.get("whatsapp", set()).isdisjoint({"shared-id"})

    def test_is_owner_with_wildcard(self) -> None:
        cfg = Config(ownerId=["webui:*"])
        assert cfg.is_owner("webui", "anyone") is True
        assert cfg.is_owner("discord", "anyone") is False

    def test_is_owner_normalizes_phone(self) -> None:
        cfg = Config(ownerId=["whatsapp:56975746099"])
        assert cfg.is_owner("whatsapp", "+56975746099") is True
        assert cfg.is_owner("whatsapp", "56975746099") is True

    def test_owner_display_name_defaults_to_operator(self) -> None:
        cfg = Config()
        assert cfg.owner_display_name == "operator"
        cfg2 = Config(ownerName="madKoding")
        assert cfg2.owner_display_name == "madKoding"

    def test_owner_id_deduplication(self) -> None:
        cfg = Config(ownerId=["discord:a", "discord:a", "DISCORD:A"])
        assert cfg.owner_id == ["discord:a"]
