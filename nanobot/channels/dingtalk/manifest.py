"""DingTalk management contract."""

from nanobot.channels._manifest import field, required_fields
from nanobot.channels.contracts import ChannelSetupSpec
from nanobot.channels.plugin import ChannelPlugin

SETUP_SPEC = ChannelSetupSpec(
    fields={
        "clientId": field(),
        "clientSecret": field("secret"),
        "allowFrom": field("list"),
    },
    required=required_fields("clientId", "clientSecret"),
    official_url="https://open.dingtalk.com/",
)

PLUGIN = ChannelPlugin(
    name="dingtalk",
    display_name="DingTalk",
    runtime=f"{__package__}.runtime:DingTalkChannel",
    setup=SETUP_SPEC,
    dependencies=(
        "dingtalk-stream>=0.24.0,<1.0.0",
        # dingtalk-stream only requires websockets>=11.0.2, so pip may resolve
        # an old 15.x that conflicts with nanobot's >=16.0,<17.0.
        "websockets>=16.0,<17.0",
    ),
    webui="webui/index.ts",
)
