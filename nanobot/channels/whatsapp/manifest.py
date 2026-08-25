"""WhatsApp management contract."""

from nanobot.channels._manifest import DIRECT_GROUP_POLICIES, field
from nanobot.channels.contracts import ChannelManagementSpec, ChannelSetupSpec
from nanobot.channels.plugin import ChannelPlugin
from nanobot.channels.whatsapp.state import local_state_present
from nanobot.channels.whatsapp.validation import validate

SETUP_SPEC = ChannelSetupSpec(
    fields={
        "allowFrom": field("list"),
        "groupPolicy": field("enum", choices=DIRECT_GROUP_POLICIES, default="open"),
        "allowSendTo": field("list"),
        "databasePath": field(writable=False, snapshot=False),
        "loginTimeoutS": field("int", default=300),
        "throttleThreshold": field("int", default=3),
        "throttleCooldownS": field("int", default=7200),
        "lidMappings": field("kv"),
        "groupWorkspaces": field("kv"),
        "dmWorkspace": field("string", help="Absolute path used as workspace for all DMs when no per-sender entry matches."),
        "dmWorkspaces": field("kv", help="Map sender id (phone number, LID, or WhatsApp JID) to an absolute workspace path."),
        "dmWorkspaceModelPreset": field("string", default="", help="Model preset name applied to sessions whose first inbound falls in the default DM workspace."),
        "groupWorkspacePresets": field("kv", help="Map group JID to a model preset name; sessions starting in that group use that preset."),
        "dmWorkspacePresets": field("kv", help="Map sender id to a model preset name; per-sender override of dmWorkspaceModelPreset."),
    },
    official_url="https://faq.whatsapp.com/",
    validator=validate,
)

PLUGIN = ChannelPlugin(
    name="whatsapp",
    display_name="WhatsApp",
    runtime=f"{__package__}.runtime:WhatsAppChannel",
    setup=SETUP_SPEC,
    management=ChannelManagementSpec(local_state_present=local_state_present),
    dependencies=(
        "neonize>=0.4.0,<0.5.0",
        "segno>=1.6.1,<2.0.0",
        "static-ffmpeg>=3.0,<4.0",
    ),
    webui="webui/index.ts",
)
