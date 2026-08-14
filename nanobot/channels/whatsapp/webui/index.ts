import type { ChannelUiContribution } from "@/channel-plugins/types";
import { chatAppGuideUrl } from "@/components/settings/channels/catalog";

export default {
  presentation: {
    displayName: "WhatsApp",
    initials: "WA",
    color: "#25D366",
    logoUrl: "https://www.whatsapp.com/favicon.ico",
    setup: {
      mode: "connect",
      command: "nanobot channels login whatsapp",
      docsUrl: chatAppGuideUrl("whatsapp"),
      manualFields: [
        { key: "channels.whatsapp.allowFrom" },
        { key: "channels.whatsapp.groupPolicy" },
        { key: "channels.whatsapp.allowSendTo" },
        { key: "channels.whatsapp.loginTimeoutS" },
        { key: "channels.whatsapp.throttleThreshold" },
        { key: "channels.whatsapp.throttleCooldownS" },
        { key: "channels.whatsapp.lidMappings" },
        { key: "channels.whatsapp.groupWorkspaces" },
        { key: "channels.whatsapp.dmWorkspace" },
        { key: "channels.whatsapp.dmWorkspaces" },
      ],
    },
  },
} satisfies ChannelUiContribution;
