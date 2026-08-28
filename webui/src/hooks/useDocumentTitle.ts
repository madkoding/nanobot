import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { displayTitle } from "@/lib/chat-groups";
import type { ChatSummary, SidebarStatePayload } from "@/lib/types";

type Args = {
  view: "chat" | "settings" | "apps" | "automations" | "skills" | "projects" | "workspace" | "todos" | "agenda" | "research";
  activeSession: ChatSummary | null;
  sidebarState: SidebarStatePayload;
};

export function useDocumentTitle({ view, activeSession, sidebarState }: Args) {
  const { t, i18n } = useTranslation();
  useEffect(() => {
    if (view === "settings") {
      document.title = t("app.documentTitle.chat", {
        title: t("settings.sidebar.title"),
      });
      return;
    }
    if (view === "apps") {
      document.title = t("app.documentTitle.chat", {
        title: t("settings.nav.apps", { defaultValue: "Apps" }),
      });
      return;
    }
    if (view === "automations") {
      document.title = t("app.documentTitle.chat", {
        title: t("settings.nav.automations", { defaultValue: "Automations" }),
      });
      return;
    }
    if (view === "skills") {
      document.title = t("app.documentTitle.chat", {
        title: t("settings.nav.skills", { defaultValue: "Skills" }),
      });
      return;
    }
    if (view === "projects") {
      document.title = t("app.documentTitle.chat", {
        title: t("sidebar.projects", { defaultValue: "Projects" }),
      });
      return;
    }
    if (view === "workspace") {
      document.title = t("app.documentTitle.chat", {
        title: t("sidebar.workspace", { defaultValue: "Workspace" }),
      });
      return;
    }
    if (view === "todos") {
      document.title = t("app.documentTitle.chat", {
        title: t("sidebar.todos", { defaultValue: "Todos" }),
      });
      return;
    }
    if (view === "agenda") {
      document.title = t("app.documentTitle.chat", {
        title: t("sidebar.agenda", { defaultValue: "Agenda" }),
      });
      return;
    }
    if (view === "research") {
      document.title = t("app.documentTitle.chat", {
        title: t("sidebar.research", { defaultValue: "Research" }),
      });
      return;
    }
    document.title = activeSession
      ? t("app.documentTitle.chat", {
          title: displayTitle(activeSession, sidebarState.title_overrides, ""),
        })
      : t("app.documentTitle.base");
  }, [activeSession, i18n.resolvedLanguage, sidebarState.title_overrides, t, view]);
}
