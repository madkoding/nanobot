import { useMemo } from "react";
import type { ChatSummary, ProjectSummary, SidebarStatePayload, WorkspacesPayload } from "@/lib/types";
import type { ActionsApi } from "@/hooks/useChatActions";
import { useProjectNames } from "@/hooks/useProjectNames";

type Args = {
  sessions: ChatSummary[];
  activeKey: string | null;
  loading: boolean;
  view: "chat" | "settings" | "apps" | "automations" | "skills" | "projects" | "workspace" | "todos" | "agenda" | "research";
  sidebarState: SidebarStatePayload;
  workspaces: WorkspacesPayload | null;
  runningChatIdList: string[];
  updatedChatIdList: string[];
  chatActions: ActionsApi["chat"];
  utility: ActionsApi["utility"];
  onOpenUtility: ActionsApi["utility"]["onOpen"];
  onOpenTodos: () => void;
  onOpenAgenda: () => void;
  onOpenResearch: () => void;
  projects: ProjectSummary[];
  onOpenProject: (id: string | null) => void;
  token: string;
  version?: string;
};

export function useSidebarProps({
  sessions,
  activeKey,
  loading,
  view,
  sidebarState,
  workspaces,
  runningChatIdList,
  updatedChatIdList,
  chatActions,
  utility,
  onOpenUtility,
      onOpenTodos,
      onOpenAgenda,
      onOpenResearch,
      projects,
  onOpenProject,
  token,
  version,
}: Args) {
  const projectNameOverrides = useProjectNames(
    "",
    token,
    sidebarState.project_name_overrides,
  );
  return useMemo(
    () => ({
      sessions,
      activeKey,
      loading,
      onNewChat: chatActions.onNew,
      onSelect: chatActions.onSelect,
      onRequestDelete: chatActions.onRequestDelete,
      onTogglePin: chatActions.onTogglePin,
      onRequestRename: chatActions.onRequestRename,
      onToggleArchive: chatActions.onToggleArchive,
      onToggleGroup: chatActions.onToggleGroup,
      onRequestRenameProject: chatActions.onRequestRenameProject,
      onNewChatInProject: chatActions.onNewInProject,
      onOpenSettings: utility.onOpenSettings,
      onOpenAutomations: () => onOpenUtility("automations"),
      onOpenProjects: () => onOpenUtility("projects"),
      onOpenWorkspace: () => onOpenUtility("workspace"),
      onOpenTodos,
      onOpenAgenda,
      onOpenResearch,
      projects,
      onOpenProject,
      onSettingsIntent: utility.onSettingsIntent,
      onOpenSearch: chatActions.onOpenSessionSearch,
      activeUtility:
        view === "automations" ||
        view === "projects" ||
        view === "workspace" ||
        view === "todos" ||
        view === "agenda" ||
        view === "research"
          ? view
          : null,
      onToggleArchived: chatActions.onToggleArchived,
      pinnedKeys: sidebarState.pinned_keys,
      archivedKeys: sidebarState.archived_keys,
      titleOverrides: sidebarState.title_overrides,
      projectNameOverrides,
      collapsedGroups: sidebarState.collapsed_groups,
      runningChatIds: runningChatIdList,
      updatedChatIds: updatedChatIdList,
      viewState: sidebarState.view,
      showArchived: sidebarState.view.show_archived,
      archivedCount: sidebarState.archived_keys.length,
      defaultWorkspacePath: workspaces?.default_scope.project_path ?? null,
      version,
    }),
    [
      sessions,
      activeKey,
      loading,
      view,
      sidebarState,
      workspaces,
      runningChatIdList,
      updatedChatIdList,
      chatActions,
      utility,
      onOpenUtility,
      onOpenTodos,
      onOpenAgenda,
      onOpenResearch,
      projects,
      onOpenProject,
      projectNameOverrides,
      version,
    ],
  );
}
