import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSessions } from "@/hooks/useSessions";
import { useSidebarState } from "@/hooks/useSidebarState";
import { useShellRoute } from "@/hooks/useShellRoute";
import { useChatActions } from "@/hooks/useChatActions";
import { useDocumentTitle } from "@/hooks/useDocumentTitle";
import { useEngineRestart } from "@/hooks/useEngineRestart";
import { useMissingSessionRedirect } from "@/hooks/useMissingSessionRedirect";
import { usePairing } from "@/hooks/usePairing";
import { useRunTracker } from "@/hooks/useRunTracker";
import { useSettingsSnapshot } from "@/hooks/useSettingsSnapshot";
import { useShellShortcuts } from "@/hooks/useShellShortcuts";
import { useSidebarProps } from "@/hooks/useSidebarProps";
import { useSkills } from "@/hooks/useSkills";
import { useThreadSessionSync } from "@/hooks/useThreadSessionSync";
import { useDeferredTitleRefresh } from "@/hooks/useDeferredTitleRefresh";
import { useHostSidebarLayout } from "@/hooks/useHostSidebarLayout";
import { useWorkspaceScope } from "@/hooks/useWorkspaceScope";
import { useTodos } from "@/hooks/useTodos";
import { useAgenda } from "@/hooks/useAgenda";
import { useProjects } from "@/hooks/useProjects";
import { useEphemeralSurfaceCleanup } from "@/hooks/useEphemeralSurfaceCleanup";
import { useDialogsState } from "@/lib/dialogs";
import type { RuntimeSurface } from "@/lib/types";
import type { NanobotClient } from "@/lib/nanobot-client";

type Args = {
  client: NanobotClient;
  token: string;
  runtimeSurface: RuntimeSurface;
  onModelNameChange: (modelName: string | null) => void;
};

export function useShellBootstrap({
  client,
  token,
  runtimeSurface,
  onModelNameChange,
}: Args) {
  const {
    sessions,
    loading,
    refresh,
    createChat,
    forkChat,
    deleteChat,
    getSessionAutomations,
  } = useSessions();
  const { state: sidebarState, update: updateSidebarState } =
    useSidebarState(sessions, !loading);
  const {
    activeKey,
    view,
    settingsSection: settingsInitialSection,
    navigate,
  } = useShellRoute();
  const dialogs = useDialogsState();
  const pairing = usePairing(token);
  const {
    visibleRequests: visiblePairingRequests,
    busyCode: pairingBusyCode,
    error: pairingError,
    onPairingAction,
    onDismissPairingRequest,
  } = pairing;
  const skills = useSkills(token);
  const projectsState = useProjects("", token);
  const settingsSnapshotApi = useSettingsSnapshot({ token });
  const { snapshot: settingsSnapshot, setSnapshot: setSettingsSnapshot } =
    settingsSnapshotApi;
  const activeChatIdRef = useRef<string | null>(null);
  const runTracker = useRunTracker({
    client,
    sessions,
    loading,
    activeChatIdRef,
  });
  const {
    runningChatIds: runningChatIdList,
    updatedChatIds: updatedChatIdList,
    setActiveChatId: setActiveChatIdTracker,
    setUpdatedChatIds,
  } = runTracker;
  const effectiveRuntimeSurface =
    settingsSnapshot?.surface ?? settingsSnapshot?.runtime_surface ?? runtimeSurface;
  const showHostChrome = effectiveRuntimeSurface === "native";
  const showMainSidebar = view !== "settings";
  const sidebarLayout = useHostSidebarLayout({ showHostChrome, showMainSidebar });
  const {
    hostSidebarOpen,
    mobileSidebarOpen,
    setMobileSidebarOpen,
    openPreview: openHostSidebarPreview,
    schedulePreviewClose: scheduleHostSidebarPreviewClose,
    closeHost: closeHostSidebar,
    openHost: openHostSidebar,
    toggleHost: toggleHostSidebar,
    closeMobile: closeMobileSidebar,
    toggleSidebar,
    hostSidebarCollapsed,
    showHostSidebarPreview,
    hostSidebarFlowWidth,
    renderHostSidebarFlowContent,
  } = sidebarLayout;

  const onOpenProject = useCallback(
    (id: string | null) => {
      setMobileSidebarOpen(false);
      navigate({ view: "projects", activeKey: id, settingsSection: "overview" });
    },
    [navigate, setMobileSidebarOpen],
  );

  const activeSession = useMemo(() => {
    if (!activeKey) return null;
    return sessions.find((s) => s.key === activeKey) ?? null;
  }, [sessions, activeKey]);
  const activeChatId = activeSession?.chatId ?? null;
  const runningChatIds = useMemo(
    () => new Set(runningChatIdList),
    [runningChatIdList],
  );
  const activeChatRunning = activeChatId
    ? runningChatIds.has(activeChatId)
    : false;
  const engineRestart = useEngineRestart({
    client,
    activeSession,
    defaultChatId: client.defaultChatId,
  });
  const workspaceScopeApi = useWorkspaceScope({
    client,
    token,
    activeSession,
    activeChatId,
    activeChatRunning,
    loading,
    shouldClearDraftScope: view === "chat" && !activeKey,
  });
  const todos = useTodos(sessions);
  const agenda = useAgenda();
  useEphemeralSurfaceCleanup(sessions, loading, deleteChat, token);
  const [todoSlug, setTodoSlug] = useState<string | null>(null);
  const {
    workspaces,
    error: workspaceError,
    setError: setWorkspaceError,
    activeWorkspaceScope,
    refresh: refreshWorkspaces,
    apply: applyWorkspaceScope,
    setDraftScope: setDraftWorkspaceScope,
    setOverrides: setWorkspaceOverrides,
    pruneOverrides: pruneWorkspaceOverrides,
  } = workspaceScopeApi;
  useThreadSessionSync({
    client,
    activeChatId,
    setActiveChatIdTracker,
    setUpdatedChatIds,
  });

  useEffect(() => {
    if (loading) return;
    const knownChatIds = new Set(sessions.map((session) => session.chatId));
    pruneWorkspaceOverrides(knownChatIds);
  }, [loading, sessions, pruneWorkspaceOverrides]);

  useMissingSessionRedirect({
    activeKey,
    loading,
    sessions,
    view,
    navigate,
  });

  const chatActions = useChatActions({
    sessions,
    activeKey,
    activeWorkspaceScope,
    sidebarState,
    updateSidebarState,
    createChat,
    forkChat,
    deleteChat,
    getSessionAutomations,
    navigate,
    setMobileSidebarOpen,
    onWorkspaceErrorCleared: () => setWorkspaceError(null),
    setWorkspaceOverrides,
    setDraftWorkspaceScope,
    setUpdatedChatIds,
    workspaces,
    loadSettingsView: () => import("@/components/settings/SettingsView"),
    dialogs,
    normalizeWorkspaceScope: (scope) => scope,
  });
  const {
    chat: {
      onCreate: onCreateChat,
      onFork: onForkChat,
      onNew: onNewChat,
      onBackToChat,
      onConfirmRename,
      onConfirmProjectRename,
      onConfirmDelete,
      onSelectSearchResult,
      onOpenSessionSearch,
    },
    utility: {
      onOpen: onOpenUtility,
      onOpenModelSettings: chatOnOpenModelSettings,
      onSettingsSectionChange: chatOnSettingsSectionChange,
    },
  } = chatActions;

  const onOpenTodoSlug = useCallback(
    (slug: string | null) => {
      setTodoSlug(slug);
      navigate({ view: "todos", activeKey, settingsSection: "overview" });
    },
    [navigate, activeKey],
  );

  const onOpenTodos = useCallback(
    (slug?: string | null) => {
      setMobileSidebarOpen(false);
      setTodoSlug(slug ?? null);
      navigate({ view: "todos", activeKey, settingsSection: "overview" });
    },
    [navigate, activeKey, setMobileSidebarOpen],
  );

  const onOpenAgenda = useCallback(() => {
    setMobileSidebarOpen(false);
    navigate({ view: "agenda", activeKey, settingsSection: "overview" });
  }, [navigate, activeKey, setMobileSidebarOpen]);

  const onOpenResearch = useCallback(() => {
    setMobileSidebarOpen(false);
    navigate({ view: "research", activeKey, settingsSection: "overview" });
  }, [navigate, activeKey, setMobileSidebarOpen]);

  useShellShortcuts({ onNewChat, onOpenSessionSearch, onOpenAgenda });

  const onTurnEnd = useDeferredTitleRefresh(activeSession, refresh);

  useDocumentTitle({ view, activeSession, sidebarState });

  useEffect(() => {
    return client.onRuntimeModelUpdate((modelName) => {
      onModelNameChange(modelName);
    });
  }, [client, onModelNameChange]);

  useEffect(() => {
    document.documentElement.classList.toggle("native-host", showHostChrome);
    return () => {
      document.documentElement.classList.remove("native-host");
    };
  }, [showHostChrome]);

  const sidebarProps = useSidebarProps({
    sessions,
    activeKey,
    loading,
    view,
    sidebarState,
    workspaces,
    runningChatIdList,
    updatedChatIdList,
    chatActions: chatActions.chat,
    utility: chatActions.utility,
    onOpenUtility,
    onOpenTodos,
    onOpenAgenda,
    onOpenResearch,
    projects: projectsState.projects,
    onOpenProject,
    token,
    version: settingsSnapshot?.version?.current,
  });

  return {
    sessions,
    loading,
    sidebarState,
    activeKey,
    view,
    settingsInitialSection,
    dialogs,
    pairing: {
      visibleRequests: visiblePairingRequests,
      busyCode: pairingBusyCode,
      error: pairingError,
      onPairingAction,
      onDismissPairingRequest,
    },
    skills,
    settingsSnapshot,
    setSettingsSnapshot,
    activeSession,
    activeChatId,
    activeChatRunning,
    engineRestart,
    workspaceError,
    activeWorkspaceScope,
    refreshWorkspaces,
    applyWorkspaceScope,
    onCreateChat,
    onForkChat,
    onNewChat,
    onBackToChat,
    onConfirmRename,
    onConfirmProjectRename,
    onConfirmDelete,
    onSelectSearchResult,
    onOpenSessionSearch,
    onOpenUtility,
    chatOnOpenModelSettings,
    chatOnSettingsSectionChange,
    onTurnEnd,
    showHostChrome,
    showMainSidebar,
    hostSidebarOpen,
    mobileSidebarOpen,
    setMobileSidebarOpen,
    openHostSidebarPreview,
    scheduleHostSidebarPreviewClose,
    closeHostSidebar,
    openHostSidebar,
    toggleHostSidebar,
    closeMobileSidebar,
    toggleSidebar,
    hostSidebarCollapsed,
    showHostSidebarPreview,
    hostSidebarFlowWidth,
    renderHostSidebarFlowContent,
    sidebarProps,
    workspaces,
    todos,
    agenda,
    todoSlug,
    onOpenTodoSlug,
    onOpenTodos,
    onOpenAgenda,
    onOpenResearch,
    onOpenProject,
  };
}

export type ShellBootstrap = ReturnType<typeof useShellBootstrap>;
