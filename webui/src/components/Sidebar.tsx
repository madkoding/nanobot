import { useState, type ReactNode } from "react";
import {
  Archive,
  CalendarClock,
  CalendarDays,
  FolderKanban,
  FolderTree,
  ListTodo,
  Menu,
  Search,
  Settings,
  Sparkles,
  SquarePen,
  Telescope,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import { ChatList } from "@/components/ChatList";
import { ConnectionBadge } from "@/components/ConnectionBadge";
import { Button } from "@/components/ui/button";
import type {
  ChatSummary,
  SidebarViewState,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import type { ProjectSummary } from "@/lib/types";

interface SidebarProps {
  sessions: ChatSummary[];
  activeKey: string | null;
  loading: boolean;
  onNewChat: () => void;
  onSelect: (key: string) => void;
  onRequestDelete: (key: string, label: string) => void;
  onTogglePin: (key: string) => void;
  onRequestRename: (key: string, label: string) => void;
  onToggleArchive: (key: string) => void;
  onToggleGroup: (groupId: string) => void;
  onRequestRenameProject: (projectKey: string, label: string) => void;
  onNewChatInProject: (projectPath: string, projectName: string) => void;
  onOpenSettings: () => void;
  onOpenAutomations: () => void;
  onOpenProjects?: () => void;
  projects?: ProjectSummary[];
  onOpenProject?: (id: string) => void;
  onOpenWorkspace?: () => void;
  onOpenTodos?: () => void;
  onOpenAgenda?: () => void;
  onOpenResearch?: () => void;
  onOpenRlaif?: () => void;
  onSettingsIntent?: () => void;
  onOpenSearch: () => void;
  activeUtility?: "automations" | "projects" | "workspace" | "todos" | "agenda" | "research" | "rlaif" | null;
  onToggleArchived: () => void;
  onCollapse: () => void;
  onExpand?: () => void;
  containActionMenus?: boolean;
  collapsed?: boolean;
  pinnedKeys?: string[];
  archivedKeys?: string[];
  titleOverrides?: Record<string, string>;
  projectNameOverrides?: Record<string, string>;
  collapsedGroups?: Record<string, boolean>;
  runningChatIds?: string[];
  updatedChatIds?: string[];
  viewState?: SidebarViewState;
  showArchived?: boolean;
  archivedCount?: number;
  defaultWorkspacePath?: string | null;
  hostChromeInset?: boolean;
  version?: string;
}

type NavigatorWithUserAgentData = Navigator & {
  userAgentData?: { platform?: string };
};

function isApplePlatform(): boolean {
  if (typeof navigator === "undefined") return false;
  const platform = navigator.platform || "";
  const userAgentPlatform =
    (navigator as NavigatorWithUserAgentData).userAgentData?.platform || "";
  return /mac|iphone|ipad|ipod/i.test(`${platform} ${userAgentPlatform}`);
}

function newChatShortcutLabel(): string {
  return isApplePlatform() ? "⌘⇧O" : "Ctrl+Shift+O";
}

export function Sidebar(props: SidebarProps) {
  const { t } = useTranslation();
  const [menuPortalContainer, setMenuPortalContainer] =
    useState<HTMLElement | null>(null);
  const collapsed = Boolean(props.collapsed);
  const toggleLabel = t("thread.header.toggleSidebar");
  const newChatShortcut = newChatShortcutLabel();
  const version = props.version;

  return (
    <nav
      ref={props.containActionMenus ? setMenuPortalContainer : undefined}
      aria-label={t("sidebar.navigation")}
      className={cn(
        "flex h-full w-full min-w-0 flex-col text-sidebar-foreground",
        props.hostChromeInset ? "bg-transparent" : "bg-sidebar",
      )}
    >
      <div
        className={cn(
          "flex items-center px-3 pb-2.5",
          props.hostChromeInset ? "pt-[2.85rem]" : "pt-3",
          collapsed ? "w-14 justify-start" : "justify-between",
        )}
      >
        <button
          type="button"
          aria-label={collapsed ? toggleLabel : undefined}
          aria-hidden={collapsed ? undefined : true}
          title={collapsed ? toggleLabel : undefined}
          onClick={collapsed ? props.onExpand : undefined}
          tabIndex={collapsed ? 0 : -1}
          className={cn(
            "flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-xl transition-colors",
            collapsed
              ? "-ml-0.5 hover:bg-sidebar-accent/75"
              : "pointer-events-none -ml-0.5",
          )}
        >
          <span className="relative inline-block shrink-0">
            <img
              src="/brand/nanobot_mark.svg"
              alt=""
              className="h-8 w-8 select-none object-contain"
              draggable={false}
            />
          </span>
        </button>
        {!collapsed && version && (
          <span
            className="ml-1 self-center text-[11px] font-medium leading-none text-muted-foreground/80 select-none"
            title={t("sidebar.version", { defaultValue: "nanobot version" })}
          >
            v{version}
          </span>
        )}
        {!collapsed && !props.hostChromeInset && (
          <Button
            variant="ghost"
            size="icon"
            aria-label={t("sidebar.collapse")}
            onClick={props.onCollapse}
            className="h-7 w-7 rounded-lg text-muted-foreground/85 hover:bg-sidebar-accent/75 hover:text-sidebar-foreground"
          >
            <Menu className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      <div
        className={cn(
          "px-2 pb-2",
          collapsed
            ? "flex w-14 flex-col items-center px-0"
            : "grid grid-cols-2 gap-1.5 lg:flex lg:flex-col lg:gap-0 lg:space-y-1.5",
        )}
      >
        <SidebarActionButton
          collapsed={collapsed}
          label={t("sidebar.newChat")}
          onClick={props.onNewChat}
          icon={<SquarePen className="h-4 w-4" />}
          shortcut={newChatShortcut}
          ariaKeyShortcuts="Meta+Shift+O Control+Shift+O"
        />
        <SidebarActionButton
          collapsed={collapsed}
          label={t("sidebar.searchAria")}
          onClick={props.onOpenSearch}
          icon={<Search className="h-4 w-4" />}
        />
        {props.onOpenProjects && (
          <SidebarActionButton
            collapsed={collapsed}
            label={t("sidebar.projects", { defaultValue: "Projects" })}
            onClick={props.onOpenProjects}
            onIntent={props.onSettingsIntent}
            active={props.activeUtility === "projects"}
            icon={<FolderKanban className="h-4 w-4" />}
          />
        )}
        {props.projects && props.projects.length > 0 ? (
          <div className="flex flex-col gap-0.5">
            {props.projects.slice(0, 8).map((project) => (
              <button
                key={project.id}
                type="button"
                onClick={() => props.onOpenProject?.(project.id)}
                className="ml-2 flex items-center gap-2 rounded-md px-2 py-1 text-left text-xs text-muted-foreground transition-colors hover:bg-muted/70 hover:text-foreground"
                title={project.name}
              >
                <FolderKanban className="h-3 w-3 shrink-0" aria-hidden />
                <span className="truncate">{project.name}</span>
              </button>
            ))}
          </div>
        ) : null}
        {props.onOpenWorkspace && (
          <SidebarActionButton
            collapsed={collapsed}
            label={t("sidebar.workspace", { defaultValue: "Workspace" })}
            onClick={props.onOpenWorkspace}
            onIntent={props.onSettingsIntent}
            active={props.activeUtility === "workspace"}
            icon={<FolderTree className="h-4 w-4" />}
          />
        )}
        {props.onOpenTodos && (
          <SidebarActionButton
            collapsed={collapsed}
            label={t("sidebar.todos", { defaultValue: "Todos" })}
            onClick={props.onOpenTodos}
            onIntent={props.onSettingsIntent}
            active={props.activeUtility === "todos"}
            icon={<ListTodo className="h-4 w-4" />}
          />
        )}
        {props.onOpenAgenda && (
          <SidebarActionButton
            collapsed={collapsed}
            label={t("sidebar.agenda", { defaultValue: "Agenda" })}
            onClick={props.onOpenAgenda}
            onIntent={props.onSettingsIntent}
            active={props.activeUtility === "agenda"}
            icon={<CalendarDays className="h-4 w-4" />}
          />
        )}
        {props.onOpenResearch && (
          <SidebarActionButton
            collapsed={collapsed}
            label={t("sidebar.research", { defaultValue: "Research" })}
            onClick={props.onOpenResearch}
            onIntent={props.onSettingsIntent}
            active={props.activeUtility === "research"}
            icon={<Telescope className="h-4 w-4" />}
          />
        )}
        {props.onOpenRlaif && (
          <SidebarActionButton
            collapsed={collapsed}
            label={t("sidebar.rlaif", { defaultValue: "RLAIF Watch" })}
            onClick={props.onOpenRlaif}
            onIntent={props.onSettingsIntent}
            active={props.activeUtility === "rlaif"}
            icon={<Sparkles className="h-4 w-4" />}
          />
        )}
        <SidebarActionButton
          collapsed={collapsed}
          label={t("sidebar.automations", { defaultValue: "Automations" })}
          onClick={props.onOpenAutomations}
          onIntent={props.onSettingsIntent}
          active={props.activeUtility === "automations"}
          icon={<CalendarClock className="h-4 w-4" />}
        />
        {props.archivedCount ? (
          <SidebarActionButton
            collapsed={collapsed}
            label={props.showArchived ? t("chat.hideArchived") : t("chat.showArchived")}
            onClick={props.onToggleArchived}
            icon={<Archive className="h-4 w-4" />}
          />
        ) : null}
      </div>
      <div
        className={cn(
          "flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden transition-opacity duration-200",
          collapsed && "pointer-events-none opacity-0",
        )}
      >
        {!collapsed && (
          <ChatList
            sessions={props.sessions}
            activeKey={props.activeKey}
            loading={props.loading}
            emptyLabel={t("chat.noSessions")}
            onSelect={props.onSelect}
            onRequestDelete={props.onRequestDelete}
            onTogglePin={props.onTogglePin}
            onRequestRename={props.onRequestRename}
            onToggleArchive={props.onToggleArchive}
            onToggleGroup={props.onToggleGroup}
            onRequestRenameProject={props.onRequestRenameProject}
            onNewChatInProject={props.onNewChatInProject}
            pinnedKeys={props.pinnedKeys}
            archivedKeys={props.archivedKeys}
            titleOverrides={props.titleOverrides}
            projectNameOverrides={props.projectNameOverrides}
            collapsedGroups={props.collapsedGroups}
            runningChatIds={props.runningChatIds}
            updatedChatIds={props.updatedChatIds}
            density={props.viewState?.density}
            showPreviews={props.viewState?.show_previews}
            showTimestamps={props.viewState?.show_timestamps}
            sort={props.viewState?.sort}
            showArchived={props.showArchived}
            defaultWorkspacePath={props.defaultWorkspacePath}
            actionMenuPortalContainer={
              props.containActionMenus ? menuPortalContainer : undefined
            }
          />
        )}
      </div>
      <div
        className={cn(
          "flex items-center gap-1 bg-sidebar/55 px-2.5 py-3 text-xs",
          collapsed && "w-14 flex-col px-0",
        )}
      >
        <SidebarActionButton
          collapsed={collapsed}
          label={t("sidebar.settings")}
          onClick={props.onOpenSettings}
          onIntent={props.onSettingsIntent}
          className={collapsed ? undefined : "flex-1"}
          icon={<Settings className="h-4 w-4" />}
        />
        <ConnectionBadge />
      </div>
    </nav>
  );
}

function SidebarActionButton({
  collapsed,
  label,
  icon,
  onClick,
  active = false,
  className,
  shortcut,
  ariaKeyShortcuts,
  onIntent,
}: {
  collapsed: boolean;
  label: string;
  icon: ReactNode;
  onClick: () => void;
  active?: boolean;
  className?: string;
  shortcut?: string;
  ariaKeyShortcuts?: string;
  onIntent?: () => void;
}) {
  const title = shortcut ? `${label} (${shortcut})` : collapsed ? label : undefined;

  return (
    <Button
      type="button"
      variant="ghost"
      aria-label={label}
      aria-current={active ? "page" : undefined}
      aria-keyshortcuts={ariaKeyShortcuts}
      title={title}
      onClick={() => onClick()}
      onFocus={onIntent}
      onPointerEnter={onIntent}
      className={cn(
        "touch-target group h-8 min-w-0 gap-2 overflow-hidden rounded-full font-medium text-sidebar-foreground/85 hover:bg-sidebar-accent/75 hover:text-sidebar-foreground",
        "transition-[width,padding,border-radius,color,background-color] duration-300 ease-out",
        collapsed
          ? "w-9 justify-center gap-0 rounded-xl px-0"
          : "w-full justify-start gap-2 px-3 text-[12.5px]",
        active && "bg-sidebar-accent text-sidebar-foreground shadow-[inset_0_0_0_1px_hsl(var(--sidebar-border)/0.55)]",
        className,
      )}
    >
      <span
        className={cn(
          "flex shrink-0 items-center justify-center transition-transform duration-300 ease-out",
          collapsed ? "translate-x-0" : "translate-x-0",
        )}
        aria-hidden
      >
        {icon}
      </span>
      <span
        className={cn(
          "min-w-0 overflow-hidden truncate whitespace-nowrap transition-[max-width,opacity,transform] duration-200 ease-out",
          collapsed
            ? "max-w-0 -translate-x-1 opacity-0"
            : "max-w-[7rem] translate-x-0 opacity-100 lg:max-w-[12rem]",
        )}
      >
        {label}
      </span>
    </Button>
  );
}
