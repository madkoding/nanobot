import { Suspense, lazy, type ReactNode } from "react";
import { ThreadShell } from "@/components/thread/ThreadShell";
import { ProjectsSurface } from "@/components/projects/ProjectsSurface";
import { WorkspaceBrowser } from "@/components/workspace/WorkspaceBrowser";
import type { ChatSummary, SettingsPayload, WorkspacesPayload, WorkspaceScopePayload } from "@/lib/types";
import type { ShellView } from "@/lib/routing";
import { cn } from "@/lib/utils";

const SettingsView = lazy(() =>
  import("@/components/settings/SettingsView").then((m) => ({ default: m.SettingsView })),
);
const TodosSurface = lazy(() =>
  import("@/components/todos/TodosSurface").then((m) => ({ default: m.TodosSurface })),
);
const AgendaSurface = lazy(() =>
  import("@/components/agenda/AgendaSurface").then((m) => ({ default: m.AgendaSurface })),
);
const ResearchSurface = lazy(() =>
  import("@/components/research/ResearchSurface").then((m) => ({ default: m.ResearchSurface })),
);

type ThreadProps = React.ComponentProps<typeof ThreadShell>;
type SettingsProps = React.ComponentProps<typeof SettingsView>;

export type MainView = ShellView;

type TodoProps = React.ComponentProps<typeof TodosSurface>;
type AgendaProps = React.ComponentProps<typeof AgendaSurface>;

type Args = {
  view: MainView;
  activeKey: string | null;
  session: ChatSummary | null;
  title: string;
  settingsInitialSection: SettingsProps["initialSection"];
  settingsSnapshot: SettingsPayload | null;
  skills: ThreadProps["skills"];
  workspaces: WorkspacesPayload | null;
  activeWorkspaceScope: WorkspaceScopePayload | null;
  activeChatRunning: boolean;
  workspaceError: string | null;
  hostChromeInset: boolean;
  isRestarting: boolean;
  onToggleSidebar: () => void;
  onNewChat: () => void;
  onOpenResearch: () => void;
  onOpenProject: (id: string | null) => void;
  onCreateChat: ThreadProps["onCreateChat"];
  onForkChat: ThreadProps["onForkChat"];
  onTurnEnd: ThreadProps["onTurnEnd"];
  theme: "light" | "dark";
  onToggleTheme: () => void;
  hostChromeTitleInset: boolean;
  onWorkspaceScopeChange: (scope: WorkspaceScopePayload) => void;
  onOpenModelSettings: () => void;
  onBackToChat: () => void;
  onModelNameChange: SettingsProps["onModelNameChange"];
  onSettingsChange: (snapshot: SettingsPayload | null) => void;
  onWorkspaceSettingsChange: () => void;
  onSectionChange: SettingsProps["onSectionChange"];
  onLogout: SettingsProps["onLogout"];
  onRestart: SettingsProps["onRestart"];
  onNativeEngineRestart: SettingsProps["onNativeEngineRestart"];
  showSidebar: boolean;
  fallback: ReactNode;
  todoSlug: TodoProps["todoSlug"];
  onOpenTodoSlug: TodoProps["onOpenSlug"];
  todos: TodoProps["todos"];
  agenda: AgendaProps["agenda"];
};

function Surface({ hidden, children }: { hidden: boolean; children: ReactNode }) {
  return <div className={cn("absolute inset-0 flex flex-col", hidden && "hidden")}>{children}</div>;
}

export function MainView(props: Args) {
  return (
    <main
      className="relative flex h-full min-w-0 flex-1 flex-col overflow-hidden bg-background"
    >
      {props.view === "chat" ? (
        <Surface hidden={false}>
          <ThreadShell
            session={props.session}
            title={props.title}
            onToggleSidebar={props.onToggleSidebar}
            onNewChat={props.onNewChat}
            onCreateChat={props.onCreateChat}
            onForkChat={props.onForkChat}
            onTurnEnd={props.onTurnEnd}
            theme={props.theme}
            onToggleTheme={props.onToggleTheme}
            hideSidebarToggleForHostChrome
            hostChromeTitleInset={props.hostChromeTitleInset}
            hideHeader={false}
            workspaceScope={props.activeWorkspaceScope}
            workspaceDefaultScope={props.workspaces?.default_scope ?? null}
            workspaceControls={props.workspaces?.controls ?? null}
            workspaceScopeDisabled={props.activeChatRunning}
            workspaceError={props.workspaceError}
            onWorkspaceScopeChange={props.onWorkspaceScopeChange}
            settingsSnapshot={props.settingsSnapshot}
            onOpenModelSettings={props.onOpenModelSettings}
            skills={props.skills}
          />
        </Surface>
      ) : props.view === "projects" ? (
        <ProjectsSurface
          activeProjectId={props.activeKey}
          onOpenProject={props.onOpenProject}
          onBackToChat={props.onBackToChat}
        />
      ) : props.view === "workspace" ? (
        <WorkspaceBrowser onBackToChat={props.onBackToChat} />
      ) : props.view === "todos" ? (
        <Surface hidden={false}>
          <Suspense fallback={props.fallback}>
            <TodosSurface
              todoSlug={props.todoSlug}
              todos={props.todos}
              onOpenSlug={props.onOpenTodoSlug}
              onBackToChat={props.onBackToChat}
            />
          </Suspense>
        </Surface>
      ) : props.view === "agenda" ? (
        <Surface hidden={false}>
          <Suspense fallback={props.fallback}>
            <AgendaSurface
              agenda={props.agenda}
              onBackToChat={props.onBackToChat}
            />
          </Suspense>
        </Surface>
      ) : props.view === "research" ? (
        <Surface hidden={false}>
          <Suspense fallback={props.fallback}>
            <ResearchSurface
              onBackToChat={props.onBackToChat}
            />
          </Suspense>
        </Surface>
      ) : (
        <Surface hidden={false}>
          <Suspense fallback={props.fallback}>
            <SettingsView
              theme={props.theme}
              initialSection={props.settingsInitialSection}
              initialSettings={props.settingsSnapshot}
              showSidebar={props.showSidebar}
              onToggleTheme={props.onToggleTheme}
              onBackToChat={props.onBackToChat}
              onModelNameChange={props.onModelNameChange}
              onSettingsChange={props.onSettingsChange}
              skills={props.skills}
              onWorkspaceSettingsChange={props.onWorkspaceSettingsChange}
              onSectionChange={props.onSectionChange}
              onLogout={props.onLogout}
              onRestart={props.onRestart}
              onNativeEngineRestart={props.onNativeEngineRestart}
              isRestarting={props.isRestarting}
              hostChromeInset={props.hostChromeInset}
            />
          </Suspense>
        </Surface>
      )}
    </main>
  );
}

