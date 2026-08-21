import type { SettingsSectionKey } from "@/lib/types";

export type ShellView =
  | "chat"
  | "settings"
  | "apps"
  | "automations"
  | "skills"
  | "projects"
  | "workspace"
  | "todos"
  | "agenda"
  | "research"
  | "rlaif";

export type ShellRoute = {
  view: ShellView;
  activeKey: string | null;
  settingsSection: SettingsSectionKey;
};

const SETTINGS_SECTION_KEYS: SettingsSectionKey[] = [
  "overview",
  "appearance",
  "models",
  "image",
  "voice",
  "browser",
  "channels",
  "apps",
  "automations",
  "skills",
  "runtime",
  "advanced",
];

const RESTART_STARTED_KEY = "nanobot-webui.restartStartedAt";
const RESTART_ROUTE_KEY = "nanobot-webui.restartRoute";
const RESTART_ROUTE_TTL_MS = 5 * 60 * 1000;

export { RESTART_STARTED_KEY, RESTART_ROUTE_KEY, RESTART_ROUTE_TTL_MS };

export function isSettingsSectionKey(value: string | null): value is SettingsSectionKey {
  return SETTINGS_SECTION_KEYS.includes(value as SettingsSectionKey);
}

export function defaultShellRoute(): ShellRoute {
  return { view: "chat", activeKey: null, settingsSection: "overview" };
}

export function shellViewForSettingsSection(section: SettingsSectionKey): ShellView {
  if (section === "automations") return section;
  return "settings";
}

function fallbackRestartHash(hash: string): boolean {
  return !hash || hash === "/" || hash === "/new";
}

export function rememberRestartRoute(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(RESTART_ROUTE_KEY, window.location.hash || "#/new");
  } catch {
    // ignore storage errors
  }
}

function maybeRestoreRestartHash(hash: string): string {
  if (typeof window === "undefined" || !fallbackRestartHash(hash)) return hash;
  try {
    const startedAt = Number(window.localStorage.getItem(RESTART_STARTED_KEY) ?? "0");
    const storedHash = window.localStorage.getItem(RESTART_ROUTE_KEY);
    if (!startedAt || !storedHash || Date.now() - startedAt > RESTART_ROUTE_TTL_MS) {
      window.localStorage.removeItem(RESTART_ROUTE_KEY);
      return hash;
    }
    window.localStorage.removeItem(RESTART_ROUTE_KEY);
    const nextHash = storedHash.startsWith("#") ? storedHash : `#${storedHash}`;
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}${nextHash}`,
    );
    return nextHash.slice(1);
  } catch {
    return hash;
  }
}

export function readShellRoute(): ShellRoute {
  if (typeof window === "undefined") return defaultShellRoute();
  const currentHash = window.location.hash.startsWith("#")
    ? window.location.hash.slice(1)
    : window.location.hash;
  const hash = maybeRestoreRestartHash(currentHash);
  if (!hash || hash === "/" || hash === "/new") return defaultShellRoute();

  const [path, query = ""] = hash.split("?", 2);
  const params = new URLSearchParams(query);
  const rawSettingsSection = params.get("section");
  const settingsSection = isSettingsSectionKey(rawSettingsSection)
    ? rawSettingsSection
    : "overview";
  const activeKey = params.get("chat")?.trim() || null;

  if (path === "/settings") {
    return {
      view: shellViewForSettingsSection(settingsSection),
      activeKey,
      settingsSection,
    };
  }
  if (path === "/apps") return { view: "apps", activeKey, settingsSection: "apps" };
  if (path === "/automations") return { view: "automations", activeKey, settingsSection: "automations" };
  if (path === "/skills") return { view: "skills", activeKey, settingsSection: "skills" };
  if (path === "/projects") return { view: "projects", activeKey, settingsSection: "overview" };
  if (path.startsWith("/projects/")) {
    const encoded = path.slice("/projects/".length);
    try {
      const id = decodeURIComponent(encoded).trim();
      return id
        ? { view: "projects", activeKey: id, settingsSection: "overview" }
        : { view: "projects", activeKey, settingsSection: "overview" };
    } catch {
      return { view: "projects", activeKey, settingsSection: "overview" };
    }
  }
  if (path === "/workspace") return { view: "workspace", activeKey, settingsSection: "overview" };
  if (path === "/todos") return { view: "todos", activeKey, settingsSection: "overview" };
  if (path === "/agenda") return { view: "agenda", activeKey, settingsSection: "overview" };
  if (path === "/research") return { view: "research", activeKey, settingsSection: "overview" };
  if (path === "/rlaif") return { view: "rlaif", activeKey, settingsSection: "overview" };
  if (path.startsWith("/chat/")) {
    const encoded = path.slice("/chat/".length);
    try {
      const key = decodeURIComponent(encoded).trim();
      return key
        ? { view: "chat", activeKey: key, settingsSection: "overview" }
        : defaultShellRoute();
    } catch {
      return defaultShellRoute();
    }
  }
  return defaultShellRoute();
}

export function shellRouteHash(route: ShellRoute): string {
  if (route.view === "chat") {
    return route.activeKey
      ? `#/chat/${encodeURIComponent(route.activeKey)}`
      : "#/new";
  }
  if (route.view === "projects" && route.activeKey) {
    const params = new URLSearchParams();
    if (route.settingsSection === "overview") return `#/projects/${encodeURIComponent(route.activeKey)}`;
    params.set("section", route.settingsSection);
    return `#/projects/${encodeURIComponent(route.activeKey)}?${params.toString()}`;
  }
  const params = new URLSearchParams();
  if (route.activeKey) params.set("chat", route.activeKey);
  if (route.view === "settings" && route.settingsSection !== "overview") {
    params.set("section", route.settingsSection);
  }
  const query = params.toString();
  return `#/${route.view}${query ? `?${query}` : ""}`;
}

export function writeShellRoute(route: ShellRoute, replace = false): void {
  if (typeof window === "undefined") return;
  const nextHash = shellRouteHash(route);
  if (window.location.hash === nextHash) return;
  if (replace) {
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}${nextHash}`,
    );
    return;
  }
  window.history.pushState(
    null,
    "",
    `${window.location.pathname}${window.location.search}${nextHash}`,
  );
}

export function markRestartStarted(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(RESTART_STARTED_KEY, String(Date.now()));
    rememberRestartRoute();
  } catch {
    // ignore
  }
}
