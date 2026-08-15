export type LocalDensity = "comfortable" | "compact";
export type LocalActivityMode = "auto" | "expanded" | "collapsed";
export type FileEditDisplayMode = "summary" | "diff" | "collapsed_diff";
export type LocalFont = "system" | "serif" | "mono";
export type LocalAccent = "default" | "blue" | "green" | "purple" | "orange" | "rose";

export interface LocalPreferences {
  density: LocalDensity;
  activityMode: LocalActivityMode;
  codeWrap: boolean;
  brandLogos: boolean;
  fileEditDisplayMode: FileEditDisplayMode;
  font: LocalFont;
  accent: LocalAccent;
}

export const LOCAL_PREFS_STORAGE_KEY = "nanobot-webui.settings-preferences";
export const LOCAL_PREFS_CHANGED_EVENT = "nanobot-webui.local-preferences-changed";

export const DEFAULT_LOCAL_PREFS: LocalPreferences = {
  density: "comfortable",
  activityMode: "collapsed",
  codeWrap: true,
  brandLogos: false,
  fileEditDisplayMode: "summary",
  font: "system",
  accent: "default",
};

export function normalizeFileEditDisplayMode(value: unknown): FileEditDisplayMode {
  return value === "diff" || value === "collapsed_diff" ? value : "summary";
}

export function normalizeActivityMode(value: unknown): LocalActivityMode {
  return value === "auto" || value === "expanded" || value === "collapsed"
    ? value
    : "collapsed";
}

export function normalizeFont(value: unknown): LocalFont {
  return value === "serif" || value === "mono" ? value : "system";
}

export function normalizeAccent(value: unknown): LocalAccent {
  const accents: LocalAccent[] = ["default", "blue", "green", "purple", "orange", "rose"];
  return accents.includes(value as LocalAccent) ? (value as LocalAccent) : "default";
}

export function readLocalPreferences(): LocalPreferences {
  try {
    const raw = window.localStorage.getItem(LOCAL_PREFS_STORAGE_KEY);
    if (!raw) return DEFAULT_LOCAL_PREFS;
    const parsed = JSON.parse(raw) as Partial<LocalPreferences>;
    return {
      density: parsed.density === "compact" ? "compact" : "comfortable",
      activityMode: normalizeActivityMode(parsed.activityMode),
      codeWrap: parsed.codeWrap !== false,
      brandLogos: parsed.brandLogos === true,
      fileEditDisplayMode: normalizeFileEditDisplayMode(parsed.fileEditDisplayMode),
      font: normalizeFont(parsed.font),
      accent: normalizeAccent(parsed.accent),
    };
  } catch {
    return DEFAULT_LOCAL_PREFS;
  }
}

export function writeLocalPreferences(preferences: LocalPreferences): void {
  try {
    window.localStorage.setItem(LOCAL_PREFS_STORAGE_KEY, JSON.stringify(preferences));
  } catch {
    // Browser-only preferences should never block settings.
  }
  window.dispatchEvent(new CustomEvent<LocalPreferences>(
    LOCAL_PREFS_CHANGED_EVENT,
    { detail: preferences },
  ));
}
