import { useEffect, useState, useCallback, type ReactNode } from "react";
import type { TFunction } from "i18next";
import {
  Brain,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Download,
  KeyRound,
  Loader2,
  RefreshCw,
  Search,
  Terminal,
  Trash2,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { ToggleButton } from "@/components/settings/ToggleButton";
import {
  deleteClawhubSkill,
  fetchClawhubBrowse,
  fetchClawhubSearch,
  fetchSkillDetail,
  fetchSkills,
  installClawhubSkill,
  toggleSkillEnabled,
  updateAllClawhubSkills,
  type ClawhubSkillSummary,
} from "@/lib/api";
import type { SkillDetail, SkillSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useClient } from "@/providers/ClientProvider";

export function SkillsCatalogSettings({
  skills,
  onRestart,
  isRestarting = false,
}: {
  skills: SkillSummary[];
  onRestart?: () => void;
  isRestarting?: boolean;
}) {
  const { token } = useClient();
  const { t } = useTranslation();
  const [localSkills, setLocalSkills] = useState<SkillSummary[]>(skills);

  const refreshSkills = useCallback(() => {
    fetchSkills(token)
      .then(({ skills: next }) => setLocalSkills(next))
      .catch(() => {
        /* keep the current list on transient failures */
      });
  }, [token]);

  const availableCount = localSkills.filter((skill) => skill.available).length;
  const [selectedSkill, setSelectedSkill] = useState<SkillSummary | null>(null);
  const [toggling, setToggling] = useState<string | null>(null);
  const [toggleError, setToggleError] = useState<string | null>(null);
  const [restartRequired, setRestartRequired] = useState(false);
  const [pendingDeleteSkill, setPendingDeleteSkill] = useState<SkillSummary | null>(null);
  const [deletingSkill, setDeletingSkill] = useState(false);

  const toggleSkill = (skill: SkillSummary, nextEnabled: boolean) => {
    setToggling(skill.name);
    setToggleError(null);
    toggleSkillEnabled(token, skill.name, nextEnabled)
      .then(() => {
        setLocalSkills((prev) =>
          prev.map((item) =>
            item.name === skill.name ? { ...item, disabled: !nextEnabled } : item,
          ),
        );
        setRestartRequired(true);
      })
      .catch(() =>
        setToggleError(
          t("settings.skills.toggleError", { defaultValue: "Could not update the skill." }),
        ),
      )
      .finally(() => setToggling(null));
  };

  const confirmDeleteSkill = () => {
    if (!pendingDeleteSkill) return;
    setDeletingSkill(true);
    setToggleError(null);
    deleteClawhubSkill(token, pendingDeleteSkill.name)
      .then(() => {
        setPendingDeleteSkill(null);
        refreshSkills();
      })
      .catch(() =>
        setToggleError(
          t("settings.skills.clawhubDeleteError", { defaultValue: "Could not delete skill." }),
        ),
      )
      .finally(() => setDeletingSkill(false));
  };

  return (
    <div className="space-y-7">
      <section className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <p className="max-w-[680px] text-[13px] leading-5 text-muted-foreground">
          {t("settings.skills.description", {
            defaultValue: "Review the instruction skills this agent can load during a conversation.",
          })}
        </p>
        <span className="text-[12px] font-medium text-muted-foreground">
          {t("settings.skills.caption", {
            available: availableCount,
            total: localSkills.length,
            defaultValue: "{{available}} available · {{total}} total",
          })}
        </span>
      </section>

      <section className="rounded-[22px] bg-settings-surface px-3 py-3 sm:px-4">
        <div className="flex items-center justify-between border-b border-border/45 pb-3">
          <h2 className="mb-2 px-1 text-[13px] font-semibold tracking-[-0.01em] text-foreground/85">
            {t("settings.skills.featured", { defaultValue: "Agent skills" })}
          </h2>
          <span className="rounded-full bg-muted px-2.5 py-1 text-[12px] font-medium text-muted-foreground">
            {localSkills.length}
          </span>
        </div>
        {localSkills.length ? (
          <div className="flex flex-col gap-2.5 py-3">
            {localSkills.map((skill) => (
              <SkillCatalogRow
                key={`${skill.source}:${skill.name}`}
                skill={skill}
                toggling={toggling === skill.name}
                onSelect={setSelectedSkill}
                onToggle={(next) => toggleSkill(skill, next)}
                onDelete={() => setPendingDeleteSkill(skill)}
              />
            ))}
          </div>
        ) : (
          <div className="px-3 py-12 text-center text-sm text-muted-foreground">
            {t("settings.skills.empty", { defaultValue: "No skills are available." })}
          </div>
        )}
        {toggleError ? (
          <div className="mx-1 mb-1 rounded-[14px] bg-destructive/10 px-3 py-2.5 text-[13px] text-destructive">
            {toggleError}
          </div>
        ) : null}
        {restartRequired ? (
          <div className="mx-1 mb-1 flex flex-wrap items-center justify-between gap-3 rounded-[14px] border border-amber-500/20 bg-amber-500/8 px-3 py-2.5 text-[13px] text-amber-800 dark:text-amber-200">
            <p>
              {t("settings.skills.restartRequired", {
                defaultValue: "Restart nanobot to apply skill changes.",
              })}
            </p>
            {onRestart ? (
              <button
                type="button"
                onClick={() => {
                  setRestartRequired(false);
                  onRestart?.();
                }}
                disabled={isRestarting}
                className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-[12px] font-medium text-amber-800 transition-colors hover:bg-amber-500/20 disabled:opacity-60 dark:text-amber-200"
              >
                {isRestarting ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                ) : (
                  <RefreshCw className="h-3.5 w-3.5" aria-hidden />
                )}
                {isRestarting
                  ? t("app.system.restarting", { defaultValue: "Restarting…" })
                  : t("app.system.restart", { defaultValue: "Restart" })}
              </button>
            ) : null}
          </div>
        ) : null}
      </section>

      <ClawhubSection
        onSkillsChanged={refreshSkills}
        installedNames={new Set(localSkills.map((skill) => skill.name))}
      />

      <SkillDetailSheet
        skill={selectedSkill}
        open={selectedSkill !== null}
        onOpenChange={(open) => {
          if (!open) setSelectedSkill(null);
        }}
      />

      <SkillDeleteDialog
        skill={pendingDeleteSkill}
        open={pendingDeleteSkill !== null}
        deleting={deletingSkill}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteSkill(null);
        }}
        onConfirm={confirmDeleteSkill}
      />
    </div>
  );
}

function ClawhubSection({
  onSkillsChanged,
  installedNames,
}: {
  onSkillsChanged: () => void;
  installedNames: Set<string>;
}) {
  const { token } = useClient();
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ClawhubSkillSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [installing, setInstalling] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [updatingAll, setUpdatingAll] = useState(false);
  const [updateResult, setUpdateResult] = useState<{
    updated: string[];
    skipped: string[];
    errors: { slug: string; error: string }[];
  } | null>(null);
  const [installed, setInstalled] = useState<Set<string>>(new Set());
  const [pendingDelete, setPendingDelete] = useState<ClawhubSkillSummary | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [browsing, setBrowsing] = useState(true);

  const loadBrowse = (targetPage: number) => {
    setLoading(true);
    setError(null);
    setBrowsing(true);
    fetchClawhubBrowse(token, targetPage)
      .then((payload) => {
        setResults(payload.results);
        setPage(payload.page);
        setTotalPages(payload.total_pages);
        setTotal(payload.total);
        if (payload.loading) {
          // Catalog is still being fetched in the background; poll until
          // it is ready so the UI shows a spinner instead of an empty list.
          window.setTimeout(() => loadBrowse(targetPage), 2_000);
          return;
        }
        if (payload.error) {
          setError(payload.error);
        }
      })
      .catch(() => setError(t("settings.skills.clawhubError", { defaultValue: "Could not load ClawHub skills." })))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadBrowse(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const runSearch = () => {
    const q = query.trim();
    if (!q) {
      loadBrowse(1);
      return;
    }
    setLoading(true);
    setError(null);
    setBrowsing(false);
    fetchClawhubSearch(token, q)
      .then(({ results: next }) => {
        setResults(next);
        setPage(1);
        setTotalPages(1);
        setTotal(next.length);
      })
      .catch(() => setError(t("settings.skills.clawhubError", { defaultValue: "Could not load ClawHub skills." })))
      .finally(() => setLoading(false));
  };

  const install = (skill: ClawhubSkillSummary) => {
    setInstalling(skill.reference);
    setError(null);
    installClawhubSkill(token, skill.reference)
      .then(({ slug }) => {
        setInstalled((prev) => new Set(prev).add(skill.reference));
        setInstalled((prev) => new Set(prev).add(slug));
        onSkillsChanged();
      })
      .catch(() => setError(t("settings.skills.clawhubInstallError", { defaultValue: "Could not install skill." })))
      .finally(() => setInstalling(null));
  };

  const confirmDelete = (skill: ClawhubSkillSummary) => {
    setDeleting(skill.reference);
    setError(null);
    deleteClawhubSkill(token, skill.slug)
      .then(() => {
        setInstalled((prev) => {
          const next = new Set(prev);
          next.delete(skill.reference);
          next.delete(skill.slug);
          return next;
        });
        onSkillsChanged();
      })
      .catch(() => setError(t("settings.skills.clawhubDeleteError", { defaultValue: "Could not delete skill." })))
      .finally(() => {
        setDeleting(null);
        setPendingDelete(null);
      });
  };

  const updateAll = () => {
    setUpdatingAll(true);
    setError(null);
    setUpdateResult(null);
    updateAllClawhubSkills(token)
      .then((result) => {
        setUpdateResult(result);
        onSkillsChanged();
      })
      .catch(() => setError(t("settings.skills.clawhubUpdateAllError", { defaultValue: "Could not update skills." })))
      .finally(() => setUpdatingAll(false));
  };

  return (
    <section className="rounded-[22px] bg-settings-surface px-3 py-3 sm:px-4">
      <div className="flex items-center justify-between border-b border-border/45 pb-3">
        <h2 className="mb-2 px-1 text-[13px] font-semibold tracking-[-0.01em] text-foreground/85">
          {t("settings.skills.clawhubTitle", { defaultValue: "Install from ClawHub" })}
        </h2>
        <span className="rounded-full bg-muted px-2.5 py-1 text-[12px] font-medium text-muted-foreground">
          clawhub.ai
        </span>
      </div>

      <p className="px-1 pb-3 pt-3 text-[13px] leading-5 text-muted-foreground">
        {t("settings.skills.clawhubDescription", {
          defaultValue: "Browse the public ClawHub registry and install skills directly.",
        })}
      </p>

      <div className="flex flex-wrap items-center gap-2 px-1 pb-3">
        <button
          type="button"
          onClick={updateAll}
          disabled={updatingAll}
          className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-[12px] border border-border/60 px-3.5 text-[13px] font-medium text-foreground/85 transition-colors hover:bg-muted/50 disabled:opacity-60"
        >
          {updatingAll ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          ) : (
            <RefreshCw className="h-3.5 w-3.5" aria-hidden />
          )}
          {t("settings.skills.clawhubUpdateAll", { defaultValue: "Update all installed" })}
        </button>
        {updateResult ? (
          <span className="text-[12px] text-muted-foreground">
            {t("settings.skills.clawhubUpdateAllResult", {
              updated: updateResult.updated.length,
              skipped: updateResult.skipped.length,
              errors: updateResult.errors.length,
              defaultValue: "{{updated}} updated · {{skipped}} skipped · {{errors}} failed",
            })}
          </span>
        ) : null}
      </div>

      <div className="flex gap-2 px-1 pb-3">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runSearch()}
            placeholder={t("settings.skills.clawhubSearchPlaceholder", { defaultValue: "Search skills…" })}
            className="h-9 w-full rounded-[12px] border border-border/50 bg-background/60 pl-9 pr-3 text-[13px] text-foreground outline-none transition-colors placeholder:text-muted-foreground/70 focus:border-ring focus:ring-1 focus:ring-ring"
          />
        </div>
        <button
          type="button"
          onClick={runSearch}
          className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-[12px] bg-foreground/90 px-3.5 text-[13px] font-medium text-background transition-colors hover:bg-foreground"
        >
          {t("settings.skills.clawhubSearch", { defaultValue: "Search" })}
        </button>
      </div>

      {error ? (
        <div className="mx-1 mb-3 rounded-[14px] bg-destructive/10 px-3 py-2.5 text-[13px] text-destructive">
          {error}
        </div>
      ) : null}

      <div className="flex items-center justify-between px-1 pb-1">
        <span className="text-[12px] font-medium text-muted-foreground">
          {browsing
            ? t("settings.skills.clawhubBrowseTotal", {
                total,
                defaultValue: "{{total}} skills · ordered by lifetime installs",
              })
            : t("settings.skills.clawhubSearchTotal", {
                total: results.length,
                defaultValue: "{{total}} results",
              })}
        </span>
        {browsing && totalPages > 1 ? (
          <span className="text-[12px] text-muted-foreground/80">
            {t("settings.skills.clawhubPage", {
              page,
              totalPages,
              defaultValue: "Page {{page}} / {{totalPages}}",
            })}
          </span>
        ) : null}
      </div>

      {loading ? (
        <div className="flex items-center gap-2 px-1 py-6 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          {t("settings.skills.loadingCatalog", { defaultValue: "Loading skills…" })}
        </div>
      ) : results.length ? (
        <div className="flex flex-col gap-2.5 py-1">
          {results.map((skill) => (
            <ClawhubRow
              key={skill.reference}
              skill={skill}
              installing={installing === skill.reference}
              installed={
                installed.has(skill.reference) ||
                installed.has(skill.slug) ||
                installedNames.has(skill.slug)
              }
              onInstall={() => install(skill)}
              onDelete={() => setPendingDelete(skill)}
            />
          ))}
        </div>
      ) : (
        <div className="px-3 py-8 text-center text-sm text-muted-foreground">
          {t("settings.skills.clawhubEmpty", { defaultValue: "No skills found. Try another search." })}
        </div>
      )}

      {browsing && totalPages > 1 ? (
        <div className="mt-3 flex items-center justify-center gap-2 px-1">
          <button
            type="button"
            disabled={loading || page <= 1}
            onClick={() => loadBrowse(page - 1)}
            className="inline-flex h-9 items-center gap-1 rounded-[12px] border border-border/60 px-3 text-[13px] font-medium text-foreground/85 transition-colors hover:bg-muted/50 disabled:opacity-50"
          >
            <ChevronLeft className="h-3.5 w-3.5" aria-hidden />
            {t("settings.skills.clawhubPrev", { defaultValue: "Previous" })}
          </button>
          <span className="min-w-[4.5rem] text-center text-[12px] font-medium text-muted-foreground">
            {t("settings.skills.clawhubPage", {
              page,
              totalPages,
              defaultValue: "Page {{page}} / {{totalPages}}",
            })}
          </span>
          <button
            type="button"
            disabled={loading || page >= totalPages}
            onClick={() => loadBrowse(page + 1)}
            className="inline-flex h-9 items-center gap-1 rounded-[12px] border border-border/60 px-3 text-[13px] font-medium text-foreground/85 transition-colors hover:bg-muted/50 disabled:opacity-50"
          >
            {t("settings.skills.clawhubNext", { defaultValue: "Next" })}
            <ChevronRight className="h-3.5 w-3.5" aria-hidden />
          </button>
        </div>
      ) : null}

      <SkillDeleteDialog
        skill={pendingDelete}
        open={pendingDelete !== null}
        deleting={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
        onConfirm={() => pendingDelete && confirmDelete(pendingDelete)}
      />
    </section>
  );
}

function SkillDeleteDialog({
  skill,
  open,
  deleting,
  onOpenChange,
  onConfirm,
}: {
  skill: { name: string } | null;
  open: boolean;
  deleting: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}) {
  const { t } = useTranslation();
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>
            {t("settings.skills.clawhubDeleteTitle", { defaultValue: "Delete skill" })}
          </DialogTitle>
          <DialogDescription>
            {t("settings.skills.clawhubDeleteDescription", {
              name: skill?.name ?? "",
              defaultValue: "Remove the '{{name}}' skill from this workspace? This cannot be undone.",
            })}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <button
            type="button"
            disabled={deleting}
            onClick={() => onOpenChange(false)}
            className="inline-flex h-9 items-center rounded-[12px] border border-border/60 px-3.5 text-[13px] font-medium text-muted-foreground transition-colors hover:bg-muted/50 disabled:opacity-60"
          >
            {t("settings.skills.cancel", { defaultValue: "Cancel" })}
          </button>
          <button
            type="button"
            disabled={deleting}
            onClick={onConfirm}
            className="inline-flex h-9 items-center gap-1.5 rounded-[12px] bg-destructive px-3.5 text-[13px] font-medium text-white transition-colors hover:bg-destructive/90 disabled:opacity-60"
          >
            {deleting ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
            ) : (
              <Trash2 className="h-3.5 w-3.5" aria-hidden />
            )}
            {t("settings.skills.clawhubDeleteConfirm", { defaultValue: "Delete" })}
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ClawhubRow({
  skill,
  installing,
  installed,
  onInstall,
  onDelete,
}: {
  skill: ClawhubSkillSummary;
  installing: boolean;
  installed: boolean;
  onInstall: () => void;
  onDelete: () => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="group flex min-w-0 flex-col gap-3 rounded-[18px] border border-border/40 bg-background/60 px-4 py-4 text-left transition-colors hover:border-border/70 hover:bg-background/85 sm:flex-row sm:items-start">
      <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[12px] bg-muted/70 text-muted-foreground">
        <Brain className="h-5 w-5" strokeWidth={1.8} aria-hidden />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
          <h3 className="text-[15px] font-semibold leading-5 text-foreground">
            {skill.name}
          </h3>
          <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium leading-none text-muted-foreground">
            {skill.owner}
          </span>
          {skill.kind === "skills-sh" ? (
            <span className="rounded-full bg-violet-500/10 px-2 py-0.5 text-[11px] font-medium leading-none text-violet-600 dark:text-violet-300">
              skills.sh
            </span>
          ) : null}
          <span className="text-[12px] leading-4 text-muted-foreground/70">
            {t("settings.skills.clawhubInstalls", {
              count: skill.lifetime_installs ?? skill.installs_60d,
              defaultValue: "{{count}} installs",
            })}
          </span>
        </div>
        <p className="mt-2 text-[13px] leading-[1.55] text-muted-foreground">
          {skill.description}
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-2 self-start sm:self-center">
        {installed ? (
          <button
            type="button"
            onClick={onDelete}
            title={t("settings.skills.clawhubDelete", { defaultValue: "Delete skill" })}
            aria-label={t("settings.skills.clawhubDelete", { defaultValue: "Delete skill" })}
            className="inline-flex items-center gap-1.5 rounded-[12px] border border-border/50 px-2.5 py-1.5 text-[12px] font-medium text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
          >
            <Trash2 className="h-3.5 w-3.5" aria-hidden />
          </button>
        ) : null}
        <button
          type="button"
          onClick={onInstall}
          disabled={installing || installed}
          className={cn(
            "inline-flex shrink-0 items-center gap-1.5 rounded-[12px] px-3 py-1.5 text-[12px] font-medium transition-colors",
            installed
              ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
              : "bg-foreground/90 text-background hover:bg-foreground",
            (installing || installed) && "cursor-default opacity-80",
          )}
        >
          {installing ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          ) : installed ? (
            <Check className="h-3.5 w-3.5" aria-hidden />
          ) : (
            <Download className="h-3.5 w-3.5" aria-hidden />
          )}
          {installing
            ? t("settings.skills.clawhubInstalling", { defaultValue: "Installing…" })
            : installed
              ? t("settings.skills.clawhubInstalled", { defaultValue: "Installed" })
              : t("settings.skills.clawhubInstall", { defaultValue: "Install" })}
        </button>
      </div>
    </div>
  );
}

function SkillCatalogRow({
  skill,
  toggling,
  onSelect,
  onToggle,
  onDelete,
}: {
  skill: SkillSummary;
  toggling: boolean;
  onSelect: (skill: SkillSummary) => void;
  onToggle: (nextEnabled: boolean) => void;
  onDelete: () => void;
}) {
  const { t } = useTranslation();
  const sourceLabel = skillSourceLabel(skill.source, t);
  const StatusIcon = skill.available ? Check : CircleAlert;
  const statusLabel = skill.available
    ? t("settings.skills.statusAvailable", { defaultValue: "Available" })
    : t("settings.skills.statusUnavailable", { defaultValue: "Unavailable" });
  const isWorkspace = skill.source === "workspace";

  return (
    <div
      className={cn(
        "group flex min-w-0 flex-col gap-3 rounded-[18px] border border-border/40 bg-background/60 px-4 py-4 text-left transition-colors hover:border-border/70 hover:bg-background/85 sm:flex-row sm:items-start",
        !skill.available && "opacity-65",
        skill.disabled && "opacity-70",
      )}
    >
      <button
        type="button"
        aria-label={t("settings.skills.openDetails", {
          name: skill.name,
          defaultValue: "Open details for {{name}}",
        })}
        onClick={() => onSelect(skill)}
        className="flex min-w-0 flex-1 items-start gap-3 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:items-center"
      >
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[12px] bg-muted/70 text-muted-foreground">
          <Brain className="h-5 w-5" strokeWidth={1.8} aria-hidden />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
            <h3 className="text-[15px] font-semibold leading-5 text-foreground">
              {skill.name}
            </h3>
            <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium leading-none text-muted-foreground">
              {sourceLabel}
            </span>
            <span
              title={!skill.available && skill.unavailable_reason ? skill.unavailable_reason : undefined}
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium leading-none",
                skill.available
                  ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                  : "bg-amber-500/10 text-amber-700 dark:text-amber-300",
              )}
            >
              <StatusIcon className="h-3 w-3" aria-hidden />
              {statusLabel}
            </span>
          </div>
          <p className="mt-2 text-[13px] leading-[1.55] text-muted-foreground">
            {skill.description}
          </p>
          {!skill.available && skill.unavailable_reason ? (
            <p className="mt-1.5 text-[12px] leading-4 text-muted-foreground/80">
              {t("settings.skills.unavailableReason", {
                reason: skill.unavailable_reason,
                defaultValue: "Missing: {{reason}}",
              })}
            </p>
          ) : null}
        </div>
      </button>
      <div className="flex shrink-0 items-center gap-2 self-start sm:self-center">
        <ToggleButton
          checked={!skill.disabled}
          disabled={toggling}
          onChange={onToggle}
          ariaLabel={t("settings.skills.toggleSkill", {
            name: skill.name,
            defaultValue: "Enable or disable {{name}}",
          })}
          label={skill.disabled ? t("settings.skills.statusDisabled", { defaultValue: "Disabled" }) : t("settings.skills.statusEnabled", { defaultValue: "Enabled" })}
        />
        {isWorkspace ? (
          <button
            type="button"
            onClick={onDelete}
            title={t("settings.skills.clawhubDelete", { defaultValue: "Delete skill" })}
            aria-label={t("settings.skills.clawhubDelete", { defaultValue: "Delete skill" })}
            className="inline-flex items-center rounded-[12px] border border-border/50 px-2.5 py-1.5 text-[12px] font-medium text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
          >
            <Trash2 className="h-3.5 w-3.5" aria-hidden />
          </button>
        ) : null}
      </div>
    </div>
  );
}

function SkillDetailSheet({
  skill,
  open,
  onOpenChange,
}: {
  skill: SkillSummary | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { token } = useClient();
  const { t } = useTranslation();
  const [detail, setDetail] = useState<SkillDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadFailed, setLoadFailed] = useState(false);

  useEffect(() => {
    if (!open || !skill) return;
    let cancelled = false;
    setDetail(null);
    setLoading(true);
    setLoadFailed(false);
    fetchSkillDetail(token, skill.name)
      .then((payload) => {
        if (!cancelled) setDetail(payload);
      })
      .catch(() => {
        if (!cancelled) setLoadFailed(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, skill, token]);

  if (!skill) return null;

  const activeSkill = detail ?? skill;
  const sourceLabel = skillSourceLabel(activeSkill.source, t);
  const statusLabel = activeSkill.available
    ? t("settings.skills.statusAvailable", { defaultValue: "Available" })
    : t("settings.skills.statusUnavailable", { defaultValue: "Unavailable" });

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-[min(34rem,calc(100vw-1rem))] max-w-none gap-0 overflow-hidden p-0 sm:max-w-none"
      >
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
          <div className="flex items-start gap-3 pr-8">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-[15px] bg-muted/70 text-muted-foreground">
              <Brain className="h-5 w-5" strokeWidth={1.8} aria-hidden />
            </div>
            <div className="min-w-0">
              <SheetTitle className="truncate text-[20px] font-semibold">
                {activeSkill.name}
              </SheetTitle>
              <SheetDescription className="sr-only">
                {t("settings.skills.detailDescription", {
                  name: activeSkill.name,
                  defaultValue: "Details for {{name}}.",
                })}
              </SheetDescription>
              <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[12px] text-muted-foreground">
                <Pill>{sourceLabel}</Pill>
                <Pill tone={activeSkill.available ? "success" : "muted"}>{statusLabel}</Pill>
              </div>
            </div>
          </div>

          {loading ? (
            <div className="mt-8 flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
              {t("settings.skills.loadingDetail", { defaultValue: "Loading skill details..." })}
            </div>
          ) : loadFailed ? (
            <div className="mt-8 rounded-[16px] bg-destructive/10 px-3 py-3 text-sm text-destructive">
              {t("settings.skills.loadFailed", { defaultValue: "Could not load skill details." })}
            </div>
          ) : (
            <div className="mt-7 space-y-6">
              <DetailSection title={t("settings.skills.descriptionTitle", { defaultValue: "Description" })}>
                <p className="text-[14px] leading-6 text-muted-foreground">{activeSkill.description}</p>
              </DetailSection>

              <div className="grid grid-cols-2 gap-2">
                <MetaItem
                  label={t("settings.skills.source", { defaultValue: "Source" })}
                  value={sourceLabel}
                />
                <MetaItem
                  label={t("settings.skills.status", { defaultValue: "Status" })}
                  value={statusLabel}
                />
              </div>

              {!activeSkill.available && activeSkill.unavailable_reason ? (
                <DetailSection
                  title={t("settings.skills.unavailableReasonLabel", {
                    defaultValue: "Unavailable reason",
                  })}
                >
                  <p className="text-[13px] leading-5 text-destructive/85">
                    {activeSkill.unavailable_reason}
                  </p>
                </DetailSection>
              ) : null}

              {detail ? <RequirementsSection detail={detail} /> : null}

              {detail ? <RawInstructionsBlock markdown={detail.raw_markdown} /> : null}
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function RawInstructionsBlock({ markdown }: { markdown: string }) {
  const { t } = useTranslation();
  const content =
    markdown ||
    t("settings.skills.rawInstructionsEmpty", {
      defaultValue: "No raw instructions.",
    });

  return (
    <details className="group rounded-[18px] border border-border/45 bg-muted/20 px-3 py-3">
      <summary className="cursor-pointer select-none text-[13px] font-medium text-foreground/90 transition-colors hover:text-foreground">
        {t("settings.skills.rawInstructions", { defaultValue: "Raw SKILL.md" })}
      </summary>
      <div className="mt-3 overflow-hidden rounded-[14px] border border-border/35 bg-background/70">
        <pre
          className={cn(
            "max-h-[min(42vh,32rem)] overflow-auto overscroll-contain px-3.5 py-3 pr-4",
            "whitespace-pre-wrap break-words font-mono text-[12px] leading-[1.7] text-foreground/62",
            "scrollbar-thin scrollbar-track-transparent",
            "[&::-webkit-scrollbar]:h-1.5 [&::-webkit-scrollbar]:w-1.5",
            "[&::-webkit-scrollbar-thumb]:bg-muted-foreground/25",
          )}
        >
          {content}
        </pre>
      </div>
    </details>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[16px] bg-muted/35 px-3 py-2.5">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate text-[13px] font-medium text-foreground">{value}</div>
    </div>
  );
}

function RequirementsSection({ detail }: { detail: SkillDetail }) {
  const { t } = useTranslation();
  const { bins, env, missing_bins, missing_env } = detail.requirements;
  const hasRequirements = bins.length > 0 || env.length > 0;

  return (
    <DetailSection title={t("settings.skills.requirements", { defaultValue: "Requirements" })}>
      {hasRequirements ? (
        <div className="space-y-3">
          {missing_bins.length ? (
            <RequirementLine
              title={t("settings.skills.missingCommands", { defaultValue: "Missing CLI" })}
              items={missing_bins}
              tone="danger"
              icon={<Terminal className="h-3.5 w-3.5" aria-hidden />}
            />
          ) : null}
          {missing_env.length ? (
            <RequirementLine
              title={t("settings.skills.missingEnvironment", { defaultValue: "Missing ENV" })}
              items={missing_env}
              tone="danger"
              icon={<KeyRound className="h-3.5 w-3.5" aria-hidden />}
            />
          ) : null}
          {bins.length ? (
            <RequirementLine
              title={t("settings.skills.commands", { defaultValue: "Commands" })}
              items={bins}
              icon={<Terminal className="h-3.5 w-3.5" aria-hidden />}
            />
          ) : null}
          {env.length ? (
            <RequirementLine
              title={t("settings.skills.environment", { defaultValue: "Environment variables" })}
              items={env}
              icon={<KeyRound className="h-3.5 w-3.5" aria-hidden />}
            />
          ) : null}
        </div>
      ) : (
        <p className="text-[13px] text-muted-foreground">
          {t("settings.skills.noRequirements", { defaultValue: "No explicit requirements." })}
        </p>
      )}
    </DetailSection>
  );
}

function DetailSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="mb-2 text-[12px] font-medium text-muted-foreground">{title}</h3>
      {children}
    </section>
  );
}

function RequirementLine({
  title,
  items,
  icon,
  tone = "muted",
}: {
  title: string;
  items: string[];
  icon: ReactNode;
  tone?: "muted" | "danger";
}) {
  return (
    <div className="space-y-1.5">
      <div
        className={cn(
          "flex items-center gap-1.5 text-[12px]",
          tone === "danger" ? "text-destructive" : "text-muted-foreground",
        )}
      >
        {icon}
        {title}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <Pill key={item}>{item}</Pill>
        ))}
      </div>
    </div>
  );
}

function Pill({
  children,
  tone = "muted",
}: {
  children: ReactNode;
  tone?: "muted" | "success";
}) {
  return (
    <span
      className={cn(
        "inline-flex max-w-full items-center rounded-full px-2 py-0.5 text-[11px] font-medium",
        tone === "success"
          ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
          : "bg-muted text-muted-foreground",
      )}
    >
      {children}
    </span>
  );
}

function skillSourceLabel(source: string, t: TFunction): string {
  if (source === "workspace") {
    return t("settings.skills.sourceWorkspace", { defaultValue: "Custom" });
  }
  if (source === "builtin") {
    return t("settings.skills.sourceBuiltin", { defaultValue: "Built-in" });
  }
  return source;
}
