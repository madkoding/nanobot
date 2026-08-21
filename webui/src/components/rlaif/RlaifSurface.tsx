import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, ChevronLeft, Loader2, RefreshCw, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useClient } from "@/providers/ClientProvider";
import { useRlaifWatch } from "@/hooks/useRlaifWatch";
import {
  actOnRlaifProposal,
  fetchRlaifProposal,
  fetchRlaifProposals,
  type RlaifProposal,
} from "@/lib/api";

interface Props {
  onBackToChat: () => void;
}

const POLL_INTERVAL_MS = 5000;

export function RlaifSurface({ onBackToChat }: Props) {
  const { t } = useTranslation();
  const watch = useRlaifWatch();
  const { token } = useClient();
  const logRef = useRef<HTMLDivElement | null>(null);
  const userScrolledRef = useRef(false);

  const [proposals, setProposals] = useState<RlaifProposal[]>([]);
  const [scannerActive, setScannerActive] = useState(false);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [expandedPatch, setExpandedPatch] = useState<string | null>(null);
  const [actionPending, setActionPending] = useState<number | null>(null);
  const [actionResult, setActionResult] = useState<{ id: number; result: string; ok: boolean } | null>(null);

  // Auto-scroll for the log card.
  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (!userScrolledRef.current || distanceFromBottom < 80) {
      el.scrollTop = el.scrollHeight;
    }
  }, [watch.log.length]);

  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    const onScroll = () => {
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      userScrolledRef.current = distanceFromBottom > 80;
    };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  const refreshProposals = useCallback(async () => {
    try {
      const payload = await fetchRlaifProposals(token);
      setScannerActive(payload.scanner_active);
      setProposals(payload.items);
    } catch (e) {
      // swallow; the rest of the surface still works
    }
  }, [token]);

  useEffect(() => {
    void refreshProposals();
    const id = window.setInterval(() => {
      void refreshProposals();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [refreshProposals]);

  const expandProposal = useCallback(
    async (id: number) => {
      if (expandedId === id) {
        setExpandedId(null);
        setExpandedPatch(null);
        return;
      }
      setExpandedId(id);
      setExpandedPatch(null);
      try {
        const full = await fetchRlaifProposal(token, id);
        setExpandedPatch(full.patch);
      } catch (e) {
        setExpandedPatch(`# error: ${e instanceof Error ? e.message : String(e)}`);
      }
    },
    [token, expandedId],
  );

  const onApprove = useCallback(
    async (id: number) => {
      setActionPending(id);
      setActionResult(null);
      try {
        const r = await actOnRlaifProposal(token, id, "approve");
        setActionResult({ id, result: r.result, ok: r.ok });
        await refreshProposals();
      } catch (e) {
        setActionResult({
          id,
          result: e instanceof Error ? e.message : String(e),
          ok: false,
        });
      } finally {
        setActionPending(null);
      }
    },
    [token, refreshProposals],
  );

  const onReject = useCallback(
    async (id: number) => {
      setActionPending(id);
      setActionResult(null);
      try {
        const r = await actOnRlaifProposal(token, id, "reject");
        setActionResult({ id, result: r.result, ok: r.ok });
        await refreshProposals();
      } catch (e) {
        setActionResult({
          id,
          result: e instanceof Error ? e.message : String(e),
          ok: false,
        });
      } finally {
        setActionPending(null);
      }
    },
    [token, refreshProposals],
  );

  const prefsReversed = useMemo(() => [...watch.preferences].reverse(), [watch.preferences]);
  const lastLog = useMemo(() => watch.log.slice(-150), [watch.log]);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-settings-canvas">
      <div className="mx-auto w-full max-w-4xl flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onBackToChat}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <ChevronLeft className="h-4 w-4" />
            {t("rlaif.backToChats", { defaultValue: "Back to chats" })}
          </button>
          <div className="mx-2 h-5 w-px bg-border" />
          <Sparkles className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-lg font-semibold text-foreground">
            {t("rlaif.title", { defaultValue: "RLAIF Watch" })}
          </h1>
        </div>

        <p className="mb-4 text-sm text-muted-foreground">
          {t("rlaif.subtitle", {
            defaultValue:
              "Scanner proposes improvements; you review and approve before they touch the repo.",
          })}
        </p>

        {watch.error ? (
          <div className="mb-3 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
            {watch.error}
          </div>
        ) : null}

        {/* Pending proposals */}
        <section
          className="mb-6 rounded-xl border border-border bg-card p-4"
          data-testid="rlaif-proposals"
        >
          <header className="mb-3 flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-foreground">
                {t("rlaif.proposalsTitle", { defaultValue: "Pending proposals" })}
              </h2>
              <p className="text-xs text-muted-foreground">
                {scannerActive
                  ? t("rlaif.proposalsScannerActive", {
                      defaultValue: "Scanner is running. {{count}} pending review.",
                    }).replace("{{count}}", String(proposals.length))
                  : t("rlaif.proposalsScannerInactive", {
                      defaultValue: "Scanner is not running.",
                    })}
              </p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => void refreshProposals()}
              aria-label={t("rlaif.refresh", { defaultValue: "Refresh" })}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </header>

          {proposals.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              {t("rlaif.noProposals", {
                defaultValue:
                  "No pending proposals. The scanner queues a patch when the critic proposes one that passes a preflight lint check.",
              })}
            </p>
          ) : (
            <ul className="space-y-2">
              {proposals.map((p) => {
                const isExpanded = expandedId === p.id;
                const isPending = actionPending === p.id;
                return (
                  <li
                    key={p.id}
                    className="rounded-lg border border-border/60 bg-background p-3"
                    data-testid="rlaif-proposal-row"
                  >
                    <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
                      <button
                        type="button"
                        onClick={() => void expandProposal(p.id)}
                        className="min-w-0 flex-1 text-left"
                      >
                        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground/70">
                          #{p.id} · {p.file}
                        </div>
                        <div className="mt-0.5 line-clamp-2 text-sm text-foreground">
                          {p.rationale || t("rlaif.noRationale", { defaultValue: "(no rationale)" })}
                        </div>
                      </button>
                      <div className="flex shrink-0 items-center gap-1.5">
                        <span className="font-mono text-[11px] text-muted-foreground">
                          {p.confidence.toFixed(2)}
                        </span>
                        <Button
                          size="sm"
                          variant="default"
                          disabled={isPending}
                          onClick={() => void onApprove(p.id)}
                          data-testid="rlaif-approve"
                        >
                          {isPending ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <Check className="h-3.5 w-3.5" />
                          )}
                          {t("rlaif.approve", { defaultValue: "Approve" })}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={isPending}
                          onClick={() => void onReject(p.id)}
                          data-testid="rlaif-reject"
                        >
                          <X className="h-3.5 w-3.5" />
                          {t("rlaif.reject", { defaultValue: "Reject" })}
                        </Button>
                      </div>
                    </div>
                    {isExpanded ? (
                      <pre className="mt-2 max-h-72 overflow-auto rounded-md border border-border/60 bg-muted/30 p-2 text-[11px] leading-5">
                        {expandedPatch ?? "loading…"}
                      </pre>
                    ) : null}
                    {actionResult && actionResult.id === p.id ? (
                      <div
                        className={
                          "mt-2 rounded-md border px-2 py-1.5 text-[11.5px] " +
                          (actionResult.ok
                            ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300"
                            : "border-destructive/30 bg-destructive/5 text-destructive")
                        }
                      >
                        {actionResult.result}
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <div className="mb-6 grid gap-4 md:grid-cols-2">
          <section className="rounded-xl border border-border bg-card p-4">
            <header className="mb-2 flex items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-semibold text-foreground">
                  {t("rlaif.preferencesTitle", { defaultValue: "Preference pairs" })}
                </h2>
                <p className="text-xs text-muted-foreground">
                  {t("rlaif.preferencesCount", {
                    defaultValue: "{{total}} stored",
                  }).replace("{{total}}", String(watch.preferencesTotal))}
                </p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={watch.refresh}
                aria-label={t("rlaif.refresh", { defaultValue: "Refresh" })}
                disabled={watch.loading}
              >
                <RefreshCw className={watch.loading ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
              </Button>
            </header>
            {prefsReversed.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                {t("rlaif.noPreferences", {
                  defaultValue: "No preference pairs yet. Approve a scanner proposal to start the dataset.",
                })}
              </p>
            ) : (
              <ul className="space-y-2">
                {prefsReversed.map((p) => {
                  const applyRaw = (p.metadata?.auto_apply ?? null) as
                    | boolean
                    | string
                    | null
                    | undefined;
                  const applyLabel =
                    applyRaw === true || applyRaw === "true"
                      ? { text: "applied", tone: "ok" as const }
                      : applyRaw === false || applyRaw === "false"
                        ? { text: "not applied", tone: "muted" as const }
                        : typeof applyRaw === "string" && applyRaw
                          ? { text: applyRaw, tone: "warn" as const }
                          : { text: "auto-apply off", tone: "muted" as const };
                  return (
                    <li
                      key={p._index}
                      className="rounded-lg border border-border/60 bg-background p-3"
                      data-testid="rlaif-pref-row"
                    >
                      <div className="mb-1 flex items-center justify-between gap-2">
                        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground/70">
                          {p.timestamp || `#${p._index}`}
                        </span>
                        <span className="font-mono text-[11px] text-muted-foreground">
                          {p.score_chosen.toFixed(2)} vs {p.score_rejected.toFixed(2)}
                        </span>
                      </div>
                      {p.task ? (
                        <p className="mb-1 line-clamp-2 text-sm text-foreground">{p.task}</p>
                      ) : null}
                      <div className="mb-1 flex flex-wrap items-center gap-1.5 text-[11px]">
                        <span
                          className={
                            applyLabel.tone === "ok"
                              ? "rounded-full bg-emerald-500/15 px-1.5 py-0.5 font-medium text-emerald-700 dark:text-emerald-300"
                              : applyLabel.tone === "warn"
                                ? "rounded-full bg-amber-500/15 px-1.5 py-0.5 font-medium text-amber-700 dark:text-amber-300"
                                : "rounded-full bg-muted px-1.5 py-0.5 font-medium text-muted-foreground"
                          }
                        >
                          {applyLabel.text}
                        </span>
                        {p.metadata?.winner_tests === false ? (
                          <span className="rounded-full bg-destructive/10 px-1.5 py-0.5 text-destructive">
                            tests failed
                          </span>
                        ) : null}
                        {p.metadata?.winner_lint === false ? (
                          <span className="rounded-full bg-destructive/10 px-1.5 py-0.5 text-destructive">
                            lint failed
                          </span>
                        ) : null}
                      </div>
                      {p.reason ? (
                        <p className="line-clamp-3 text-xs text-muted-foreground">{p.reason}</p>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          <section className="rounded-xl border border-border bg-card p-4">
            <header className="mb-2 flex items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-semibold text-foreground">
                  {t("rlaif.logTitle", { defaultValue: "Gateway log (filtered)" })}
                </h2>
                <p className="text-xs text-muted-foreground">
                  {t("rlaif.logCount", {
                    defaultValue: "{{total}} matching lines",
                  }).replace("{{total}}", String(watch.logTotal))}
                </p>
              </div>
              {watch.loading ? (
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              ) : null}
            </header>
            <div
              ref={logRef}
              data-testid="rlaif-log"
              className="h-72 overflow-y-auto rounded-md border border-border/60 bg-background p-2 font-mono text-[11.5px] leading-5"
            >
              {lastLog.length === 0 ? (
                <p className="px-2 py-1 text-xs text-muted-foreground">
                  {t("rlaif.noLog", {
                    defaultValue: "No RLAIF log lines yet.",
                  })}
                </p>
              ) : (
                <ul className="space-y-0.5">
                  {lastLog.map((line) => (
                    <li
                      key={`${line.line_no}`}
                      className="whitespace-pre-wrap break-words text-foreground/85"
                    >
                      {line.text}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <p className="mt-2 truncate text-[11px] text-muted-foreground" title={watch.logPath ?? ""}>
              {watch.logPath ?? t("rlaif.noLogPath", { defaultValue: "log path unavailable" })}
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
