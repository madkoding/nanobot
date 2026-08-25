import { useCallback, useEffect, useRef, useState } from "react";

import { useClient } from "@/providers/ClientProvider";
import {
  fetchRlaifLog,
  fetchRlaifPreferences,
  type RlaifLogItem,
  type RlaifLogPayload,
  type RlaifPreferenceItem,
  type RlaifPreferencesPayload,
} from "@/lib/api";

const POLL_INTERVAL_MS = 2000;
const MAX_LOG_LINES = 400;

export type RlaifWatchState = {
  preferences: RlaifPreferenceItem[];
  preferencesTotal: number;
  preferencesPath: string | null;
  log: RlaifLogItem[];
  logTotal: number;
  logPath: string | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
};

export function useRlaifWatch(): RlaifWatchState {
  const { token } = useClient();
  const [preferences, setPreferences] = useState<RlaifPreferenceItem[]>([]);
  const [preferencesTotal, setPreferencesTotal] = useState(0);
  const [preferencesPath, setPreferencesPath] = useState<string | null>(null);
  const [log, setLog] = useState<RlaifLogItem[]>([]);
  const [logTotal, setLogTotal] = useState(0);
  const [logPath, setLogPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const prefCursorRef = useRef<number>(-1);
  const logCursorRef = useRef<number>(-1);
  const tokenRef = useRef(token);
  tokenRef.current = token;

  const tick = useCallback(async () => {
    const tk = tokenRef.current;
    if (!tk) return;
    setLoading(true);
    try {
      const [pref, lg] = await Promise.all([
        fetchRlaifPreferences(tk, { sinceIndex: prefCursorRef.current }),
        fetchRlaifLog(tk, { sinceLine: logCursorRef.current, maxLines: MAX_LOG_LINES }),
      ]);
      applyPreferenceDelta(pref);
      applyLogDelta(lg);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const applyPreferenceDelta = useCallback((payload: RlaifPreferencesPayload) => {
    setPreferencesPath(payload.path);
    setPreferencesTotal(payload.total);
    if (payload.items.length > 0) {
      prefCursorRef.current = payload.next_index - 1;
      setPreferences((prev) => [...prev, ...payload.items]);
    } else if (prefCursorRef.current < 0 && payload.total > 0) {
      prefCursorRef.current = payload.total - 1;
    }
  }, []);

  const applyLogDelta = useCallback((payload: RlaifLogPayload) => {
    setLogPath(payload.path);
    setLogTotal(payload.total);
    if (payload.items.length > 0) {
      logCursorRef.current = payload.next_line;
      setLog((prev) => {
        const next = [...prev, ...payload.items];
        return next.length > MAX_LOG_LINES ? next.slice(-MAX_LOG_LINES) : next;
      });
    } else if (logCursorRef.current < 0 && payload.total > 0) {
      logCursorRef.current = payload.total - 1;
    }
  }, []);

  useEffect(() => {
    void tick();
    const id = window.setInterval(() => {
      void tick();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [tick]);

  return {
    preferences,
    preferencesTotal,
    preferencesPath,
    log,
    logTotal,
    logPath,
    loading,
    error,
    refresh: () => {
      void tick();
    },
  };
}
