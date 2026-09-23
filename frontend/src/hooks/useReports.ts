import { useCallback, useEffect, useRef, useState } from "react";

import { listReports } from "../api";
import type { ReportSummary } from "../types";

const POLL_INTERVAL_MS = 2000;

function hasProcessingReport(reports: ReportSummary[]): boolean {
  return reports.some((report) => report.status === "processing");
}

/**
 * Loads the report list and keeps polling it every `POLL_INTERVAL_MS` while
 * any report is still `processing`, so upload progress updates live.
 *
 * @returns The current reports, a loading flag, and a manual refresh function.
 */
export function useReports(): {
  reports: ReportSummary[];
  loading: boolean;
  refresh: () => Promise<void>;
} {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  const refresh = useCallback(async () => {
    const next = await listReports();
    setReports(next);
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    return () => clearTimeout(timerRef.current);
  }, [refresh]);

  useEffect(() => {
    if (!hasProcessingReport(reports)) {
      return;
    }
    timerRef.current = setTimeout(refresh, POLL_INTERVAL_MS);
    return () => clearTimeout(timerRef.current);
  }, [reports, refresh]);

  return { reports, loading, refresh };
}
