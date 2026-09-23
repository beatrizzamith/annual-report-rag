import { useEffect, useState } from "react";

import { deleteReport, getExtractions } from "../api";
import Badge from "../components/Badge";
import GoalsTable from "../components/GoalsTable";
import SortableHeader, { type SortDirection } from "../components/SortableHeader";
import UploadForm from "../components/UploadForm";
import { useReports } from "../hooks/useReports";
import type { ExtractionItem, FtePayload, ReportSummary } from "../types";

type ReportSortKey = "company" | "fiscal_year" | "page_count" | "status";

/** Compares two reports by the active sort key; missing page counts sort last either way. */
function compareReports(a: ReportSummary, b: ReportSummary, key: ReportSortKey): number {
  switch (key) {
    case "company":
      return a.company.localeCompare(b.company);
    case "fiscal_year":
      return a.fiscal_year - b.fiscal_year;
    case "page_count":
      if (a.page_count == null) return b.page_count == null ? 0 : 1;
      if (b.page_count == null) return -1;
      return a.page_count - b.page_count;
    case "status":
      return a.status.localeCompare(b.status);
    default:
      return 0;
  }
}

/** Status cell: a live progress line while processing, or a terminal state. */
function StatusCell({ report }: { report: ReportSummary }) {
  if (report.status === "ready") {
    return <Badge tone="success">Ready</Badge>;
  }
  if (report.status === "failed") {
    return (
      <span title={report.error ?? undefined}>
        <Badge tone="danger">Failed — upload again to retry</Badge>
      </span>
    );
  }
  const percent = Math.round(report.progress * 100);
  return (
    <div className="flex items-center gap-2">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-500" />
      </span>
      <span className="text-gray-500 text-sm capitalize">
        {report.stage ?? "processing"}… {percent}%
      </span>
    </div>
  );
}

/** Extracted-data summary cell: FTE presence and goal count, once ready. */
function ExtractedCell({ report }: { report: ReportSummary }) {
  if (report.status !== "ready") {
    return <span className="text-gray-300">—</span>;
  }
  const fteLabel = report.extraction_counts.fte > 0 ? "FTE" : "no FTE";
  return (
    <span className="text-gray-600">
      {fteLabel}, {report.extraction_counts.sustainability_goal} goals
    </span>
  );
}

/** The FTE workforce card: value, metric type, verbatim quote and page. */
function FteCard({ item }: { item: ExtractionItem }) {
  const payload = item.payload as FtePayload;
  const metricLabel = payload.metric.replace(/_/g, " ");
  return (
    <div className="rounded-lg border border-gray-200 p-4 space-y-2 bg-gradient-to-br from-white to-gray-50/50">
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-semibold text-gray-900">{payload.value_text ?? "—"}</span>
        <span className="text-xs text-gray-400">
          {metricLabel}
          {payload.scope ? ` · ${payload.scope}` : ""}
        </span>
      </div>
      <blockquote className="border-l-2 border-brand-200 pl-3 italic text-gray-500 text-sm">
        &ldquo;{item.quote}&rdquo;
      </blockquote>
      <div className="text-xs text-gray-400">p.{item.page}</div>
    </div>
  );
}

/** Empty state shown before any report has been uploaded. */
function EmptyState() {
  return (
    <div className="text-center py-14 text-gray-400">
      <div className="text-3xl mb-2">📄</div>
      <p className="text-sm">No reports uploaded yet. Upload a PDF above to get started.</p>
    </div>
  );
}

/** Reports page: upload files, track ingestion progress, and review extracted data. */
export default function ReportsPage() {
  const { reports, loading, refresh } = useReports();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [extractions, setExtractions] = useState<ExtractionItem[]>([]);
  const [sortKey, setSortKey] = useState<ReportSortKey>("company");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");

  useEffect(() => {
    if (selectedId === null) {
      setExtractions([]);
      return;
    }
    getExtractions(selectedId).then(setExtractions);
  }, [selectedId]);

  useEffect(() => {
    if (selectedId !== null) {
      return;
    }
    const firstReady = reports.find((report) => report.status === "ready");
    if (firstReady) {
      setSelectedId(firstReady.id);
    }
  }, [reports, selectedId]);

  const selectedReport = reports.find((report) => report.id === selectedId) ?? null;
  const fteItem = extractions.find((item) => item.kind === "fte");
  const goalItems = extractions.filter((item) => item.kind === "sustainability_goal");

  const sortedReports = [...reports].sort((a, b) => {
    const comparison = compareReports(a, b, sortKey);
    return sortDirection === "asc" ? comparison : -comparison;
  });

  function handleSort(key: ReportSortKey) {
    if (key === sortKey) {
      setSortDirection((previous) => (previous === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDirection("asc");
    }
  }

  async function handleDelete(event: React.MouseEvent, reportId: number) {
    event.stopPropagation();
    const confirmed = window.confirm(
      "Delete this report? Its chunks and extracted data are removed so the file can be re-uploaded and re-tested from scratch.",
    );
    if (!confirmed) {
      return;
    }
    await deleteReport(reportId);
    if (selectedId === reportId) {
      setSelectedId(null);
    }
    refresh();
  }

  return (
    <div className="space-y-6">
      <UploadForm onUploaded={refresh} />

      {loading ? (
        <p className="text-sm text-gray-400">Loading reports…</p>
      ) : reports.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="rounded-xl border border-gray-200 bg-white shadow-card overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50/80 text-left text-xs text-gray-400 border-b border-gray-100">
              <tr>
                <SortableHeader
                  label="Company"
                  sortKey="company"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                  className="px-4 py-3 font-medium"
                />
                <SortableHeader
                  label="Year"
                  sortKey="fiscal_year"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                  className="px-4 py-3 font-medium"
                />
                <SortableHeader
                  label="Pages"
                  sortKey="page_count"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                  className="px-4 py-3 font-medium"
                />
                <SortableHeader
                  label="Status"
                  sortKey="status"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                  className="px-4 py-3 font-medium"
                />
                <th className="px-4 py-3 font-medium">Extracted</th>
                <th className="px-4 py-3 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {sortedReports.map((report) => (
                <tr
                  key={report.id}
                  onClick={() => setSelectedId(report.id)}
                  className={`border-t border-gray-100 cursor-pointer transition-colors ${
                    selectedId === report.id ? "bg-brand-50/60" : "hover:bg-gray-50"
                  }`}
                >
                  <td className="px-4 py-3 font-medium text-gray-800">{report.company}</td>
                  <td className="px-4 py-3 text-gray-600">{report.fiscal_year}</td>
                  <td className="px-4 py-3 text-gray-600">{report.page_count ?? "—"}</td>
                  <td className="px-4 py-3">
                    <StatusCell report={report} />
                  </td>
                  <td className="px-4 py-3">
                    <ExtractedCell report={report} />
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={(event) => handleDelete(event, report.id)}
                      className="text-gray-300 hover:text-rose-600 transition-colors"
                      title="Delete report"
                    >
                      🗑
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selectedReport && selectedReport.status === "ready" && (
        <div className="rounded-xl border border-gray-200 bg-white shadow-card p-5 space-y-5">
          <h2 className="font-semibold text-gray-800">
            {selectedReport.company} — Annual Report {selectedReport.fiscal_year}
          </h2>

          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2.5">
              Workforce
            </h3>
            {fteItem ? (
              <FteCard item={fteItem} />
            ) : (
              <p className="text-sm text-gray-400 italic">No FTE figure extracted for this report.</p>
            )}
          </section>

          <section>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2.5">
              Sustainability goals
            </h3>
            <GoalsTable goals={goalItems} />
          </section>
        </div>
      )}
    </div>
  );
}
