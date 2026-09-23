import { useMemo, useState } from "react";

import Badge from "./Badge";
import SortableHeader, { type SortDirection } from "./SortableHeader";
import type { ExtractionItem, GoalPayload } from "../types";

const CATEGORY_LABELS: Record<string, string> = {
  climate: "Climate",
  energy: "Energy",
  circularity_waste: "Circularity & waste",
  water: "Water",
  biodiversity: "Biodiversity",
  social_people: "Social & people",
  governance: "Governance",
  other: "Other",
};

type SortKey = "category" | "title" | "target_year" | "page";

const ALL = "all";

/** Renders one sustainability-goal row, with its verbatim quote and page. */
function GoalRow({ item }: { item: ExtractionItem }) {
  const payload = item.payload as GoalPayload;
  return (
    <tr className="border-t border-gray-100 align-top hover:bg-gray-50/60 transition-colors">
      <td className="px-3 py-3 whitespace-nowrap">
        <Badge tone="info">{CATEGORY_LABELS[payload.category] ?? payload.category}</Badge>
      </td>
      <td className="px-3 py-3 max-w-sm">
        <div className="font-medium text-gray-800">{payload.title}</div>
        <div className="text-gray-400 italic text-xs mt-1 leading-relaxed">
          &ldquo;{item.quote}&rdquo;
        </div>
      </td>
      <td className="px-3 py-3 text-gray-600">{payload.target ?? "—"}</td>
      <td className="px-3 py-3 text-gray-600">{payload.target_year ?? "—"}</td>
      <td className="px-3 py-3 text-gray-600">{item.page}</td>
    </tr>
  );
}

/**
 * Compares two goals by the active sort key, nulls (e.g. no `target_year`)
 * always last regardless of direction so filtering out "—" rows to the
 * bottom doesn't flip depending on which way the user sorted.
 */
function compareGoals(a: ExtractionItem, b: ExtractionItem, key: SortKey): number {
  const payloadA = a.payload as GoalPayload;
  const payloadB = b.payload as GoalPayload;
  switch (key) {
    case "category": {
      const labelA = CATEGORY_LABELS[payloadA.category] ?? payloadA.category;
      const labelB = CATEGORY_LABELS[payloadB.category] ?? payloadB.category;
      return labelA.localeCompare(labelB);
    }
    case "title":
      return payloadA.title.localeCompare(payloadB.title);
    case "target_year": {
      if (payloadA.target_year == null) return payloadB.target_year == null ? 0 : 1;
      if (payloadB.target_year == null) return -1;
      return payloadA.target_year - payloadB.target_year;
    }
    case "page":
      return a.page - b.page;
    default:
      return 0;
  }
}

/** Table of a report's extracted sustainability goals: filterable, sortable. */
export default function GoalsTable({ goals }: { goals: ExtractionItem[] }) {
  const [categoryFilter, setCategoryFilter] = useState<string>(ALL);
  const [yearFilter, setYearFilter] = useState<string>(ALL);
  const [sortKey, setSortKey] = useState<SortKey>("category");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");

  const categories = useMemo(() => {
    const present = new Set(goals.map((g) => (g.payload as GoalPayload).category));
    return [...present].sort((a, b) => (CATEGORY_LABELS[a] ?? a).localeCompare(CATEGORY_LABELS[b] ?? b));
  }, [goals]);

  const years = useMemo(() => {
    const present = new Set(
      goals
        .map((g) => (g.payload as GoalPayload).target_year)
        .filter((year): year is number => year != null),
    );
    return [...present].sort((a, b) => a - b);
  }, [goals]);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDirection((previous) => (previous === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDirection("asc");
    }
  }

  const visible = goals
    .filter((g) => categoryFilter === ALL || (g.payload as GoalPayload).category === categoryFilter)
    .filter(
      (g) => yearFilter === ALL || String((g.payload as GoalPayload).target_year ?? "") === yearFilter,
    )
    .sort((a, b) => {
      const comparison = compareGoals(a, b, sortKey);
      return sortDirection === "asc" ? comparison : -comparison;
    });

  if (goals.length === 0) {
    return (
      <p className="text-sm text-gray-400 italic">No sustainability goals extracted for this report.</p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5 text-gray-500">
          Category
          <select
            value={categoryFilter}
            onChange={(event) => setCategoryFilter(event.target.value)}
            className="border border-gray-200 rounded-md px-2 py-1 text-gray-700 outline-none focus:ring-2 focus:ring-brand-500/30"
          >
            <option value={ALL}>All</option>
            {categories.map((category) => (
              <option key={category} value={category}>
                {CATEGORY_LABELS[category] ?? category}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-gray-500">
          Year
          <select
            value={yearFilter}
            onChange={(event) => setYearFilter(event.target.value)}
            className="border border-gray-200 rounded-md px-2 py-1 text-gray-700 outline-none focus:ring-2 focus:ring-brand-500/30"
          >
            <option value={ALL}>All</option>
            {years.map((year) => (
              <option key={year} value={String(year)}>
                {year}
              </option>
            ))}
          </select>
        </label>
        {(categoryFilter !== ALL || yearFilter !== ALL) && (
          <span className="text-gray-400">
            {visible.length} of {goals.length} goals
          </span>
        )}
      </div>

      {visible.length === 0 ? (
        <p className="text-sm text-gray-400 italic">No goals match these filters.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-gray-100">
          <table className="w-full text-sm bg-white">
            <thead>
              <tr className="text-left text-xs text-gray-400 border-b border-gray-100 bg-gray-50/60">
                <SortableHeader
                  label="Category"
                  sortKey="category"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                />
                <SortableHeader
                  label="Goal"
                  sortKey="title"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                />
                <th className="px-3 py-2.5 font-medium">Target</th>
                <SortableHeader
                  label="Year"
                  sortKey="target_year"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                />
                <SortableHeader
                  label="Page"
                  sortKey="page"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                />
              </tr>
            </thead>
            <tbody>
              {visible.map((goal) => (
                <GoalRow key={goal.id} item={goal} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
