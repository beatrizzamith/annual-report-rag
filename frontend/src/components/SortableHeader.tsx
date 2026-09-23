export type SortDirection = "asc" | "desc";

/**
 * A `<th>` whose label is a button that toggles sorting by this column.
 *
 * Generic over the caller's own sort-key union (e.g. `"company" | "year"`)
 * so each table gets type-checked sort keys without this component needing
 * to know what they are.
 */
export default function SortableHeader<Key extends string>({
  label,
  sortKey,
  activeKey,
  direction,
  onSort,
  className = "px-3 py-2.5 font-medium",
}: {
  label: string;
  sortKey: Key;
  activeKey: Key | null;
  direction: SortDirection;
  onSort: (key: Key) => void;
  /** Overrides the `<th>`'s padding/weight classes to match the caller's table. */
  className?: string;
}) {
  const active = activeKey === sortKey;
  return (
    <th className={className}>
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className={`flex items-center gap-1 transition-colors hover:text-gray-700 ${
          active ? "text-gray-700" : ""
        }`}
      >
        {label}
        <span className={`text-[9px] ${active ? "opacity-100" : "opacity-0"}`}>
          {direction === "asc" ? "▲" : "▼"}
        </span>
      </button>
    </th>
  );
}
