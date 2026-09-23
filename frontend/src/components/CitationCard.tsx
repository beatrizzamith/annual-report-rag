import { Fragment, useState } from "react";

import Badge from "./Badge";
import type { CitationGroup, CitationValue } from "../citations";

/** Small numbered circle matching the citation marker cited inline above. */
function CitationNumber({ number }: { number: number }) {
  return (
    <span className="inline-flex items-center justify-center h-4 w-4 rounded-full bg-brand-100 text-brand-700 text-[10px] font-semibold shrink-0">
      {number}
    </span>
  );
}

/**
 * Splits a chunk's full text around a quote, for highlighting, if the quote
 * appears in it verbatim (by plain JS string search, not the server's
 * normalised comparison, so this can miss on whitespace/quote-style
 * differences even for a server-verified quote -- falls back to no
 * highlighting rather than guessing).
 *
 * @param chunkText - The cited chunk's full text.
 * @param quote - The citation's quote to locate within it.
 * @returns `[before, match, after]`, or null if `quote` isn't found as a
 *   literal substring.
 */
function splitAroundQuote(chunkText: string, quote: string): [string, string, string] | null {
  const index = chunkText.indexOf(quote);
  if (index === -1) {
    return null;
  }
  return [chunkText.slice(0, index), chunkText.slice(index, index + quote.length), chunkText.slice(index + quote.length)];
}

/**
 * Splits a quote into plain and "highlighted" segments around every value's
 * `value_text`, for emphasising the figure(s) a verified quote backs.
 *
 * Only meaningful to call when the group is verified: verification already
 * requires every value's `value_text` to be a literal substring of the
 * quote (see `verify_value_in_quote`), so a verified group's values are
 * always findable here -- this never has to fall back to "not found".
 *
 * @param quote - The citation's verbatim quote.
 * @param values - The figures this quote backs, each highlighted where its
 *   `value_text` appears.
 * @returns Segments covering the whole quote in order, each marked whether
 *   it falls inside one of the values' matched spans.
 */
function highlightValues(
  quote: string,
  values: CitationValue[],
): { text: string; highlighted: boolean }[] {
  const spans: [number, number][] = [];
  for (const value of values) {
    const index = quote.indexOf(value.value_text);
    if (index !== -1) {
      spans.push([index, index + value.value_text.length]);
    }
  }
  if (spans.length === 0) {
    return [{ text: quote, highlighted: false }];
  }
  spans.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [spans[0]];
  for (const [start, end] of spans.slice(1)) {
    const last = merged[merged.length - 1];
    if (start <= last[1]) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }

  const segments: { text: string; highlighted: boolean }[] = [];
  let cursor = 0;
  for (const [start, end] of merged) {
    if (start > cursor) {
      segments.push({ text: quote.slice(cursor, start), highlighted: false });
    }
    segments.push({ text: quote.slice(start, end), highlighted: true });
    cursor = end;
  }
  if (cursor < quote.length) {
    segments.push({ text: quote.slice(cursor), highlighted: false });
  }
  return segments;
}

/**
 * Shows one citation group: the source report and page, the quote the model
 * gave for it, and whether that quote matches the source text exactly. The
 * extracted label/value (`group.values`) is deliberately not restated as
 * its own line here -- this card is about grounding the claim in its
 * source, not repeating the figure a reader can already see in the answer
 * text above it. Where a value's text does appear, inside a *verified*
 * quote, it's bolded in place instead: a lighter-weight way to show which
 * specific figure(s) the quote backs without a redundant value line.
 *
 * The quote is always shown, matched or not: an exact match means it is
 * confirmed word-for-word in the source; not matching does not mean the
 * underlying figure is wrong, only that this exact wording could not be
 * found verbatim (it may be paraphrased, or drawn from elsewhere in the
 * source) -- withholding the quote in that case would hide the one thing
 * that lets a reader judge for themselves, which is worse than showing an
 * unconfirmed one.
 *
 * A group is every citation that shares one source *and* one quote, so two
 * figures quoted from the same sentence render as one card instead of two
 * cards repeating the identical quote under the identical number.
 *
 * "Show full source" reveals the cited chunk's whole text, not just the
 * one quoted sentence -- useful when a chunk backs several separate
 * citation cards (e.g. one bullet-list page becomes one card per bullet),
 * so a reader can see them together in their original context instead of
 * as disconnected fragments, and judge an unmatched quote against the
 * surrounding text rather than just the isolated (possibly mismatched)
 * excerpt.
 *
 * Deliberately smaller than the answer text above it: this is supporting
 * evidence, not the main content, and shouldn't visually compete with it.
 */
export default function CitationCard({ group, number }: { group: CitationGroup; number?: number }) {
  const [showFullSource, setShowFullSource] = useState(false);
  const highlighted = group.chunkText ? splitAroundQuote(group.chunkText, group.quote) : null;

  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-card p-3 space-y-1.5 text-xs">
      <div className="flex justify-between items-start gap-3">
        <span className="font-medium text-gray-600 flex items-center gap-1.5">
          {number !== undefined && <CitationNumber number={number} />}
          Cited source
        </span>
        <span className="text-gray-400 shrink-0 whitespace-nowrap">
          {group.report} · p.{group.page}
        </span>
      </div>
      <blockquote
        className={`border-l-2 pl-2.5 italic text-gray-500 ${
          group.verified ? "border-brand-200" : "border-amber-200"
        }`}
      >
        &ldquo;
        {group.verified && group.values.length > 0
          ? highlightValues(group.quote, group.values).map((segment, index) =>
              segment.highlighted ? (
                <strong key={index} className="font-semibold text-gray-700 not-italic">
                  {segment.text}
                </strong>
              ) : (
                <Fragment key={index}>{segment.text}</Fragment>
              ),
            )
          : group.quote}
        &rdquo;
      </blockquote>
      <div className="flex items-center gap-2.5">
        {group.verified ? (
          <span title="This exact wording was found word-for-word in the source text.">
            <Badge tone="success">✓ Exact match</Badge>
          </span>
        ) : (
          <span title="This exact wording wasn't found in the source text as printed above -- it may be paraphrased, or drawn from a different part of the source. Not a sign the figure itself is wrong.">
            <Badge tone="warning">⚠ Not an exact match</Badge>
          </span>
        )}
        {group.chunkText && (
          <button
            type="button"
            onClick={() => setShowFullSource((previous) => !previous)}
            className="text-gray-400 hover:text-gray-600 underline decoration-dotted underline-offset-2 transition-colors"
          >
            {showFullSource ? "Hide full source" : "Show full source"}
          </button>
        )}
      </div>
      {showFullSource && group.chunkText && (
        <div className="mt-1 rounded-md bg-gray-50 border border-gray-100 p-2.5 text-gray-600 leading-relaxed max-h-64 overflow-y-auto">
          {highlighted ? (
            <>
              {highlighted[0]}
              <mark className="bg-brand-100 text-brand-900 rounded px-0.5">{highlighted[1]}</mark>
              {highlighted[2]}
            </>
          ) : (
            group.chunkText
          )}
        </div>
      )}
    </div>
  );
}
