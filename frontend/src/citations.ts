import type { Citation } from "./types";

const CITATION_PATTERN = /\[S(\d+)\]/g;

/** One piece of answer text: either plain text, or a citation marker. */
export interface ContentSegment {
  text: string;
  citationNumber: number | null;
}

export interface ParsedCitations {
  segments: ContentSegment[];
  numberBySourceId: Map<string, number>;
}

/**
 * Splits answer text on inline "[S3]"-style citation markers and assigns
 * each distinct source a sequential display number in order of first
 * mention, so the same number can be shown inline and on its evidence card.
 *
 * @param content - The assistant's answer text, possibly containing markers.
 * @param citations - The turn's verified citations; any whose source is
 *   not mentioned inline still receives a trailing display number.
 * @returns The text split into plain-text and citation-marker segments,
 *   plus a source-id-to-display-number map covering every citation.
 */
export function parseCitations(content: string, citations: Citation[]): ParsedCitations {
  const numberBySourceId = new Map<string, number>();
  const segments: ContentSegment[] = [];
  let lastIndex = 0;

  CITATION_PATTERN.lastIndex = 0;
  let match = CITATION_PATTERN.exec(content);
  while (match !== null) {
    const sourceId = `S${match[1]}`;
    if (!numberBySourceId.has(sourceId)) {
      numberBySourceId.set(sourceId, numberBySourceId.size + 1);
    }
    if (match.index > lastIndex) {
      segments.push({ text: content.slice(lastIndex, match.index), citationNumber: null });
    }
    segments.push({ text: "", citationNumber: numberBySourceId.get(sourceId) ?? null });
    lastIndex = CITATION_PATTERN.lastIndex;
    match = CITATION_PATTERN.exec(content);
  }
  if (lastIndex < content.length) {
    segments.push({ text: content.slice(lastIndex), citationNumber: null });
  }

  for (const citation of citations) {
    if (citation.source_id && !numberBySourceId.has(citation.source_id)) {
      numberBySourceId.set(citation.source_id, numberBySourceId.size + 1);
    }
  }

  return { segments, numberBySourceId };
}

/** One figure carried by a citation group (empty when the group backs non-numeric prose). */
export interface CitationValue {
  label: string | null;
  value_text: string;
  unit: string | null;
  period: string | null;
}

/**
 * One evidence card's worth of citations: every citation sharing a source,
 * merged into one card instead of one identical-looking card each.
 */
export interface CitationGroup {
  number: number | undefined;
  sourceId: string | undefined;
  report: string;
  page: string;
  quote: string;
  chunkText: string | undefined;
  verified: boolean;
  values: CitationValue[];
}

/**
 * Groups citations that share both a source and a quote into one evidence
 * card each.
 *
 * The model can cite the same source more than once in one answer (e.g. two
 * figures in the same sentence), which previously rendered as several cards
 * showing the identical quote under the identical citation number. Grouping
 * keeps that one number pointing at one card, with every figure it backs
 * listed together -- but only when they share the *same* quote.
 *
 * Two citations can share a source without sharing a quote: a source chunk
 * commonly holds several sentences or table cells, and the model may
 * legitimately quote a different one for each figure (a real case: one
 * chunk had adjacent "Internal employees ..." and "External employees ..."
 * table cells, each individually verbatim, backing different numbers).
 * Grouping those by source alone would keep only the first citation's
 * quote for the whole card -- silently showing an "Exact match" badge over
 * a quote that does not actually contain the other figures shown next to
 * it. Requiring the quote to match too keeps each card's quote honest for
 * every value in it, at the cost of two cards sharing one citation number
 * when their quotes genuinely differ -- both cards are still correctly
 * labelled with that same source's number, since they are both really
 * from it.
 *
 * @param citations - The turn's verified citations, in the order returned.
 * @param numberBySourceId - Display numbers from `parseCitations`, so
 *   groups without a source id still get a stable (undefined) position.
 * @returns One group per distinct (source, quote) pair, in order of first
 *   appearance.
 */
export function groupCitations(
  citations: Citation[],
  numberBySourceId: Map<string, number>,
): CitationGroup[] {
  const groups: CitationGroup[] = [];
  const groupByKey = new Map<string, CitationGroup>();

  for (const citation of citations) {
    const key = `${citation.source_id ?? `__ungrouped_${groups.length}`}::${citation.quote}`;
    let group = groupByKey.get(key);
    if (!group) {
      group = {
        number: citation.source_id ? numberBySourceId.get(citation.source_id) : undefined,
        sourceId: citation.source_id,
        report: citation.report,
        page: citation.page,
        quote: citation.quote,
        chunkText: citation.chunk_text,
        verified: citation.verified,
        values: [],
      };
      groupByKey.set(key, group);
      groups.push(group);
    } else {
      // Same quote, but a citation can still individually fail verification
      // (e.g. its value_text isn't actually inside the quote even though
      // the quote itself is verbatim) -- one unverified value should not
      // hide behind a card whose badge only reflects the first citation.
      group.verified = group.verified && citation.verified;
    }
    if (citation.value_text) {
      group.values.push({
        label: citation.label,
        value_text: citation.value_text,
        unit: citation.unit,
        period: citation.period,
      });
    }
  }

  return groups;
}
