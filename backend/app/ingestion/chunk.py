"""Splits parsed PDF content into retrieval-friendly chunks while keeping tables intact.

`text` is the canonical verbatim text used for checks against the source.
`embed_text` adds a context header so retrieval understands the chunk without
contaminating the stored quote text.
"""

import re
from dataclasses import dataclass

from app.ingestion.parse import ParsedDocument, TableItem, TextItem

TARGET_TOKENS = 450
HARD_MAX_TOKENS = 700
CHARS_PER_TOKEN = 4
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN
HARD_MAX_CHARS = HARD_MAX_TOKENS * CHARS_PER_TOKEN
OVERLAP_RATIO = 0.12
CAPTION_MAX_CHARS = 200

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class PreparedChunk:
    """One chunk ready to be stored and used for retrieval."""

    page_start: int
    page_end: int
    kind: str  # text | table
    text: str


def _page_label(page_start: int, page_end: int) -> str:
    """Formats a page range for display.

    Args:
        page_start: The first physical page.
        page_end: The last physical page.

    Returns:
        `"12"` when the range is a single page, otherwise `"12-13"`.
    """
    return str(page_start) if page_start == page_end else f"{page_start}-{page_end}"


def build_embed_text(company: str, year: int, page_start: int, page_end: int, text: str) -> str:
    """Prefixes chunk text with a context header for embedding.

    Args:
        company: The report's company name.
        year: The report's fiscal year.
        page_start: The chunk's first physical page.
        page_end: The chunk's last physical page.
        text: The chunk's canonical verbatim text.

    Returns:
        The header followed by `text`. Never stored as `chunks.text`, so it
        never contaminates a verbatim quote.
    """
    header = f"{company} | Annual Report {year} | page {_page_label(page_start, page_end)}"
    return f"{header}\n\n{text}"


def _caption_from_paragraph(paragraph: str) -> str:
    """Extracts a short caption line from the paragraph preceding a table.

    Args:
        paragraph: The paragraph immediately before a table in reading
            order.

    Returns:
        `paragraph` itself if short, otherwise its last sentence (or its
        last `CAPTION_MAX_CHARS` characters if no sentence boundary is
        found).
    """
    paragraph = paragraph.strip()
    if len(paragraph) <= CAPTION_MAX_CHARS:
        return paragraph
    parts = [p for p in _SENTENCE_SPLIT.split(paragraph) if p]
    tail = parts[-1] if parts else paragraph
    return tail[-CAPTION_MAX_CHARS:]


def _render_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Renders a table as GitHub-flavoured markdown.

    Args:
        header: The column headers.
        rows: The table's data rows.

    Returns:
        The table as a markdown string, with a header row followed by a
        separator row and then the data rows.
    """
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _split_long_paragraph(page: int, text: str) -> list[PreparedChunk]:
    """Splits one oversized paragraph into overlapping chunks.

    Used only when a single paragraph exceeds `HARD_MAX_CHARS` on its own;
    normal chunk boundaries fall on paragraph boundaries and need no
    overlap.

    Args:
        page: The physical page the paragraph is on.
        text: The paragraph's canonical verbatim text.

    Returns:
        One or more text chunks covering the whole paragraph, each with a
        small overlap to preserve continuity between split pieces.
    """
    overlap_chars = int(HARD_MAX_CHARS * OVERLAP_RATIO)
    prepared_chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + HARD_MAX_CHARS, n)
        if end < n:
            next_space = text.find(" ", end)
            if next_space != -1 and next_space - end < 50:
                end = next_space
        piece = text[start:end].strip()
        if piece:
            prepared_chunks.append(PreparedChunk(page, page, "text", piece))
        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)
    return prepared_chunks


def _paragraph_buffer_length(paragraph_buffer: list[tuple[int, str]]) -> int:
    """Computes the length of the paragraphs in `paragraph_buffer` once joined.

    Args:
        paragraph_buffer: Consecutive paragraphs that have not yet been
            flushed into a chunk.

    Returns:
        The character length of `"\\n\\n".join(text for _, text in paragraph_buffer)`,
        computed without building that string.
    """
    if not paragraph_buffer:
        return 0
    return sum(len(text) for _, text in paragraph_buffer) + 2 * (len(paragraph_buffer) - 1)


def _flush_paragraph_buffer(paragraph_buffer: list[tuple[int, str]]) -> PreparedChunk | None:
    """Builds one text chunk from a buffered paragraph run.

    Args:
        paragraph_buffer: Paragraphs to pack, each as `(page, text)`, in order.

    Returns:
        A text chunk spanning from the first paragraph's page to the last,
        or None if there are no paragraphs to flush.
    """
    if not paragraph_buffer:
        return None
    page_start = paragraph_buffer[0][0]
    page_end = paragraph_buffer[-1][0]
    text = "\n\n".join(paragraph_text for _, paragraph_text in paragraph_buffer)
    return PreparedChunk(page_start, page_end, "text", text)


def _pack_text_items(paragraph_buffer: list[tuple[int, str]]) -> list[PreparedChunk]:
    """Packs consecutive paragraphs into chunks up to `TARGET_CHARS`.

    Paragraphs are combined until adding the next one would exceed
    `TARGET_CHARS`; a paragraph that on its own exceeds `HARD_MAX_CHARS` is
    split instead (see `_split_long_paragraph`).

    Args:
        paragraph_buffer: Paragraphs to pack, each as `(page, text)`, in
            reading order.

    Returns:
        The resulting text chunks, in order.
    """
    prepared_chunks = []
    active_paragraphs = []

    for page, text in paragraph_buffer:
        if len(text) > HARD_MAX_CHARS:
            flushed = _flush_paragraph_buffer(active_paragraphs)
            if flushed is not None:
                prepared_chunks.append(flushed)
            active_paragraphs = []
            prepared_chunks.extend(_split_long_paragraph(page, text))
            continue

        would_exceed_target = (
            active_paragraphs
            and _paragraph_buffer_length(active_paragraphs) + 2 + len(text) > TARGET_CHARS
        )
        if would_exceed_target:
            # Flush the current paragraph run before adding a larger one so each
            # chunk stays close to the target size without breaking paragraph
            # boundaries unnecessarily.
            flushed = _flush_paragraph_buffer(active_paragraphs)
            if flushed is not None:
                prepared_chunks.append(flushed)
            active_paragraphs = []
        active_paragraphs.append((page, text))

    flushed = _flush_paragraph_buffer(active_paragraphs)
    if flushed is not None:
        prepared_chunks.append(flushed)
    return prepared_chunks


def _table_chunk_text(header: list[str], rows: list[list[str]], caption: str | None) -> str:
    """Renders a table (or a partial slice of its rows) as chunk text.

    Args:
        header: The table's column headers, repeated in every part of a
            split table.
        rows: The rows to render (the full table, or one part of a split).
        caption: The preceding paragraph's caption line, or None. Only
            meaningful for the first part of a split table; callers pass
            None for later parts.

    Returns:
        The caption followed by the markdown table, or just the markdown
        table if there is no caption.
    """
    markdown = _render_markdown_table(header, rows)
    return f"{caption}\n{markdown}" if caption else markdown


def _chunk_table(item: TableItem, caption: str | None) -> list[PreparedChunk]:
    """Turns one parsed table into one or more prepared table chunks.

    The table is never split unless it exceeds `HARD_MAX_CHARS`, in which
    case it is split by rows with the header repeated in every part.

    Args:
        item: The parsed table, with its header, rows and page number.
        caption: The preceding paragraph's caption line, or None. Prefixed
            only to the first chunk produced.

    Returns:
        One table chunk, or several if the table had to be split.
    """
    whole_text = _table_chunk_text(item.header, item.rows, caption)
    if len(whole_text) <= HARD_MAX_CHARS or not item.rows:
        return [PreparedChunk(item.page, item.page, "table", whole_text)]

    prepared_chunks = []
    current_rows = []
    for row in item.rows:
        candidate_rows = [*current_rows, row]
        candidate_caption = caption if not prepared_chunks else None
        candidate_text = _table_chunk_text(item.header, candidate_rows, candidate_caption)
        if len(candidate_text) > HARD_MAX_CHARS and current_rows:
            flushed_caption = caption if not prepared_chunks else None
            flushed_text = _table_chunk_text(item.header, current_rows, flushed_caption)
            prepared_chunks.append(PreparedChunk(item.page, item.page, "table", flushed_text))
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        flushed_caption = caption if not prepared_chunks else None
        flushed_text = _table_chunk_text(item.header, current_rows, flushed_caption)
        prepared_chunks.append(PreparedChunk(item.page, item.page, "table", flushed_text))
    return prepared_chunks


def _flush_paragraph_run(
    prepared_chunks: list[PreparedChunk], paragraph_buffer: list[tuple[int, str, int]]
) -> list[tuple[int, str, int]]:
    """Packs a buffered paragraph run into chunks and appends them in place.

    Args:
        prepared_chunks: The chunks accumulated so far; extended in place
            with whatever `paragraph_buffer` packs into.
        paragraph_buffer: Consecutive same-region paragraphs to flush, each
            as `(page, text, region_index)`.

    Returns:
        A fresh empty buffer, for the caller to keep accumulating into.
    """
    if not paragraph_buffer:
        return paragraph_buffer
    page_and_text = [(page, text) for page, text, _ in paragraph_buffer]
    prepared_chunks.extend(_pack_text_items(page_and_text))
    return []


def chunk_document(parsed: ParsedDocument) -> list[PreparedChunk]:
    """Turns a parsed document into retrieval-ready chunks.

    Paragraphs are packed together up to `TARGET_CHARS`; each table is kept
    whole (or split by rows if oversized) and prefixed with the paragraph
    that precedes it as a caption. A region change is treated as a natural
    chunk boundary so left-column and right-column content do not collapse
    into a single annual-report paragraph block.

    Args:
        parsed: The document's parsed text and table items, in reading
            order.

    Returns:
        The prepared chunks, in document order.
    """
    prepared_chunks: list[PreparedChunk] = []
    paragraph_buffer: list[tuple[int, str, int]] = []

    for item in parsed.items:
        if isinstance(item, TextItem):
            if paragraph_buffer and paragraph_buffer[-1][2] != item.region_index:
                paragraph_buffer = _flush_paragraph_run(prepared_chunks, paragraph_buffer)
            paragraph_buffer.append((item.page, item.text, item.region_index))
            continue

        if paragraph_buffer:
            caption = _caption_from_paragraph(paragraph_buffer[-1][1])
            paragraph_buffer = _flush_paragraph_run(prepared_chunks, paragraph_buffer)
            prepared_chunks.extend(_chunk_table(item, caption))
            continue

        prepared_chunks.extend(_chunk_table(item, None))

    _flush_paragraph_run(prepared_chunks, paragraph_buffer)
    return prepared_chunks
