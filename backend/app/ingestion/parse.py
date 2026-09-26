"""Parses PDF pages into text and table items while keeping page boundaries intact.

Uses PyMuPDF. Text blocks and tables are merged into a reading-order stream
per page, and table bounding boxes are excluded from plain-text blocks so
numbers are not duplicated.
"""

import logging
import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pymupdf
from pydantic import BaseModel, Field

from app.core.text import normalize_text

logger = logging.getLogger(__name__)

MIN_PAGE_CHARS = 50
_OVERLAP_THRESHOLD = 0.5  # fraction of a text block's area inside a table bbox to drop it

# `find_tables()` is CPU-bound and holds the GIL (threads gave no speedup), so
# large reports parse in worker processes. Below this many pages, process
# start-up costs more than it saves.
_MIN_PAGES_FOR_PARALLEL = 20
_PAGES_PER_TASK = 5  # amortises each worker's one-time `pymupdf.open()` over several pages
_MAX_WORKERS = 8  # bounds memory: each worker holds its own open copy of the PDF

# A block that repeats verbatim on most pages is running-header/footer or
# side-navigation chrome (e.g. a left-rail chapter menu), not page content.
_BOILERPLATE_PAGE_FRACTION = 0.5
_BOILERPLATE_MIN_PAGES = 10

# Gap (in points) between blocks' left edges beyond which they count as
# separate columns. In-column indentation (bullets, quotes) stays below it.
_COLUMN_GAP_THRESHOLD = 40.0


class TextItem(BaseModel):
    """One paragraph of body text, in reading order."""

    page: int
    text: str = Field(description="One cleaned paragraph, with no embedded newlines.")
    x0: float = 0.0
    y0: float = 0.0
    region_index: int = 0


class TableItem(BaseModel):
    """One detected table, in reading order."""

    page: int
    header: list[str]
    rows: list[list[str]]
    x0: float = 0.0
    y0: float = 0.0
    region_index: int = 0


ParsedItem = TextItem | TableItem


class ParsedDocument(BaseModel):
    """The full output of parsing one PDF."""

    items: list[ParsedItem]
    page_count: int
    skipped_pages: list[int]


def _rect_area(rect: pymupdf.Rect) -> float:
    """Computes a rectangle's area, treating a degenerate one as zero.

    Args:
        rect: A rectangle, possibly empty or degenerate.

    Returns:
        The rectangle's area, or 0 if its width or height is negative.
    """
    return max(0.0, rect.width) * max(0.0, rect.height)


def _overlap_fraction(block_rect: pymupdf.Rect, table_rect: pymupdf.Rect) -> float:
    """Computes how much of a text block's area a table's bbox covers.

    Args:
        block_rect: A text block's bounding box.
        table_rect: A detected table's bounding box.

    Returns:
        The fraction of `block_rect`'s area that lies inside `table_rect`,
        in [0, 1]. 0 if `block_rect` has zero area.
    """
    inter = block_rect & table_rect
    block_area = _rect_area(block_rect)
    if block_area == 0:
        return 0.0
    return _rect_area(inter) / block_area


def _clean_paragraph(lines: list[str]) -> str:
    """Joins a text block's lines into one normalised, single-line paragraph.

    Args:
        lines: A text block's lines, in reading order, as PDF-extracted
            strings (still ligatures, curly quotes, wrapped mid-word, etc.).

    Returns:
        The lines normalised (see `normalize_text`) and joined into one
        single-line paragraph with collapsed whitespace.
    """
    raw = "\n".join(lines)
    cleaned = normalize_text(raw)
    single_line = " ".join(cleaned.split("\n"))
    return " ".join(single_line.split())


def _block_text(block: dict) -> str:
    """Extracts and cleans a PyMuPDF text block's text.

    Args:
        block: One `type == 0` (text) block from PyMuPDF's
            `page.get_text("dict")` output.

    Returns:
        The block's text as one cleaned paragraph.
    """
    lines = []
    for line in block.get("lines", []):
        spans_text = "".join(span.get("text", "") for span in line.get("spans", []))
        if spans_text.strip():
            lines.append(spans_text)
    return _clean_paragraph(lines)


def _extract_table_rows(table: "pymupdf.table.Table") -> tuple[list[str], list[list[str]]] | None:
    """Extracts and normalises a detected table's header and body rows.

    Args:
        table: A table detected by `page.find_tables()`.

    Returns:
        A `(header, body_rows)` tuple with every cell normalised, or None if
        the table extracted to no usable content.
    """
    raw_rows = table.extract()
    if not raw_rows:
        return None
    clean_rows = [
        [normalize_text(cell or "").replace("\n", " ").replace("|", "/").strip() for cell in row]
        for row in raw_rows
    ]
    header, *body = clean_rows
    body = [row for row in body if any(cell for cell in row)]
    if not any(header) and not body:
        return None
    return header, body


def _column_starts(x0_values: list[float]) -> list[float]:
    """Finds column boundaries from a page's blocks' left edges.

    Sorts the x0 values and starts a new column wherever consecutive values
    are more than `_COLUMN_GAP_THRESHOLD` apart, so multi-column pages (e.g.
    a narrow side-nav plus several body columns) are split by their actual
    layout instead of a fixed fraction of page width.

    Args:
        x0_values: Every block's left x-coordinate on one page.

    Returns:
        The left edge of each detected column, ascending. `[0.0]` if
        `x0_values` is empty.
    """
    if not x0_values:
        return [0.0]
    ordered = sorted(x0_values)
    starts = [ordered[0]]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current - previous > _COLUMN_GAP_THRESHOLD:
            starts.append(current)
    return starts


def _region_for(x0: float, column_starts: list[float]) -> int:
    """Finds which detected column a block belongs to.

    Args:
        x0: The block's left x-coordinate.
        column_starts: Column left edges, ascending, from `_column_starts`.

    Returns:
        The index of the rightmost column whose start is at or before `x0`.
    """
    region = 0
    for index, start in enumerate(column_starts):
        if x0 >= start:
            region = index
    return region


def _collect_boilerplate_texts(document: "pymupdf.Document") -> set[str]:
    """Finds text blocks that repeat verbatim across most pages.

    Running headers/footers and side-navigation chrome (chapter menus,
    logos) render as ordinary text blocks with identical text on nearly
    every page. They carry no page-specific information and, left in,
    interleave with real paragraphs once sorted by position.

    Args:
        document: The open PDF document.

    Returns:
        The set of block texts that appear on at least half the document's
        pages (minimum `_BOILERPLATE_MIN_PAGES`).
    """
    page_counts: Counter[str] = Counter()
    for page in document:
        page_dict = page.get_text("dict", sort=True)
        texts_on_page = set()
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            text = _block_text(block)
            if text:
                texts_on_page.add(text)
        page_counts.update(texts_on_page)

    threshold = max(_BOILERPLATE_MIN_PAGES, int(document.page_count * _BOILERPLATE_PAGE_FRACTION))
    return {text for text, count in page_counts.items() if count >= threshold}


def _reading_order_key(item: ParsedItem) -> tuple[int, int, float, float]:
    """Builds the sort key `_sort_items_for_reading_order` sorts items by.

    Args:
        item: A parsed text or table item.

    Returns:
        `(page, region, y0, x0)`, so sorting by this key groups items by
        page, then column, then top-to-bottom, then left-to-right within a
        row.
    """
    return (
        int(getattr(item, "page", 1)),
        int(getattr(item, "region_index", 0)),
        float(getattr(item, "y0", 0.0)),
        float(getattr(item, "x0", 0.0)),
    )


def _sort_items_for_reading_order(items: list[ParsedItem]) -> list[ParsedItem]:
    """Sorts page items by region, then y-position, then x-position.

    This preserves the natural reading order of annual-report pages with
    multi-column layouts, left-side narratives, and right-side summary panels.
    Assumes every item's `region_index` has already been set by the caller.

    Args:
        items: Parsed text and table items, in any order.

    Returns:
        `items`, sorted into reading order.
    """
    return sorted(items, key=_reading_order_key)


def _detect_tables(
    page: "pymupdf.Page", page_number: int
) -> list[tuple["pymupdf.Rect", list[str], list[list[str]]]]:
    """Finds a page's tables, each as its bounding box, header and rows.

    A failure inside PyMuPDF's table detection is logged and treated as "no
    tables on this page": the page's text is still parsed, just without
    structured tables.

    Args:
        page: The page to search.
        page_number: The page's 1-based physical page number, for logging.

    Returns:
        One `(bounding_box, header, rows)` tuple per usable table.
    """
    try:
        found_tables = list(page.find_tables().tables)
    except Exception:
        logger.warning("table detection failed", extra={"extra_fields": {"page": page_number}})
        return []

    detected = []
    for table in found_tables:
        extracted = _extract_table_rows(table)
        if extracted is None:
            continue
        header, rows = extracted
        detected.append((pymupdf.Rect(table.bbox), header, rows))
    return detected


def _collect_text_blocks(
    page: "pymupdf.Page", table_rects: list["pymupdf.Rect"], boilerplate: set[str]
) -> list[tuple["pymupdf.Rect", str]]:
    """Collects a page's plain-text blocks, minus tables and boilerplate.

    Args:
        page: The page to read text from.
        table_rects: Bounding boxes of the page's detected tables.
        boilerplate: Block texts to drop as running headers/footers or
            side-navigation chrome.

    Returns:
        One `(bounding_box, text)` pair per kept block.
    """
    page_dict = page.get_text("dict", sort=True)
    text_blocks = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block, 1 = image
            continue
        rect = pymupdf.Rect(block["bbox"])
        overlaps_a_table = any(
            _overlap_fraction(rect, table_rect) > _OVERLAP_THRESHOLD for table_rect in table_rects
        )
        if overlaps_a_table:
            # Table cells are already captured as structured rows; keeping the
            # overlapping text would duplicate the same numbers in plain text.
            continue
        text = _block_text(block)
        if text and text not in boilerplate:
            text_blocks.append((rect, text))
    return text_blocks


def _parse_page(page: "pymupdf.Page", page_number: int, boilerplate: set[str]) -> list[ParsedItem]:
    """Parses one page into text and table items, in reading order.

    Args:
        page: The page to parse.
        page_number: The page's 1-based physical page number.
        boilerplate: Block texts to drop as running headers/footers or
            side-navigation chrome (see `_collect_boilerplate_texts`).

    Returns:
        The page's text and table items, ordered by detected column,
        top-to-bottom and left-to-right, with text blocks that overlap a
        detected table excluded (so table numbers are not duplicated as
        plain text) and boilerplate blocks dropped.
    """
    pending_tables = _detect_tables(page, page_number)
    table_rects = [rect for rect, _, _ in pending_tables]
    pending_text = _collect_text_blocks(page, table_rects, boilerplate)

    column_starts = _column_starts(
        [rect.x0 for rect, *_ in pending_tables] + [rect.x0 for rect, _ in pending_text]
    )

    items = [
        TableItem(
            page=page_number,
            header=header,
            rows=rows,
            x0=float(rect.x0),
            y0=float(rect.y0),
            region_index=_region_for(float(rect.x0), column_starts),
        )
        for rect, header, rows in pending_tables
    ]
    items.extend(
        TextItem(
            page=page_number,
            text=text,
            x0=float(rect.x0),
            y0=float(rect.y0),
            region_index=_region_for(float(rect.x0), column_starts),
        )
        for rect, text in pending_text
    )

    return _sort_items_for_reading_order(items)


def _item_chars(item: ParsedItem) -> int:
    """Counts a parsed item's characters, for the page-length check.

    Args:
        item: A parsed text or table item.

    Returns:
        Its total character count, used to decide whether a page has
        enough text to keep.
    """
    if isinstance(item, TextItem):
        return len(item.text)
    return sum(len(cell) for row in item.rows for cell in row)


def _worker_count(page_count: int) -> int:
    """Decides how many worker processes to use for a report of this size.

    Args:
        page_count: The report's total page count.

    Returns:
        1 (i.e. run sequentially, no process pool) for reports under
        `_MIN_PAGES_FOR_PARALLEL` pages, otherwise a worker count bounded by
        `_MAX_WORKERS`, the machine's core count, and the number of
        `_PAGES_PER_TASK`-sized tasks there are to hand out.
    """
    if page_count < _MIN_PAGES_FOR_PARALLEL:
        return 1
    task_count = -(-page_count // _PAGES_PER_TASK)  # ceiling division
    return max(1, min(_MAX_WORKERS, os.cpu_count() or 1, task_count))


def _page_ranges(page_count: int, pages_per_task: int) -> list[tuple[int, int]]:
    """Splits a page count into contiguous, `pages_per_task`-sized ranges.

    Args:
        page_count: The report's total page count.
        pages_per_task: The maximum number of pages per range.

    Returns:
        `(start_page, end_page)` pairs, both 1-based and inclusive, covering
        every page exactly once.
    """
    return [
        (start, min(start + pages_per_task - 1, page_count))
        for start in range(1, page_count + 1, pages_per_task)
    ]


# Set by `_init_worker`: each worker opens the PDF once and reuses it across tasks.
_worker_doc: "pymupdf.Document | None" = None


def _init_worker(path: Path) -> None:
    """Opens this worker process's own PDF handle, once, at pool start-up.

    Args:
        path: Path to the PDF file being parsed.
    """
    global _worker_doc
    _worker_doc = pymupdf.open(path)


def _parse_page_task(
    page_range: tuple[int, int], boilerplate: set[str]
) -> list[tuple[int, list[ParsedItem]]]:
    """Parses one contiguous range of pages in a pool worker process.

    Args:
        page_range: `(start_page, end_page)`, both 1-based and inclusive.
        boilerplate: Block texts to drop, as collected by
            `_collect_boilerplate_texts` before the pool was started.

    Returns:
        `(page_number, page_items)` pairs, one per page in `page_range`.
    """
    assert _worker_doc is not None, "_parse_page_task run outside an initialized pool worker"
    start, end = page_range
    return [
        (page_number, _parse_page(_worker_doc[page_number - 1], page_number, boilerplate))
        for page_number in range(start, end + 1)
    ]


def _parse_pages_in_parallel(
    path: Path,
    page_count: int,
    boilerplate: set[str],
    worker_count: int,
    on_progress: Callable[[float], None] | None,
) -> dict[int, list[ParsedItem]]:
    """Parses every page of a report across a pool of worker processes.

    Each worker opens its own `pymupdf.Document` (PyMuPDF documents aren't
    shareable across processes) and handles several `_PAGES_PER_TASK`-sized
    page ranges, so `path` is reopened `worker_count` times, not once per
    page or per task.

    Args:
        path: Path to the PDF file.
        page_count: The report's total page count.
        boilerplate: Block texts to drop from every page.
        worker_count: How many worker processes to start.
        on_progress: Optional callback invoked as each task completes, with
            the fraction of pages parsed so far, in (0, 1].

    Returns:
        Every page's parsed items, keyed by 1-based page number.
    """
    items_by_page: dict[int, list[ParsedItem]] = {}
    completed_pages = 0

    with ProcessPoolExecutor(
        max_workers=worker_count, initializer=_init_worker, initargs=(path,)
    ) as executor:
        futures = {
            executor.submit(_parse_page_task, page_range, boilerplate): page_range
            for page_range in _page_ranges(page_count, _PAGES_PER_TASK)
        }
        for future in as_completed(futures):
            for page_number, page_items in future.result():
                items_by_page[page_number] = page_items
            start, end = futures[future]
            completed_pages += end - start + 1
            if on_progress:
                on_progress(completed_pages / page_count)

    return items_by_page


def _assemble_parsed_document(
    page_count: int, items_by_page: dict[int, list[ParsedItem]]
) -> ParsedDocument:
    """Builds the final `ParsedDocument` from each page's parsed items.

    Shared by the sequential and parallel paths in `parse_pdf`, so the
    `MIN_PAGE_CHARS` skip rule is applied identically either way, and pages
    end up in order regardless of the order they were parsed in.

    Args:
        page_count: The report's total page count.
        items_by_page: Every page's parsed items, keyed by 1-based page
            number.

    Returns:
        The parsed document, with pages under `MIN_PAGE_CHARS` characters
        omitted from `items` and listed in `skipped_pages`.
    """
    items = []
    skipped_pages = []
    for page_number in range(1, page_count + 1):
        page_items = items_by_page[page_number]
        page_chars = sum(_item_chars(item) for item in page_items)
        if page_chars < MIN_PAGE_CHARS:
            skipped_pages.append(page_number)
            logger.info(
                "page skipped: too little text (likely image-only)",
                extra={"extra_fields": {"page": page_number, "chars": page_chars}},
            )
        else:
            items.extend(page_items)
    return ParsedDocument(items=items, page_count=page_count, skipped_pages=skipped_pages)


def parse_pdf(path: Path, on_progress: Callable[[float], None] | None = None) -> ParsedDocument:
    """Parses a PDF into per-page text and table items.

    Table detection dominates parse time, so progress is reported as pages
    complete. Reports under `_MIN_PAGES_FOR_PARALLEL` pages parse in this
    process; larger ones use a pool of worker processes (see
    `_parse_pages_in_parallel`).

    Args:
        path: Path to the PDF file.
        on_progress: Optional callback invoked as pages complete, with the
            fraction of pages parsed so far, in (0, 1].

    Returns:
        The parsed document. Pages yielding fewer than `MIN_PAGE_CHARS`
        characters (likely image-only) are omitted from `items` and listed
        in `skipped_pages`.
    """
    with pymupdf.open(path) as document:
        page_count = document.page_count
        boilerplate = _collect_boilerplate_texts(document)
        worker_count = _worker_count(page_count)

        if worker_count == 1:
            items_by_page = {}
            for page_number in range(1, page_count + 1):
                page = document[page_number - 1]
                items_by_page[page_number] = _parse_page(page, page_number, boilerplate)
                if on_progress:
                    on_progress(page_number / page_count)
            return _assemble_parsed_document(page_count, items_by_page)

    # The parallel path reopens the PDF once per worker process, so the
    # document above is closed (via `with`) before handing off `path`.
    items_by_page = _parse_pages_in_parallel(
        path, page_count, boilerplate, worker_count, on_progress
    )
    return _assemble_parsed_document(page_count, items_by_page)
