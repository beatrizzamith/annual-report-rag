from app.ingestion.chunk import (
    HARD_MAX_CHARS,
    build_embed_text,
    chunk_document,
)
from app.ingestion.parse import (
    ParsedDocument,
    TableItem,
    TextItem,
    _sort_items_for_reading_order,
)


def _doc(items):
    page_count = max((i.page for i in items), default=1)
    return ParsedDocument(items=items, page_count=page_count, skipped_pages=[])


def test_sort_items_for_reading_order_keeps_left_column_before_right_column():
    left = TextItem(1, "Left-hand introduction.", x0=100.0, y0=100.0, region_index=0)
    right = TextItem(1, "Right-hand summary.", x0=900.0, y0=100.0, region_index=2)

    ordered = _sort_items_for_reading_order([right, left])

    assert [item.text for item in ordered] == [
        "Left-hand introduction.",
        "Right-hand summary.",
    ]


def test_short_paragraphs_are_packed_into_one_chunk():
    items = [TextItem(1, "First paragraph."), TextItem(1, "Second paragraph.")]
    chunks = chunk_document(_doc(items))
    assert len(chunks) == 1
    assert chunks[0].kind == "text"
    assert "First paragraph." in chunks[0].text
    assert "Second paragraph." in chunks[0].text
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_chunk_records_correct_page_range_across_a_page_break():
    items = [TextItem(1, "Page one paragraph."), TextItem(2, "Page two paragraph.")]
    chunks = chunk_document(_doc(items))
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_long_run_of_paragraphs_is_split_into_multiple_chunks():
    items = [TextItem(1, "word " * 200) for _ in range(10)]  # each ~1000 chars
    chunks = chunk_document(_doc(items))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= HARD_MAX_CHARS


def test_a_single_oversized_paragraph_is_split_with_overlap():
    huge = "word " * 1000  # ~5000 chars, well over the hard max
    chunks = chunk_document(_doc([TextItem(1, huge)]))
    assert len(chunks) > 1
    # a small overshoot is allowed: splits snap forward to the next word
    # boundary rather than cutting a word in half
    word_boundary_slack = 50
    assert all(len(chunk.text) <= HARD_MAX_CHARS + word_boundary_slack for chunk in chunks)
    # consecutive pieces should overlap somewhat
    assert chunks[0].text[-20:] in chunks[1].text or chunks[1].text[:20] in chunks[0].text


def test_table_is_never_split_when_it_fits():
    header = ["Metric", "2025", "2024"]
    rows = [["Revenue", "100", "90"], ["Costs", "50", "45"]]
    chunks = chunk_document(_doc([TableItem(12, header, rows)]))
    assert len(chunks) == 1
    assert chunks[0].kind == "table"
    assert chunks[0].page_start == 12
    assert chunks[0].page_end == 12
    assert "Revenue" in chunks[0].text


def test_oversized_table_is_split_by_rows_with_header_repeated():
    header = ["Metric", "Value"]
    rows = [[f"Row {i}", "x" * 50] for i in range(80)]
    chunks = chunk_document(_doc([TableItem(3, header, rows)]))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.kind == "table"
        assert "Metric" in chunk.text  # header repeated in every part
        assert len(chunk.text) <= HARD_MAX_CHARS + len(header) * 10  # small caption slack


def test_table_gets_preceding_paragraph_as_caption():
    items = [
        TextItem(1, "Table 3: Workforce by region"),
        TableItem(1, ["Region", "FTE"], [["EMEA", "1000"]]),
    ]
    chunks = chunk_document(_doc(items))
    table_chunk = next(chunk for chunk in chunks if chunk.kind == "table")
    assert "Table 3: Workforce by region" in table_chunk.text


def test_build_embed_text_adds_header_without_altering_stored_text_elsewhere():
    embed_text = build_embed_text("Shell", 2025, 10, 11, "Verbatim body text.")
    assert embed_text.startswith("Shell | Annual Report 2025 | page 10-11")
    assert "Verbatim body text." in embed_text
