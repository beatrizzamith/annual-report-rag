"""Tests for turning ranked chunks into the model's tagged, budgeted evidence."""

from app.answer.context import assemble_context, render_context_text
from app.db.models import Chunk, Report


def _report(report_id: int, company: str) -> Report:
    return Report(
        id=report_id,
        sha256=f"hash-{report_id}",
        filename=f"{company}.pdf",
        company=company,
        fiscal_year=2025,
        page_count=100,
        status="ready",
        stage=None,
        progress=1.0,
        error=None,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _chunk(chunk_id: int, report_id: int, page: int, text: str, page_end: int | None = None):
    return Chunk(
        id=chunk_id,
        report_id=report_id,
        page_start=page,
        page_end=page_end or page,
        kind="text",
        text=text,
        embed_text=text,
        embedding=None,
    )


REPORTS = {1: _report(1, "Shell"), 2: _report(2, "ABN")}


def test_sources_are_ordered_by_report_then_page_and_tagged_in_that_order():
    ranked = [
        _chunk(10, 2, 5, "abn"),
        _chunk(11, 1, 9, "shell late"),
        _chunk(12, 1, 2, "shell early"),
    ]

    sources = assemble_context(ranked, REPORTS, token_budget=1000)

    assert [(s.source_id, s.chunk_id) for s in sources] == [("S1", 12), ("S2", 11), ("S3", 10)]


def test_a_chunk_retrieved_twice_appears_once():
    ranked = [_chunk(10, 1, 1, "same"), _chunk(10, 1, 1, "same")]

    assert len(assemble_context(ranked, REPORTS, token_budget=1000)) == 1


def test_a_chunk_that_does_not_fit_is_dropped_whole_but_smaller_later_ones_still_fit():
    # Budget of 15 tokens is 60 characters: big (50) fits, medium (40) would
    # overflow, small (10) still fits.
    ranked = [
        _chunk(1, 1, 1, "b" * 50),
        _chunk(2, 1, 2, "m" * 40),
        _chunk(3, 1, 3, "s" * 10),
    ]

    sources = assemble_context(ranked, REPORTS, token_budget=15)

    assert [source.chunk_id for source in sources] == [1, 3]
    assert all(len(source.text) in (50, 10) for source in sources)


def test_page_label_is_a_single_page_or_a_range():
    ranked = [_chunk(1, 1, 4, "one page"), _chunk(2, 1, 6, "two pages", page_end=7)]

    sources = assemble_context(ranked, REPORTS, token_budget=1000)

    assert [source.pages for source in sources] == ["4", "6-7"]


def test_no_chunks_gives_no_sources_and_empty_context_text():
    assert assemble_context([], REPORTS, token_budget=1000) == []
    assert render_context_text([]) == ""


def test_context_text_wraps_each_source_in_a_tag_with_its_metadata():
    sources = assemble_context([_chunk(1, 1, 3, "Shell had 81,000 FTE.")], REPORTS, 1000)

    assert render_context_text(sources) == (
        '<source id="S1" report="Shell Annual Report 2025" company="Shell" year="2025" '
        'pages="3" kind="text">\nShell had 81,000 FTE.\n</source>'
    )
