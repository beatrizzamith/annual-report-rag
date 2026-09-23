"""Tests for `parse_pdf`'s progress reporting.

Table detection dominates parsing time on table-heavy reports, so progress
is reported after every page rather than only once parsing finishes (see
`pipeline._report_parse_progress`).
"""

from pathlib import Path

from app.ingestion.parse import parse_pdf


def test_on_progress_is_called_once_per_page_in_increasing_order(fixture_pdf_path: Path):
    fractions = []
    parse_pdf(fixture_pdf_path, fractions.append)

    assert len(fractions) == 2  # the fixture PDF has 2 pages
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_parsing_without_a_callback_still_works(fixture_pdf_path: Path):
    parsed = parse_pdf(fixture_pdf_path)
    assert parsed.page_count == 2
