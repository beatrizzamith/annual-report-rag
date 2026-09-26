"""Tests for choosing which reports a question is searched against."""

from app.db.models import Report
from app.retrieval.scope import scope_reports


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


READY_REPORTS = [_report(1, "Shell"), _report(2, "ABN"), _report(3, "Heineken")]


def test_question_naming_a_company_is_scoped_to_that_report():
    assert scope_reports("How many FTE does Shell have?", READY_REPORTS) == [1]


def test_company_match_ignores_case():
    assert scope_reports("what is HEINEKEN's revenue?", READY_REPORTS) == [3]


def test_question_naming_no_company_searches_every_report():
    assert scope_reports("Which company has the most FTE?", READY_REPORTS) is None


def test_question_naming_two_companies_is_scoped_to_both():
    assert scope_reports("Compare Shell and ABN headcount", READY_REPORTS) == [1, 2]


def test_no_ready_reports_means_no_scope():
    assert scope_reports("How many FTE does Shell have?", []) is None
