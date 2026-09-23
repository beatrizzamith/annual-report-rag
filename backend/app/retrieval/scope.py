"""Limits retrieval to reports whose company name matches the question text."""

from app.db.models import Report


def scope_reports(question: str, ready_reports: list[Report]) -> list[int] | None:
    """Picks which ready reports a question should be searched against.

    Args:
        question: The standalone question (after follow-up rewriting).
        ready_reports: Every report currently in the `ready` status.

    Returns:
        The ids of reports whose company name appears in `question`, or
        None if none match (meaning "search every ready report").
    """
    question_lower = question.lower()
    matching_report_ids = [
        report.id for report in ready_reports if report.company.lower() in question_lower
    ]
    return matching_report_ids or None
