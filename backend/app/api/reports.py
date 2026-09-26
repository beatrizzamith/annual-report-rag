"""Report endpoints: upload, list, delete, and read pre-extracted data."""

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Form, UploadFile
from fastapi.responses import JSONResponse

from app.api.deps import AppState, get_state
from app.core.errors import LLMUnavailableError, ReportNotFoundError
from app.db.models import Report
from app.db.repositories import ChunksRepo, ExtractionsRepo, ReportsRepo
from app.ingestion.intake import intake_upload
from app.ingestion.pipeline import run_ingestion

router = APIRouter()
logger = logging.getLogger(__name__)


def _report_dict(
    report: Report, extraction_counts: dict[int, dict[str, int]] | None = None
) -> dict:
    """Renders a report as the JSON shape returned by the reports endpoints.

    Args:
        report: The report to render.
        extraction_counts: A `{report_id: {"fte": n, "sustainability_goal": n}}`
            map, as built by `list_reports`. Reports missing from it (e.g.
            not yet `ready`) render with zero counts.

    Returns:
        A JSON-serialisable dict describing the report and its extraction
        counts.
    """
    counts = (extraction_counts or {}).get(report.id, {"fte": 0, "sustainability_goal": 0})
    return {
        "id": report.id,
        "filename": report.filename,
        "company": report.company,
        "fiscal_year": report.fiscal_year,
        "page_count": report.page_count,
        "status": report.status,
        "stage": report.stage,
        "progress": report.progress,
        "error": report.error,
        "created_at": report.created_at,
        "extraction_counts": counts,
    }


def _schedule_ingestion(background_tasks: BackgroundTasks, state: AppState, report: Report) -> None:
    """Queues ingestion for a report, or fails it fast when no LLM is configured.

    Args:
        background_tasks: FastAPI's background task queue.
        state: The shared application state.
        report: The report that still needs ingesting.

    Raises:
        LLMUnavailableError: No LLM provider is configured; the report is
            marked failed so a re-upload retries it once a key is added.
    """
    if state.llm is None:
        ReportsRepo(state.conn).mark_failed(
            report.id, "llm_auth_failed: no LLM provider configured"
        )
        logger.warning(
            "report upload rejected: llm not configured",
            extra={"extra_fields": {"report_id": report.id}},
        )
        raise LLMUnavailableError(
            "No LLM provider is configured; add an API key to .env and re-upload."
        )
    background_tasks.add_task(
        run_ingestion, report.id, state.settings, state.conn, state.vector_index, state.llm
    )


@router.post("/reports")
async def upload_report(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    company: str = Form(...),
    fiscal_year: int = Form(...),
    state: AppState = Depends(get_state),
) -> JSONResponse:
    """Uploads a PDF and schedules ingestion for the report.

    Args:
        background_tasks: FastAPI's background task queue; ingestion is
            scheduled on it rather than run inline.
        file: The uploaded PDF.
        company: The company name typed at upload.
        fiscal_year: The fiscal year typed at upload.
        state: The shared application state.

    Returns:
        A 200 response with the existing report if it was already `ready`
        or is already `processing`; otherwise a 202 response with the
        (possibly newly created) report, and ingestion scheduled.

    Raises:
        InvalidFileError: The upload is empty or not a PDF.
        FileTooLargeError: The upload exceeds the configured size limit.
        LLMUnavailableError: No LLM provider is configured, so ingestion
            cannot run.
    """
    reports_repo = ReportsRepo(state.conn)
    logger.info(
        "report upload received",
        extra={
            "extra_fields": {
                "filename": file.filename,
                "company": company,
                "fiscal_year": fiscal_year,
                "llm_configured": state.llm is not None,
            }
        },
    )

    result = intake_upload(
        stream=file.file,
        filename=file.filename or "report.pdf",
        company=company,
        fiscal_year=fiscal_year,
        settings=state.settings,
        reports_repo=reports_repo,
    )

    if result.needs_ingestion:
        _schedule_ingestion(background_tasks, state, result.report)

    status_code = 200 if not result.needs_ingestion and result.report.status == "ready" else 202
    logger.info(
        "report upload handled",
        extra={
            "extra_fields": {
                "report_id": result.report.id,
                "status": result.report.status,
                "needs_ingestion": result.needs_ingestion,
                "http_status": status_code,
            }
        },
    )
    return JSONResponse(content=_report_dict(result.report), status_code=status_code)


@router.get("/reports")
def list_reports(state: AppState = Depends(get_state)) -> list[dict]:
    """Lists every report with its status and extraction counts.

    Args:
        state: The shared application state.

    Returns:
        Every report, most recently created first, as JSON-serialisable
        dicts. The frontend polls this endpoint while any report is
        `processing`.
    """
    reports_repo = ReportsRepo(state.conn)
    extractions_repo = ExtractionsRepo(state.conn)

    reports = reports_repo.list()
    counts: dict[int, dict[str, int]] = {}
    for report in reports:
        if report.status != "ready":
            continue
        items = extractions_repo.list_by_report(report.id)
        counts[report.id] = {
            "fte": sum(1 for i in items if i.kind == "fte"),
            "sustainability_goal": sum(1 for i in items if i.kind == "sustainability_goal"),
        }
    logger.info("reports listing requested", extra={"extra_fields": {"count": len(reports)}})
    return [_report_dict(report, counts) for report in reports]


@router.get("/reports/{report_id}/extractions")
def get_extractions(report_id: int, state: AppState = Depends(get_state)) -> list[dict]:
    """Returns one report's pre-extracted FTE and sustainability goals.

    Args:
        report_id: The report whose extracted items to return.
        state: The shared application state.

    Returns:
        The report's extracted items (empty list if it has none, or does
        not exist), each with its quote, page and verification flag.
    """
    extractions_repo = ExtractionsRepo(state.conn)
    items = extractions_repo.list_by_report(report_id)
    logger.info(
        "report extractions requested",
        extra={"extra_fields": {"report_id": report_id, "count": len(items)}},
    )
    return [
        {
            "id": item.id,
            "kind": item.kind,
            "payload": json.loads(item.payload),
            "quote": item.quote,
            "page": item.page,
            "verified": item.verified,
        }
        for item in items
    ]


@router.delete("/reports/{report_id}")
def delete_report(report_id: int, state: AppState = Depends(get_state)) -> dict:
    """Permanently deletes a report so it can be re-uploaded and re-tested.

    This is useful when you want to force a clean retry of the same file.

    Args:
        report_id: The report to delete.
        state: The shared application state.

    Returns:
        `{"deleted": true}` on success.

    Raises:
        ReportNotFoundError: No report has that id.
    """
    reports_repo = ReportsRepo(state.conn)
    if reports_repo.get(report_id) is None:
        raise ReportNotFoundError(f"No report with id {report_id}")

    chunk_ids = {chunk.id for chunk in ChunksRepo(state.conn).get_by_report(report_id)}
    reports_repo.delete(report_id)
    state.vector_index.remove(chunk_ids)
    logger.info(
        "report deleted",
        extra={"extra_fields": {"report_id": report_id, "chunk_count": len(chunk_ids)}},
    )

    return {"deleted": True}
