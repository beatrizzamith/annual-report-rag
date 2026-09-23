"""Runs the report-ingestion pipeline in order: parse, chunk, embed, then extract.

A single ingestion lock avoids SQLite write contention and reduces provider
rate-limit bursts when multiple reports are queued.
"""

import logging
import sqlite3
import threading
import time
from functools import partial
from pathlib import Path

from app.config import Settings
from app.db.repositories import ChunksRepo, ExtractionsRepo, ReportsRepo
from app.extraction.fte import extract_fte
from app.extraction.goals import extract_goals
from app.ingestion.chunk import build_embed_text, chunk_document
from app.ingestion.embed import embed_and_index_report
from app.ingestion.parse import parse_pdf
from app.llm.client import LLMClient
from app.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)


def _log_stage_complete(report_id: int, stage: str) -> None:
    """Logs one ingestion stage finishing, for progress visibility in the logs.

    Args:
        report_id: The report being ingested.
        stage: The stage that just finished (e.g. "parse_and_chunk", "embed").
    """
    logger.info(
        "ingestion stage complete", extra={"extra_fields": {"report_id": report_id, "stage": stage}}
    )


INGESTION_LOCK = threading.Lock()

_STAGE_PARSE_END = 0.3
_STAGE_CHUNK_END = 0.5
_STAGE_EMBED_END = 0.8


def _report_parse_progress(
    reports_repo: ReportsRepo, report_id: int, span: float, fraction: float
) -> None:
    """Maps a page-level parse fraction onto the overall ingestion progress.

    Table detection can take a long time on large reports, so this helper
    updates the progress bar after each page instead of only once parsing
    finishes.

    Args:
        reports_repo: Repository for updating the report's progress.
        report_id: The report being ingested.
        span: How much overall progress the parse stage spans.
        fraction: The parse stage's completion fraction, in [0, 1].
    """
    reports_repo.update_progress(report_id, "parse", span * fraction)


def _parse_and_chunk(
    report_id: int,
    pdf_path: Path,
    company: str,
    fiscal_year: int,
    reports_repo: ReportsRepo,
    chunks_repo: ChunksRepo,
) -> None:
    """Parses a PDF, chunks it, and stores the chunks.

    Args:
        report_id: The report being ingested.
        pdf_path: Path to the report's stored PDF.
        company: The report's company name, used in each chunk's embed text.
        fiscal_year: The report's fiscal year, used in each chunk's embed text.
        reports_repo: Repository for updating the report's stage and progress.
        chunks_repo: Repository for persisting the resulting chunks.
    """
    reports_repo.update_progress(report_id, "parse", 0.0)
    on_parse_progress = partial(_report_parse_progress, reports_repo, report_id, _STAGE_PARSE_END)
    parsed = parse_pdf(pdf_path, on_parse_progress)
    reports_repo.mark_page_count(report_id, parsed.page_count)
    if parsed.skipped_pages:
        logger.info(
            "pages skipped (too little text)",
            extra={"extra_fields": {"report_id": report_id, "pages": parsed.skipped_pages}},
        )
    reports_repo.update_progress(report_id, "chunk", _STAGE_PARSE_END)

    prepared_chunks = chunk_document(parsed)
    rows = [
        (
            chunk.page_start,
            chunk.page_end,
            chunk.kind,
            chunk.text,
            build_embed_text(company, fiscal_year, chunk.page_start, chunk.page_end, chunk.text),
            None,
        )
        for chunk in prepared_chunks
    ]
    chunks_repo.insert_many(report_id, rows)
    logger.info(
        "chunking complete", extra={"extra_fields": {"report_id": report_id, "chunks": len(rows)}}
    )
    reports_repo.update_progress(report_id, "embed", _STAGE_CHUNK_END)


def _report_embed_progress(
    reports_repo: ReportsRepo, report_id: int, start: float, span: float, fraction: float
) -> None:
    """Maps an embedding-batch fraction onto the overall ingestion progress.

    Args:
        reports_repo: Repository for updating the report's progress.
        report_id: The report being ingested.
        start: The overall progress value where the embed stage begins.
        span: How much overall progress the embed stage spans.
        fraction: The embed stage's own completion fraction, in [0, 1].
    """
    reports_repo.update_progress(report_id, "embed", start + span * fraction)


def _embed(
    report_id: int,
    reports_repo: ReportsRepo,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    llm: LLMClient,
) -> None:
    """Embeds every not-yet-embedded chunk for one report.

    Args:
        report_id: The report being ingested.
        reports_repo: Repository for updating the report's stage and progress.
        chunks_repo: Repository for reading chunks and writing embeddings.
        vector_index: The shared in-memory index new embeddings are added to.
        llm: The LLM client used to compute embeddings.
    """
    span = _STAGE_EMBED_END - _STAGE_CHUNK_END
    on_progress = partial(_report_embed_progress, reports_repo, report_id, _STAGE_CHUNK_END, span)
    embed_and_index_report(report_id, llm, chunks_repo, vector_index, on_progress)
    reports_repo.update_progress(report_id, "extract", _STAGE_EMBED_END)


def _extract(
    report_id: int,
    report_fiscal_year: int,
    reports_repo: ReportsRepo,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    llm: LLMClient,
    extractions_repo: ExtractionsRepo,
    model_name: str,
) -> None:
    """Runs the FTE and sustainability-goal extraction pass for one report.

    Run one after the other, not concurrently: both read through
    `chunks_repo`/`vector_index` on the same `sqlite3.Connection`, and
    despite `check_same_thread=False`, Python's sqlite3 module is not safe
    for concurrent statement execution from two threads on one connection
    (confirmed by a real `sqlite3.InterfaceError` when this was tried).
    Concurrency is instead applied inside `extract_goals`, across its
    map-reduce batches' LLM calls, which don't touch the database.

    Args:
        report_id: The report being ingested.
        report_fiscal_year: The report's own fiscal year, passed through to
            `extract_goals` to filter out already-fulfilled goals.
        reports_repo: Repository for updating the report's stage and progress.
        chunks_repo: Repository used for retrieval-guided extraction.
        vector_index: The shared in-memory index used for semantic search.
        llm: The LLM client used for extraction calls.
        extractions_repo: Repository for persisting extracted items.
        model_name: The chat model name recorded on each extraction row.
    """
    extract_fte(report_id, chunks_repo, vector_index, llm, extractions_repo, model_name)
    extract_goals(
        report_id, report_fiscal_year, chunks_repo, vector_index, llm, extractions_repo, model_name
    )
    reports_repo.update_progress(report_id, "extract", 1.0)


def run_ingestion(
    report_id: int,
    settings: Settings,
    conn: sqlite3.Connection,
    vector_index: VectorIndex,
    llm: LLMClient,
) -> None:
    """Runs the full ingestion pipeline for one report.

    Serialises against other calls via `INGESTION_LOCK`. Any parsing,
    chunking, embedding or extraction error is recorded on the report
    instead of being propagated to the caller.

    Args:
        report_id: The report to ingest. It must already exist with its PDF
            stored under `settings.pdfs_dir`.
        settings: Application settings, including the data directory and chat
            model name.
        conn: The shared SQLite connection.
        vector_index: The shared in-memory index new embeddings are added to.
        llm: The LLM client used for embedding and extraction calls.
    """
    reports_repo = ReportsRepo(conn)
    chunks_repo = ChunksRepo(conn)
    extractions_repo = ExtractionsRepo(conn)

    report = reports_repo.get(report_id)
    if report is None:
        logger.error(
            "run_ingestion: unknown report id", extra={"extra_fields": {"report_id": report_id}}
        )
        return

    pdf_path = settings.pdfs_dir / f"{report.sha256}.pdf"
    logger.info(
        "ingestion started",
        extra={
            "extra_fields": {
                "report_id": report_id,
                "company": report.company,
                "fiscal_year": report.fiscal_year,
                "pdf_path": str(pdf_path),
            }
        },
    )

    started_at = time.monotonic()
    with INGESTION_LOCK:
        try:
            _parse_and_chunk(
                report_id, pdf_path, report.company, report.fiscal_year, reports_repo, chunks_repo
            )
            _log_stage_complete(report_id, "parse_and_chunk")
            _embed(report_id, reports_repo, chunks_repo, vector_index, llm)
            _log_stage_complete(report_id, "embed")
            _extract(
                report_id,
                report.fiscal_year,
                reports_repo,
                chunks_repo,
                vector_index,
                llm,
                extractions_repo,
                settings.chat_model,
            )
            reports_repo.mark_ready(report_id)
            elapsed = time.monotonic() - started_at
            logger.info(
                "ingestion complete",
                extra={
                    "extra_fields": {"report_id": report_id, "elapsed_seconds": round(elapsed, 3)}
                },
            )
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            logger.exception(
                "ingestion failed",
                extra={
                    "extra_fields": {"report_id": report_id, "elapsed_seconds": round(elapsed, 3)}
                },
            )
            reports_repo.mark_failed(report_id, str(exc))
