"""End-to-end tests for the ingestion pipeline using a real SQLite database and a fake LLM.

The fake model responds using the data retrieved in the current run, so the
pipeline is exercised end to end rather than depending on hard-coded ranking
assumptions.
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest

from app.config import Settings
from app.db.connection import connect
from app.db.repositories import ChunksRepo, ExtractionsRepo, ReportsRepo
from app.extraction.schemas import FteExtraction, GoalsExtraction, SustainabilityGoal
from app.ingestion.intake import intake_upload
from app.ingestion.pipeline import run_ingestion
from app.llm.fakes import FakeEmbedder, FakeLLM
from app.retrieval.vector_index import VectorIndex
from tests.fixtures.build_fixture import FTE_SENTENCE, GOAL_SENTENCE

EMBEDDING_DIM = 32
_SOURCE_BLOCK = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)


def _extract_source_blocks(prompt: str) -> dict[str, str]:
    """Args:
        prompt: An extraction prompt containing `<source id="...">` blocks.

    Returns:
        A `{source_id: block_text}` map for every source in the prompt.
    """
    return {match.group(1): match.group(2) for match in _SOURCE_BLOCK.finditer(prompt)}


def _find_source_id_containing(blocks: dict[str, str], needle: str) -> str:
    """Args:
        blocks: Source blocks as returned by `_extract_source_blocks`.
        needle: Text to search for within each block.

    Returns:
        The id of the first block containing `needle`.

    Raises:
        AssertionError: No block contains `needle` (i.e. retrieval missed
            the fixture's known sentence).
    """
    for source_id, text in blocks.items():
        if needle in text:
            return source_id
    raise AssertionError(f"no retrieved source contained: {needle!r}")


def _fte_response_from_last_call(fake_llm: FakeLLM) -> FteExtraction:
    """Builds a realistic FTE extraction response from the last prompt sent.

    Copies verbatim from whichever retrieved source actually contains the
    fixture's known FTE sentence, exactly as a real model is instructed to.

    Args:
        fake_llm: The `FakeLLM` whose most recent call to respond to.

    Returns:
        An `FteExtraction` referencing the source that contains the fixture's
        FTE sentence.
    """
    _, user_prompt = fake_llm.calls[-1]
    blocks = _extract_source_blocks(user_prompt)
    source_id = _find_source_id_containing(blocks, FTE_SENTENCE)
    return FteExtraction(
        found=True,
        value_text="12,345",
        metric="average_fte",
        as_of="the reporting year",
        scope="Group",
        quote=FTE_SENTENCE,
        source_id=source_id,
        notes=None,
    )


def _goals_response_from_last_call(fake_llm: FakeLLM) -> GoalsExtraction:
    """Builds a realistic goals extraction response from the last prompt sent.

    Args:
        fake_llm: The `FakeLLM` whose most recent call to respond to.

    Returns:
        A `GoalsExtraction` with one goal, referencing the source that
        contains the fixture's known sustainability-goal sentence.
    """
    _, user_prompt = fake_llm.calls[-1]
    blocks = _extract_source_blocks(user_prompt)
    source_id = _find_source_id_containing(blocks, GOAL_SENTENCE)
    goal = SustainabilityGoal(
        title="Net-zero emissions",
        category="climate",
        target="net-zero Scope 1 and 2 emissions by 2035",
        target_year=2035,
        baseline="2019 levels",
        quote=GOAL_SENTENCE,
        source_id=source_id,
    )
    return GoalsExtraction(goals=[goal])


def _build_settings(data_dir: Path) -> Settings:
    """Args:
        data_dir: Directory the test's SQLite database and PDFs live under.

    Returns:
        Settings suitable for a test run (no real LLM credentials needed,
        since tests use `FakeLLM` directly rather than `build_llm_client`).
    """
    return Settings(data_dir=data_dir, chat_model="fake-model")


@dataclass
class IngestedReport:
    """A report ingested end to end with a working `FakeLLM`, for tests
    that only need to inspect the result."""

    settings: Settings
    conn: sqlite3.Connection
    report_id: int


def _ingest_fixture(settings: Settings, conn: sqlite3.Connection, fixture_pdf_path: Path) -> int:
    """Uploads and fully ingests the fixture PDF with a working `FakeLLM`.

    Args:
        settings: Settings pointing at the test's data directory.
        conn: The test's SQLite connection.
        fixture_pdf_path: Path to the generated fixture PDF.

    Returns:
        The ingested report's id.
    """
    reports_repo = ReportsRepo(conn)
    with open(fixture_pdf_path, "rb") as stream:
        result = intake_upload(stream, "fixture.pdf", "TestCo", 2025, settings, reports_repo)

    fake_llm = FakeLLM(embedder=FakeEmbedder(dim=EMBEDDING_DIM))
    fake_llm.queue_response(partial(_fte_response_from_last_call, fake_llm))
    fake_llm.queue_response(partial(_goals_response_from_last_call, fake_llm))
    vector_index = VectorIndex(dim=EMBEDDING_DIM)

    run_ingestion(result.report.id, settings, conn, vector_index, fake_llm)
    return result.report.id


@pytest.fixture
def ingested_report(fixture_pdf_path: Path, data_dir: Path) -> IngestedReport:
    settings = _build_settings(data_dir)
    conn = connect(settings.db_path)
    report_id = _ingest_fixture(settings, conn, fixture_pdf_path)
    return IngestedReport(settings=settings, conn=conn, report_id=report_id)


def test_ingestion_produces_verified_fte_and_goal(ingested_report: IngestedReport):
    reports_repo = ReportsRepo(ingested_report.conn)
    report = reports_repo.get(ingested_report.report_id)
    assert report is not None
    assert report.status == "ready"
    assert report.page_count == 2

    chunks_repo = ChunksRepo(ingested_report.conn)
    chunks = chunks_repo.get_by_report(ingested_report.report_id)
    assert chunks, "expected at least one chunk"
    assert all(c.embedding is not None for c in chunks)
    assert any(c.kind == "table" and "6,145" in c.text for c in chunks)
    assert any(FTE_SENTENCE in c.text for c in chunks)

    extractions_repo = ExtractionsRepo(ingested_report.conn)
    items = extractions_repo.list_by_report(ingested_report.report_id)
    fte_items = [item for item in items if item.kind == "fte"]
    goal_items = [item for item in items if item.kind == "sustainability_goal"]

    assert len(fte_items) == 1
    assert fte_items[0].verified is True
    assert json.loads(fte_items[0].payload)["value"] == 12345.0

    assert len(goal_items) == 1
    assert goal_items[0].verified is True


def test_data_persists_across_a_new_connection(ingested_report: IngestedReport):
    new_conn = connect(ingested_report.settings.db_path)
    try:
        report = ReportsRepo(new_conn).get(ingested_report.report_id)
        assert report is not None
        assert report.status == "ready"
        assert len(ChunksRepo(new_conn).get_by_report(ingested_report.report_id)) > 0
        assert len(ExtractionsRepo(new_conn).list_by_report(ingested_report.report_id)) == 2
    finally:
        new_conn.close()


def test_deleting_a_report_removes_it_and_lets_the_same_file_be_reingested(
    ingested_report: IngestedReport, fixture_pdf_path: Path
):
    reports_repo = ReportsRepo(ingested_report.conn)
    chunks_repo = ChunksRepo(ingested_report.conn)
    extractions_repo = ExtractionsRepo(ingested_report.conn)
    deleted_id = ingested_report.report_id

    reports_repo.delete(deleted_id)

    assert reports_repo.get(deleted_id) is None
    assert chunks_repo.get_by_report(deleted_id) == []
    assert extractions_repo.list_by_report(deleted_id) == []

    # Re-uploading the same bytes starts a fresh report rather than being
    # treated as already-ingested, since the sha256 lookup no longer finds
    # a row for it (SQLite may reuse the freed id, which is fine).
    with open(fixture_pdf_path, "rb") as stream:
        result = intake_upload(
            stream, "fixture.pdf", "TestCo", 2025, ingested_report.settings, reports_repo
        )
    assert result.needs_ingestion is True
    assert result.report.status == "processing"


def test_uploading_the_same_pdf_twice_does_not_duplicate(
    ingested_report: IngestedReport, fixture_pdf_path: Path
):
    reports_repo = ReportsRepo(ingested_report.conn)
    chunks_repo = ChunksRepo(ingested_report.conn)
    chunk_count_before = len(chunks_repo.get_by_report(ingested_report.report_id))

    with open(fixture_pdf_path, "rb") as stream:
        result = intake_upload(
            stream, "fixture.pdf", "TestCo", 2025, ingested_report.settings, reports_repo
        )

    assert result.needs_ingestion is False
    assert result.report.id == ingested_report.report_id
    assert len(chunks_repo.get_by_report(ingested_report.report_id)) == chunk_count_before


def test_a_failed_report_is_retried_cleanly_by_reuploading(fixture_pdf_path: Path, data_dir: Path):
    settings = _build_settings(data_dir)
    conn = connect(settings.db_path)
    reports_repo = ReportsRepo(conn)

    with open(fixture_pdf_path, "rb") as stream:
        first_upload = intake_upload(stream, "fixture.pdf", "TestCo", 2025, settings, reports_repo)

    # No structured-output response is queued, so extraction raises and
    # ingestion is marked failed (parsing, chunking and embedding succeed
    # first, so this also verifies partial progress is cleaned up on retry).
    broken_llm = FakeLLM(embedder=FakeEmbedder(dim=EMBEDDING_DIM))
    run_ingestion(
        first_upload.report.id, settings, conn, VectorIndex(dim=EMBEDDING_DIM), broken_llm
    )

    failed_report = reports_repo.get(first_upload.report.id)
    assert failed_report is not None
    assert failed_report.status == "failed"
    assert failed_report.error

    with open(fixture_pdf_path, "rb") as stream:
        retry_upload = intake_upload(stream, "fixture.pdf", "TestCo", 2025, settings, reports_repo)
    assert retry_upload.needs_ingestion is True
    assert retry_upload.report.id == first_upload.report.id

    working_llm = FakeLLM(embedder=FakeEmbedder(dim=EMBEDDING_DIM))
    working_llm.queue_response(partial(_fte_response_from_last_call, working_llm))
    working_llm.queue_response(partial(_goals_response_from_last_call, working_llm))
    run_ingestion(
        retry_upload.report.id, settings, conn, VectorIndex(dim=EMBEDDING_DIM), working_llm
    )

    final_report = reports_repo.get(first_upload.report.id)
    assert final_report is not None
    assert final_report.status == "ready"

    extractions_repo = ExtractionsRepo(conn)
    items = extractions_repo.list_by_report(first_upload.report.id)
    assert len(items) == 2  # exactly one fte + one goal, not duplicated from the failed attempt
