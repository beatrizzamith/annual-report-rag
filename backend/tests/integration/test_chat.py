"""Integration tests covering verified answers, missing answers, and follow-up rewriting."""

from pathlib import Path

import pytest

from app.answer.schemas import Citation, ModelAnswer
from app.answer.service import answer_question
from app.api.deps import build_vector_index
from app.core.errors import NoReportsLoadedError
from app.db.connection import connect
from app.db.repositories import MessagesRepo
from app.llm.fakes import FakeEmbedder, FakeLLM
from app.retrieval.rewrite import FollowUpRewrite
from tests.integration.test_ingestion_pipeline import (
    EMBEDDING_DIM,
    IngestedReport,
    _build_settings,
    _ingest_fixture,
)


@pytest.fixture
def ingested_report(fixture_pdf_path: Path, data_dir: Path) -> IngestedReport:
    settings = _build_settings(data_dir)
    conn = connect(settings.db_path)
    report_id = _ingest_fixture(settings, conn, fixture_pdf_path)
    return IngestedReport(settings=settings, conn=conn, report_id=report_id)


def _answer_with(model_answer: ModelAnswer, ingested_report: IngestedReport, message: str):
    """Runs `answer_question` with a `FakeLLM` scripted to return `model_answer`.

    Args:
        model_answer: The structured answer the fake model should return.
        ingested_report: A report already ingested (chunked and embedded).
        message: The chat message to answer.

    Returns:
        The `ChatResult` from `answer_question`.
    """
    fake_llm = FakeLLM(embedder=FakeEmbedder(dim=EMBEDDING_DIM))
    fake_llm.queue_response(model_answer)
    vector_index = build_vector_index(ingested_report.conn, EMBEDDING_DIM)
    return answer_question(
        message,
        ingested_report.conn,
        vector_index,
        fake_llm,
        ingested_report.settings.context_token_budget,
    )


def test_answered_question_returns_a_verified_citation(ingested_report: IngestedReport):
    model_answer = ModelAnswer(
        status="answered",
        answer="The Group averaged 12,345 FTE during the year [S1].",
        citations=[
            Citation(
                label="Average FTE",
                value_text="12,345",
                unit=None,
                period="the year",
                quote=(
                    "The Group employed an average of 12,345 FTE during the year, "
                    "measured as full-time equivalents across all operating segments."
                ),
                source_id="S1",
            )
        ],
        caveats=[],
    )
    result = _answer_with(model_answer, ingested_report, "How many FTE does the company have?")

    assert result.status == "answered"
    assert len(result.citations) == 1
    assert result.citations[0].verified is True
    assert result.interpreted_as is None


def test_not_found_status_is_passed_through(ingested_report: IngestedReport):
    model_answer = ModelAnswer(
        status="not_found",
        answer="The reports do not disclose climate change adaptation spend.",
        citations=[],
        caveats=[],
    )
    result = _answer_with(
        model_answer, ingested_report, "How much was spent on climate adaptation?"
    )

    assert result.status == "not_found"
    assert result.citations == []


def test_fabricated_quote_is_flagged_and_answer_downgraded(ingested_report: IngestedReport):
    model_answer = ModelAnswer(
        status="answered",
        answer="The Group employed exactly one million robots [S1].",
        citations=[
            Citation(
                label="Robots",
                value_text="one million",
                unit=None,
                period=None,
                quote="The Group employed exactly one million robots.",
                source_id="S1",
            )
        ],
        caveats=[],
    )
    result = _answer_with(model_answer, ingested_report, "How many FTE does the company have?")

    assert len(result.citations) == 1
    assert result.citations[0].verified is False
    assert result.status == "partial"  # downgraded: every citation failed verification
    assert "Quotes could not be matched exactly" in " ".join(result.caveats)


def test_no_reports_loaded_raises_typed_error(data_dir: Path):
    settings = _build_settings(data_dir)
    conn = connect(settings.db_path)
    fake_llm = FakeLLM(embedder=FakeEmbedder(dim=EMBEDDING_DIM))
    vector_index = build_vector_index(conn, EMBEDDING_DIM)

    with pytest.raises(NoReportsLoadedError):
        answer_question(
            "How many FTE?", conn, vector_index, fake_llm, settings.context_token_budget
        )


def test_followup_is_rewritten_and_used_for_retrieval(ingested_report: IngestedReport):
    # First turn establishes history.
    first_answer = ModelAnswer(
        status="answered",
        answer="The Group averaged 12,345 FTE during the year [S1].",
        citations=[],
        caveats=[],
    )
    _answer_with(first_answer, ingested_report, "How many FTE does the company have?")

    # Second turn: a follow-up with no topic of its own. The fake LLM is
    # scripted to first return a rewrite, then an answer; if the rewrite's
    # output were NOT used for retrieval, the FTE sentence would not be found
    # and this second call would have no candidate chunks.
    fake_llm = FakeLLM(embedder=FakeEmbedder(dim=EMBEDDING_DIM))
    fake_llm.queue_response(
        FollowUpRewrite(standalone_question="How many FTE does the company have?")
    )
    second_answer = ModelAnswer(
        status="answered", answer="Still 12,345 FTE [S1].", citations=[], caveats=[]
    )
    fake_llm.queue_response(second_answer)

    vector_index = build_vector_index(ingested_report.conn, EMBEDDING_DIM)
    result = answer_question(
        "and last year?",
        ingested_report.conn,
        vector_index,
        fake_llm,
        ingested_report.settings.context_token_budget,
    )

    assert result.interpreted_as == "How many FTE does the company have?"
    assert result.status == "answered"

    messages = MessagesRepo(ingested_report.conn).list()
    assert len(messages) == 4  # 2 from the first turn + 2 from this one
    assert messages[-1].interpreted_as == "How many FTE does the company have?"
