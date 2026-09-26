"""Coordinates the end-to-end chat pipeline: rewrite, scope, search, answer, verify, and store."""

import json
import logging
import re
import sqlite3

from pydantic import BaseModel, Field

from app.answer.context import AssembledSource, assemble_context, render_context_text
from app.answer.generate import generate_answer
from app.answer.schemas import AnswerStatus, ModelAnswer
from app.core.errors import NoReportsLoadedError
from app.db.models import Report
from app.db.repositories import ChunksRepo, MessagesRepo, ReportsRepo
from app.extraction.verifier import verify_citation
from app.llm.client import LLMClient
from app.retrieval.hybrid import hybrid_search
from app.retrieval.rewrite import rewrite_followup
from app.retrieval.scope import scope_reports
from app.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)

SEARCH_TOP_K = 20
CONTEXT_TOP_K = 8
# Prior chat messages fed to the follow-up rewrite and the answer prompt. Kept
# small so the most recent turn dominates instead of competing with older ones.
RECENT_MESSAGES_LIMIT = 2

_INLINE_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")

# A figure worth backing with a citation: thousands-grouped (122,779), decimal
# (8.7) or a bare run of 4+ digits. It only flags "looks like a specific
# figure"; it does not parse one, unlike `parse_number`.
_REPORTABLE_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,}")
# A bare 4-digit number in this range is almost always a year ("by 2028").
_PLAUSIBLE_YEAR = re.compile(r"^(19|20)\d\d$")


class VerifiedCitation(BaseModel):
    """One citation from a model answer, after verbatim verification.

    `label`/`value_text`/`unit`/`period` are None for a citation backing
    non-numeric prose rather than a specific figure.
    """

    label: str | None
    value_text: str | None
    unit: str | None
    period: str | None
    quote: str
    chunk_text: str = Field(description="The cited chunk's full text, for 'show full source'.")
    report: str
    page: str
    verified: bool
    source_id: str = Field(description='Matches the "[S3]"-style marker in the answer text.')


class ChatResult(BaseModel):
    """The full result of answering one chat question."""

    status: AnswerStatus
    answer: str
    citations: list[VerifiedCitation]
    caveats: list[str]
    interpreted_as: str | None


def _verify_citations(
    model_answer: ModelAnswer, sources_by_id: dict[str, AssembledSource]
) -> list[VerifiedCitation]:
    """Verifies every citation the model returned against its cited source.

    Args:
        model_answer: The model's structured answer.
        sources_by_id: The assembled sources for this turn, keyed by
            source id.

    Returns:
        One `VerifiedCitation` per citation whose `source_id` matches a
        real assembled source. Citations pointing at an unknown source id
        (a fabricated citation) are dropped rather than shown as evidence.
    """
    verified_citations = []
    for citation in model_answer.citations:
        source = sources_by_id.get(citation.source_id)
        if source is None:
            logger.warning(
                "chat: model cited an unknown source_id",
                extra={"extra_fields": {"source_id": citation.source_id}},
            )
            continue
        verification = verify_citation(citation.quote, citation.value_text, source.text)
        verified_citations.append(
            VerifiedCitation(
                label=citation.label,
                value_text=citation.value_text,
                unit=citation.unit,
                period=citation.period,
                quote=citation.quote,
                chunk_text=source.text,
                report=source.report_label,
                page=source.pages,
                verified=verification.verified,
                source_id=citation.source_id,
            )
        )
    return verified_citations


def _downgrade_if_all_unverified(
    status: AnswerStatus, citations: list[VerifiedCitation], caveats: list[str]
) -> AnswerStatus:
    """Downgrades an `answered` status to `partial` when nothing verified.

    Args:
        status: The model's reported status.
        citations: The verified citations for this turn.
        caveats: The model's caveats list; a note is appended in place when
            downgrading.

    Returns:
        `"partial"` if `status` was `"answered"` but every citation failed
        verification; `status` unchanged otherwise.
    """
    if status == "answered" and citations and all(not cited.verified for cited in citations):
        caveats.append("Quotes could not be matched exactly to the source text")
        return "partial"
    return status


def _drop_citations_if_not_found(
    status: AnswerStatus, citations: list[VerifiedCitation]
) -> list[VerifiedCitation]:
    """Clears citations for a `not_found` answer.

    A source that was retrieved and then rejected is not evidence, so showing
    it as a citation card would mislead. The prompt already asks the model
    not to cite here; this enforces it in code.

    Args:
        status: The model's reported status, before any downgrade.
        citations: The verified citations produced for this turn.

    Returns:
        An empty list when `status` is `"not_found"`; `citations`
        unchanged otherwise.
    """
    return [] if status == "not_found" else citations


def _find_orphan_citation_ids(answer_text: str, citations: list[VerifiedCitation]) -> list[str]:
    """Finds inline `[S#]` markers with no matching citation entry.

    The prompt requires every marker to have an entry; this enforces it in code.

    Args:
        answer_text: The model's answer prose, possibly containing `[S#]`
            markers.
        citations: The verified citations produced for this turn.

    Returns:
        Source ids (e.g. `"S3"`) cited inline but missing from `citations`,
        in order of first appearance, without duplicates.
    """
    cited_ids = {citation.source_id for citation in citations}
    orphan_ids: list[str] = []
    for match in _INLINE_CITATION_PATTERN.finditer(answer_text):
        source_id = f"S{match.group(1)}"
        if source_id not in cited_ids and source_id not in orphan_ids:
            orphan_ids.append(source_id)
    return orphan_ids


def _find_unbacked_numbers(answer_text: str, citations: list[VerifiedCitation]) -> list[str]:
    """Finds reportable numbers in the answer prose that no verified citation backs.

    Verification only checks a citation's own `value_text`, so a number typed
    straight into the prose would otherwise go unchecked. A number counts as
    backed if it appears verbatim in a verified citation's quote. Being a
    literal check, it can flag a restated or rounded number, which is why the
    caller adds a caveat instead of hiding the answer.

    Args:
        answer_text: The model's answer prose.
        citations: The verified citations produced for this turn.

    Returns:
        Reportable numbers (as printed in the answer) with no verified
        citation whose quote contains them, in order of first appearance,
        without duplicates.
    """
    verified_quotes = " ".join(citation.quote for citation in citations if citation.verified)
    unbacked: list[str] = []
    for match in _REPORTABLE_NUMBER.finditer(answer_text):
        number = match.group(0)
        if _PLAUSIBLE_YEAR.fullmatch(number):
            continue
        if number in verified_quotes:
            continue
        if number not in unbacked:
            unbacked.append(number)
    return unbacked


def _retrieve_sources(
    question: str,
    ready_reports: list[Report],
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    llm: LLMClient,
    context_token_budget: int,
) -> list[AssembledSource]:
    """Scopes to the relevant reports, searches them, and assembles the evidence.

    Args:
        question: The standalone question to search with.
        ready_reports: Every report that has finished ingestion.
        chunks_repo: Repository for keyword search and chunk lookup.
        vector_index: The shared in-memory index used for semantic search.
        llm: The LLM client used to embed the question.
        context_token_budget: The approximate token budget for the evidence.

    Returns:
        The tagged sources that fit the budget, each carrying a source id
        the model can cite.
    """
    scoped_report_ids = scope_reports(question, ready_reports)
    chunk_ids = hybrid_search(
        question,
        scoped_report_ids,
        chunks_repo,
        vector_index,
        llm,
        top_k=SEARCH_TOP_K,
        final_top_k=CONTEXT_TOP_K,
    )
    looked_up_chunks = [chunks_repo.get(chunk_id) for chunk_id in chunk_ids]
    ranked_chunks = [chunk for chunk in looked_up_chunks if chunk is not None]
    reports_by_id = {report.id: report for report in ready_reports}
    sources = assemble_context(ranked_chunks, reports_by_id, context_token_budget)
    scoped_report_count = (
        len(scoped_report_ids) if scoped_report_ids is not None else len(ready_reports)
    )
    logger.info(
        "chat retrieval complete",
        extra={
            "extra_fields": {
                "scoped_reports": scoped_report_count,
                "scoped_report_filter": "all" if scoped_report_ids is None else "matched_company",
                "retrieved_chunks": len(ranked_chunks),
                "assembled_sources": len(sources),
            }
        },
    )
    return sources


def _check_answer(
    structured_answer: ModelAnswer, sources: list[AssembledSource], interpreted_as: str | None
) -> ChatResult:
    """Verifies the model's answer in code and attaches any caveats.

    Every citation is checked against its source, and the answer text is
    checked for dangling citation markers and unbacked numbers.

    Args:
        structured_answer: The model's structured answer.
        sources: The sources the model was shown for this turn.
        interpreted_as: The rewritten standalone question, when there was one.

    Returns:
        The final chat result: status (possibly downgraded), answer,
        verified citations, and the model's caveats plus any added here.
    """
    sources_by_id = {source.source_id: source for source in sources}
    citations = _verify_citations(structured_answer, sources_by_id)
    citations = _drop_citations_if_not_found(structured_answer.status, citations)
    caveats = list(structured_answer.caveats)
    status = _downgrade_if_all_unverified(structured_answer.status, citations, caveats)

    orphan_ids = _find_orphan_citation_ids(structured_answer.answer, citations)
    if orphan_ids:
        logger.warning(
            "chat: answer cites source(s) with no matching citation entry",
            extra={"extra_fields": {"source_ids": orphan_ids}},
        )
        caveats.append("Some inline citations could not be matched to evidence")

    unbacked_numbers = _find_unbacked_numbers(structured_answer.answer, citations)
    if unbacked_numbers:
        logger.warning(
            "chat: answer states number(s) no verified citation backs",
            extra={"extra_fields": {"numbers": unbacked_numbers}},
        )
        caveats.append("Some figures in the answer are not confirmed by a verified citation")

    return ChatResult(
        status=status,
        answer=structured_answer.answer,
        citations=citations,
        caveats=caveats,
        interpreted_as=interpreted_as,
    )


def _store_turn(messages_repo: MessagesRepo, message: str, result: ChatResult) -> None:
    """Persists the user's message and the assistant's reply.

    Args:
        messages_repo: Repository for the chat history.
        message: The user's question, exactly as typed.
        result: The verified reply to store, with its citations.
    """
    citations_json = json.dumps([citation.model_dump() for citation in result.citations])
    messages_repo.insert("user", message, citations=None, interpreted_as=None)
    messages_repo.insert(
        "assistant",
        result.answer,
        citations=citations_json,
        interpreted_as=result.interpreted_as,
    )


def answer_question(
    message: str,
    conn: sqlite3.Connection,
    vector_index: VectorIndex,
    llm: LLMClient,
    context_token_budget: int,
) -> ChatResult:
    """Answers one chat question end to end.

    Args:
        message: The user's question, exactly as typed.
        conn: The application's SQLite connection.
        vector_index: The shared in-memory index used for semantic search.
        llm: The LLM client used for rewrite, embedding and generation.
        context_token_budget: The approximate token budget for the assembled
            evidence context.

    Returns:
        The status, final answer, verified citations, caveats, and the
        rewritten standalone question when one was created.

    Raises:
        NoReportsLoadedError: No report is currently `ready`.
    """
    chunks_repo = ChunksRepo(conn)
    messages_repo = MessagesRepo(conn)

    ready_reports = ReportsRepo(conn).list_ready()
    if not ready_reports:
        logger.warning("chat request rejected: no ready reports")
        raise NoReportsLoadedError("No reports have finished ingestion yet.")

    recent_messages = messages_repo.list(limit=RECENT_MESSAGES_LIMIT)
    standalone_question, was_rewritten = rewrite_followup(message, recent_messages, llm)
    interpreted_as = standalone_question if was_rewritten else None
    logger.info(
        "chat answer generation started",
        extra={
            "extra_fields": {
                "ready_reports": len(ready_reports),
                "message_length": len(message),
                "rewritten": was_rewritten,
            }
        },
    )

    sources = _retrieve_sources(
        standalone_question, ready_reports, chunks_repo, vector_index, llm, context_token_budget
    )
    context_text = render_context_text(sources)
    structured_answer = generate_answer(message, interpreted_as, context_text, recent_messages, llm)

    result = _check_answer(structured_answer, sources, interpreted_as)
    _store_turn(messages_repo, message, result)
    logger.info(
        "chat answer completed",
        extra={
            "extra_fields": {
                "status": result.status,
                "citations": len(result.citations),
                "caveats": len(result.caveats),
            }
        },
    )
    return result
