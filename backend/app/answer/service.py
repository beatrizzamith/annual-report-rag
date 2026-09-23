"""Coordinates the end-to-end chat pipeline: rewrite, scope, search, answer, verify, and store."""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass

from app.answer.context import AssembledSource, assemble_context, render_context_text
from app.answer.generate import generate_answer
from app.answer.schemas import ModelAnswer
from app.core.errors import NoReportsLoadedError
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
# How much prior chat history feeds the follow-up rewrite and the answer
# prompt. Widened from 1 (decision #28) to 2: with a longer history (the
# original 4), a follow-up could get matched against an earlier question in
# that history instead of the actual most recent one -- the most recent
# turn should dominate, not compete with several others (decision #36).
RECENT_MESSAGES_LIMIT = 2

_INLINE_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")

# A "reportable" number: thousands-grouped (122,779), decimal (8.7, also
# covers a percent's digits since a bare 4+ digit run is redundant once a
# decimal already matched), or a bare run of 4+ digits (23126). Deliberately
# not the general `parse_number`/`_NUMBER_PATTERN` machinery in
# `app.core.text` (magnitude words, localized separators, signs) -- this
# only needs to flag "does this look like a specific figure", not parse one.
_REPORTABLE_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,}")
# A bare 4-digit number in this range is almost always a calendar year
# mention ("by 2028"), not a claimed figure -- `_REPORTABLE_NUMBER`'s
# comma-grouped and decimal branches never match a plain 4-digit run, so
# this only ever excludes that one shape.
_PLAUSIBLE_YEAR = re.compile(r"^(19|20)\d\d$")


@dataclass
class VerifiedCitation:
    """One citation from a model answer, after verbatim verification.

    `label`/`value_text`/`unit`/`period` are None for a citation backing
    non-numeric prose rather than a specific figure.
    """

    label: str | None
    value_text: str | None
    unit: str | None
    period: str | None
    quote: str
    chunk_text: str  # the cited chunk's full text, for "show full source" in the UI
    report: str
    page: str
    verified: bool
    source_id: str  # matches the "[S3]"-style marker cited inline in the answer text


@dataclass
class ChatResult:
    """The full result of answering one chat question."""

    status: str  # answered | partial | not_found
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
    status: str, citations: list[VerifiedCitation], caveats: list[str]
) -> str:
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
    if status == "answered" and citations and all(not c.verified for c in citations):
        caveats.append("Quotes could not be matched exactly to the source text")
        return "partial"
    return status


def _drop_citations_if_not_found(
    status: str, citations: list[VerifiedCitation]
) -> list[VerifiedCitation]:
    """Clears citations for a `not_found` answer.

    A source the model points to when it couldn't answer the question is
    not evidence for anything -- showing a citation card next to "the
    sources do not cover this" reads as if the source were somehow
    related, when it was only retrieved and then rejected. The prompt
    asks the model not to cite in this case (see `prompts/answer.md`),
    but that is a model instruction, not an enforced property, in keeping
    with the project's rule that nothing shown to the user is trusted
    from the model without a check (SPEC.md section 1, "grounded or
    silent").

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

    The prompt requires every inline citation to have a matching entry (see
    `prompts/answer.md`), but that is a model instruction, not an enforced
    property — this is the code-level check, in keeping with the project's
    rule that nothing shown to the user is trusted from the model without a
    check (SPEC.md section 1, "grounded or silent").

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
    """Finds reportable numbers in the answer prose no verified citation backs.

    `value_text` is verified against its own citation's quote (see
    `verify_citation`), but the model is also free to type a number
    straight into the answer prose, next to a citation, without ever
    putting it through that check -- a real case found this exactly: an
    answer stated an unrelated risk-weighted-assets figure as an employee
    count, backed by a citation whose quote was about a completely
    different fact and never contained that number. This is the
    code-level check for that gap, in keeping with the project's rule that
    nothing shown to the user is trusted from the model without a check
    (SPEC.md section 1, "grounded or silent").

    A number counts as backed if it appears verbatim in a *verified*
    citation's quote -- an unverified citation's quote isn't itself
    confirmed to be in the source, so it can't back anything else either.
    This is a literal substring check, same spirit and same limits as
    `verify_quote`: it can miss a number restated in a different form
    (a unit conversion, a rounded figure) and flag a harmless case, which
    is why this adds a caveat rather than hiding the answer.

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
    reports_repo = ReportsRepo(conn)
    chunks_repo = ChunksRepo(conn)
    messages_repo = MessagesRepo(conn)

    ready_reports = reports_repo.list_ready()
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

    reports_by_id = {report.id: report for report in ready_reports}
    scoped_report_ids = scope_reports(standalone_question, ready_reports)

    chunk_ids = hybrid_search(
        standalone_question,
        scoped_report_ids,
        chunks_repo,
        vector_index,
        llm,
        top_k=SEARCH_TOP_K,
        final_top_k=CONTEXT_TOP_K,
    )
    ranked_chunks = [c for c in (chunks_repo.get(cid) for cid in chunk_ids) if c is not None]
    sources = assemble_context(ranked_chunks, reports_by_id, context_token_budget)
    context_text = render_context_text(sources)
    cited_sources_by_id = {source.source_id: source for source in sources}
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

    structured_answer = generate_answer(message, interpreted_as, context_text, recent_messages, llm)

    citations = _verify_citations(structured_answer, cited_sources_by_id)
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

    citations_json = json.dumps([citation.__dict__ for citation in citations])
    messages_repo.insert("user", message, citations=None, interpreted_as=None)
    messages_repo.insert(
        "assistant",
        structured_answer.answer,
        citations=citations_json,
        interpreted_as=interpreted_as,
    )
    logger.info(
        "chat answer completed",
        extra={
            "extra_fields": {
                "status": status,
                "citations": len(citations),
                "caveats": len(caveats),
                "orphan_citations": len(orphan_ids),
                "unbacked_numbers": len(unbacked_numbers),
            }
        },
    )

    return ChatResult(
        status=status,
        answer=structured_answer.answer,
        citations=citations,
        caveats=caveats,
        interpreted_as=interpreted_as,
    )
