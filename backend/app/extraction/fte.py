"""Extracts workforce figures such as FTE totals and related metrics."""

import json
import logging

from app.core.prompts import load_prompt
from app.db.repositories import ChunksRepo, ExtractionsRepo
from app.extraction.context import build_source_context
from app.extraction.schemas import FteExtraction
from app.extraction.verifier import verify_extraction
from app.llm.client import LLMClient
from app.retrieval.hybrid import multi_query_search
from app.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)

QUERIES = [
    "total number of employees FTE full-time equivalent",
    "workforce headcount at year end",
    "average number of employees",
    # EU annual reports have disclosed workforce figures under CSRD/ESRS S1
    # as "own workforce"/"own employees" since 2024; without these, a report
    # phrasing it that way (observed in a real report during manual testing)
    # can be missed even though a generic "employees" query is also present.
    "own employees own workforce",
    "number of employees as of 31 December",
    # A real report (ABN AMRO, manual testing) states its headline figure as
    # "internal employees" alongside "external employees" -- a phrasing none
    # of the queries above use the exact wording of, and the surrounding
    # ESG-bullet/key-figures-table chunks are noisy enough (many unrelated
    # facts packed in) that this generic "employees" phrasing wasn't enough
    # for either chunk to reach the fused top-12 on its own.
    "internal employees external employees",
]
TOP_K_PER_QUERY = 20
# Widened from 8 alongside the extra queries above: more queries competing
# in the same reciprocal-rank-fusion pool need more room, or a chunk that
# ranks well for only one phrasing can get crowded out by chunks that rank
# well across several. Cost impact is one-time, per report, and small (a
# few thousand extra input tokens on the single FTE extraction call).
FINAL_TOP_K = 12


def extract_fte(
    report_id: int,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    llm: LLMClient,
    extractions_repo: ExtractionsRepo,
    model_name: str,
) -> int | None:
    """Runs retrieval-guided FTE extraction for one report.

    Args:
        report_id: The report to extract from. Must already be chunked and
            embedded.
        chunks_repo: Repository for retrieval and chunk lookup.
        vector_index: The in-memory index used for semantic search.
        llm: The LLM client used for retrieval embeddings and extraction.
        extractions_repo: Repository the extracted item is persisted to.
        model_name: The chat model name, recorded on the extraction row.

    Returns:
        The new `extractions` row id, or None if no figure was found or the
        model's answer could not be verified against a real chunk (nothing
        is stored in either case; the UI shows a dash).
    """
    chunk_ids = multi_query_search(
        QUERIES, [report_id], chunks_repo, vector_index, llm, TOP_K_PER_QUERY, FINAL_TOP_K
    )
    if not chunk_ids:
        logger.info(
            "fte extraction: no candidate chunks", extra={"extra_fields": {"report_id": report_id}}
        )
        return None

    chunks = [c for c in (chunks_repo.get(cid) for cid in chunk_ids) if c is not None]
    context, source_map = build_source_context(chunks)
    system_prompt = load_prompt("extract_fte.md")
    user_prompt = f"{context}\n\nExtract the workforce (FTE) figure following the rules."

    extraction_result = llm.complete_structured(
        system_prompt, user_prompt, FteExtraction, temperature=0
    )

    is_incomplete = (
        not extraction_result.found
        or not extraction_result.quote
        or not extraction_result.source_id
    )
    if is_incomplete:
        logger.info("fte extraction: not found", extra={"extra_fields": {"report_id": report_id}})
        return None

    chunk_id = source_map.get(extraction_result.source_id)
    source_chunk = chunks_repo.get(chunk_id) if chunk_id else None
    if source_chunk is None:
        logger.warning(
            "fte extraction: model cited an unknown source_id",
            extra={
                "extra_fields": {"report_id": report_id, "source_id": extraction_result.source_id}
            },
        )
        return None

    quote_verification = verify_extraction(
        extraction_result.quote, extraction_result.value_text, source_chunk.text
    )
    payload = extraction_result.model_dump()
    payload["value"] = quote_verification.value

    return extractions_repo.insert(
        report_id=report_id,
        kind="fte",
        payload=json.dumps(payload),
        quote=extraction_result.quote,
        page=source_chunk.page_start,
        chunk_id=chunk_id,
        verified=quote_verification.verified,
        model=model_name,
    )
