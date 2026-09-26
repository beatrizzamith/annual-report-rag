"""Extracts workforce figures such as FTE totals and related metrics."""

import json
import logging

from app.core.prompts import load_prompt
from app.db.models import Chunk
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
    # EU reports disclose workforce under CSRD/ESRS S1 as "own workforce".
    "own employees own workforce",
    "number of employees as of 31 December",
    # Some reports state the headline figure as "internal" vs "external" employees.
    "internal employees external employees",
]
TOP_K_PER_QUERY = 20
# More queries share one fusion pool, so it needs more room than a single
# query would; otherwise a chunk that ranks well for one phrasing is crowded out.
FINAL_TOP_K = 12


def _find_candidate_chunks(
    report_id: int, chunks_repo: ChunksRepo, vector_index: VectorIndex, llm: LLMClient
) -> list[Chunk]:
    """Retrieves the chunks most likely to state the workforce figure.

    Args:
        report_id: The report to search.
        chunks_repo: Repository for retrieval and chunk lookup.
        vector_index: The in-memory index used for semantic search.
        llm: The LLM client used to embed the search queries.

    Returns:
        The fused top chunks across every FTE query, best match first.
    """
    chunk_ids = multi_query_search(
        QUERIES, [report_id], chunks_repo, vector_index, llm, TOP_K_PER_QUERY, FINAL_TOP_K
    )
    looked_up_chunks = [chunks_repo.get(chunk_id) for chunk_id in chunk_ids]
    return [chunk for chunk in looked_up_chunks if chunk is not None]


def _store_extraction(
    report_id: int,
    extraction_result: FteExtraction,
    source_chunk: Chunk,
    extractions_repo: ExtractionsRepo,
    model_name: str,
) -> int:
    """Verifies the model's FTE answer against its source chunk and stores it.

    Args:
        report_id: The report the figure belongs to.
        extraction_result: The model's structured answer.
        source_chunk: The chunk the model cited as its source.
        extractions_repo: Repository the extracted item is persisted to.
        model_name: The chat model name, recorded on the extraction row.

    Returns:
        The new `extractions` row id. The row is stored either way, flagged
        verified or unverified.
    """
    verification = verify_extraction(
        extraction_result.quote, extraction_result.value_text, source_chunk.text
    )
    payload = extraction_result.model_dump()
    payload["value"] = verification.value

    return extractions_repo.insert(
        report_id=report_id,
        kind="fte",
        payload=json.dumps(payload),
        quote=extraction_result.quote,
        page=source_chunk.page_start,
        chunk_id=source_chunk.id,
        verified=verification.verified,
        model=model_name,
    )


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
    chunks = _find_candidate_chunks(report_id, chunks_repo, vector_index, llm)
    if not chunks:
        logger.info(
            "fte extraction: no candidate chunks", extra={"extra_fields": {"report_id": report_id}}
        )
        return None

    context, source_map = build_source_context(chunks)
    system_prompt = load_prompt("extract_fte.md")
    user_prompt = f"{context}\n\nExtract the workforce (FTE) figure following the rules."
    extraction_result = llm.complete_structured(
        system_prompt, user_prompt, FteExtraction, temperature=0
    )

    if not (extraction_result.found and extraction_result.quote and extraction_result.source_id):
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

    return _store_extraction(
        report_id, extraction_result, source_chunk, extractions_repo, model_name
    )
