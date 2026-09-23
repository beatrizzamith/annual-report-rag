"""Combines keyword and semantic retrieval to rank the most relevant chunks."""

from app.db.repositories import ChunksRepo
from app.llm.client import LLMClient
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword import build_fts_query
from app.retrieval.vector_index import VectorIndex

TOP_K_PER_SEARCH = 20


def _hybrid_search_with_vector(
    query: str,
    query_vector: list[float],
    report_ids: list[int] | None,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    top_k: int,
    final_top_k: int | None,
) -> list[int]:
    """Fuses BM25 and cosine search for one query, given its embedding.

    Shared by `hybrid_search` (which embeds `query` itself) and
    `multi_query_search` (which embeds every query in one batched call
    up front, since each embedding call is a network round trip and
    embedding them one at a time serialises that latency across queries).

    Args:
        query: The search query, for keyword search.
        query_vector: `query`'s embedding, for semantic search.
        report_ids: Restrict the search to these reports, or None to
            search every ready report.
        chunks_repo: Repository providing keyword search and chunk lookup.
        vector_index: The in-memory index providing semantic search.
        top_k: The maximum number of results to keep from each of the
            keyword and semantic searches before fusion.
        final_top_k: The maximum number of fused results to return.
            Defaults to `top_k` if None.

    Returns:
        Fused chunk ids, best match first, length at most `final_top_k`.
    """
    fts_query = build_fts_query(query)
    keyword_ids = [cid for cid, _ in chunks_repo.keyword_search(fts_query, report_ids, top_k)]

    report_chunk_ids = None
    if report_ids is not None:
        report_chunk_ids = [
            chunk.id for report_id in report_ids for chunk in chunks_repo.get_by_report(report_id)
        ]
    vector_ids = [cid for cid, _ in vector_index.search(query_vector, report_chunk_ids, top_k)]

    merged_results = reciprocal_rank_fusion([keyword_ids, vector_ids], top_k=final_top_k or top_k)
    return [chunk_id for chunk_id, _ in merged_results]


def hybrid_search(
    query: str,
    report_ids: list[int] | None,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    llm: LLMClient,
    top_k: int = TOP_K_PER_SEARCH,
    final_top_k: int | None = None,
) -> list[int]:
    """Fuses BM25 and cosine search for one query.

    Args:
        query: The search query (a question, or one of pre-extraction's
            targeted queries).
        report_ids: Restrict the search to these reports, or None to
            search every ready report.
        chunks_repo: Repository providing keyword search and chunk lookup.
        vector_index: The in-memory index providing semantic search.
        llm: The LLM client used to embed `query`.
        top_k: The maximum number of results to keep from each of the
            keyword and semantic searches before fusion.
        final_top_k: The maximum number of fused results to return.
            Defaults to `top_k` (chat wants 20 per search but only the top
            8 after fusion, hence the separate parameter).

    Returns:
        Fused chunk ids, best match first, length at most `final_top_k`.
    """
    query_vector = llm.embed([query])[0]
    return _hybrid_search_with_vector(
        query, query_vector, report_ids, chunks_repo, vector_index, top_k, final_top_k
    )


def multi_query_search(
    queries: list[str],
    report_ids: list[int] | None,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    llm: LLMClient,
    top_k_per_query: int = TOP_K_PER_SEARCH,
    final_top_k: int = 8,
) -> list[int]:
    """Runs `hybrid_search` for each query and fuses the results into one list.

    Pre-extraction issues several targeted queries per field to improve
    recall; this is what lets it do so with one fused result.

    Embeds every query in a single batched call rather than one call per
    query — an embedding call is a network round trip, and pre-extraction's
    5-8 queries otherwise serialise that latency one after another for no
    benefit (the embedding model batches a request's texts internally).

    Args:
        queries: The queries to run, e.g. several phrasings for "FTE".
        report_ids: Restrict the search to these reports, or None to
            search every ready report.
        chunks_repo: Repository providing keyword search and chunk lookup.
        vector_index: The in-memory index providing semantic search.
        llm: The LLM client used to embed each query.
        top_k_per_query: The maximum number of results to keep per query
            before the final fusion.
        final_top_k: The maximum number of results to return after fusing
            every query's results together.

    Returns:
        Fused chunk ids, best match first, length at most `final_top_k`.
    """
    if not queries:
        return []
    query_vectors = llm.embed(queries)
    ranked_lists = [
        _hybrid_search_with_vector(
            query, query_vector, report_ids, chunks_repo, vector_index, top_k_per_query, None
        )
        for query, query_vector in zip(queries, query_vectors, strict=True)
    ]
    merged_results = reciprocal_rank_fusion(ranked_lists, top_k=final_top_k)
    return [chunk_id for chunk_id, _ in merged_results]
