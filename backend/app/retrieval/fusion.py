"""Combines ranked result lists without requiring score calibration."""

RRF_DAMPING_CONSTANT = 60 # Conventional value for the RRF damping constant (also known as K).


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], top_k: int, k: int = RRF_DAMPING_CONSTANT
) -> list[tuple[int, float]]:
    """Fuses several ranked lists of chunk ids into one.

    Args:
        ranked_lists: Ranked lists of chunk ids, each ordered best-first
            (e.g. one from BM25, one from cosine similarity).
        top_k: The maximum number of results to return.
        k: The RRF damping constant (conventionally called
            `k`); higher values reduce the influence of rank differences
            further down each list.

    Returns:
        `(chunk_id, fused_score)` pairs, ordered best-first, of length at
        most `top_k`.
    """
    fused_scores = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                k + rank + 1
            )
    ranked_pairs = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked_pairs[:top_k]
