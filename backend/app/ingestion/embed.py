"""Embeds chunks and keeps the in-memory vector index in sync with the database."""

import logging
from collections.abc import Callable

import numpy as np

from app.db.repositories import ChunksRepo
from app.llm.client import LLMClient
from app.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)

# Fewer, larger batches mean fewer sequential network round trips per
# report. 256 chunks at up to 700 tokens each is at most ~180k tokens per
# request, comfortably under embedding endpoints' per-request limits (which
# cap at 2048 items regardless). Diminishing returns past this: measured on
# a 462-page report, the whole embed stage was ~15% of total ingestion time,
# dwarfed by parsing — bigger batches shave seconds off a bottleneck that
# isn't the bottleneck.
BATCH_SIZE = 256


def vector_to_blob(vector: list[float]) -> bytes:
    """Serialises an embedding vector for storage in `chunks.embedding`.

    Args:
        vector: An embedding vector.

    Returns:
        The vector serialised as float32 bytes.
    """
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> list[float]:
    """Deserialises an embedding vector previously stored by `vector_to_blob`.

    Args:
        blob: Bytes previously produced by `vector_to_blob`.

    Returns:
        The deserialised embedding vector.
    """
    return np.frombuffer(blob, dtype=np.float32).tolist()


def embed_and_index_report(
    report_id: int,
    llm: LLMClient,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Embeds every not-yet-embedded chunk of a report, in batches.

    Writes each embedding back to SQLite and adds it to the shared
    in-memory vector index.

    Args:
        report_id: The report whose chunks to embed.
        llm: The LLM client used to compute embeddings.
        chunks_repo: Repository for reading chunks and writing embeddings.
        vector_index: The shared in-memory index new embeddings are added to.
        on_progress: Optional callback invoked with the completion fraction
            (in [0, 1]) after each batch.
    """
    chunks = [c for c in chunks_repo.get_by_report(report_id) if c.embedding is None]
    total = len(chunks)
    if total == 0:
        if on_progress:
            on_progress(1.0)
        return

    done = 0
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = llm.embed([c.embed_text for c in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            chunks_repo.set_embedding(chunk.id, vector_to_blob(vector))
        # One add_many per batch, not one add() per chunk (see VectorIndex.add_many).
        vector_index.add_many([c.id for c in batch], vectors)
        chunks_repo.commit()
        done += len(batch)
        if on_progress:
            on_progress(done / total)
        logger.info(
            "embedding batch complete",
            extra={"extra_fields": {"report_id": report_id, "done": done, "total": total}},
        )
