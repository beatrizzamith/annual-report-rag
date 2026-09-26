"""Embeds chunks and keeps the in-memory vector index in sync with the database."""

import logging
from collections.abc import Callable

import numpy as np

from app.db.repositories import ChunksRepo
from app.llm.client import LLMClient
from app.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)

# Larger batches mean fewer network round trips; 256 chunks stays well under
# the embedding endpoint's per-request limits. Embedding is a small share of
# ingestion time, so bigger batches would gain little.
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
    chunks = [chunk for chunk in chunks_repo.get_by_report(report_id) if chunk.embedding is None]
    total = len(chunks)
    if total == 0:
        if on_progress:
            on_progress(1.0)
        return

    done = 0
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = llm.embed([chunk.embed_text for chunk in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            chunks_repo.set_embedding(chunk.id, vector_to_blob(vector))
        vector_index.add_many([chunk.id for chunk in batch], vectors)
        chunks_repo.commit()
        done += len(batch)
        if on_progress:
            on_progress(done / total)
        logger.info(
            "embedding batch complete",
            extra={"extra_fields": {"report_id": report_id, "done": done, "total": total}},
        )
