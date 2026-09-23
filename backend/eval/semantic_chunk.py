"""Experimental chunker that groups paragraphs at embedding-similarity
breakpoints instead of a fixed character target.

Not used by the real ingestion pipeline -- this exists to answer one
question empirically, via `eval/compare_chunking.py`: does semantic
grouping retrieve better than the paragraph packer already in production
(`app.ingestion.chunk.chunk_document`)? Table handling and paragraph-run
boundaries (page, region) are reused as-is from that module; only how a
run's paragraphs are grouped into chunks differs.
"""

from collections.abc import Callable

import numpy as np

from app.ingestion.chunk import (
    HARD_MAX_CHARS,
    PreparedChunk,
    _caption_from_paragraph,
    _chunk_table,
    _paragraph_buffer_length,
    _split_long_paragraph,
)
from app.ingestion.parse import ParsedDocument, TextItem

EmbedFn = Callable[[list[str]], list[list[float]]]

# Break between two paragraphs when their embedding similarity falls in the
# bottom quartile of their run's similarities -- breakpoints land wherever
# the topic shifts most, rather than at a fixed size.
_BREAKPOINT_PERCENTILE = 25
_EMBED_BATCH_SIZE = 256


def _embed_in_batches(texts: list[str], embed: EmbedFn) -> list[list[float]]:
    """Embeds every text in as few batched calls as possible.

    Args:
        texts: The texts to embed, in order.
        embed: The embedding function (a provider round trip per batch).

    Returns:
        One embedding vector per text, in the same order.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        vectors.extend(embed(texts[start : start + _EMBED_BATCH_SIZE]))
    return vectors


def _cosine_similarities(vectors: list[list[float]]) -> list[float]:
    """Computes cosine similarity between each consecutive pair of vectors.

    Args:
        vectors: Embedding vectors, in paragraph order.

    Returns:
        `len(vectors) - 1` similarities; `result[i]` is the similarity
        between `vectors[i]` and `vectors[i + 1]`.
    """
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    return [float(np.dot(unit[i], unit[i + 1])) for i in range(len(unit) - 1)]


def _semantic_groups_to_chunks(
    paragraphs: list[tuple[int, str]], vectors: list[list[float]]
) -> list[PreparedChunk]:
    """Splits one buffer of under-cap paragraphs at its weakest similarity points.

    Args:
        paragraphs: Consecutive paragraphs as `(page, text)`, none exceeding
            `HARD_MAX_CHARS` on its own.
        vectors: One embedding per paragraph, same order.

    Returns:
        One text chunk per group, `HARD_MAX_CHARS` still enforced as a
        ceiling regardless of similarity.
    """
    if len(paragraphs) == 1:
        page, text = paragraphs[0]
        return [PreparedChunk(page, page, "text", text)]

    similarities = _cosine_similarities(vectors)
    threshold = float(np.percentile(similarities, _BREAKPOINT_PERCENTILE))

    groups: list[list[tuple[int, str]]] = [[paragraphs[0]]]
    for i in range(1, len(paragraphs)):
        candidate_length = _paragraph_buffer_length(groups[-1]) + 2 + len(paragraphs[i][1])
        if similarities[i - 1] < threshold or candidate_length > HARD_MAX_CHARS:
            groups.append([paragraphs[i]])
        else:
            groups[-1].append(paragraphs[i])

    return [
        PreparedChunk(group[0][0], group[-1][0], "text", "\n\n".join(text for _, text in group))
        for group in groups
    ]


def _chunk_run(
    paragraphs: list[tuple[int, str]],
    run_index: int,
    vector_by_key: dict[tuple[int, int], list[float]],
) -> list[PreparedChunk]:
    """Chunks one run of consecutive same-region paragraphs.

    Args:
        paragraphs: The run's paragraphs as `(page, text)`, in order.
        run_index: This run's index, matching the keys in `vector_by_key`.
        vector_by_key: Every under-cap paragraph's embedding, keyed by
            `(run_index, paragraph_index)`.

    Returns:
        The run's paragraphs turned into one or more text chunks.
    """
    chunks: list[PreparedChunk] = []
    buffer: list[tuple[int, str]] = []
    buffer_vectors: list[list[float]] = []

    for paragraph_index, (page, text) in enumerate(paragraphs):
        if len(text) > HARD_MAX_CHARS:
            if buffer:
                chunks.extend(_semantic_groups_to_chunks(buffer, buffer_vectors))
                buffer, buffer_vectors = [], []
            chunks.extend(_split_long_paragraph(page, text))
            continue
        buffer.append((page, text))
        buffer_vectors.append(vector_by_key[(run_index, paragraph_index)])

    if buffer:
        chunks.extend(_semantic_groups_to_chunks(buffer, buffer_vectors))
    return chunks


def chunk_document_semantic(parsed: ParsedDocument, embed: EmbedFn) -> list[PreparedChunk]:
    """Turns a parsed document into chunks, grouping paragraphs semantically.

    Tables are kept whole and paragraph runs are split on page-region
    changes exactly as `app.ingestion.chunk.chunk_document` does; only how a
    run's paragraphs are grouped into chunks differs (embedding-similarity
    breakpoints instead of a fixed character target). Every under-cap
    paragraph in the document is embedded in one batched pass up front, not
    one embedding call per run, so this scales the same way production
    embedding does.

    Args:
        parsed: The document's parsed text and table items, in reading order.

    Returns:
        The prepared chunks, in document order.
    """
    runs: list[list[tuple[int, str]]] = []
    ordered: list[tuple] = []
    current_run: list[tuple[int, str, int]] = []

    def close_run() -> None:
        nonlocal current_run
        if current_run:
            runs.append([(page, text) for page, text, _ in current_run])
            ordered.append(("run", len(runs) - 1))
            current_run = []

    for item in parsed.items:
        if isinstance(item, TextItem):
            if current_run and current_run[-1][2] != item.region_index:
                close_run()
            current_run.append((item.page, item.text, item.region_index))
            continue
        caption = _caption_from_paragraph(current_run[-1][1]) if current_run else None
        close_run()
        ordered.append(("table", item, caption))
    close_run()

    embed_keys: list[tuple[int, int]] = []
    embed_texts: list[str] = []
    for run_index, paragraphs in enumerate(runs):
        for paragraph_index, (_, text) in enumerate(paragraphs):
            if len(text) <= HARD_MAX_CHARS:
                embed_keys.append((run_index, paragraph_index))
                embed_texts.append(text)
    vectors = _embed_in_batches(embed_texts, embed) if embed_texts else []
    vector_by_key = dict(zip(embed_keys, vectors, strict=True))

    prepared_chunks: list[PreparedChunk] = []
    for entry in ordered:
        if entry[0] == "table":
            _, item, caption = entry
            prepared_chunks.extend(_chunk_table(item, caption))
        else:
            _, run_index = entry
            prepared_chunks.extend(_chunk_run(runs[run_index], run_index, vector_by_key))
    return prepared_chunks
