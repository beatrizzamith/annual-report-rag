"""Experimental chunker that groups paragraphs at embedding-similarity breakpoints.

Not used by ingestion; `eval/compare_chunking.py` compares it with the
production paragraph packer. Table handling and run boundaries are reused from
`app.ingestion.chunk`; only how paragraphs are grouped differs.
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

# Break between paragraphs whose similarity is in the bottom quartile of the run,
# i.e. where the topic shifts most.
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
        return [PreparedChunk(page_start=page, page_end=page, kind="text", text=text)]

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
        PreparedChunk(
            page_start=group[0][0],
            page_end=group[-1][0],
            kind="text",
            text="\n\n".join(text for _, text in group),
        )
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


def _close_run(
    current_run: list[tuple[int, str, int]],
    runs: list[list[tuple[int, str]]],
    ordered: list[tuple],
) -> None:
    """Moves a finished run into `runs` and records its place in `ordered`.

    Args:
        current_run: The run being built, as `(page, text, region_index)`.
            Left for the caller to reset.
        runs: Every closed run so far, as `(page, text)` lists; extended in place.
        ordered: The document's runs and tables in reading order; extended
            in place with `("run", index_into_runs)`.
    """
    if not current_run:
        return
    runs.append([(page, text) for page, text, _ in current_run])
    ordered.append(("run", len(runs) - 1))


def _split_into_runs(
    parsed: ParsedDocument,
) -> tuple[list[list[tuple[int, str]]], list[tuple]]:
    """Splits a document into paragraph runs and tables, in reading order.

    A run is consecutive paragraphs from one page region; a table or a
    region change ends the current run.

    Args:
        parsed: The document's parsed text and table items, in reading order.

    Returns:
        `(runs, ordered)`: every run as `(page, text)` paragraphs, and the
        reading-order sequence of `("run", run_index)` and
        `("table", item, caption)` entries.
    """
    runs: list[list[tuple[int, str]]] = []
    ordered: list[tuple] = []
    current_run: list[tuple[int, str, int]] = []

    for item in parsed.items:
        if isinstance(item, TextItem):
            if current_run and current_run[-1][2] != item.region_index:
                _close_run(current_run, runs, ordered)
                current_run = []
            current_run.append((item.page, item.text, item.region_index))
            continue
        caption = _caption_from_paragraph(current_run[-1][1]) if current_run else None
        _close_run(current_run, runs, ordered)
        current_run = []
        ordered.append(("table", item, caption))
    _close_run(current_run, runs, ordered)
    return runs, ordered


def _embed_run_paragraphs(
    runs: list[list[tuple[int, str]]], embed: EmbedFn
) -> dict[tuple[int, int], list[float]]:
    """Embeds every under-cap paragraph of every run in one batched pass.

    Args:
        runs: The document's paragraph runs.
        embed: The embedding function.

    Returns:
        Each embedded paragraph's vector, keyed by
        `(run_index, paragraph_index)`. Paragraphs over `HARD_MAX_CHARS`
        are split without embedding, so they have no entry.
    """
    keys: list[tuple[int, int]] = []
    texts: list[str] = []
    for run_index, paragraphs in enumerate(runs):
        for paragraph_index, (_, text) in enumerate(paragraphs):
            if len(text) <= HARD_MAX_CHARS:
                keys.append((run_index, paragraph_index))
                texts.append(text)
    vectors = _embed_in_batches(texts, embed) if texts else []
    return dict(zip(keys, vectors, strict=True))


def chunk_document_semantic(parsed: ParsedDocument, embed: EmbedFn) -> list[PreparedChunk]:
    """Turns a parsed document into chunks, grouping paragraphs semantically.

    Tables and paragraph runs are handled as in `chunk_document`; only the
    grouping differs (embedding-similarity breakpoints, not a size target).
    Paragraphs are embedded in one batched pass up front, not per run.

    Args:
        parsed: The document's parsed text and table items, in reading order.

    Returns:
        The prepared chunks, in document order.
    """
    runs, ordered = _split_into_runs(parsed)
    vector_by_key = _embed_run_paragraphs(runs, embed)

    prepared_chunks: list[PreparedChunk] = []
    for entry in ordered:
        if entry[0] == "table":
            _, table_item, caption = entry
            prepared_chunks.extend(_chunk_table(table_item, caption))
        else:
            _, run_index = entry
            prepared_chunks.extend(_chunk_run(runs[run_index], run_index, vector_by_key))
    return prepared_chunks
