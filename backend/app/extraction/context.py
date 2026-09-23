"""Builds the `<source>`-tagged context text handed to extraction prompts."""

from app.db.models import Chunk


def _page_label(chunk: Chunk) -> str:
    """Formats a chunk's page range for display in its `<source>` tag.

    Args:
        chunk: The chunk to label.

    Returns:
        `"12"` when the chunk is on a single page, otherwise `"12-13"`.
    """
    return (
        str(chunk.page_start)
        if chunk.page_start == chunk.page_end
        else (f"{chunk.page_start}-{chunk.page_end}")
    )


def build_source_context(chunks: list[Chunk]) -> tuple[str, dict[str, int]]:
    """Renders retrieved chunks as `<source>`-tagged text for a prompt.

    Args:
        chunks: The chunks to include, in the order they should appear.

    Returns:
        A tuple of the context text and a `{source_id: chunk_id}` map, so
        the model's chosen `source_id` can be resolved back to a chunk.
    """
    blocks = []
    source_map = {}
    for index, chunk in enumerate(chunks, start=1):
        source_id = f"S{index}"
        # The model returns a source tag like "S3"; this map restores the
        # original chunk id so a later verification step can point back to the
        # exact text that produced the extracted fact.
        source_map[source_id] = chunk.id
        blocks.append(
            f'<source id="{source_id}" pages="{_page_label(chunk)}" kind="{chunk.kind}">\n'
            f"{chunk.text}\n</source>"
        )
    return "\n\n".join(blocks), source_map
