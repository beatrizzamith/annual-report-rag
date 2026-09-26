"""Builds the evidence text passed to the model in a token-limited format."""

from pydantic import BaseModel, Field

from app.db.models import Chunk, Report

CHARS_PER_TOKEN = 4


class AssembledSource(BaseModel):
    """One chunk selected for the model's context, tagged with a source id."""

    source_id: str = Field(description='Tag the model cites, e.g. "S3".')
    chunk_id: int
    report_label: str
    company: str
    year: int
    pages: str
    kind: str
    text: str


def _page_label(page_start: int, page_end: int) -> str:
    """Formats a page range for a source tag.

    Args:
        page_start: The chunk's first physical page.
        page_end: The chunk's last physical page.

    Returns:
        `"12"` when the range is a single page, otherwise `"12-13"`.
    """
    return str(page_start) if page_start == page_end else f"{page_start}-{page_end}"


def assemble_context(
    ranked_chunks: list[Chunk], reports_by_id: dict[int, Report], token_budget: int
) -> list[AssembledSource]:
    """Selects and orders the chunks that will make up the model's context.

    Deduplicates by chunk id, fills in rank order up to `token_budget`
    (dropping whole chunks that would not fit rather than truncating one),
    then reorders by report and page so each report's evidence reads in
    document order.

    Args:
        ranked_chunks: Retrieved chunks, best match first.
        reports_by_id: Every ready report, keyed by id, for the source tags.
        token_budget: The approximate token budget for all sources combined
            (characters divided by `CHARS_PER_TOKEN`).

    Returns:
        The selected chunks as tagged `AssembledSource` entries, ordered by
        report then page (not by rank).
    """
    seen_chunk_ids = set()
    unique_chunks = []
    for chunk in ranked_chunks:
        if chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.id)
        unique_chunks.append(chunk)

    budget_chars = token_budget * CHARS_PER_TOKEN
    selected_chunks = []
    total_chars = 0
    for chunk in unique_chunks:
        if total_chars + len(chunk.text) > budget_chars:
            continue
        selected_chunks.append(chunk)
        total_chars += len(chunk.text)

    selected_chunks.sort(key=lambda chunk: (chunk.report_id, chunk.page_start))

    assembled_sources = []
    for index, chunk in enumerate(selected_chunks, start=1):
        report = reports_by_id[chunk.report_id]
        assembled_sources.append(
            AssembledSource(
                source_id=f"S{index}",
                chunk_id=chunk.id,
                report_label=f"{report.company} Annual Report {report.fiscal_year}",
                company=report.company,
                year=report.fiscal_year,
                pages=_page_label(chunk.page_start, chunk.page_end),
                kind=chunk.kind,
                text=chunk.text,
            )
        )
    return assembled_sources


def render_context_text(sources: list[AssembledSource]) -> str:
    """Renders assembled sources as the `<source>`-tagged text for a prompt.

    Args:
        sources: Sources produced by `assemble_context`.

    Returns:
        The sources rendered as `<source>` blocks, separated by blank
        lines. Empty string if `sources` is empty.
    """
    blocks = [
        f'<source id="{source.source_id}" report="{source.report_label}" '
        f'company="{source.company}" year="{source.year}" pages="{source.pages}" '
        f'kind="{source.kind}">\n{source.text}\n</source>'
        for source in sources
    ]
    return "\n\n".join(blocks)
