"""Builds a gold-standard question set from a report's own chunks, for
evaluating retrieval and generation (see `eval/run_eval.py`).

Run against reports already ingested into the real database:

    python -m eval.build_gold_set

Read-only against the app database: only SELECT queries are issued (via
`ReportsRepo`/`ChunksRepo`), so it never touches real chat history or
report data.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from app.config import get_settings
from app.core.prompts import load_prompt
from app.db.connection import connect
from app.db.models import Chunk, Report
from app.db.repositories import ChunksRepo, ReportsRepo
from app.extraction.verifier import verify_quote
from app.llm.client import LLMClient, build_llm_client

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SAMPLES_PER_REPORT = 6
# Below the report-wide median chunk length (see chunk.py's ~1800-char
# target): a short chunk is disproportionately likely to be a table of
# contents entry, a cover page, or a disclosure index rather than real
# prose or a populated table, and the model correctly (but wastefully)
# marks most of those `usable=False`.
MIN_CHUNK_CHARS = 400


class EvalQuestion(BaseModel):
    """One candidate question the model wrote from a chunk, before verification."""

    usable: bool
    question: str = ""
    expected_quote: str = ""


class GoldSetEntry(TypedDict):
    """One verified gold-set question, as written to and read back from JSON."""

    report_id: int
    company: str
    fiscal_year: int
    chunk_id: int
    page_start: int
    page_end: int
    kind: str
    question: str
    expected_quote: str


def _sample_chunks(chunks: list[Chunk], sample_size: int) -> list[Chunk]:
    """Picks a small, representative sample of one report's chunks.

    Args:
        chunks: The report's chunks, in page order.
        sample_size: The maximum number to pick.

    Returns:
        Up to `sample_size` chunks: one table chunk if the report has any,
        the rest text chunks spread evenly across the report rather than
        clustered at the start.
    """
    usable_chunks = [c for c in chunks if len(c.text) >= MIN_CHUNK_CHARS]
    if len(usable_chunks) <= sample_size:
        return usable_chunks

    sample = []
    table_chunks = [c for c in usable_chunks if c.kind == "table"]
    if table_chunks:
        # Longest table text is a simple proxy for "most populated with
        # real data", as opposed to a sparse disclosure index with mostly
        # blank cells.
        sample.append(max(table_chunks, key=lambda c: len(c.text)))

    text_chunks = [c for c in usable_chunks if c.kind == "text"]
    remaining = max(0, sample_size - len(sample))
    step = max(1, len(text_chunks) // remaining) if remaining else 1
    sample.extend(text_chunks[::step][:remaining])
    return sample


def _generate_question(
    chunk: Chunk, report: Report, system_prompt: str, llm: LLMClient
) -> GoldSetEntry | None:
    """Asks the model for one question-and-answer pair grounded in one chunk.

    Args:
        chunk: The chunk to write a question from.
        report: The chunk's report, for the company/year shown to the model.
        system_prompt: The loaded `eval_question.md` prompt.
        llm: The LLM client used for the generation call.

    Returns:
        A gold-set entry dict, or None if the model judged the chunk
        unusable, or its `expected_quote` did not verify verbatim against
        the chunk's text (a bad gold entry is worse than a missing one).
    """
    user_prompt = (
        f'<excerpt company="{report.company}" year="{report.fiscal_year}" '
        f'page="{chunk.page_start}">\n{chunk.text}\n</excerpt>'
    )
    result = llm.complete_structured(system_prompt, user_prompt, EvalQuestion, temperature=0.3)

    if not result.usable or not result.question or not result.expected_quote:
        return None
    if not verify_quote(result.expected_quote, chunk.text):
        logger.warning(
            "dropped: model's expected_quote did not verify verbatim (chunk %s)", chunk.id
        )
        return None

    return {
        "report_id": report.id,
        "company": report.company,
        "fiscal_year": report.fiscal_year,
        "chunk_id": chunk.id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "kind": chunk.kind,
        "question": result.question,
        "expected_quote": result.expected_quote,
    }


def build_gold_set(samples_per_report: int) -> list[GoldSetEntry]:
    """Builds gold-set entries for every ready report in the database.

    Args:
        samples_per_report: How many chunks to sample per report.

    Returns:
        One entry per sampled chunk that produced a verified question,
        across every `ready` report.

    Raises:
        LLMAuthFailedError: No LLM provider is configured.
    """
    settings = get_settings()
    conn = connect(settings.db_path)
    llm = build_llm_client(settings)
    reports_repo = ReportsRepo(conn)
    chunks_repo = ChunksRepo(conn)
    system_prompt = load_prompt("eval_question.md")

    entries = []
    for report in reports_repo.list_ready():
        chunks = _sample_chunks(chunks_repo.get_by_report(report.id), samples_per_report)
        logger.info("%s (%d): sampling %d chunks", report.company, report.fiscal_year, len(chunks))
        for chunk in chunks:
            entry = _generate_question(chunk, report, system_prompt, llm)
            if entry:
                entries.append(entry)
    return entries


def main() -> None:
    """CLI entry point: parses arguments, builds the gold set, writes it to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-per-report",
        type=int,
        default=DEFAULT_SAMPLES_PER_REPORT,
        help="How many chunks to sample per ready report (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the gold-set JSON. Defaults to <data_dir>/eval/gold_set.json"
        " -- inside the already-gitignored data directory, since the gold set quotes real,"
        " potentially private report content verbatim.",
    )
    args = parser.parse_args()

    out_path = args.out or (get_settings().data_dir / "eval" / "gold_set.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = build_gold_set(args.samples_per_report)
    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    logger.info("wrote %d gold-set entries to %s", len(entries), out_path)


if __name__ == "__main__":
    main()
