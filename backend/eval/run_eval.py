"""Runs the gold set through the real retrieval and generation pipeline and reports hit rates.

    python -m eval.run_eval

Runs against a temporary copy of the database: `answer_question` stores
messages, and an eval must not add its questions to the real chat history.
"""

import argparse
import json
import logging
import sqlite3
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from app.answer.service import CONTEXT_TOP_K, SEARCH_TOP_K, answer_question
from app.api.deps import build_vector_index
from app.config import get_settings
from app.db.connection import connect
from app.db.models import Report
from app.db.repositories import ChunksRepo, ReportsRepo
from app.extraction.verifier import verify_quote
from app.llm.client import LLMClient, build_llm_client
from app.retrieval.hybrid import hybrid_search
from app.retrieval.scope import scope_reports
from app.retrieval.vector_index import VectorIndex
from eval.build_gold_set import GoldSetEntry
from eval.console import configure_console_logging

logger = logging.getLogger(__name__)


class QuestionResult(BaseModel):
    """One gold question's outcome against retrieval and generation."""

    question: str
    company: str
    page_start: int
    retrieval_hit: bool
    answer_status: str
    generation_grounded: bool = Field(
        description="True when a verified citation backs the expected quote."
    )


def _quotes_overlap(first_quote: str, second_quote: str) -> bool:
    """Checks whether two quotes plausibly refer to the same source text.

    Uses the app's own verification comparison, so it ignores the same
    punctuation, whitespace and table-formatting differences the gold set's
    expected quotes were validated with.

    Args:
        first_quote: A quote.
        second_quote: Another quote to compare it against.

    Returns:
        True if either quote is found inside the other.
    """
    return verify_quote(first_quote, second_quote) or verify_quote(second_quote, first_quote)


def _copy_database(source_path: Path) -> Path:
    """Copies the app database to a temporary file, WAL-safe.

    Args:
        source_path: Path to the live app database.

    Returns:
        Path to the temporary copy. The caller is responsible for deleting
        it (or its containing directory) when done.
    """
    tmp_path = Path(tempfile.mkdtemp(prefix="annual_report_eval_")) / "eval.db"
    source_conn = sqlite3.connect(source_path)
    dest_conn = sqlite3.connect(tmp_path)
    try:
        source_conn.backup(dest_conn)
    finally:
        source_conn.close()
        dest_conn.close()
    return tmp_path


def _check_retrieval(
    entry: GoldSetEntry,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    llm: LLMClient,
    ready_reports: list[Report],
) -> bool:
    """Checks whether the same retrieval chat uses finds the expected chunk.

    Args:
        entry: One gold-set entry.
        chunks_repo: Repository for keyword search and chunk lookup.
        vector_index: The in-memory index used for semantic search.
        llm: The LLM client used to embed the query.
        ready_reports: Every currently `ready` report, for scoping.

    Returns:
        True if any chunk in the fused top results contains
        `entry["expected_quote"]` verbatim.
    """
    scoped_report_ids = scope_reports(entry["question"], ready_reports)
    chunk_ids = hybrid_search(
        entry["question"],
        scoped_report_ids,
        chunks_repo,
        vector_index,
        llm,
        top_k=SEARCH_TOP_K,
        final_top_k=CONTEXT_TOP_K,
    )
    looked_up_chunks = [chunks_repo.get(chunk_id) for chunk_id in chunk_ids]
    retrieved_chunks = [chunk for chunk in looked_up_chunks if chunk is not None]
    return any(_quotes_overlap(entry["expected_quote"], chunk.text) for chunk in retrieved_chunks)


def run_eval(gold_set_path: Path, db_path: Path | None = None) -> list[QuestionResult]:
    """Runs every gold-set question through retrieval and full generation.

    Args:
        gold_set_path: Path to the gold-set JSON built by
            `eval/build_gold_set.py`.
        db_path: The database to evaluate against. Defaults to the live app
            database; pass a different one (e.g. a database re-ingested with
            an alternative chunker) to evaluate that instead -- the gold
            set's questions and expected quotes are plain text, not tied to
            any particular database's row ids, so the same gold set works
            against either.

    Returns:
        One `QuestionResult` per gold-set entry.

    Raises:
        FileNotFoundError: No gold set exists at `gold_set_path` yet; run
            `eval/build_gold_set.py` first.
    """
    entries: list[GoldSetEntry] = json.loads(gold_set_path.read_text(encoding="utf-8"))
    settings = get_settings()
    db_copy_path = _copy_database(db_path or settings.db_path)

    conn = connect(db_copy_path)
    llm = build_llm_client(settings)
    chunks_repo = ChunksRepo(conn)
    ready_reports = ReportsRepo(conn).list_ready()
    vector_index = build_vector_index(conn, settings.embedding_dim)

    results = []
    for entry in entries:
        retrieval_hit = _check_retrieval(entry, chunks_repo, vector_index, llm, ready_reports)

        chat_result = answer_question(
            entry["question"], conn, vector_index, llm, settings.context_token_budget
        )
        generation_grounded = any(
            citation.verified and _quotes_overlap(entry["expected_quote"], citation.quote)
            for citation in chat_result.citations
        )

        results.append(
            QuestionResult(
                question=entry["question"],
                company=entry["company"],
                page_start=entry["page_start"],
                retrieval_hit=retrieval_hit,
                answer_status=chat_result.status,
                generation_grounded=generation_grounded,
            )
        )
    return results


def summarize(results: list[QuestionResult]) -> dict:
    """Builds the summary dict shared by the console report and the JSON report.

    Args:
        results: Every gold question's outcome.

    Returns:
        A dict with total/rate counts and the per-question results, suitable
        for `json.dump`.
    """
    total = len(results)
    retrieval_hits = sum(result.retrieval_hit for result in results)
    grounded = sum(result.generation_grounded for result in results)
    not_found = sum(result.answer_status == "not_found" for result in results)

    return {
        "total": total,
        "retrieval_hit_rate": retrieval_hits / total if total else None,
        "retrieval_hits": retrieval_hits,
        "generation_grounded_rate": grounded / total if total else None,
        "generation_grounded": grounded,
        "answered_not_found": not_found,
        "results": [
            {
                "question": result.question,
                "company": result.company,
                "page_start": result.page_start,
                "retrieval_hit": result.retrieval_hit,
                "answer_status": result.answer_status,
                "generation_grounded": result.generation_grounded,
            }
            for result in results
        ],
    }


def format_report(results: list[QuestionResult]) -> str:
    """Formats hit-rate summaries and every failing question, for manual review.

    Args:
        results: Every gold question's outcome.

    Returns:
        A multi-line, human-readable report for the caller to log.
    """
    total = len(results)
    if total == 0:
        return "No gold-set entries to evaluate."

    retrieval_hits = sum(result.retrieval_hit for result in results)
    grounded = sum(result.generation_grounded for result in results)
    not_found = sum(result.answer_status == "not_found" for result in results)

    lines = [
        f"\n{total} gold-set questions",
        f"  retrieval hit rate:   {retrieval_hits}/{total} ({retrieval_hits / total:.0%})",
        f"  generation grounded:  {grounded}/{total} ({grounded / total:.0%})",
        f"  answered not_found:   {not_found}/{total}",
    ]

    failures = [
        result for result in results if not result.retrieval_hit or not result.generation_grounded
    ]
    if failures:
        lines.append(f"\n{len(failures)} question(s) worth a manual look:")
        for failure in failures:
            flags = []
            if not failure.retrieval_hit:
                flags.append("retrieval miss")
            if not failure.generation_grounded:
                flags.append("ungrounded/unverified answer")
            lines.append(
                f"  [{', '.join(flags)}] {failure.company} p.{failure.page_start}: "
                f'"{failure.question}"'
            )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point: parses arguments, runs the eval, logs the report."""
    configure_console_logging(logger)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-set",
        type=Path,
        default=None,
        help="Path to the gold-set JSON. Defaults to <data_dir>/eval/gold_set.json, matching"
        " eval/build_gold_set.py's default output location.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Path to write the JSON results report. Defaults to <data_dir>/eval/results.json,"
        " next to the gold set.",
    )
    args = parser.parse_args()
    args.gold_set = args.gold_set or (get_settings().data_dir / "eval" / "gold_set.json")
    args.json_out = args.json_out or (get_settings().data_dir / "eval" / "results.json")

    if not args.gold_set.exists():
        raise FileNotFoundError(
            f"No gold set at {args.gold_set}. Run `python -m eval.build_gold_set` first."
        )

    results = run_eval(args.gold_set)
    logger.info("%s", format_report(results))
    args.json_out.write_text(json.dumps(summarize(results), indent=2), encoding="utf-8")
    logger.info("\nWrote JSON report to %s", args.json_out)


if __name__ == "__main__":
    main()
