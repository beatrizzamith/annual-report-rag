"""Empirically compares the production paragraph chunker against the
experimental semantic chunker (`eval/semantic_chunk.py`), on the same gold
set.

Re-ingests every ready report's PDF into a second, throwaway SQLite database
using the semantic chunker -- the real app database and its chunks are never
touched -- then runs `eval.run_eval`'s exact retrieval/generation checks
against both databases and prints a side-by-side comparison.

    python -m eval.compare_chunking

Requires a gold set already built (`python -m eval.build_gold_set`) and
ready reports already ingested in the real database (their PDFs are re-read
from `settings.pdfs_dir`, so re-ingestion costs one parse + one full
re-embed per report, same as a normal upload).
"""

import argparse
import json
import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.db.connection import connect
from app.db.repositories import ChunksRepo, ReportsRepo
from app.ingestion.chunk import build_embed_text
from app.ingestion.embed import embed_and_index_report
from app.ingestion.parse import parse_pdf
from app.llm.client import LLMClient, build_llm_client
from app.retrieval.vector_index import VectorIndex
from eval.run_eval import print_report, run_eval, summarize
from eval.semantic_chunk import chunk_document_semantic

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _delete_database(db_path: Path) -> None:
    """Removes a SQLite database file and its WAL/SHM siblings, if present.

    Args:
        db_path: Path to the database file to remove.
    """
    for path in (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        path.unlink(missing_ok=True)


def rebuild_with_semantic_chunker(db_path: Path, settings: Settings, llm: LLMClient) -> None:
    """Re-ingests every ready report into a fresh database, chunked semantically.

    Reads each report's PDF from `settings.pdfs_dir` (the same files the
    real database's rows point at) and re-runs parse, chunk and embed with
    `chunk_document_semantic` in place of the production chunker. Extraction
    (FTE, goals) is skipped -- `run_eval` only exercises retrieval and chat
    generation, neither of which reads the extractions table.

    Args:
        db_path: Where to create the new database. Deleted first if it
            already exists, so re-runs start clean.
        settings: Application settings, for `pdfs_dir` and the embedding
            dimension.
        llm: The LLM client used for embedding calls.
    """
    _delete_database(db_path)

    source_conn = connect(settings.db_path)
    source_reports = ReportsRepo(source_conn).list_ready()

    conn = connect(db_path)
    reports_repo = ReportsRepo(conn)
    chunks_repo = ChunksRepo(conn)
    vector_index = VectorIndex(settings.embedding_dim)

    for report in source_reports:
        pdf_path = settings.pdfs_dir / f"{report.sha256}.pdf"
        new_report = reports_repo.create(
            report.sha256, report.filename, report.company, report.fiscal_year
        )

        parsed = parse_pdf(pdf_path)
        prepared_chunks = chunk_document_semantic(parsed, llm.embed)
        rows = [
            (
                chunk.page_start,
                chunk.page_end,
                chunk.kind,
                chunk.text,
                build_embed_text(
                    report.company, report.fiscal_year, chunk.page_start, chunk.page_end, chunk.text
                ),
                None,
            )
            for chunk in prepared_chunks
        ]
        chunks_repo.insert_many(new_report.id, rows)
        embed_and_index_report(new_report.id, llm, chunks_repo, vector_index)
        reports_repo.mark_ready(new_report.id)
        logger.info("%s (%d): %d chunks (semantic)", report.company, report.fiscal_year, len(rows))


def main() -> None:
    """CLI entry point: rebuilds the semantic database, runs both evals, compares."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-set",
        type=Path,
        default=None,
        help="Path to the gold-set JSON. Defaults to <data_dir>/eval/gold_set.json.",
    )
    args = parser.parse_args()

    settings = get_settings()
    gold_set_path = args.gold_set or (settings.data_dir / "eval" / "gold_set.json")
    if not gold_set_path.exists():
        raise FileNotFoundError(
            f"No gold set at {gold_set_path}. Run `python -m eval.build_gold_set` first."
        )

    llm = build_llm_client(settings)
    semantic_db_path = settings.data_dir / "eval" / "chunking_experiment" / "semantic.db"

    logger.info("Re-ingesting every ready report with the semantic chunker...")
    rebuild_with_semantic_chunker(semantic_db_path, settings, llm)

    logger.info("\n=== Paragraph chunking (production) ===")
    paragraph_results = run_eval(gold_set_path)
    print_report(paragraph_results)

    logger.info("\n=== Semantic chunking (experimental) ===")
    semantic_results = run_eval(gold_set_path, db_path=semantic_db_path)
    print_report(semantic_results)

    comparison = {
        "paragraph": summarize(paragraph_results),
        "semantic": summarize(semantic_results),
    }
    out_path = settings.data_dir / "eval" / "chunking_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nWrote comparison to {out_path}")


if __name__ == "__main__":
    main()
