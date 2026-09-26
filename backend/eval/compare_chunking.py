"""Compares the production paragraph chunker with the experimental semantic chunker.

Re-ingests every ready report into a throwaway database using the semantic
chunker (the real database is never touched), runs the same eval against both,
and logs the results side by side.

    python -m eval.compare_chunking

Needs a gold set (`python -m eval.build_gold_set`); each report's PDF is
re-parsed and re-embedded.
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
from eval.console import configure_console_logging
from eval.run_eval import format_report, run_eval, summarize
from eval.semantic_chunk import chunk_document_semantic

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

    Re-runs parse, chunk and embed on each report's stored PDF, using the
    semantic chunker. Extraction is skipped: the eval only exercises retrieval
    and chat, which don't read the extractions table.

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
    configure_console_logging(logger)
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
    logger.info("%s", format_report(paragraph_results))

    logger.info("\n=== Semantic chunking (experimental) ===")
    semantic_results = run_eval(gold_set_path, db_path=semantic_db_path)
    logger.info("%s", format_report(semantic_results))

    comparison = {
        "paragraph": summarize(paragraph_results),
        "semantic": summarize(semantic_results),
    }
    out_path = settings.data_dir / "eval" / "chunking_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info("\nWrote comparison to %s", out_path)


if __name__ == "__main__":
    main()
