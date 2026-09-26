"""Re-checks every stored extraction against the current verifier, without re-extracting.

    python -m eval.reverify_extractions

Only the verification rules change when `verifier.py` improves, so this
recomputes `verified` for each stored (quote, chunk_text) pair and updates just
the rows that changed. It runs against the live database (unlike `run_eval.py`)
and never touches any other column.
"""

import json
import logging

from app.config import get_settings
from app.db.connection import connect
from app.db.models import Extraction
from app.db.repositories import ChunksRepo, ExtractionsRepo
from app.extraction.verifier import verify_extraction, verify_quote
from eval.console import configure_console_logging

logger = logging.getLogger(__name__)


def _recompute_verified(extraction: Extraction, chunk_text: str) -> bool:
    """Recomputes one extraction's `verified` flag with the current verifier.

    Args:
        extraction: The stored extraction row.
        chunk_text: Its source chunk's full text.

    Returns:
        The freshly computed verification result, using `verify_extraction`
        (quote plus value) for an FTE row and `verify_quote` (quote only,
        matching `extract_goals`' own check) for a sustainability goal.
    """
    if extraction.kind == "fte":
        payload = json.loads(extraction.payload)
        return verify_extraction(extraction.quote, payload.get("value_text"), chunk_text).verified
    return verify_quote(extraction.quote, chunk_text)


def main() -> None:
    """Re-verifies every stored extraction and reports how many flipped."""
    configure_console_logging(logger)
    settings = get_settings()
    conn = connect(settings.db_path)
    extractions_repo = ExtractionsRepo(conn)
    chunks_repo = ChunksRepo(conn)

    extractions = extractions_repo.list_all()
    changed = 0
    skipped_no_chunk = 0
    for extraction in extractions:
        if extraction.chunk_id is None:
            skipped_no_chunk += 1
            continue
        chunk = chunks_repo.get(extraction.chunk_id)
        if chunk is None:
            skipped_no_chunk += 1
            continue
        fresh_verified = _recompute_verified(extraction, chunk.text)
        if fresh_verified != extraction.verified:
            extractions_repo.update_verified(extraction.id, fresh_verified)
            changed += 1

    logger.info("Re-checked %d extraction(s).", len(extractions))
    logger.info("  now verified that weren't, or vice versa: %d", changed)
    if skipped_no_chunk:
        logger.info("  skipped (no resolvable source chunk): %d", skipped_no_chunk)


if __name__ == "__main__":
    main()
