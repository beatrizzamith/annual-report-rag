"""Re-checks every stored FTE and sustainability-goal extraction against the
current `verify_quote`/`verify_extraction` logic, without re-running
extraction itself.

    python -m eval.reverify_extractions

A quote and its source chunk are fixed at ingestion time; only the
verification *rules* change when `app/extraction/verifier.py` is improved.
Re-running full ingestion (parse, chunk, embed, extract) just to pick up a
verifier fix wastes an LLM extraction pass and can't change anything the fix
actually touches -- this reads each stored (quote, chunk_text) pair straight
from the database, recomputes `verified` with the current code, and updates
only the rows whose result actually changed.

Runs against the live app database directly (unlike `eval/run_eval.py`,
which works on a throwaway copy): the whole point is to update real,
already-ingested rows in place. It never touches `quote`, `payload`, or any
other stored field -- only `verified`.
"""

import json
import logging

from app.config import get_settings
from app.db.connection import connect
from app.db.repositories import ChunksRepo, ExtractionsRepo
from app.extraction.verifier import verify_extraction, verify_quote

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def _recompute_verified(extraction, chunk_text: str) -> bool:
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

    print(f"Re-checked {len(extractions)} extraction(s).")
    print(f"  now verified that weren't, or vice versa: {changed}")
    if skipped_no_chunk:
        print(f"  skipped (no resolvable source chunk): {skipped_no_chunk}")


if __name__ == "__main__":
    main()
