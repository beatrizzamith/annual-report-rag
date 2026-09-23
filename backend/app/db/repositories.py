"""Thin repositories over the SQLite tables in schema.sql.

Each function takes a connection explicitly rather than hiding a global, so
tests can pass an in-memory database.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from app.db.models import Chunk, Extraction, Message, Report


def _now() -> str:
    """Returns the current UTC time for timestamping new rows.

    Returns:
        The current UTC time as an ISO 8601 string.
    """
    return datetime.now(UTC).isoformat()


class ReportsRepo:
    """CRUD and status transitions for the `reports` table."""

    def __init__(self, conn: sqlite3.Connection):
        """Wraps a database connection for report operations.

        Args:
            conn: An open, schema-initialised SQLite connection.
        """
        self.conn = conn

    def create(self, sha256: str, filename: str, company: str, fiscal_year: int) -> Report:
        """Inserts a new report row in the `processing` status.

        Args:
            sha256: The uploaded file's SHA-256 hash (unique, idempotency key).
            filename: The original filename, for display.
            company: The company name typed at upload.
            fiscal_year: The fiscal year typed at upload.

        Returns:
            The newly created report.
        """
        cursor = self.conn.execute(
            """INSERT INTO reports (sha256, filename, company, fiscal_year, status, stage,
                                     progress, created_at)
               VALUES (?, ?, ?, ?, 'processing', 'parse', 0, ?)""",
            (sha256, filename, company, fiscal_year, _now()),
        )
        self.conn.commit()
        report = self.get(cursor.lastrowid)
        assert report is not None
        return report

    def get(self, report_id: int) -> Report | None:
        """Fetches a report by id.

        Args:
            report_id: The report's id.

        Returns:
            The report, or None if no report has that id.
        """
        row = self.conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        return Report(**dict(row)) if row else None

    def get_by_sha256(self, sha256: str) -> Report | None:
        """Fetches a report by its uploaded file hash.

        Args:
            sha256: The uploaded file's SHA-256 hash.

        Returns:
            The report with that hash, or None if none exists yet.
        """
        row = self.conn.execute("SELECT * FROM reports WHERE sha256 = ?", (sha256,)).fetchone()
        return Report(**dict(row)) if row else None

    def list(self) -> list[Report]:
        """Lists every report.

        Returns:
            Every report, most recently created first.
        """
        rows = self.conn.execute("SELECT * FROM reports ORDER BY created_at DESC").fetchall()
        return [Report(**dict(row)) for row in rows]

    def list_ready(self) -> list[Report]:
        """Lists every ready report.

        Returns:
            Every report with status `ready`, oldest first.
        """
        rows = self.conn.execute(
            "SELECT * FROM reports WHERE status = 'ready' ORDER BY created_at"
        ).fetchall()
        return [Report(**dict(row)) for row in rows]

    def update_progress(self, report_id: int, stage: str, progress: float) -> None:
        """Updates a report's ingestion stage and progress fraction.

        Args:
            report_id: The report being ingested.
            stage: The current pipeline stage, such as "parse", "chunk",
                "embed", or "extract".
            progress: Overall ingestion progress, in [0, 1].
        """
        self.conn.execute(
            "UPDATE reports SET stage = ?, progress = ? WHERE id = ?",
            (stage, progress, report_id),
        )
        self.conn.commit()

    def mark_page_count(self, report_id: int, page_count: int) -> None:
        """Records a report's total page count.

        Args:
            report_id: The report being ingested.
            page_count: The PDF's total page count.
        """
        self.conn.execute("UPDATE reports SET page_count = ? WHERE id = ?", (page_count, report_id))
        self.conn.commit()

    def mark_ready(self, report_id: int) -> None:
        """Marks a report as ready and clears the current stage.

        Args:
            report_id: The report that finished ingestion successfully.
        """
        self.conn.execute(
            "UPDATE reports SET status = 'ready', stage = NULL, progress = 1 WHERE id = ?",
            (report_id,),
        )
        self.conn.commit()

    def mark_failed(self, report_id: int, error: str) -> None:
        """Marks a report as failed with an error message.

        Args:
            report_id: The report whose ingestion failed.
            error: A human-readable description of the failure.
        """
        self.conn.execute(
            "UPDATE reports SET status = 'failed', error = ? WHERE id = ?",
            (error, report_id),
        )
        self.conn.commit()

    def _delete_chunks_and_extractions(self, report_id: int) -> None:
        """Deletes a report's chunks and extracted items while keeping the report row.

        Args:
            report_id: The report whose chunks and extractions to remove.
        """
        self.conn.execute("DELETE FROM extractions WHERE report_id = ?", (report_id,))
        self.conn.execute("DELETE FROM chunks WHERE report_id = ?", (report_id,))

    def reset_for_retry(self, report_id: int) -> None:
        """Deletes partial rows and resets a failed report for another run.

        Args:
            report_id: The failed report to reset. Its existing chunks and
                extractions are deleted so the same report id can be ingested
                again from the start.
        """
        self._delete_chunks_and_extractions(report_id)
        self.conn.execute(
            """UPDATE reports
               SET status = 'processing', stage = 'parse', progress = 0, error = NULL
               WHERE id = ?""",
            (report_id,),
        )
        self.conn.commit()

    def delete(self, report_id: int) -> None:
        """Permanently deletes a report and all of its extracted data.

        Args:
            report_id: The report to delete. Its stored PDF is left on disk
                (named by content hash); re-uploading the same file simply
                creates a fresh report reusing that file.
        """
        self._delete_chunks_and_extractions(report_id)
        self.conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        self.conn.commit()


class ChunksRepo:
    """CRUD and search over the `chunks` table (and its FTS5 index)."""

    def __init__(self, conn: sqlite3.Connection):
        """Wraps a database connection for chunk operations.

        Args:
            conn: An open, schema-initialised SQLite connection.
        """
        self.conn = conn

    def insert_many(
        self,
        report_id: int,
        rows: list[tuple[int, int, str, str, str, bytes | None]],
    ) -> list[int]:
        """Inserts chunks for one report in insertion order.

        Args:
            report_id: The report the chunks belong to.
            rows: One tuple per chunk:
                `(page_start, page_end, kind, text, embed_text, embedding)`.

        Returns:
            The new chunks' ids, in the same order as `rows`.
        """
        ids = []
        for page_start, page_end, kind, text, embed_text, embedding in rows:
            cursor = self.conn.execute(
                """INSERT INTO chunks (report_id, page_start, page_end, kind, text,
                                        embed_text, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (report_id, page_start, page_end, kind, text, embed_text, embedding),
            )
            ids.append(cursor.lastrowid)
        self.conn.commit()
        return ids

    def set_embedding(self, chunk_id: int, embedding: bytes) -> None:
        """Writes a chunk's embedding vector.

        Args:
            chunk_id: The chunk to update.
            embedding: The embedding, serialised as float32 bytes.
        """
        self.conn.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (embedding, chunk_id))

    def commit(self) -> None:
        """Commits the current transaction after a batch of embedding writes."""
        self.conn.commit()

    def get(self, chunk_id: int) -> Chunk | None:
        """Fetches a chunk by id.

        Args:
            chunk_id: The chunk's id.

        Returns:
            The chunk, or None if no chunk has that id.
        """
        row = self.conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return Chunk(**dict(row)) if row else None

    def get_by_report(self, report_id: int) -> list[Chunk]:
        """Lists a report's chunks.

        Args:
            report_id: The report whose chunks to return.

        Returns:
            The report's chunks, ordered by their starting page.
        """
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE report_id = ? ORDER BY page_start", (report_id,)
        ).fetchall()
        return [Chunk(**dict(row)) for row in rows]

    def all_with_embeddings(self) -> list[Chunk]:
        """Lists every chunk that already has an embedding.

        Returns:
            Every chunk that already has an embedding across all reports.
        """
        rows = self.conn.execute("SELECT * FROM chunks WHERE embedding IS NOT NULL").fetchall()
        return [Chunk(**dict(row)) for row in rows]

    def keyword_search(
        self, query: str, report_ids: list[int] | None, top_k: int
    ) -> list[tuple[int, float]]:
        """Runs a BM25 keyword search over the `chunks_fts` index.

        Args:
            query: An FTS5 match expression (see `retrieval.keyword`).
            report_ids: Restrict the search to these reports' chunks, or
                None to search every report.
            top_k: The maximum number of results to return.

        Returns:
            `(chunk_id, bm25_score)` pairs, best match first. Empty if
            `query` is blank or `report_ids` is an empty list.
        """
        if not query.strip():
            return []
        if report_ids is not None:
            if not report_ids:
                return []
            placeholders = ",".join("?" for _ in report_ids)
            sql = f"""
                SELECT c.id, bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ? AND c.report_id IN ({placeholders})
                ORDER BY score
                LIMIT ?
            """
            params = (query, *report_ids, top_k)
        else:
            sql = """
                SELECT c.id, bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """
            params = (query, top_k)
        rows = self.conn.execute(sql, params).fetchall()
        return [(row["id"], row["score"]) for row in rows]


class ExtractionsRepo:
    """CRUD over the `extractions` table (pre-extracted FTE and goals)."""

    def __init__(self, conn: sqlite3.Connection):
        """Wraps a database connection for extraction operations.

        Args:
            conn: An open, schema-initialised SQLite connection.
        """
        self.conn = conn

    def insert(
        self,
        report_id: int,
        kind: str,
        payload: str,
        quote: str,
        page: int,
        chunk_id: int | None,
        verified: bool,
        model: str,
    ) -> int:
        """Inserts one extracted item.

        Args:
            report_id: The report the item was extracted from.
            kind: "fte" or "sustainability_goal".
            payload: The item's fields, JSON-encoded.
            quote: The verbatim quote backing the item.
            page: The physical page the quote was found on.
            chunk_id: The source chunk's id, or None if it could not be
                resolved.
            verified: Whether `quote` was found verbatim in the source chunk.
            model: The chat model name used for extraction.

        Returns:
            The new extraction row's id.
        """
        cursor = self.conn.execute(
            """INSERT INTO extractions (report_id, kind, payload, quote, page, chunk_id,
                                         verified, model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (report_id, kind, payload, quote, page, chunk_id, int(verified), model),
        )
        self.conn.commit()
        return cursor.lastrowid

    def list_by_report(self, report_id: int) -> list[Extraction]:
        """Lists a report's extracted items.

        Args:
            report_id: The report whose extracted items to return.

        Returns:
            The report's extracted items, grouped by kind.
        """
        rows = self.conn.execute(
            "SELECT * FROM extractions WHERE report_id = ? ORDER BY kind, id", (report_id,)
        ).fetchall()
        return [Extraction(**{**dict(row), "verified": bool(row["verified"])}) for row in rows]

    def list_all(self) -> list[Extraction]:
        """Lists every extracted item across all reports.

        Used to re-run verification against the current `verify_quote`
        logic without re-running extraction itself (see
        `eval/reverify_extractions.py`) -- a quote and its source chunk
        never change after ingestion, only the verification rules do, so
        there's no need to touch the LLM-extracted content to refresh
        `verified`.

        Returns:
            Every extracted item, ordered by report and kind.
        """
        rows = self.conn.execute(
            "SELECT * FROM extractions ORDER BY report_id, kind, id"
        ).fetchall()
        return [Extraction(**{**dict(row), "verified": bool(row["verified"])}) for row in rows]

    def update_verified(self, extraction_id: int, verified: bool) -> None:
        """Updates one extraction's stored verification result in place.

        Args:
            extraction_id: The extraction row to update.
            verified: The freshly computed verification result.
        """
        self.conn.execute(
            "UPDATE extractions SET verified = ? WHERE id = ?", (int(verified), extraction_id)
        )
        self.conn.commit()


class MessagesRepo:
    """CRUD over the `messages` table (chat history)."""

    def __init__(self, conn: sqlite3.Connection):
        """Wraps a database connection for message operations.

        Args:
            conn: An open, schema-initialised SQLite connection.
        """
        self.conn = conn

    def insert(
        self, role: str, content: str, citations: str | None, interpreted_as: str | None
    ) -> int:
        """Inserts one chat message.

        Args:
            role: "user" or "assistant".
            content: The message text.
            citations: The assistant's verified citations, JSON-encoded, or
                None for a user message.
            interpreted_as: The standalone question the message was
                rewritten to, or None if it was not rewritten.

        Returns:
            The new message row's id.
        """
        cursor = self.conn.execute(
            """INSERT INTO messages (role, content, citations, interpreted_as, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (role, content, citations, interpreted_as, _now()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def list(self, limit: int | None = None) -> list[Message]:
        """Lists chat messages.

        Args:
            limit: If given, return only the most recent `limit` messages
                (still ordered oldest first); otherwise return all of them.

        Returns:
            Chat messages in chronological order.
        """
        sql = "SELECT * FROM messages ORDER BY id"
        if limit:
            sql += f" DESC LIMIT {int(limit)}"
        rows = self.conn.execute(sql).fetchall()
        messages = [Message(**dict(row)) for row in rows]
        return list(reversed(messages)) if limit else messages

    def delete_all(self) -> None:
        """Deletes every chat message to start a fresh conversation."""
        self.conn.execute("DELETE FROM messages")
        self.conn.commit()
