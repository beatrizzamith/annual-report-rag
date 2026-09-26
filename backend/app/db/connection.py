"""Sets up the SQLite connection, applies the schema, and validates FTS5 support."""

import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class FTS5NotSupportedError(RuntimeError):
    """The installed SQLite build lacks the FTS5 full-text search extension."""


def check_fts5(conn: sqlite3.Connection) -> None:
    """Verifies that the SQLite build backing `conn` supports FTS5.

    Args:
        conn: An open SQLite connection.

    Raises:
        FTS5NotSupportedError: The SQLite build lacks the FTS5 extension.
    """
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.fts5_check USING fts5(x)")
        conn.execute("DROP TABLE temp.fts5_check")
    except sqlite3.OperationalError as exc:
        raise FTS5NotSupportedError(
            "This Python's SQLite build lacks FTS5. Use the Docker image, "
            "or install a Python build with FTS5 support."
        ) from exc


def connect(db_path: Path) -> sqlite3.Connection:
    """Opens the application database, applying pragmas and the schema.

    Args:
        db_path: Path to the SQLite database file. Its parent directory is
            created if missing.

    Returns:
        A connection in WAL mode with foreign keys enabled, `Row`-typed
        results, and the schema in schema.sql already applied.

    Raises:
        FTS5NotSupportedError: The SQLite build lacks the FTS5 extension.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    check_fts5(conn)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def mark_interrupted_reports_failed(conn: sqlite3.Connection) -> int:
    """Fails any report left `processing` by a previous, interrupted run.

    Called at startup: a report stuck in `processing` means the app was
    killed mid-ingestion. The user re-uploads the same file to retry.

    Args:
        conn: An open, schema-initialised SQLite connection.

    Returns:
        The number of reports marked failed.
    """
    cursor = conn.execute(
        "UPDATE reports SET status = 'failed', error = ? WHERE status = 'processing'",
        ("interrupted by restart",),
    )
    conn.commit()
    return cursor.rowcount
