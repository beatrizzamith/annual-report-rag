-- SQLite schema with WAL mode for safer local writes and simpler recovery.

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY,
    sha256 TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    company TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    page_count INTEGER,
    status TEXT NOT NULL DEFAULT 'processing',   -- processing | ready | failed
    stage TEXT,                                  -- parse | chunk | embed | extract
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    report_id INTEGER NOT NULL REFERENCES reports(id),
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    kind TEXT NOT NULL,           -- text | table
    text TEXT NOT NULL,           -- canonical verbatim text
    embed_text TEXT NOT NULL,     -- text plus a context header
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_report_id ON chunks(report_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS extractions (
    id INTEGER PRIMARY KEY,
    report_id INTEGER NOT NULL REFERENCES reports(id),
    kind TEXT NOT NULL,           -- fte | sustainability_goal
    payload TEXT NOT NULL,        -- JSON, validated by Pydantic on write
    quote TEXT NOT NULL,
    page INTEGER NOT NULL,
    chunk_id INTEGER,
    verified INTEGER NOT NULL,    -- 1 only if the quote was found verbatim in the chunk text
    model TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_extractions_report_id ON extractions(report_id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    role TEXT NOT NULL,           -- user | assistant
    content TEXT NOT NULL,
    citations TEXT,               -- JSON: citations with quote, report, page, verified
    interpreted_as TEXT,          -- standalone question, when the rewrite changed it
    created_at TEXT NOT NULL
);
