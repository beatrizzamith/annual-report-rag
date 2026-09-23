"""Typed records mirroring the tables in schema.sql."""

from dataclasses import dataclass


@dataclass
class Report:
    """One uploaded annual report and its ingestion status."""

    id: int
    sha256: str
    filename: str
    company: str
    fiscal_year: int
    page_count: int | None
    status: str
    stage: str | None
    progress: float
    error: str | None
    created_at: str


@dataclass
class Chunk:
    """One retrieval unit: verbatim text plus its embedding, if computed."""

    id: int
    report_id: int
    page_start: int
    page_end: int
    kind: str
    text: str
    embed_text: str
    embedding: bytes | None


@dataclass
class Extraction:
    """One pre-extracted fact (FTE or a sustainability goal) with its
    verification result."""

    id: int
    report_id: int
    kind: str
    payload: str
    quote: str
    page: int
    chunk_id: int | None
    verified: bool
    model: str


@dataclass
class Message:
    """One chat message, with its citations when it is an assistant reply."""

    id: int
    role: str
    content: str
    citations: str | None
    interpreted_as: str | None
    created_at: str
