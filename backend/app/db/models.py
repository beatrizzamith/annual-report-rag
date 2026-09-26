"""Typed records mirroring the tables in schema.sql."""

from typing import Literal

from pydantic import BaseModel, Field


class Report(BaseModel):
    """One uploaded annual report and its ingestion status."""

    id: int
    sha256: str = Field(description="SHA-256 of the uploaded file; the idempotency key.")
    filename: str
    company: str
    fiscal_year: int
    page_count: int | None
    status: Literal["processing", "ready", "failed"]
    stage: Literal["parse", "chunk", "embed", "extract"] | None = Field(
        description="The ingestion step running now; None once finished."
    )
    progress: float = Field(description="Overall ingestion progress, from 0 to 1.")
    error: str | None
    created_at: str


class Chunk(BaseModel):
    """One retrieval unit: verbatim text plus its embedding, if computed."""

    id: int
    report_id: int
    page_start: int
    page_end: int
    kind: Literal["text", "table"]
    text: str = Field(description="Canonical verbatim text; quotes are verified against this.")
    embed_text: str = Field(description="`text` plus a context header, used only for embedding.")
    embedding: bytes | None


class Extraction(BaseModel):
    """One pre-extracted fact (FTE or a sustainability goal) with its verification result."""

    id: int
    report_id: int
    kind: Literal["fte", "sustainability_goal"]
    payload: str = Field(description="The extracted item as JSON.")
    quote: str
    page: int
    chunk_id: int | None
    verified: bool = Field(description="True only if the quote was found verbatim in its chunk.")
    model: str = Field(description="The chat model that produced this extraction.")


class Message(BaseModel):
    """One chat message, with its citations when it is an assistant reply."""

    id: int
    role: Literal["user", "assistant"]
    content: str
    citations: str | None = Field(description="Citations as JSON; None for a user message.")
    interpreted_as: str | None = Field(
        description="The standalone question, when the follow-up rewrite changed the message."
    )
    created_at: str
