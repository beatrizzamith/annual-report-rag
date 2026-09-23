"""Shared app state, built once at startup (see main.py's lifespan) and
handed to routes as FastAPI dependencies."""

import sqlite3
from dataclasses import dataclass

from fastapi import Request

from app.config import Settings
from app.db.repositories import ChunksRepo
from app.ingestion.embed import blob_to_vector
from app.llm.client import LLMClient
from app.retrieval.vector_index import VectorIndex


@dataclass
class AppState:
    """Process-wide state shared by every request."""

    settings: Settings
    conn: sqlite3.Connection
    vector_index: VectorIndex
    llm: LLMClient | None


def build_vector_index(conn: sqlite3.Connection, dim: int) -> VectorIndex:
    """Loads every already-embedded chunk into a fresh in-memory index.

    Called once at startup so restarts don't need to re-embed anything.

    Args:
        conn: The application's SQLite connection.
        dim: The embedding dimension the index should be built for.

    Returns:
        A `VectorIndex` populated from every chunk that already has an
        embedding, across all reports.
    """
    chunks = ChunksRepo(conn).all_with_embeddings()
    index = VectorIndex(dim)
    # One batched add_many, not one add() per chunk: with thousands of
    # chunks now indexed, adding them one at a time made this O(N^2) and
    # was the main cause of a slow app startup.
    ids = [chunk.id for chunk in chunks]
    vectors = [blob_to_vector(chunk.embedding) for chunk in chunks]
    index.add_many(ids, vectors)
    return index


def get_state(request: Request) -> AppState:
    """FastAPI dependency returning the shared application state.

    Args:
        request: The current request.

    Returns:
        The `AppState` built by `main.lifespan` at startup.
    """
    return request.app.state.app_state
