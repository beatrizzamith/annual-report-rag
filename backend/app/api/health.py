"""Liveness and configuration check, used by the UI to show setup problems."""

from fastapi import APIRouter, Depends

from app.api.deps import AppState, get_state
from app.db.connection import FTS5NotSupportedError, check_fts5

router = APIRouter()


@router.get("/health")
def health(state: AppState = Depends(get_state)) -> dict:
    """Reports whether the app is healthy and which services are configured.

    Args:
        state: The shared application state, including the database connection
            and current settings.

    Returns:
        A dictionary with the service status, whether the LLM is configured,
        whether FTS5 is available, and how many chunks are indexed.
    """
    fts5_ok = True
    try:
        check_fts5(state.conn)
    except FTS5NotSupportedError:
        fts5_ok = False

    return {
        "status": "ok",
        "llm_configured": state.settings.llm_configured,
        "llm_provider": state.settings.llm_provider,
        "fts5_available": fts5_ok,
        "chunks_indexed": state.vector_index.size,
    }
