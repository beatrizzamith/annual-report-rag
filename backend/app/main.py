"""FastAPI app factory and startup/shutdown lifecycle (DB connect, vector index, LLM client)."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import chat, health, reports
from app.api.deps import AppState, build_vector_index
from app.config import get_settings
from app.core.errors import AppError, LLMAuthFailedError
from app.core.logging import configure_logging
from app.db.connection import connect, mark_interrupted_reports_failed
from app.db.repositories import MessagesRepo
from app.llm.client import build_llm_client

logger = logging.getLogger(__name__)

# Built frontend assets, present only in the Docker image; in development
# Vite serves the frontend and proxies /api here.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Not a `Settings` field: `create_app()` runs at import, which must not load config.
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:8000"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Builds the shared `AppState` on startup and closes it on shutdown.

    Connects to SQLite, fails reports interrupted by a restart, clears old chat
    history, loads embeddings into the vector index, and builds the LLM client
    (None if unconfigured, so the app still starts and `/api/health` reports it).

    Args:
        app: The FastAPI application being started.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    conn = connect(settings.db_path)
    interrupted = mark_interrupted_reports_failed(conn)
    if interrupted:
        logger.warning(
            "reports interrupted by a previous restart",
            extra={"extra_fields": {"count": interrupted}},
        )

    # A restart is a natural "new conversation" boundary, so chat history
    # (unlike reports and extractions) starts fresh on every run.
    MessagesRepo(conn).delete_all()

    vector_index = build_vector_index(conn, settings.embedding_dim)
    logger.info("vector index loaded", extra={"extra_fields": {"chunks": vector_index.size}})

    try:
        llm = build_llm_client(settings)
    except LLMAuthFailedError as exc:
        logger.warning(
            "LLM client not configured; ingestion and chat will fail until it is",
            extra={"extra_fields": {"error": str(exc)}},
        )
        llm = None

    app.state.app_state = AppState(settings=settings, conn=conn, vector_index=vector_index, llm=llm)
    yield
    conn.close()


async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Converts a typed `AppError` into the API's JSON error body.

    Args:
        request: The request that raised the error (unused, required by
            FastAPI's exception-handler signature).
        exc: The application error to render.

    Returns:
        A JSON response with the error's status code, error code, and message.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "message": exc.message},
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Converts a FastAPI validation error into the API's JSON error body.

    Args:
        request: The request that failed validation (unused, required by
            FastAPI's exception-handler signature).
        exc: The validation error raised by FastAPI.

    Returns:
        A 422 JSON response with error code `invalid_request`.
    """
    return JSONResponse(
        status_code=422,
        content={"error_code": "invalid_request", "message": str(exc)},
    )


def create_app() -> FastAPI:
    """Builds the FastAPI application: middleware, error handlers, routers.

    Returns:
        The configured FastAPI application, not yet started (its lifespan
        runs when it is served, e.g. by uvicorn or `TestClient`).
    """
    app = FastAPI(title="Annual Report RAG Assistant", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)

    app.include_router(health.router, prefix="/api")
    app.include_router(reports.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")

    # Registered last: only reached when no /api/* route matched above.
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
