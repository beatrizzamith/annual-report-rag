"""FastAPI app factory and startup/shutdown lifecycle (DB connect, vector index, LLM client)."""

import logging
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
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.db.connection import connect, mark_interrupted_reports_failed
from app.db.repositories import MessagesRepo
from app.llm.client import build_llm_client

logger = logging.getLogger(__name__)

# Populated by the Docker build: the built frontend's static assets. When
# these are absent during local development, Vite serves the frontend and
# proxies /api requests to this backend.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Builds the shared `AppState` on startup and closes it on shutdown.

    Connects to SQLite, marks any report interrupted by a previous restart
    as `failed`, clears chat history from previous runs, loads existing
    embeddings into an in-memory vector index, and builds the LLM client
    (or leaves it None if not configured, so the app still starts and
    reports the problem via `/api/health`).

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

    # *(decision)* Chat is the one thing that starts fresh on every run,
    # unlike reports/chunks/extractions (still all on disk and reloaded
    # above/below): a restart is a natural "new conversation" boundary, and
    # nothing else in the app depends on chat history surviving one.
    MessagesRepo(conn).delete_all()

    vector_index = build_vector_index(conn, settings.embedding_dim)
    logger.info("vector index loaded", extra={"extra_fields": {"chunks": vector_index.size}})

    try:
        llm = build_llm_client(settings)
    except Exception as exc:
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
        allow_origins=["http://localhost:5173", "http://localhost:8000"],
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
