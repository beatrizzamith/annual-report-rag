"""Chat endpoints: ask a question, list history, clear the conversation."""

import json
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.answer.service import ChatResult, answer_question
from app.api.deps import AppState, get_state
from app.core.errors import LLMUnavailableError
from app.db.repositories import MessagesRepo

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Body of `POST /api/chat`."""

    message: str


def serialize_chat_result(result: ChatResult) -> dict:
    """Renders a chat result as the JSON payload returned by the API."""
    return {
        "status": result.status,
        "answer": result.answer,
        "citations": [citation.__dict__ for citation in result.citations],
        "caveats": result.caveats,
        "interpreted_as": result.interpreted_as,
    }


@router.post("/chat")
def chat(request_body: ChatRequest, state: AppState = Depends(get_state)) -> dict:
    """Answers a user question using the loaded reports and current chat history.

    Args:
        request_body: The request body containing the user's message.
        state: The shared application state.

    Returns:
        A JSON payload containing the status, final answer, verified evidence,
        caveats, and the rewritten standalone question when relevant.

    Raises:
        NoReportsLoadedError: No report has finished ingestion yet.
        LLMUnavailableError: No LLM provider is configured.
    """
    logger.info(
        "chat request received",
        extra={
            "extra_fields": {
                "message_length": len(request_body.message),
                "llm_configured": state.llm is not None,
            }
        },
    )
    if state.llm is None:
        logger.warning("chat rejected: llm not configured")
        raise LLMUnavailableError(
            "No LLM provider is configured; add an API key to .env and restart."
        )
    chat_result = answer_question(
        request_body.message,
        state.conn,
        state.vector_index,
        state.llm,
        state.settings.context_token_budget,
    )
    logger.info(
        "chat request completed",
        extra={
            "extra_fields": {
                "status": chat_result.status,
                "citation_count": len(chat_result.citations),
                "caveat_count": len(chat_result.caveats),
                "interpreted_as": chat_result.interpreted_as is not None,
            }
        },
    )
    return serialize_chat_result(chat_result)


@router.get("/messages")
def list_messages(state: AppState = Depends(get_state)) -> list[dict]:
    """Returns the full chat history in chronological order."""
    messages_repo = MessagesRepo(state.conn)
    messages = messages_repo.list()
    logger.info("chat history requested", extra={"extra_fields": {"count": len(messages)}})
    return [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "citations": json.loads(message.citations) if message.citations else None,
            "interpreted_as": message.interpreted_as,
            "created_at": message.created_at,
        }
        for message in messages
    ]


@router.delete("/messages")
def clear_messages(state: AppState = Depends(get_state)) -> dict:
    """Clears the chat history so the user can start a fresh conversation.

    Args:
        state: The shared application state.

    Returns:
        `{"cleared": true}` on success.
    """
    logger.info("clearing chat history")
    MessagesRepo(state.conn).delete_all()
    return {"cleared": True}
