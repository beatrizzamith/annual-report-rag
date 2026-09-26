"""Rewrites follow-up questions into standalone search queries when conversation context exists."""

import logging

from pydantic import BaseModel

from app.core.errors import LLMAuthFailedError, LLMRateLimitedError, LLMUnavailableError
from app.core.message_formatting import format_recent_messages
from app.core.prompts import load_prompt
from app.db.models import Message
from app.llm.client import LLMClient

logger = logging.getLogger(__name__)


class FollowUpRewrite(BaseModel):
    """Structured output of the follow-up rewrite call."""

    standalone_question: str


def rewrite_followup(
    message: str, recent_messages: list[Message], llm: LLMClient
) -> tuple[str, bool]:
    """Rewrites a message into a standalone question, when there is history.

    Skipped entirely for a first message (no history). If the rewrite call
    fails or times out, falls back to the raw message rather than blocking
    the chat turn.

    Args:
        message: The user's new message, as typed.
        recent_messages: The last few chat messages, oldest first. Empty
            for a first message.
        llm: The LLM client used for the rewrite call.

    Returns:
        A tuple of the standalone question to use for retrieval, and
        whether it differs from `message` (used by the caller to decide
        whether to show "Interpreted as: ...").
    """
    if not recent_messages:
        return message, False

    system_prompt = load_prompt("rewrite.md")
    history_text = format_recent_messages(recent_messages)
    user_prompt = f"Conversation so far:\n{history_text}\n\nNew message: {message}"

    try:
        result = llm.complete_structured(system_prompt, user_prompt, FollowUpRewrite, temperature=0)
    except (LLMAuthFailedError, LLMRateLimitedError, LLMUnavailableError) as exc:
        logger.warning(
            "follow-up rewrite failed, using raw message",
            extra={"extra_fields": {"error": str(exc)}},
        )
        return message, False

    standalone_question = result.standalone_question.strip()
    if not standalone_question:
        return message, False
    return standalone_question, standalone_question != message.strip()
