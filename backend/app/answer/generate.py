"""Runs the model call that produces the answer and its structured evidence."""

from app.answer.schemas import ModelAnswer
from app.core.message_formatting import format_recent_messages
from app.core.prompts import load_prompt
from app.db.models import Message
from app.llm.client import LLMClient

_NO_SOURCES_PLACEHOLDER = "(no relevant sources were found for this question)"


def generate_answer(
    message: str,
    interpreted_as: str | None,
    context_text: str,
    recent_messages: list[Message],
    llm: LLMClient,
) -> ModelAnswer:
    """Runs the answer generation call.

    Args:
        message: The user's question, exactly as typed this turn.
        interpreted_as: The standalone question the follow-up rewrite
            produced, or None if there was no history or the rewrite left
            the message unchanged.
        context_text: The `<source>`-tagged evidence text (may be empty if
            retrieval found nothing).
        recent_messages: The last few chat messages, oldest first, for
            follow-up context. Never includes old source text.
        llm: The LLM client used for the call.

    Returns:
        The model's structured answer, not yet verified.
    """
    system_prompt = load_prompt("answer.md")

    parts = [f"<sources>\n{context_text or _NO_SOURCES_PLACEHOLDER}\n</sources>"]
    if recent_messages:
        parts.append(f"Conversation so far:\n{format_recent_messages(recent_messages)}")

    question_line = f"Question: {message}"
    if interpreted_as:
        question_line += f"\nInterpreted as: {interpreted_as}"
    parts.append(question_line)

    user_prompt = "\n\n".join(parts)
    return llm.complete_structured(system_prompt, user_prompt, ModelAnswer, temperature=0)
