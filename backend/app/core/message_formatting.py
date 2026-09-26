"""Formats recent chat messages for inclusion in a prompt.

This keeps the prompt short while preserving the last few turns of user and
assistant text without including older source excerpts.
"""

from app.db.models import Message

MAX_CHARS_PER_MESSAGE = 500


def format_recent_messages(messages: list[Message]) -> str:
    """Renders recent chat messages as plain "Role: text" lines.

    How many messages to include is the caller's decision (see
    `RECENT_MESSAGES_LIMIT` in `app.answer.service`), so this does not cap
    the count a second time.

    Args:
        messages: Chat history, oldest first. Each message's text is
            truncated to `MAX_CHARS_PER_MESSAGE` characters.

    Returns:
        The formatted history, or an empty string if `messages` is empty.
    """
    formatted_lines = []
    for message in messages:
        role = "User" if message.role == "user" else "Assistant"
        text = message.content[:MAX_CHARS_PER_MESSAGE]
        formatted_lines.append(f"{role}: {text}")
    return "\n".join(formatted_lines)
