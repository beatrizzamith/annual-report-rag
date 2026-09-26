"""Tests for rendering recent chat history into a prompt."""

from app.core.message_formatting import MAX_CHARS_PER_MESSAGE, format_recent_messages
from app.db.models import Message


def _message(role: str, content: str) -> Message:
    return Message(
        id=1,
        role=role,
        content=content,
        citations=None,
        interpreted_as=None,
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_messages_are_labelled_by_role_in_order():
    messages = [_message("user", "How many FTE?"), _message("assistant", "About 80,000.")]

    assert format_recent_messages(messages) == "User: How many FTE?\nAssistant: About 80,000."


def test_no_messages_gives_an_empty_string():
    assert format_recent_messages([]) == ""


def test_a_long_message_is_truncated():
    formatted = format_recent_messages([_message("user", "x" * (MAX_CHARS_PER_MESSAGE + 100))])

    assert formatted == "User: " + "x" * MAX_CHARS_PER_MESSAGE
