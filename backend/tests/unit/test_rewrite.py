"""Tests for rewriting follow-up questions into standalone search queries."""

from app.core.errors import LLMUnavailableError
from app.db.models import Message
from app.llm.fakes import FakeLLM
from app.retrieval.rewrite import FollowUpRewrite, rewrite_followup


def _previous_message() -> Message:
    return Message(
        id=1,
        role="user",
        content="How much did Shell spend on adaptation in 2024?",
        citations=None,
        interpreted_as=None,
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_first_message_is_returned_unchanged_without_calling_the_model():
    llm = FakeLLM()

    question, was_rewritten = rewrite_followup("How many FTE does Shell have?", [], llm)

    assert question == "How many FTE does Shell have?"
    assert was_rewritten is False
    assert llm.calls == []


def test_follow_up_is_replaced_by_the_standalone_question():
    llm = FakeLLM()
    llm.queue_response(
        FollowUpRewrite(standalone_question="How much did Shell spend on adaptation in 2025?")
    )

    question, was_rewritten = rewrite_followup("and 2025?", [_previous_message()], llm)

    assert question == "How much did Shell spend on adaptation in 2025?"
    assert was_rewritten is True


def test_rewrite_identical_to_the_message_is_not_reported_as_rewritten():
    llm = FakeLLM()
    llm.queue_response(FollowUpRewrite(standalone_question="What is Shell's FTE count?"))

    question, was_rewritten = rewrite_followup(
        "What is Shell's FTE count?", [_previous_message()], llm
    )

    assert question == "What is Shell's FTE count?"
    assert was_rewritten is False


def test_failed_rewrite_call_falls_back_to_the_raw_message():
    llm = FakeLLM()
    llm.queue_response(LLMUnavailableError("timeout"))

    question, was_rewritten = rewrite_followup("and 2025?", [_previous_message()], llm)

    assert question == "and 2025?"
    assert was_rewritten is False
