"""Tests for turning a question into a safe full-text search expression."""

from app.retrieval.keyword import build_fts_query


def test_stop_words_are_dropped_and_tokens_are_quoted_and_or_combined():
    assert build_fts_query("How many FTE does Shell have?") == '"many" OR "fte" OR "shell"'


def test_repeated_tokens_appear_once_and_case_is_ignored():
    assert build_fts_query("FTE fte FTE") == '"fte"'


def test_a_question_of_only_stop_words_falls_back_to_all_its_tokens():
    assert build_fts_query("what is the") == '"what" OR "is" OR "the"'


def test_text_without_any_tokens_gives_an_empty_query():
    assert build_fts_query("?! - ,") == ""


def test_search_operators_and_quotes_cannot_break_out_of_the_expression():
    assert build_fts_query('shell" OR *') == '"shell"'
