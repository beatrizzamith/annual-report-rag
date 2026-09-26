"""Tests for how the eval decides two quotes refer to the same passage."""

from eval.run_eval import _quotes_overlap


def test_a_quote_matches_a_table_row_despite_pipes_and_spacing():
    assert _quotes_overlap("Revenue 100 90", "| Revenue | 100 | 90 |")


def test_a_short_quote_matches_a_longer_one_that_contains_it():
    longer = "These loans are valued using a model for which the prepayment rate is the key input."
    assert _quotes_overlap("the prepayment rate is the key input", longer)
    assert _quotes_overlap(longer, "the prepayment rate is the key input")


def test_quotes_about_different_facts_do_not_match():
    assert not _quotes_overlap("Shell employed 81,000 people.", "ABN AMRO employed 22,000 people.")


def test_a_different_number_in_otherwise_identical_text_does_not_match():
    assert not _quotes_overlap("A target of 2030 was set.", "A target of 2030s was set.")
