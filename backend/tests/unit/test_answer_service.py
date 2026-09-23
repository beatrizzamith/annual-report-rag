"""Tests for the code-level guarantees around what citations are shown to the user."""

from pathlib import Path

from app.answer.service import (
    VerifiedCitation,
    _drop_citations_if_not_found,
    _find_orphan_citation_ids,
    _find_unbacked_numbers,
)


def _citation(source_id: str, quote: str = "irrelevant", verified: bool = True) -> VerifiedCitation:
    return VerifiedCitation(
        label=None,
        value_text=None,
        unit=None,
        period=None,
        quote=quote,
        chunk_text="irrelevant",
        report="Annual Report 2025",
        page="1",
        verified=verified,
        source_id=source_id,
    )


def test_no_orphans_when_every_marker_has_a_citation():
    answer = "Revenue grew [S1], driven by strong volumes [S2]."
    assert _find_orphan_citation_ids(answer, [_citation("S1"), _citation("S2")]) == []


def test_finds_a_marker_with_no_matching_citation():
    answer = "Our strategy focuses on decarbonisation [S1]."
    assert _find_orphan_citation_ids(answer, []) == ["S1"]


def test_deduplicates_repeated_orphan_markers_in_order_of_first_appearance():
    answer = "We aim for net zero [S3]. This is reaffirmed later [S3] and again [S1]."
    assert _find_orphan_citation_ids(answer, []) == ["S3", "S1"]


def test_no_markers_means_no_orphans():
    assert _find_orphan_citation_ids("Revenue grew this year.", []) == []


def test_number_backed_by_a_verified_quote_is_not_flagged():
    answer = "ABN AMRO reduced its workforce by 5,200 FTEs [S1]."
    citation = _citation("S1", quote="a net total reduction of the global workforce by 5,200 FTEs")
    assert _find_unbacked_numbers(answer, [citation]) == []


def test_number_with_no_verified_citation_containing_it_is_flagged():
    # The real case this guards: an unrelated figure stated as fact next to
    # a citation whose quote is genuinely verified, just about something else.
    answer = "ABN AMRO had 122,779 employees [S2]."
    citation = _citation("S2", quote="a reduction of approximately 1,500 FTEs was realised")
    assert _find_unbacked_numbers(answer, [citation]) == ["122,779"]


def test_number_backed_only_by_an_unverified_citation_is_still_flagged():
    citation = _citation("S1", quote="contains 122,779 somewhere", verified=False)
    assert _find_unbacked_numbers("ABN AMRO had 122,779 employees [S1].", [citation]) == ["122,779"]


def test_plausible_year_mentions_are_not_flagged():
    answer = "The target is set for 2028, following a 2024 baseline."
    assert _find_unbacked_numbers(answer, []) == []


def test_a_four_digit_number_outside_year_range_is_flagged():
    assert _find_unbacked_numbers("Total assets reached 8123 million.", []) == ["8123"]


def test_deduplicates_repeated_unbacked_numbers_in_order_of_first_appearance():
    answer = (
        "Revenue was 45,000. Later restated, revenue was still 45,000, unlike 12,000 elsewhere."
    )
    assert _find_unbacked_numbers(answer, []) == ["45,000", "12,000"]


def test_answer_prompt_forbids_irrelevant_details_in_not_found_answers():
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "answer.md"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "do not mention related but irrelevant" in prompt_text.lower()


def test_citations_are_dropped_for_a_not_found_answer():
    # Real case: the model found a source that only says "FTE is disclosed
    # elsewhere" and, despite not answering the question, still cited it --
    # showing a source card next to "not found" reads as if that source
    # were relevant evidence, when it was retrieved and then rejected.
    assert _drop_citations_if_not_found("not_found", [_citation("S1")]) == []


def test_citations_are_kept_for_an_answered_or_partial_status():
    citations = [_citation("S1")]
    assert _drop_citations_if_not_found("answered", citations) == citations
    assert _drop_citations_if_not_found("partial", citations) == citations
