from app.extraction.verifier import (
    verify_citation,
    verify_extraction,
    verify_quote,
    verify_value_in_quote,
)

SOURCE = (
    "The Group employed an average of 103,100 FTE during 2025, "
    "compared with 96,442 FTE in the prior year."
)


def test_exact_quote_is_verified():
    quote = "The Group employed an average of 103,100 FTE during 2025"
    assert verify_quote(quote, SOURCE) is True


def test_fabricated_quote_is_rejected():
    quote = "The Group employed exactly one million robots in 2025"
    assert verify_quote(quote, SOURCE) is False


def test_quote_verification_is_whitespace_and_quote_style_tolerant():
    quote = "The Group employed an average of  103,100 FTE during 2025"  # double space
    assert verify_quote(quote, SOURCE) is True


def test_empty_quote_is_never_verified():
    assert verify_quote("", SOURCE) is False
    assert verify_quote(None, SOURCE) is False


def test_a_hyphen_our_own_dehyphenation_incorrectly_stripped_does_not_fail_verification():
    # Real case: the source PDF wraps "well-below-2C" across a line break,
    # so our own de-hyphenation heuristic (built for genuine line-wrap
    # artifacts) strips the hyphen, leaving the stored chunk text reading
    # "wellbelow-2C" -- a bug in our stored text, not in a faithfully
    # quoted "well-below-2C".
    source = "We aim to align with a wellbelow-2C pathway, in line with the Paris Agreement."
    quote = "We aim to align with a well-below-2C pathway, in line with the Paris Agreement."
    assert verify_quote(quote, source) is True


def test_a_paraphrase_is_still_rejected_even_with_hyphen_folding():
    # Hyphen folding must not become a loophole for genuine wording drift:
    # this differs by more than a hyphen ("our intention to" vs "we plan
    # to"), so it must still fail.
    source = "We announced that we intend to update our well-below-2C strategy."
    quote = "We announced our intention to update our well-below-2C strategy."
    assert verify_quote(quote, source) is False


def test_a_sentence_split_across_two_chunked_paragraphs_still_verifies():
    # Real case: a PDF bullet wrapped across two separate text blocks, so
    # the chunker joined them as two paragraphs with a blank-line gap
    # ("chunk_document" joins consecutive paragraphs with "\n\n"), leaving
    # the stored text reading "...FLAG (forest, land and\n\nagriculture)
    # emissions...". The model quotes the sentence continuously, with a
    # single space where the source has that gap -- a chunking artefact,
    # not a wrong quote.
    source = (
        "Reach net zero in Scope 1 and 2 by 2030\n\n"
        "Reduce Scope 3 FLAG (forest, land and\n\n"
        "agriculture) emissions by 30% and non-FLAG by 25% by 2030\n\n"
        "Maximise circularity"
    )
    quote = (
        "Reduce Scope 3 FLAG (forest, land and agriculture) emissions by 30% "
        "and non-FLAG by 25% by 2030"
    )
    assert verify_quote(quote, source) is True


def test_a_paraphrase_is_still_rejected_even_with_whitespace_collapsed():
    # Whitespace collapsing must not become a loophole either: this differs
    # by an actual word ("cut" vs "reduced"), so it must still fail even
    # though it spans the same paragraph-break gap as the case above.
    source = "Reduce Scope 3 FLAG (forest, land and\n\nagriculture) emissions by 30% by 2030"
    quote = "Cut Scope 3 FLAG (forest, land and agriculture) emissions by 30% by 2030"
    assert verify_quote(quote, source) is False


def test_a_hyphenated_word_split_across_a_paragraph_break_still_verifies():
    # Real case (Heineken, page 165): the chunker stored a bullet's
    # wrapped second line as its own paragraph, so "water-stressed" ended
    # up split as "water-\n\nstressed" (hyphen then TWO newlines) rather
    # than the single-newline line-wrap dehyphenation was built for.
    source = (
        "Towards healthy watersheds and nature\n\n"
        "¢ Fully balance water used in our products in water-\n\n"
        "stressed areas by 2030"
    )
    quote = "Fully balance water used in our products in water-stressed areas by 2030"
    assert verify_quote(quote, source) is True


def test_a_paraphrase_is_still_rejected_across_a_hyphenated_paragraph_break():
    # Same safety check as the other fallback tiers: this differs by an
    # actual word ("Balance" vs "balancing"), so it must still fail even
    # though it spans the same hyphen-and-paragraph-break gap.
    source = "Fully balance water used in our products in water-\n\nstressed areas by 2030"
    quote = "Fully balancing water used in our products in water-stressed areas by 2030"
    assert verify_quote(quote, source) is False


def test_a_truncated_quote_closed_with_an_added_period_still_verifies():
    # Real case (Shell): the model quoted only the first clause of a
    # longer sentence and closed it with a period, even though the source
    # continues past that point with a comma, not a full stop.
    source = (
        "This is a real challenge. We have a target to become a "
        "net-zero emissions energy business by 2050, and we are "
        "committed to that."
    )
    quote = "We have a target to become a net-zero emissions energy business by 2050."
    assert verify_quote(quote, source) is True


def test_a_truncated_quote_still_verifies_against_a_bullet_with_no_punctuation():
    # Real case (Heineken): a bullet in the source ends with a line break,
    # not a period, but the model still closed its quote with one.
    source = (
        "Reach net zero carbon\n\n¢ Reach net zero across our value chain by 2040\n\n"
        "¢ Reach net zero in Scope 1 and 2 by 2030"
    )
    quote = "Reach net zero across our value chain by 2040."
    assert verify_quote(quote, source) is True


def test_an_added_period_does_not_let_a_truncated_quote_merge_into_a_longer_number():
    # Canonicalising away the period must not let "2030" match inside
    # "2030s": that would silently accept a different figure as the same
    # one. The boundary padding in `_canonical_form` is what prevents this.
    source = "We expect this trend to continue through the 2030s and beyond."
    quote = "We expect this trend to continue through the 2030."
    assert verify_quote(quote, source) is False


def test_verification_is_tolerant_of_punctuation_it_was_never_specifically_fixed_for():
    # The point of canonicalising to words-and-numbers-only, rather than
    # patching one punctuation shape at a time, is that it also covers
    # shapes nobody hit yet: a semicolon where the quote has a comma, and
    # a quote that truncates right before a trailing footnote marker --
    # still a clean word-boundary prefix, unlike a footnote digit glued
    # directly onto the preceding word with no space (a known, still-
    # rejected residual gap).
    source = "Revenue grew 12%; driven by strong demand in Europe [3]."
    quote = "Revenue grew 12%, driven by strong demand in Europe."
    assert verify_quote(quote, source) is True


def test_value_inside_quote_is_verified():
    quote = "an average of 103,100 FTE"
    assert verify_value_in_quote("103,100", quote) is True


def test_value_not_inside_quote_is_rejected():
    quote = "an average of 103,100 FTE"
    assert verify_value_in_quote("96,442", quote) is False


def test_verify_citation_combines_quote_and_value_checks():
    quote = "an average of 103,100 FTE during 2025"
    result = verify_citation(quote, "103,100", SOURCE)
    assert result.verified is True
    assert result.value == 103_100.0


def test_verify_citation_fails_when_value_not_in_quote_even_if_quote_matches():
    quote = SOURCE  # quote matches the source verbatim...
    result = verify_citation(quote, "1,234", SOURCE)  # ...but this value is not in it
    assert result.verified is False


def test_verify_citation_without_value_text_only_checks_the_quote():
    result = verify_citation(SOURCE, None, SOURCE)
    assert result.verified is True
    assert result.value is None


def test_verify_extraction_behaves_the_same_as_verify_citation():
    quote = "an average of 103,100 FTE during 2025"
    result = verify_extraction(quote, "103,100", SOURCE)
    assert result.verified is True
    assert result.value == 103_100.0
