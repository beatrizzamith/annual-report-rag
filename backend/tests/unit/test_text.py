from app.core.text import normalize_text, parse_number


def test_normalize_dehyphenates_line_break_words():
    assert normalize_text("sustain-\nability") == "sustainability"


def test_normalize_dehyphenates_across_a_paragraph_break_too():
    # Real case (Heineken, page 165): the chunker treated a bullet's
    # wrapped second line as its own paragraph, storing the split as
    # "water-\n\nstressed" (two newlines) rather than the single-newline
    # line-wrap the dehyphenation regex originally looked for.
    assert normalize_text("water-\n\nstressed") == "waterstressed"


def test_normalize_strips_nbsp_and_soft_hyphen():
    text = "EUR 312 million­ target"
    assert normalize_text(text) == "EUR 312 million target"


def test_normalize_unifies_curly_quotes_and_dashes():
    assert normalize_text("“net‑zero” – 2030") == '"net-zero" - 2030'


def test_normalize_collapses_whitespace_and_blank_lines():
    text = "line one\n\n\n\nline two   with   spaces"
    assert normalize_text(text) == "line one\n\nline two with spaces"


def test_normalize_is_idempotent():
    text = "Some “Report” text with­ quirks"
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_parse_number_us_style_thousands_and_decimal():
    assert parse_number("1,234.5") == 1234.5


def test_parse_number_european_style():
    assert parse_number("1.234,5") == 1234.5


def test_parse_number_space_thousands():
    assert parse_number("1 234") == 1234.0


def test_parse_number_million_word():
    assert parse_number("12.3 million") == 12_300_000.0


def test_parse_number_k_suffix():
    assert parse_number("43k") == 43_000.0


def test_parse_number_percentage_returns_bare_number():
    assert parse_number("12%") == 12.0


def test_parse_number_none_when_no_digits():
    assert parse_number("not a number") is None


def test_parse_number_none_for_empty_string():
    assert parse_number("") is None
