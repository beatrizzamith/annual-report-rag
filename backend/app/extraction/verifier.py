"""Checks whether a model claim matches the source text closely enough to trust it.

The same verification is used for pre-extraction (FTE, goals) and chat
citations, so a quote is only marked as verified when it matches the source
chunk after normalisation and, if a value is claimed, that value also
appears inside the quote.
"""

import re
from dataclasses import dataclass

from app.core.text import normalize_text, parse_number

_NON_WORD_RUN = re.compile(r"[^a-z0-9]+")


def _canonical_form(text: str) -> str:
    """Reduces text to its words and numbers only, padded for boundary-safe matching.

    Hyphens are deleted outright rather than turned into a separator: PDF
    line-wraps and our own de-hyphenation sometimes eat a compound word's
    hyphen on one side but not the other (a real case: source stored as
    "wellbelow-2C", quote faithfully written "well-below-2C") -- both sides
    need every hyphen gone to line back up. Every *other* run of
    punctuation or whitespace (periods, commas, footnote brackets, a
    chunker-inserted "\\n\\n" paragraph gap) becomes exactly one space
    instead: a quote and source that agree on every word but disagree on
    how it's punctuated or line-wrapped should match, but two words must
    never be allowed to fuse into one, or a genuinely different number
    could look like a match (e.g. deleting the space in "2030s" would
    make it indistinguishable from "2030").

    The result is padded with a leading and trailing space so a caller
    doing a plain substring check gets word-boundary safety for free: for
    `_canonical_form(quote)` to be found inside `_canonical_form(source)`,
    the match must be flanked by real word boundaries on both sides, not
    just happen to be a prefix of a longer word or number in the source.

    Args:
        text: Raw text -- a quote or a source chunk -- to canonicalise.

    Returns:
        `text` lower-cased, stripped of everything but letters, digits and
        single-space separators, and padded with one space on each end.
    """
    text = normalize_text(text).lower().replace("-", "")
    text = _NON_WORD_RUN.sub(" ", text).strip()
    return f" {text} "


def verify_quote(quote: str | None, source_text: str) -> bool:
    """Checks whether a quoted snippet appears in the source text after canonicalisation.

    Compares the quote and source in a canonical form that ignores
    hyphenation, whitespace and punctuation differences (see
    `_canonical_form`) -- covering, in one pass, every specific gap this
    project has hit in practice: a PDF line-wrap eating a compound word's
    hyphen, a paragraph break re-introduced where a sentence was actually
    continuous, and a model closing a truncated quote with a period the
    source doesn't have there. It only ever treats two texts as matching
    because they agree on every word and number, never because a wrong
    word or a different number happens to look similar once punctuation is
    stripped away.

    Args:
        quote: The quoted text to verify.
        source_text: The full source text that should contain the quote.

    Returns:
        True when the quote's words and numbers appear, in order and at a
        genuine word boundary, inside the source text's.
    """
    if not quote or not quote.strip():
        return False
    return _canonical_form(quote) in _canonical_form(source_text)


def verify_value_in_quote(value_text: str | None, quote: str | None) -> bool:
    """Checks whether a reported value appears inside the quoted sentence.

    Args:
        value_text: The numeric or textual value claimed in the answer.
        quote: The source quote that should contain that value.

    Returns:
        True when `value_text` appears in `quote` after normalisation.
    """
    if not value_text or not quote:
        return False
    return normalize_text(value_text) in normalize_text(quote)


@dataclass
class VerificationResult:
    verified: bool
    value: float | None


def _verify_quote_and_value(
    quote: str | None, value_text: str | None, source_text: str
) -> VerificationResult:
    """Verifies a claimed quote and value against the canonical source text.

    Shared implementation behind `verify_citation` and `verify_extraction`:
    the check itself is identical either way, only what's calling it
    differs, so the public API is two correctly-named wrappers over this
    one private function rather than one function named after just one of
    its callers.

    Args:
        quote: The alleged quote to check.
        value_text: The value claimed to appear inside the quote, if any.
        source_text: The full source chunk text to compare against.

    Returns:
        A result containing whether the claim verified and the parsed numeric
        value when one was supplied.
    """
    quote_ok = verify_quote(quote, source_text)
    value_ok = verify_value_in_quote(value_text, quote) if value_text else True
    value = parse_number(value_text) if value_text else None
    return VerificationResult(verified=quote_ok and value_ok, value=value)


def verify_citation(
    quote: str | None, value_text: str | None, source_text: str
) -> VerificationResult:
    """Verifies a chat citation's claimed quote and value against the source.

    Args:
        quote: The citation's alleged quote to check.
        value_text: The value claimed to appear inside the quote, if any
            (None for a citation backing non-numeric prose).
        source_text: The full source chunk text to compare against.

    Returns:
        A result containing whether the citation verified and the parsed
        numeric value when one was supplied.
    """
    return _verify_quote_and_value(quote, value_text, source_text)


def verify_extraction(
    quote: str | None, value_text: str | None, source_text: str
) -> VerificationResult:
    """Verifies a pre-extracted field's (e.g. FTE) claimed quote and value.

    Args:
        quote: The extraction's alleged quote to check.
        value_text: The value claimed to appear inside the quote, if any.
        source_text: The full source chunk text to compare against.

    Returns:
        A result containing whether the extraction verified and the parsed
        numeric value when one was supplied.
    """
    return _verify_quote_and_value(quote, value_text, source_text)
