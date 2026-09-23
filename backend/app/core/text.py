"""Text normalisation used for both storage and verbatim verification.

Using one function for both means the two can never drift apart: whatever
`normalize_text` does to a chunk when it is stored is exactly what it does to
a quote when that quote is checked against the chunk.
"""

import re
import unicodedata

_SOFT_HYPHEN = "­"
_NBSP_LIKE = "     "
_QUOTE_MAP = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
}
_DASH_MAP = {
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
}
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n+(\w)")
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _join_hyphenated_word(match: re.Match[str]) -> str:
    """Builds the replacement for one line-break-hyphenated word match.

    Args:
        match: A match of `_HYPHEN_LINEBREAK`, whose two groups are the
            characters immediately before and after the hyphen-newline.

    Returns:
        The two captured characters joined with no hyphen or newline
        between them.
    """
    return match.group(1) + match.group(2)


def _dehyphenate(text: str) -> str:
    """Joins words split by a line-break hyphen, e.g. "sustain-\\nability".

    Matches one or more consecutive newlines after the hyphen, not just
    one: the same line-wrap can end up stored as a `"\\n\\n"` paragraph
    break instead of a single `"\\n"` when the chunker treats the wrapped
    fragment as its own paragraph (a real case: "water-\\n\\nstressed",
    from a bullet whose second line PyMuPDF read as a separate block). The
    word is split by the PDF layout either way, so both must rejoin the
    same word.

    Args:
        text: Raw text that may contain hyphen-newline line breaks.

    Returns:
        The text with such line breaks removed and the word rejoined.
    """
    return _HYPHEN_LINEBREAK.sub(_join_hyphenated_word, text)


def normalize_text(text: str) -> str:
    """Canonicalises text for storage and for verbatim comparison.

    Applied once at write time to produce the text stored in `chunks.text`,
    and again at read time to the quote being verified, so both sides of a
    verbatim check go through the identical transformation.

    Args:
        text: Raw text extracted from a PDF, or a quote to be verified
            against such text.

    Returns:
        The canonicalised text: de-hyphenated, NFKC-normalised, with
        non-breaking spaces and soft hyphens removed, quote and dash
        variants unified, and whitespace collapsed.
    """
    text = _dehyphenate(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_SOFT_HYPHEN, "")
    for ch in _NBSP_LIKE:
        text = text.replace(ch, " ")
    for src, dst in _QUOTE_MAP.items():
        text = text.replace(src, dst)
    for src, dst in _DASH_MAP.items():
        text = text.replace(src, dst)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


_NUMBER_WORD_MULTIPLIERS = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "mn": 1_000_000,
    "million": 1_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
}

_NUMBER_PATTERN = re.compile(
    r"(?P<sign>[-+])?"
    r"(?P<number>\d[\d.,\s ]*\d|\d)"
    r"\s*"
    r"(?P<word>thousand|million|billion|mn|bn|k)?"
    r"\s*(?P<percent>%)?",
    re.IGNORECASE,
)


def _parse_localized_number(raw: str) -> float | None:
    """Parses a number string in either `1,234.5` or `1.234,5` style.

    Args:
        raw: The digit-and-separator substring matched by `_NUMBER_PATTERN`
            (no sign, magnitude word, or percent sign).

    Returns:
        The parsed value, or None if `raw` is not a valid number once its
        thousands/decimal separators are resolved.
    """
    raw = raw.replace(" ", "").replace(" ", "")
    has_comma = "," in raw
    has_dot = "." in raw

    if has_comma and has_dot:
        if raw.rfind(",") > raw.rfind("."):
            # European style: '.' is a thousands separator, ',' is decimal.
            raw = raw.replace(".", "").replace(",", ".")
        else:
            # US style: ',' is a thousands separator, '.' is decimal.
            raw = raw.replace(",", "")
    elif has_comma:
        # Ambiguous: '1,234' is thousands, but '1,5' is a decimal. We treat
        # a trailing comma with 1-2 digits as a decimal separator and otherwise
        # strip the comma as a thousands separator.
        if re.search(r",\d{1,2}$", raw) and not re.search(r",\d{3}\b", raw):
            raw = raw.replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif has_dot:
        # '1.234' (thousands) vs '1.5' (decimal): a dot followed by exactly
        # 3 digits with no more digits after is treated as a thousands sep.
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", raw):
            raw = raw.replace(".", "")

    try:
        return float(raw)
    except ValueError:
        return None


def parse_number(text: str) -> float | None:
    """Best-effort numeric parse of a printed value, e.g. from `value_text`.

    Handles thousands/decimal separators in both US and European style,
    magnitude words/suffixes ("12.3 million", "43k"), and percentages
    (returned as the bare number, e.g. "12%" -> 12.0).

    Args:
        text: The printed value to parse, e.g. "1,234.5" or "43k".

    Returns:
        The parsed number, or None if no number could be found in `text`.
    """
    if not text:
        return None
    match = _NUMBER_PATTERN.search(text)
    if not match or not match.group("number"):
        return None

    value = _parse_localized_number(match.group("number"))
    if value is None:
        return None

    if match.group("sign") == "-":
        value = -value

    word = match.group("word")
    if word:
        value *= _NUMBER_WORD_MULTIPLIERS[word.lower()]

    return value
