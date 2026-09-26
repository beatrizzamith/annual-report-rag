"""Builds FTS5 match queries from natural-language input."""

import re

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "than",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "did",
    "do",
    "does",
    "much",
}

_TOKEN = re.compile(r"[A-Za-z0-9%.]+")


def build_fts_query(text: str) -> str:
    """Builds a SQLite FTS5 `MATCH` expression from free text.

    Lower-cases, strips stop words, quotes each remaining token (so FTS5
    treats punctuation like periods literally) and OR-combines them.

    Args:
        text: The question or query text to convert.

    Returns:
        An FTS5 match expression, e.g. `"fte" OR "employees"`. Empty string
        if `text` contains no usable tokens.
    """
    tokens = _TOKEN.findall(text.lower())
    keywords = [token for token in tokens if token not in _STOPWORDS and len(token) > 1]
    if not keywords:
        keywords = tokens
    if not keywords:
        return ""
    quoted = [f'"{token}"' for token in dict.fromkeys(keywords)]
    return " OR ".join(quoted)
