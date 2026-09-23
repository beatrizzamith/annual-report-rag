"""Structured output schema for the answer-generation model call."""

from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    """One piece of evidence backing an inline [S3]-style citation in an answer.

    Every inline citation must have one of these, so a citation can always
    be inspected. `label`/`value_text`/`unit`/`period` are only filled in
    when the citation backs a specific figure; a citation for descriptive,
    non-numeric prose leaves them empty and carries just the quote.

    Field order matters here, not just for readability: structured output
    is generated in schema-declaration order, so `source_id`/`quote` are
    declared before the derived fields deliberately -- the model copies the
    real source text first, then derives label/value/unit/period *from*
    that already-copied quote, rather than writing a claim first and having
    to retrofit a supporting quote for it afterwards (the pattern behind a
    real fabricated-quote case this project hit: an accurate claim backed
    by a quote that did not actually appear anywhere in the source).
    """

    source_id: str  # "S3"
    quote: str  # verbatim sentence or table row backing the citation
    label: str | None = None  # "Climate change adaptation spend"
    value_text: str | None = None  # exactly as printed: "USD 312 million"
    unit: str | None = None
    period: str | None = None  # "FY2025"


class ModelAnswer(BaseModel):
    """The model's full structured response to a question.

    Field order matters here too, for the same reason as `Citation`:
    `citations` is declared before `answer` so the model selects and copies
    its evidence first, then writes prose constrained to what it already
    committed to as verbatim -- not the other way around.
    """

    citations: list[Citation] = []
    status: Literal["answered", "partial", "not_found"]
    answer: str  # plain text, cites sources inline as [S3]
    caveats: list[str] = []  # e.g. "Report covers FY2024, not 2025"
