"""Structured output schema for the answer-generation model call."""

from typing import Literal

from pydantic import BaseModel, Field

AnswerStatus = Literal["answered", "partial", "not_found"]


class Citation(BaseModel):
    """One piece of evidence backing an inline [S3]-style citation in an answer.

    `label`/`value_text`/`unit`/`period` are only filled in when the citation
    backs a specific figure; a non-numeric one carries just the quote.

    Field order matters: the model generates fields in declaration order, so
    `quote` comes first and the model copies real source text before deriving
    values from it, instead of writing a claim and then inventing a quote.
    """

    source_id: str = Field(description='The source tag this evidence comes from, e.g. "S3".')
    quote: str = Field(description="Verbatim sentence or table row backing the citation.")
    label: str | None = Field(
        default=None, description='What the figure is, e.g. "Climate change adaptation spend".'
    )
    value_text: str | None = Field(
        default=None, description='The figure exactly as printed, e.g. "USD 312 million".'
    )
    unit: str | None = None
    period: str | None = Field(default=None, description='The period it covers, e.g. "FY2025".')


class ModelAnswer(BaseModel):
    """The model's full structured response to a question.

    As in `Citation`, order matters: `citations` comes before `answer`, so
    the model commits to its evidence first and then writes prose around it.
    """

    citations: list[Citation] = Field(default_factory=list)
    status: AnswerStatus
    answer: str = Field(description="Plain text that cites its sources inline as [S3].")
    caveats: list[str] = Field(
        default_factory=list, description='Warnings, e.g. "Report covers FY2024, not 2025".'
    )
