"""Structured-output schemas for the two pre-extraction fields: FTE and sustainability goals."""

from typing import Literal

from pydantic import BaseModel, Field


class FteExtraction(BaseModel):
    """Structured output of the FTE extraction call.

    Field order matters: the model generates fields in declaration order, so
    `quote` comes first and the model copies real source text before deriving
    the value from it.
    """

    quote: str | None = Field(
        default=None, description="Verbatim sentence or table row that contains value_text."
    )
    source_id: str | None = Field(
        default=None, description="Which supplied chunk the quote came from (S1..Sn)."
    )
    found: bool
    value_text: str | None = Field(
        default=None, description='The figure exactly as printed, e.g. "103,100".'
    )
    metric: Literal["fte", "headcount", "average_fte", "other"] = "other"
    as_of: str | None = Field(
        default=None, description='The date or period it applies to, e.g. "31 December 2025".'
    )
    scope: str | None = Field(
        default=None, description='Who it covers, e.g. "Group, consolidated".'
    )
    notes: str | None = Field(
        default=None, description='Caveats, e.g. "Report gives headcount only, not FTE".'
    )


class SustainabilityGoal(BaseModel):
    """One sustainability goal found in a report.

    Field order matters, as in `FteExtraction`: `quote` and `source_id` come
    first so the model copies real source text before deriving the rest.
    """

    quote: str = Field(description="Verbatim sentence(s) stating the goal.")
    source_id: str
    title: str = Field(description='A short label, e.g. "Net-zero emissions".')
    category: Literal[
        "climate",
        "energy",
        "circularity_waste",
        "water",
        "biodiversity",
        "social_people",
        "governance",
        "other",
    ]
    target: str | None = Field(
        default=None, description="The target wording, copied verbatim if short."
    )
    target_year: int | None = None
    baseline: str | None = Field(default=None, description='The baseline, e.g. "2019 levels".')


class GoalsExtraction(BaseModel):
    """Structured output of one goals-extraction call over a batch of chunks."""

    goals: list[SustainabilityGoal]
