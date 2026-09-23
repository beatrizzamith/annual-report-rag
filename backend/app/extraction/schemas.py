"""Structured-output schemas for the two pre-extraction fields: FTE and sustainability goals."""

from typing import Literal

from pydantic import BaseModel


class FteExtraction(BaseModel):
    """Field order matters, not just readability: structured output is
    generated in schema-declaration order, so `quote`/`source_id` are
    declared before the derived fields deliberately -- the model copies the
    real source text (or finds none) first, then derives `found`/
    `value_text`/`metric`/etc. *from* that already-copied quote, rather
    than deciding there is a figure and retrofitting a supporting quote
    for it afterwards.
    """

    quote: str | None = None  # verbatim sentence or table row that contains value_text
    source_id: str | None = None  # which supplied chunk the quote came from (S1..Sn)
    found: bool
    value_text: str | None = None  # exactly as printed, e.g. "103,100" or "approx. 43,000 FTE"
    metric: Literal["fte", "headcount", "average_fte", "other"] = "other"
    as_of: str | None = None  # "31 December 2025", "FY2025", "average 2025"
    scope: str | None = None  # "Group, consolidated"
    notes: str | None = None  # e.g. "Report gives headcount only, not FTE"


class SustainabilityGoal(BaseModel):
    """Field order matters here too, for the same reason as `FteExtraction`:
    `quote`/`source_id` are declared before the derived fields, so the
    model copies the real source text first and derives
    `title`/`category`/`target`/etc. from it, not the other way around.
    """

    quote: str  # verbatim sentence(s) stating the goal
    source_id: str
    title: str  # short label written by the model, e.g. "Net-zero emissions"
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
    target: str | None = None  # the target wording, copied verbatim if short
    target_year: int | None = None
    baseline: str | None = None  # e.g. "2019 levels"


class GoalsExtraction(BaseModel):
    goals: list[SustainabilityGoal]
