"""Tests for goals.py's code-level cleanup: dropping fulfilled goals and de-duplicating."""

from app.db.models import Chunk
from app.extraction.goals import _dedupe, _is_still_open
from app.extraction.schemas import SustainabilityGoal

REPORT_FISCAL_YEAR = 2025


def _goal(
    target_year: int | None,
    title: str = "Some goal",
    target: str | None = "some target",
    category: str = "climate",
    quote: str = "irrelevant",
) -> SustainabilityGoal:
    return SustainabilityGoal(
        title=title,
        category=category,
        target=target,
        target_year=target_year,
        baseline=None,
        quote=quote,
        source_id="S1",
    )


def _chunk(id_: int) -> Chunk:
    return Chunk(
        id=id_,
        report_id=1,
        page_start=id_,
        page_end=id_,
        kind="text",
        text="t",
        embed_text="t",
        embedding=None,
    )


def test_a_target_year_after_the_report_is_open():
    assert _is_still_open(_goal(2030), REPORT_FISCAL_YEAR) is True


def test_a_target_year_before_the_report_is_not_open():
    assert _is_still_open(_goal(2020), REPORT_FISCAL_YEAR) is False


def test_a_target_year_equal_to_the_report_is_not_open():
    # The report is published after its own fiscal year ends, so a target
    # dated to that same year has necessarily already passed by then.
    assert _is_still_open(_goal(REPORT_FISCAL_YEAR), REPORT_FISCAL_YEAR) is False


def test_no_target_year_is_always_open():
    assert _is_still_open(_goal(None), REPORT_FISCAL_YEAR) is True


def test_reworded_restatements_of_the_same_goal_are_merged():
    # A real case: three map-reduce batches independently titled the same
    # net-zero-by-2050 commitment differently.
    scored_goals = [
        (
            _goal(
                2050,
                "Net-zero economy alignment",
                "transition to a net-zero economy by 2050",
                quote="short",
            ),
            _chunk(25),
        ),
        (
            _goal(
                2050,
                "Net-zero emissions economy",
                "netzero emissions economy by 2050",
                quote="a longer quote",
            ),
            _chunk(28),
        ),
        (
            _goal(2050, "Net-zero emissions by 2050", "net-zero emissions by 2050", quote="q"),
            _chunk(29),
        ),
    ]

    deduped = _dedupe(scored_goals)

    assert len(deduped) == 1
    kept_goal, kept_chunk = deduped[0]
    assert kept_goal.title == "Net-zero emissions economy"  # the longest quote wins
    assert kept_chunk.id == 28


def test_goals_with_different_targets_are_not_merged_even_if_same_category_and_year():
    scored_goals = [
        (_goal(2030, "Emissions reduction", "-50% Scope 1 and 2 emissions by 2030"), _chunk(1)),
        (_goal(2030, "Renewable electricity", "100% renewable electricity by 2030"), _chunk(2)),
    ]

    deduped = _dedupe(scored_goals)

    assert len(deduped) == 2


def test_similar_wording_is_not_merged_across_different_categories_or_years():
    same_year_different_category = [
        (_goal(2030, "Net-zero emissions", "net-zero by 2030", category="climate"), _chunk(1)),
        (
            _goal(2030, "Net-zero waste", "net-zero waste by 2030", category="circularity_waste"),
            _chunk(2),
        ),
    ]
    assert len(_dedupe(same_year_different_category)) == 2

    same_category_different_year = [
        (_goal(2030, "Net-zero emissions by 2030", "net-zero by 2030"), _chunk(1)),
        (_goal(2050, "Net-zero emissions by 2050", "net-zero by 2050"), _chunk(2)),
    ]
    assert len(_dedupe(same_category_different_year)) == 2
