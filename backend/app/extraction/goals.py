"""Extracts sustainability goals and their supporting metadata.

Goals span a whole chapter, more than a few top-ranked chunks can cover, so
this gathers every plausible chunk and extracts from them in batches (map),
then merges and deduplicates the results (reduce).
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from functools import partial

from app.core.prompts import load_prompt
from app.db.models import Chunk
from app.db.repositories import ChunksRepo, ExtractionsRepo
from app.extraction.context import build_source_context
from app.extraction.schemas import GoalsExtraction, SustainabilityGoal
from app.extraction.verifier import verify_quote
from app.llm.client import LLMClient
from app.retrieval.hybrid import multi_query_search
from app.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)

QUERIES = [
    "net zero ambition",
    "emission reduction target 2030 scope 1 2 3",
    "we aim to",
    "science based targets",
    "renewable energy target",
    "biodiversity water waste circularity target",
    "diversity inclusion target",
    "sustainability strategy goals",
]
SEMANTIC_TOP_K_PER_QUERY = 20
SEMANTIC_FINAL_TOP_K = 40

# Chunks mentioning these terms count as sustainability content. Goals are often
# phrased without "target" ("we aim to..."), so topic terms catch more of them.
_SUSTAINABILITY_TERMS = re.compile(
    r"\b("
    r"sustainab\w*|climate|emissions?|carbon|co2e?|scope\s*[123]\b|renewable|"
    r"circular\w*|biodiversity|water|waste|net[- ]zero|greenhouse gas|ghg|"
    r"esrs|human rights|diversity|inclusion|community|deforestation|energy efficiency"
    r")\b",
    re.IGNORECASE,
)

MAX_CANDIDATE_CHUNKS = 150 
BATCH_SIZE = 10
MAX_GOALS = 60
# Batches are independent, so they run concurrently instead of as ~15 sequential
# round trips. Capped so a large report doesn't trip the provider's rate limit.
MAX_CONCURRENT_BATCHES = 5

_WHITESPACE_AND_HYPHENS = re.compile(r"[\s\-]+")

ScoredGoal = tuple[SustainabilityGoal, Chunk]

# Minimum similarity of two goals' (title + target) text for them to count as
# the same commitment worded differently. Tuned on real restated goals.
_DUPLICATE_SIMILARITY_THRESHOLD = 0.6


def _dedupe_text(goal: SustainabilityGoal) -> str:
    """Builds the text two goals' similarity is compared on.

    Args:
        goal: A goal to build comparison text for.

    Returns:
        `goal`'s title and target, lower-cased with all whitespace and
        hyphens removed -- so "net-zero", "net zero" and "netzero" (all
        seen across real reports for the same phrase) compare as identical,
        rather than diluting the similarity ratio with punctuation noise.
    """
    combined = f"{goal.title} {goal.target or ''}".lower()
    return _WHITESPACE_AND_HYPHENS.sub("", combined)


def _are_likely_duplicates(first_goal: SustainabilityGoal, second_goal: SustainabilityGoal) -> bool:
    """Checks whether two goals are probably the same commitment, reworded.

    Gated on an exact category and target-year match first, so two
    genuinely distinct goals that happen to be worded similarly (e.g. two
    different targets both due in 2030) are never merged just because their
    titles overlap.

    Args:
        first_goal: A goal to compare.
        second_goal: Another goal to compare it against.

    Returns:
        True if both goals share a category and target year, and their
        title-plus-target text is at least `_DUPLICATE_SIMILARITY_THRESHOLD`
        similar.
    """
    if (
        first_goal.category != second_goal.category
        or first_goal.target_year != second_goal.target_year
    ):
        return False
    ratio = SequenceMatcher(None, _dedupe_text(first_goal), _dedupe_text(second_goal)).ratio()
    return ratio >= _DUPLICATE_SIMILARITY_THRESHOLD


def _dedupe(scored_goals: list[ScoredGoal]) -> list[ScoredGoal]:
    """Removes duplicate goals, keeping the best-evidenced version of each.

    Batches can't see each other, so one goal restated in two places comes
    back with two different titles. Goals are compared by fuzzy similarity,
    not exact title, in code with no extra model call.

    Args:
        scored_goals: Goals extracted by the model, each paired with its
            source chunk, possibly containing the same goal more than once.

    Returns:
        One `(goal, chunk)` pair per detected duplicate group: whichever
        had the longest quote.
    """
    kept: list[ScoredGoal] = []
    for goal, chunk in scored_goals:
        duplicate_index = next(
            (i for i, (kept_goal, _) in enumerate(kept) if _are_likely_duplicates(goal, kept_goal)),
            None,
        )
        if duplicate_index is None:
            kept.append((goal, chunk))
        elif len(goal.quote) > len(kept[duplicate_index][0].quote):
            kept[duplicate_index] = (goal, chunk)
    return kept


def _rank(scored_goals: list[ScoredGoal]) -> list[ScoredGoal]:
    """Orders goals so the most concrete commitments sort first.

    Args:
        scored_goals: De-duplicated `(goal, chunk)` pairs to rank.

    Returns:
        `scored_goals` sorted so entries with both a target and a target
        year come first, so the UI shows the most concrete commitments
        first when there are more than `MAX_GOALS`.
    """
    return sorted(
        scored_goals, key=lambda pair: 0 if (pair[0].target and pair[0].target_year) else 1
    )


def _is_still_open(goal: SustainabilityGoal, report_fiscal_year: int) -> bool:
    """Checks whether a goal's target year could still be in the future.

    A deterministic backstop for the prompt's "skip fulfilled commitments"
    rule: a report is published after its fiscal year ends, so a target dated
    to that year or earlier has already passed. A goal with no `target_year`
    is always kept.

    Args:
        goal: A goal the model extracted.
        report_fiscal_year: The report's own fiscal year.

    Returns:
        False only for a goal whose `target_year` is at or before
        `report_fiscal_year`.
    """
    return goal.target_year is None or goal.target_year > report_fiscal_year


def _candidate_chunks(
    report_id: int,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    llm: LLMClient,
) -> list[Chunk]:
    """Gathers every chunk that plausibly belongs to the sustainability chapter.

    Unions a keyword pass over the whole report with a semantic search pass,
    so the set approximates the whole chapter rather than a few top-ranked
    chunks. Each pass catches phrasings the other misses.

    Args:
        report_id: The report to gather candidates from.
        chunks_repo: Repository for chunk lookup and search.
        vector_index: The in-memory index used for semantic search.
        llm: The LLM client used to embed the semantic queries.

    Returns:
        Candidate chunks, in document order, capped at
        `MAX_CANDIDATE_CHUNKS`.
    """
    matched: dict[int, Chunk] = {
        chunk.id: chunk
        for chunk in chunks_repo.get_by_report(report_id)
        if _SUSTAINABILITY_TERMS.search(chunk.text)
    }

    semantic_ids = multi_query_search(
        QUERIES,
        [report_id],
        chunks_repo,
        vector_index,
        llm,
        SEMANTIC_TOP_K_PER_QUERY,
        SEMANTIC_FINAL_TOP_K,
    )
    for chunk_id in semantic_ids:
        if chunk_id not in matched:
            chunk = chunks_repo.get(chunk_id)
            if chunk is not None:
                matched[chunk_id] = chunk

    ordered = sorted(matched.values(), key=lambda chunk: (chunk.page_start, chunk.id))
    return ordered[:MAX_CANDIDATE_CHUNKS]


def _batched(chunks: list[Chunk], batch_size: int) -> list[list[Chunk]]:
    """Splits chunks into fixed-size batches, preserving order.

    Args:
        chunks: The chunks to split.
        batch_size: The maximum number of chunks per batch.

    Returns:
        Consecutive slices of `chunks`, each at most `batch_size` long.
    """
    return [chunks[start : start + batch_size] for start in range(0, len(chunks), batch_size)]


def _extract_batch(
    batch: list[Chunk],
    system_prompt: str,
    llm: LLMClient,
) -> list[ScoredGoal]:
    """Runs one extraction call over a batch of chunks (the "map" step).

    Args:
        batch: The chunks to extract goals from.
        system_prompt: The loaded `extract_goals.md` prompt.
        llm: The LLM client used for the extraction call.

    Returns:
        Every goal the model returned for this batch, paired with the
        chunk its `source_id` resolved to. A goal citing an unresolvable
        `source_id` is dropped and logged.
    """
    context, source_map = build_source_context(batch)
    user_prompt = f"{context}\n\nExtract sustainability goals following the rules."
    result = llm.complete_structured(system_prompt, user_prompt, GoalsExtraction, temperature=0)

    resolved = []
    for goal in result.goals:
        source_chunk_id = source_map.get(goal.source_id)
        source_chunk = next((chunk for chunk in batch if chunk.id == source_chunk_id), None)
        if source_chunk is None:
            logger.warning(
                "goals extraction: model cited an unknown source_id",
                extra={"extra_fields": {"source_id": goal.source_id}},
            )
            continue
        resolved.append((goal, source_chunk))
    return resolved


def extract_goals(
    report_id: int,
    report_fiscal_year: int,
    chunks_repo: ChunksRepo,
    vector_index: VectorIndex,
    llm: LLMClient,
    extractions_repo: ExtractionsRepo,
    model_name: str,
) -> list[int]:
    """Runs map-reduce sustainability-goal extraction for one report.

    Gathers every chunk plausibly in the sustainability chapter (map input),
    extracts goals from each batch of them, then drops any already-fulfilled
    commitment, deduplicates, and ranks the rest (reduce) before storing the
    best `MAX_GOALS`.

    Args:
        report_id: The report to extract from. Must already be chunked and
            embedded.
        report_fiscal_year: The report's own fiscal year, for filtering out
            goals whose target year has already passed (see `_is_still_open`).
        chunks_repo: Repository for retrieval and chunk lookup.
        vector_index: The in-memory index used for semantic search.
        llm: The LLM client used for retrieval embeddings and extraction.
        extractions_repo: Repository extracted items are persisted to.
        model_name: The chat model name, recorded on each extraction row.

    Returns:
        The new `extractions` row ids, one per stored goal. Empty if there
        were no candidate chunks or the model found no still-open goals.
    """
    candidates = _candidate_chunks(report_id, chunks_repo, vector_index, llm)
    if not candidates:
        logger.info(
            "goals extraction: no candidate chunks",
            extra={"extra_fields": {"report_id": report_id}},
        )
        return []

    system_prompt = load_prompt("extract_goals.md")
    batches = _batched(candidates, BATCH_SIZE)
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_BATCHES, len(batches))) as executor:
        extract = partial(_extract_batch, system_prompt=system_prompt, llm=llm)
        batch_results = executor.map(extract, batches)
    scored_goals = [pair for pairs in batch_results for pair in pairs]
    open_goals = [
        (goal, chunk) for goal, chunk in scored_goals if _is_still_open(goal, report_fiscal_year)
    ]

    ranked = _rank(_dedupe(open_goals))[:MAX_GOALS]

    inserted_ids = []
    for goal, source_chunk in ranked:
        verified = verify_quote(goal.quote, source_chunk.text)
        inserted_ids.append(
            extractions_repo.insert(
                report_id=report_id,
                kind="sustainability_goal",
                payload=json.dumps(goal.model_dump()),
                quote=goal.quote,
                page=source_chunk.page_start,
                chunk_id=source_chunk.id,
                verified=verified,
                model=model_name,
            )
        )
    return inserted_ids
