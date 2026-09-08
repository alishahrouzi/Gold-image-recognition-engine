"""S2.9 baseline retrieval error-analysis utilities.

This module consumes the existing S2.7 ``RetrievalResult`` contract and the
S2.8 ``QueryEvaluationRecord`` contract. It does not alter retrieval, metrics,
model training, or dataset contents.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from src.retrieval.result import RetrievalCandidate, RetrievalResult

from .result import QueryEvaluationRecord


@dataclass(frozen=True)
class ErrorAnalysisRecord:
    """Per-query evidence retained for S2.9 analysis and manual review."""

    query_id: str
    query_image_path: str | None
    query_category: str | None
    query_product_group: str
    first_positive_rank: int | None
    top1_correct: bool
    top5_correct: bool
    top10_correct: bool
    top1_candidate: dict[str, Any] | None
    top5_candidates: tuple[dict[str, Any], ...]
    top10_candidates: tuple[dict[str, Any], ...]
    top10_product_evidence: tuple[dict[str, Any], ...]
    top1_similarity: float | None
    best_observed_positive_similarity: float | None
    top1_to_positive_similarity_margin: float | None
    excluded: bool = False
    exclusion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_dict(candidate: RetrievalCandidate, *, image_path: str | None = None) -> dict[str, Any]:
    return {
        "image_id": candidate.image_id,
        "product_group": candidate.product_group,
        "category": candidate.category,
        "similarity": float(candidate.similarity),
        "rank": int(candidate.rank),
        "image_path": image_path if image_path is not None else candidate.image_path,
    }


def _product_evidence(
    candidates: Sequence[dict[str, Any]],
    *,
    query_product_group: str,
) -> tuple[dict[str, Any], ...]:
    """Collapse the observed Top-10 image candidates to first-seen products.

    This is deliberately named ``top10_product_evidence`` rather than Top-10
    product retrieval: the underlying S2.7 retriever returns ten images, so
    this field only describes products represented in those ten image results.
    """
    products: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        product_group = str(candidate["product_group"])
        if product_group in products:
            continue
        products[product_group] = {
            "product_rank": len(products) + 1,
            "product_group": product_group,
            "category": candidate.get("category"),
            "best_observed_similarity": float(candidate["similarity"]),
            "representative_image_id": candidate["image_id"],
            "representative_image_path": candidate.get("image_path"),
            "is_query_product": product_group == query_product_group,
        }
    return tuple(products.values())


def build_error_record(
    result: RetrievalResult,
    evaluation: QueryEvaluationRecord,
    *,
    query_image_path: str | None = None,
    candidate_paths: Mapping[str, str] | None = None,
) -> ErrorAnalysisRecord:
    """Combine an S2.7 result and S2.8 score into one auditable record."""
    candidate_paths = candidate_paths or {}
    candidates = tuple(
        _candidate_dict(candidate, image_path=candidate_paths.get(candidate.image_id))
        for candidate in result.candidates
    )
    top1_similarity = float(candidates[0]["similarity"]) if candidates else None
    positive_similarities = [
        float(candidate["similarity"])
        for candidate in candidates
        if str(candidate["product_group"]) == evaluation.query_group_id
    ]
    best_positive = max(positive_similarities) if positive_similarities else None
    margin = (
        top1_similarity - best_positive
        if top1_similarity is not None and best_positive is not None
        else None
    )
    return ErrorAnalysisRecord(
        query_id=evaluation.query_id,
        query_image_path=query_image_path,
        query_category=evaluation.category,
        query_product_group=evaluation.query_group_id,
        first_positive_rank=evaluation.first_positive_rank,
        top1_correct=bool(evaluation.top1_hit),
        top5_correct=bool(evaluation.top5_hit),
        top10_correct=bool(evaluation.top10_hit),
        top1_candidate=candidates[0] if candidates else None,
        top5_candidates=candidates[:5],
        top10_candidates=candidates[:10],
        top10_product_evidence=_product_evidence(
            candidates[:10], query_product_group=evaluation.query_group_id
        ),
        top1_similarity=top1_similarity,
        best_observed_positive_similarity=best_positive,
        top1_to_positive_similarity_margin=margin,
        excluded=evaluation.excluded,
        exclusion_reason=evaluation.exclusion_reason,
    )


def _valid(records: Iterable[ErrorAnalysisRecord]) -> list[ErrorAnalysisRecord]:
    return [record for record in records if not record.excluded]


def category_summary(records: Sequence[ErrorAnalysisRecord]) -> dict[str, dict[str, Any]]:
    """Return deterministic per-category retrieval/error statistics."""
    grouped: dict[str, list[ErrorAnalysisRecord]] = defaultdict(list)
    for record in _valid(records):
        grouped[record.query_category or "unknown"].append(record)

    output: dict[str, dict[str, Any]] = {}
    for category in sorted(grouped):
        rows = grouped[category]
        ranks = [r.first_positive_rank for r in rows if r.first_positive_rank is not None]
        margins = [
            r.top1_to_positive_similarity_margin
            for r in rows
            if r.top1_to_positive_similarity_margin is not None
        ]
        output[category] = {
            "num_queries": len(rows),
            "top1_correct_count": sum(r.top1_correct for r in rows),
            "top5_correct_count": sum(r.top5_correct for r in rows),
            "top10_correct_count": sum(r.top10_correct for r in rows),
            "top1": mean(r.top1_correct for r in rows),
            "top5": mean(r.top5_correct for r in rows),
            "top10": mean(r.top10_correct for r in rows),
            "mean_first_positive_rank": mean(ranks) if ranks else None,
            "median_first_positive_rank": median(ranks) if ranks else None,
            "no_positive_in_top10_count": sum(r.first_positive_rank is None for r in rows),
            "mean_top1_to_positive_similarity_margin": mean(margins) if margins else None,
        }
    return output


def category_confusion(records: Sequence[ErrorAnalysisRecord]) -> dict[str, dict[str, int]]:
    """Count query-category to incorrect Top-1 candidate-category pairs."""
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for record in _valid(records):
        candidate = record.top1_candidate
        if record.top1_correct or candidate is None:
            continue
        query_category = record.query_category or "unknown"
        candidate_category = str(candidate.get("category") or "unknown")
        matrix[query_category][candidate_category] += 1
    return {
        query_category: dict(sorted(counter.items()))
        for query_category, counter in sorted(matrix.items())
    }


def _rank_key(record: ErrorAnalysisRecord) -> tuple[int, str]:
    return (record.first_positive_rank if record.first_positive_rank is not None else 10**9, record.query_id)


def _similarity_margin_key(record: ErrorAnalysisRecord) -> tuple[float, str]:
    margin = record.top1_to_positive_similarity_margin
    return (abs(margin) if margin is not None else float("inf"), record.query_id)


def _select_pool(candidates: Sequence[ErrorAnalysisRecord], per_group: int, seed: int) -> list[ErrorAnalysisRecord]:
    pool = sorted(candidates, key=_rank_key)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:per_group], key=_rank_key)


def _select_category_stratified(
    candidates: Sequence[ErrorAnalysisRecord],
    *,
    per_category: int,
    seed: int,
) -> list[ErrorAnalysisRecord]:
    grouped: dict[str, list[ErrorAnalysisRecord]] = defaultdict(list)
    for record in candidates:
        grouped[record.query_category or "unknown"].append(record)

    selected: list[ErrorAnalysisRecord] = []
    for category in sorted(grouped):
        rows = sorted(grouped[category], key=_rank_key)
        rng = random.Random(seed + sum(ord(char) for char in category))
        rng.shuffle(rows)
        selected.extend(sorted(rows[:per_category], key=_rank_key))
    return selected


def select_representative_examples(
    records: Sequence[ErrorAnalysisRecord],
    *,
    per_group: int = 10,
    seed: int = 42,
    per_category: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Select deterministic examples, including category-stratified evidence."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    if per_category < 1:
        raise ValueError("per_category must be positive")

    valid = _valid(records)
    groups: dict[str, list[ErrorAnalysisRecord]] = {
        "top1_correct": [r for r in valid if r.top1_correct],
        "top1_wrong_top5_correct": [r for r in valid if not r.top1_correct and r.top5_correct],
        "top5_wrong_top10_correct": [r for r in valid if not r.top5_correct and r.top10_correct],
        "top10_wrong": [r for r in valid if not r.top10_correct],
        "no_positive_top10": [r for r in valid if r.first_positive_rank is None],
        "high_similarity_wrong": [
            r for r in valid if not r.top1_correct and r.top1_candidate is not None
        ],
        "low_similarity_correct": [
            r for r in valid if r.top1_correct and r.top1_candidate is not None
        ],
        "smallest_observed_positive_margin": [
            r
            for r in valid
            if r.top1_to_positive_similarity_margin is not None and not r.top1_correct
        ],
    }

    output: dict[str, list[dict[str, Any]]] = {}
    for name, candidates in groups.items():
        if name == "high_similarity_wrong":
            selected = sorted(
                candidates,
                key=lambda r: (-float(r.top1_similarity), r.query_id),
            )[:per_group]
        elif name == "low_similarity_correct":
            selected = sorted(
                candidates,
                key=lambda r: (float(r.top1_similarity), r.query_id),
            )[:per_group]
        elif name == "smallest_observed_positive_margin":
            selected = sorted(candidates, key=_similarity_margin_key)[:per_group]
        else:
            selected = _select_pool(candidates, per_group, seed)
        output[name] = [record.to_dict() for record in selected]

    output["category_stratified_top10_wrong"] = [
        record.to_dict()
        for record in _select_category_stratified(
            [r for r in valid if not r.top10_correct],
            per_category=per_category,
            seed=seed,
        )
    ]
    output["category_stratified_high_similarity_wrong"] = [
        record.to_dict()
        for record in _select_category_stratified(
            [r for r in valid if not r.top1_correct and r.top1_candidate is not None],
            per_category=per_category,
            seed=seed,
        )
    ]
    return output


def build_summary(records: Sequence[ErrorAnalysisRecord]) -> dict[str, Any]:
    """Summarize S2.9 evidence without making visual-cause claims."""
    valid = _valid(records)
    ranks = [r.first_positive_rank for r in valid if r.first_positive_rank is not None]
    margins = [
        r.top1_to_positive_similarity_margin
        for r in valid
        if r.top1_to_positive_similarity_margin is not None
    ]
    return {
        "number_of_queries": len(records),
        "valid_queries": len(valid),
        "excluded_queries": len(records) - len(valid),
        "top1_correct": sum(r.top1_correct for r in valid),
        "top5_correct": sum(r.top5_correct for r in valid),
        "top10_correct": sum(r.top10_correct for r in valid),
        "positive_outside_top10": sum(r.first_positive_rank is None for r in valid),
        "mean_first_positive_rank": mean(ranks) if ranks else None,
        "median_first_positive_rank": median(ranks) if ranks else None,
        "mean_top1_to_positive_similarity_margin": mean(margins) if margins else None,
    }
