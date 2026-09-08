"""S2.8 evaluation over existing S2.7 retrieval results.

This module does not extract embeddings, compute cosine similarity, or
implement Top-K. It consumes ``RetrievalResult`` values produced by
``TopKRetriever`` / ``BaselineRetrievalPipeline`` and scores them with
the existing ``metrics`` functions.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable, Mapping, Optional, Sequence

from src.retrieval.gallery import Gallery
from src.retrieval.result import RetrievalCandidate, RetrievalResult
from src.retrieval.topk import TopKRetriever

from . import metrics
from .errors import EvaluationError
from .leakage import assert_gallery_metadata_aligned, assert_same_split_protocol
from .result import (
    EVALUATION_K,
    HIT_CUTOFFS,
    QueryEvaluationRecord,
    RetrievalMetricResult,
)


def _require_non_empty_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _validate_candidates(
    candidates: Sequence[RetrievalCandidate],
    *,
    query_image_id: str,
    requested_k: int,
) -> None:
    previous_score = math.inf
    seen_image_ids: set[str] = set()
    invalid_scores = 0
    for index, candidate in enumerate(candidates, start=1):
        image_id = _require_non_empty_id(candidate.image_id, "candidate.image_id")
        _require_non_empty_id(candidate.product_group, "candidate.product_group")
        if image_id in seen_image_ids:
            raise EvaluationError(f"Duplicate candidate image_id {image_id!r}.")
        seen_image_ids.add(image_id)
        if image_id == query_image_id:
            raise EvaluationError(
                f"Query image {query_image_id!r} appeared in the ranked gallery; "
                "self-image exclusion is required."
            )
        if candidate.rank != index:
            raise EvaluationError(
                f"Candidate ranks are not 1-indexed sequential order: "
                f"expected rank {index}, got {candidate.rank}."
            )
        score = float(candidate.similarity)
        if not math.isfinite(score):
            invalid_scores += 1
            raise EvaluationError(
                f"Non-finite similarity at rank {index}: {candidate.similarity!r}."
            )
        if score > previous_score:
            raise EvaluationError(
                "Retrieval ranking is not ordered by non-increasing similarity."
            )
        previous_score = score
    if requested_k < 1 or not isinstance(requested_k, int) or isinstance(requested_k, bool):
        raise EvaluationError(f"K must be a positive integer, got {requested_k!r}.")
    if invalid_scores:
        raise EvaluationError(f"Encountered {invalid_scores} NaN/Inf similarity scores.")


def _unique_product_order(candidates: Sequence[RetrievalCandidate]) -> list[str]:
    """Preserve first-seen product identity from the S2.7 ranking."""
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        group_id = candidate.product_group
        if group_id not in seen:
            seen.add(group_id)
            ordered.append(group_id)
    return ordered


def score_retrieval_result(
    result: RetrievalResult,
    *,
    query_group_id: str,
    query_image_id: str,
    split: str,
    gallery_split: str,
    requested_k: int = EVALUATION_K,
    gallery_positive_images: int = 0,
    gallery_positive_products: int = 0,
    category: Optional[str] = None,
    self_match_excluded: bool = False,
) -> QueryEvaluationRecord:
    """Score one S2.7 ``RetrievalResult`` against product-level ground truth."""
    query_id = _require_non_empty_id(result.query_id, "query_id")
    query_group_id = _require_non_empty_id(query_group_id, "query_group_id")
    query_image_id = _require_non_empty_id(query_image_id, "query_image_id")
    _validate_candidates(
        result.candidates,
        query_image_id=query_image_id,
        requested_k=requested_k,
    )

    product_order = _unique_product_order(result.candidates)
    is_positive_product = [group_id == query_group_id for group_id in product_order]
    positive_product_count = sum(1 for flag in is_positive_product if flag)
    short_ranking = len(result.candidates) < requested_k

    if gallery_positive_products <= 0 and gallery_positive_images <= 0:
        return QueryEvaluationRecord(
            query_id=query_id,
            query_group_id=query_group_id,
            split=split,
            gallery_split=gallery_split,
            first_positive_rank=None,
            top1_hit=0,
            top5_hit=0,
            top10_hit=0,
            reciprocal_rank=0.0,
            number_of_candidates=len(result.candidates),
            number_of_positive_candidates=0,
            gallery_positive_images=gallery_positive_images,
            gallery_positive_products=gallery_positive_products,
            unique_products_in_ranking=len(product_order),
            requested_k=requested_k,
            short_ranking=short_ranking,
            excluded=True,
            exclusion_reason="no_positive_in_gallery",
            category=category,
            self_match_excluded=self_match_excluded,
        )

    first_rank = metrics.first_positive_rank(is_positive_product)
    return QueryEvaluationRecord(
        query_id=query_id,
        query_group_id=query_group_id,
        split=split,
        gallery_split=gallery_split,
        first_positive_rank=first_rank,
        top1_hit=metrics.hit_at_k(is_positive_product, 1),
        top5_hit=metrics.hit_at_k(is_positive_product, 5),
        top10_hit=metrics.hit_at_k(is_positive_product, 10),
        reciprocal_rank=metrics.reciprocal_rank(is_positive_product),
        number_of_candidates=len(result.candidates),
        number_of_positive_candidates=positive_product_count,
        gallery_positive_images=gallery_positive_images,
        gallery_positive_products=gallery_positive_products,
        unique_products_in_ranking=len(product_order),
        requested_k=requested_k,
        short_ranking=short_ranking,
        excluded=False,
        exclusion_reason=None,
        category=category,
        self_match_excluded=self_match_excluded,
    )


def evaluate_retrieval(
    records: Sequence[QueryEvaluationRecord],
) -> RetrievalMetricResult:
    """Aggregate per-query records into dataset-level Top-K / MRR."""
    if not records:
        raise EvaluationError("evaluate_retrieval requires at least one query record.")

    valid = [record for record in records if not record.excluded]
    short_rankings = sum(1 for record in valid if record.short_ranking)
    ranks = [
        record.first_positive_rank
        for record in valid
        if record.first_positive_rank is not None
    ]
    zero_positive_queries = sum(1 for record in records if record.excluded)
    with_positive = len(valid)
    missed_in_k = sum(1 for record in valid if record.first_positive_rank is None)

    if valid:
        top1 = sum(record.top1_hit for record in valid) / len(valid)
        top5 = sum(record.top5_hit for record in valid) / len(valid)
        top10 = sum(record.top10_hit for record in valid) / len(valid)
        mrr = metrics.mean_reciprocal_rank([record.reciprocal_rank for record in valid])
    else:
        top1 = top5 = top10 = mrr = 0.0

    diagnostics = {
        "number_of_queries": len(records),
        "queries_with_at_least_one_positive": with_positive,
        "queries_with_zero_positives": zero_positive_queries,
        "queries_with_positive_outside_topk": missed_in_k,
        "short_ranking_queries": short_rankings,
        "mean_first_positive_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "median_first_positive_rank": statistics.median(ranks) if ranks else None,
        "min_first_positive_rank": min(ranks) if ranks else None,
        "max_first_positive_rank": max(ranks) if ranks else None,
        "self_match_exclusions": sum(1 for record in records if record.self_match_excluded),
        "invalid_query_count": 0,
        "invalid_candidate_count": 0,
        "nan_inf_count": 0,
        "hit_cutoffs": list(HIT_CUTOFFS),
    }
    return RetrievalMetricResult(
        top1=top1,
        top5=top5,
        top10=top10,
        mrr=mrr,
        num_queries=len(records),
        num_valid_queries=len(valid),
        num_excluded_queries=len(records) - len(valid),
        query_records=tuple(records),
        diagnostics=diagnostics,
    )


def _gallery_group_index(metadata: Sequence[Mapping[str, object]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        group_id = str(item["product_group"])
        groups[group_id].append(index)
    return groups


def evaluate_leave_one_out_gallery(
    gallery: Gallery,
    *,
    k: int = EVALUATION_K,
    query_split: str = "train",
    gallery_split: str = "train",
) -> RetrievalMetricResult:
    """Evaluate every gallery image as a query against the same gallery.

    The query embedding is the already-extracted S2.7 gallery embedding.
    ``exclude_image_id`` removes the query image; other images from the same
    ``product_group`` remain valid positives.
    """
    assert_same_split_protocol(query_split, gallery_split)
    assert_gallery_metadata_aligned(gallery.metadata, gallery.size)
    if k != EVALUATION_K:
        raise EvaluationError(
            f"S2.8 derives Top-1/Top-5/Top-10/MRR from a single retrieval with k={EVALUATION_K}."
        )

    splits = {str(item.get("split", query_split)) for item in gallery.metadata}
    if splits - {query_split, ""}:
        unexpected = sorted(split for split in splits if split and split != query_split)
        if unexpected:
            raise EvaluationError(
                f"Gallery contains unexpected splits {unexpected}; "
                f"S2.8 baseline gallery/query split is {query_split!r}."
            )

    retriever = TopKRetriever(gallery.embeddings, gallery.metadata)
    group_index = _gallery_group_index(gallery.metadata)
    records: list[QueryEvaluationRecord] = []

    for index, item in enumerate(gallery.metadata):
        query_image_id = str(item["image_id"])
        query_group_id = str(item["product_group"])
        category = str(item.get("category") or "")
        member_indices = group_index[query_group_id]
        gallery_positive_images = len(member_indices) - 1
        gallery_positive_products = 1 if gallery_positive_images > 0 else 0

        result = retriever.retrieve(
            gallery.embeddings[index],
            query_id=query_image_id,
            k=k,
            exclude_image_id=query_image_id,
        )
        records.append(
            score_retrieval_result(
                result,
                query_group_id=query_group_id,
                query_image_id=query_image_id,
                split=query_split,
                gallery_split=gallery_split,
                requested_k=k,
                gallery_positive_images=gallery_positive_images,
                gallery_positive_products=gallery_positive_products,
                category=category or None,
                self_match_excluded=True,
            )
        )

    aggregated = evaluate_retrieval(records)
    category_rows = _per_category_metrics(aggregated.query_records)
    diagnostics = dict(aggregated.diagnostics)
    diagnostics["per_category"] = category_rows
    diagnostics["gallery_size"] = gallery.size
    diagnostics["gallery_groups"] = len(group_index)
    diagnostics["evaluated_groups"] = len(
        {record.query_group_id for record in aggregated.query_records if not record.excluded}
    )
    return RetrievalMetricResult(
        top1=aggregated.top1,
        top5=aggregated.top5,
        top10=aggregated.top10,
        mrr=aggregated.mrr,
        num_queries=aggregated.num_queries,
        num_valid_queries=aggregated.num_valid_queries,
        num_excluded_queries=aggregated.num_excluded_queries,
        query_records=aggregated.query_records,
        diagnostics=diagnostics,
    )


def _per_category_metrics(records: Sequence[QueryEvaluationRecord]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[QueryEvaluationRecord]] = defaultdict(list)
    for record in records:
        if record.excluded:
            continue
        grouped[record.category or "unknown"].append(record)
    payload: dict[str, dict[str, float | int]] = {}
    for category, rows in grouped.items():
        payload[category] = {
            "num_queries": len(rows),
            "top1": sum(row.top1_hit for row in rows) / len(rows),
            "top5": sum(row.top5_hit for row in rows) / len(rows),
            "top10": sum(row.top10_hit for row in rows) / len(rows),
            "mrr": metrics.mean_reciprocal_rank([row.reciprocal_rank for row in rows]),
        }
    return payload


def ranking_from_groups(product_groups: Iterable[str], query_group_id: str) -> list[bool]:
    """Test helper: map an ordered product list to Positive/Negative flags."""
    query_group_id = _require_non_empty_id(query_group_id, "query_group_id")
    flags: list[bool] = []
    seen: set[str] = set()
    for group_id in product_groups:
        group_id = _require_non_empty_id(str(group_id), "candidate.product_group")
        if group_id in seen:
            continue
        seen.add(group_id)
        flags.append(group_id == query_group_id)
    return flags
