"""S3.11 retrieval evaluation for the metric-learning product search path."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.evaluation.errors import EvaluationError
from src.retrieval.gallery import Gallery
from src.retrieval.ranking import ProductRanker
from src.retrieval.search import SimilaritySearchEngine
from src.retrieval.search_result import ProductSearchResult


EVALUATION_K = 10
EVALUATION_CUTOFFS = (1, 5, 10)


@dataclass(frozen=True)
class RetrievalQueryRecord:
    """Product-level retrieval outcome for one leave-one-image-out query."""

    query_image_id: str
    query_product_id: str
    category: str | None
    ranked_product_ids: tuple[str, ...]
    first_positive_rank: int | None
    top1_hit: int
    top5_hit: int
    top10_hit: int
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    gallery_positive_images: int
    excluded: bool
    exclusion_reason: str | None


@dataclass(frozen=True)
class MetricLearningRetrievalResult:
    """Aggregated S3.11 retrieval metrics and per-query diagnostics."""

    top1: float
    top5: float
    top10: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    num_queries: int
    num_valid_queries: int
    num_excluded_queries: int
    query_records: tuple[RetrievalQueryRecord, ...]
    diagnostics: Mapping[str, Any]


def _product_id(item: Mapping[str, Any]) -> str:
    value = item.get("product_id", item.get("product_group"))
    if value is None or not str(value).strip():
        raise EvaluationError("Gallery metadata contains an empty product ID.")
    return str(value)


def _image_id(item: Mapping[str, Any]) -> str:
    value = item.get("image_id", item.get("image"))
    if value is None or not str(value).strip():
        raise EvaluationError("Gallery metadata contains an empty image ID.")
    return str(value)


def _validate_gallery(gallery: Gallery) -> None:
    if gallery.size == 0:
        raise EvaluationError("S3.11 requires a non-empty gallery.")
    seen_images: set[str] = set()
    for index, item in enumerate(gallery.metadata):
        image_id = _image_id(item)
        if image_id in seen_images:
            raise EvaluationError(f"Duplicate gallery image_id at row {index}: {image_id!r}.")
        seen_images.add(image_id)
        if not math.isfinite(float(gallery.embeddings[index].norm().item())):
            raise EvaluationError(f"Non-finite embedding at gallery row {index}.")


def _hit_at_k(ranked_product_ids: Sequence[str], query_product_id: str, k: int) -> int:
    return int(query_product_id in ranked_product_ids[:k])


def _first_positive_rank(ranked_product_ids: Sequence[str], query_product_id: str) -> int | None:
    for rank, product_id in enumerate(ranked_product_ids, start=1):
        if product_id == query_product_id:
            return rank
    return None


def _score_query(
    *,
    query_image_id: str,
    query_product_id: str,
    category: str | None,
    result: ProductSearchResult,
    gallery_positive_images: int,
) -> RetrievalQueryRecord:
    if gallery_positive_images <= 0:
        return RetrievalQueryRecord(
            query_image_id=query_image_id,
            query_product_id=query_product_id,
            category=category,
            ranked_product_ids=tuple(),
            first_positive_rank=None,
            top1_hit=0,
            top5_hit=0,
            top10_hit=0,
            precision_at_1=0.0,
            precision_at_5=0.0,
            precision_at_10=0.0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            reciprocal_rank=0.0,
            gallery_positive_images=0,
            excluded=True,
            exclusion_reason="no_same_product_gallery_image",
        )

    ranked = ProductRanker().rank(result)
    ranked_product_ids = tuple(candidate.product_id for candidate in ranked.candidates)
    first_rank = _first_positive_rank(ranked_product_ids, query_product_id)
    hits = {k: _hit_at_k(ranked_product_ids, query_product_id, k) for k in EVALUATION_CUTOFFS}
    # There is exactly one relevant product for a product-identity query:
    # the query's own product. Therefore Recall@K is 1 iff that product is
    # retrieved, while Precision@K is hits divided by K.
    return RetrievalQueryRecord(
        query_image_id=query_image_id,
        query_product_id=query_product_id,
        category=category,
        ranked_product_ids=ranked_product_ids,
        first_positive_rank=first_rank,
        top1_hit=hits[1],
        top5_hit=hits[5],
        top10_hit=hits[10],
        precision_at_1=hits[1] / 1.0,
        precision_at_5=hits[5] / 5.0,
        precision_at_10=hits[10] / 10.0,
        recall_at_1=float(hits[1]),
        recall_at_5=float(hits[5]),
        recall_at_10=float(hits[10]),
        reciprocal_rank=(1.0 / first_rank) if first_rank is not None else 0.0,
        gallery_positive_images=gallery_positive_images,
        excluded=False,
        exclusion_reason=None,
    )


def _per_category(records: Sequence[RetrievalQueryRecord]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[RetrievalQueryRecord]] = {}
    for record in records:
        if record.excluded:
            continue
        grouped.setdefault(record.category or "unknown", []).append(record)
    output: dict[str, dict[str, float | int]] = {}
    for category, rows in sorted(grouped.items()):
        output[category] = {
            "num_queries": len(rows),
            "top1": sum(row.top1_hit for row in rows) / len(rows),
            "top5": sum(row.top5_hit for row in rows) / len(rows),
            "top10": sum(row.top10_hit for row in rows) / len(rows),
            "precision_at_1": sum(row.precision_at_1 for row in rows) / len(rows),
            "precision_at_5": sum(row.precision_at_5 for row in rows) / len(rows),
            "precision_at_10": sum(row.precision_at_10 for row in rows) / len(rows),
            "recall_at_1": sum(row.recall_at_1 for row in rows) / len(rows),
            "recall_at_5": sum(row.recall_at_5 for row in rows) / len(rows),
            "recall_at_10": sum(row.recall_at_10 for row in rows) / len(rows),
            "mrr": sum(row.reciprocal_rank for row in rows) / len(rows),
        }
    return output


def evaluate_gallery(gallery: Gallery, *, k: int = EVALUATION_K) -> MetricLearningRetrievalResult:
    """Evaluate a metric-learning gallery using leave-one-image-out product retrieval.

    The gallery embeddings are reused as query embeddings, exactly as in S2.8,
    but the retrieval path is the S3.8 product search followed by S3.9 ranking.
    The query image itself is excluded. Products with no other gallery image
    are excluded from metric aggregation because their ground-truth product
    cannot be retrieved from the gallery.
    """
    if k != EVALUATION_K:
        raise EvaluationError(
            f"S3.11 requires k={EVALUATION_K} so Top-1/5/10 and MRR share one ranking."
        )
    _validate_gallery(gallery)

    group_indices: dict[str, list[int]] = {}
    for index, item in enumerate(gallery.metadata):
        group_indices.setdefault(_product_id(item), []).append(index)

    search_engine = SimilaritySearchEngine(gallery.embeddings, gallery.metadata)
    records: list[RetrievalQueryRecord] = []
    for index, item in enumerate(gallery.metadata):
        image_id = _image_id(item)
        product_id = _product_id(item)
        category = str(item.get("category") or "") or None
        positive_images = len(group_indices[product_id]) - 1
        result = search_engine.search(
            gallery.embeddings[index],
            query_id=image_id,
            k=k,
            exclude_image_id=image_id,
        )
        records.append(
            _score_query(
                query_image_id=image_id,
                query_product_id=product_id,
                category=category,
                result=result,
                gallery_positive_images=positive_images,
            )
        )

    valid = tuple(record for record in records if not record.excluded)
    if not valid:
        raise EvaluationError("S3.11 has no valid queries with a same-product gallery image.")

    ranks = [record.first_positive_rank for record in valid if record.first_positive_rank is not None]
    diagnostics = {
        "queries_with_at_least_one_positive": len(valid),
        "queries_with_zero_positives": len(records) - len(valid),
        "positive_outside_top10": sum(record.top10_hit == 0 for record in valid),
        "mean_first_positive_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "median_first_positive_rank": statistics.median(ranks) if ranks else None,
        "min_first_positive_rank": min(ranks) if ranks else None,
        "max_first_positive_rank": max(ranks) if ranks else None,
        "self_match_exclusions": len(records),
        "short_rankings": sum(len(record.ranked_product_ids) < k for record in valid),
        "nan_inf_count": 0,
        "per_category": _per_category(valid),
    }
    return MetricLearningRetrievalResult(
        top1=sum(record.top1_hit for record in valid) / len(valid),
        top5=sum(record.top5_hit for record in valid) / len(valid),
        top10=sum(record.top10_hit for record in valid) / len(valid),
        precision_at_1=sum(record.precision_at_1 for record in valid) / len(valid),
        precision_at_5=sum(record.precision_at_5 for record in valid) / len(valid),
        precision_at_10=sum(record.precision_at_10 for record in valid) / len(valid),
        recall_at_1=sum(record.recall_at_1 for record in valid) / len(valid),
        recall_at_5=sum(record.recall_at_5 for record in valid) / len(valid),
        recall_at_10=sum(record.recall_at_10 for record in valid) / len(valid),
        mrr=sum(record.reciprocal_rank for record in valid) / len(valid),
        num_queries=len(records),
        num_valid_queries=len(valid),
        num_excluded_queries=len(records) - len(valid),
        query_records=tuple(records),
        diagnostics=diagnostics,
    )


def result_to_dict(result: MetricLearningRetrievalResult) -> dict[str, Any]:
    """Serialize the S3.11 aggregate without exposing the full query records."""
    return {
        "metrics": {
            "top1": result.top1,
            "top5": result.top5,
            "top10": result.top10,
            "precision_at_1": result.precision_at_1,
            "precision_at_5": result.precision_at_5,
            "precision_at_10": result.precision_at_10,
            "recall_at_1": result.recall_at_1,
            "recall_at_5": result.recall_at_5,
            "recall_at_10": result.recall_at_10,
            "mrr": result.mrr,
        },
        "query_counts": {
            "total": result.num_queries,
            "valid": result.num_valid_queries,
            "excluded": result.num_excluded_queries,
        },
        "diagnostics": dict(result.diagnostics),
    }
