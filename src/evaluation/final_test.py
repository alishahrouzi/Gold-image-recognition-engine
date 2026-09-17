"""Final unseen-image evaluation and robustness protocol (S4.9).

This module intentionally evaluates a frozen checkpoint against the original
Dataset 1 train gallery. Validation and test images are never added to the
gallery. Because Dataset 1 validation/test product groups are singleton groups,
exact product retrieval is not treated as the primary final-test metric.
Instead, this protocol measures category-aware retrieval generalization and
exports ranked candidates for optional human visual-relevance annotation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from PIL import Image, ImageEnhance

from data.loaders.image_loader import load_rgb_image
from data.preprocessing import ImagePreprocessor
from data.types import Sample
from retrieval.embedding import EmbeddingExtractor
from retrieval.gallery import Gallery
from retrieval.ranking import ProductRanker
from retrieval.search import SimilaritySearchEngine

S4_9_POLICY = "s4.9-final-unseen-image-evaluation-v1"
S4_9_K = 10
S4_9_TRANSFORMS = ("original", "rotation", "brightness", "center_crop", "resize")


@dataclass(frozen=True)
class FinalEvaluationResult:
    """Serializable final-test result for one frozen model checkpoint."""

    model: str
    split: str
    metrics: dict[str, float]
    query_counts: dict[str, int]
    diagnostics: dict[str, Any]
    query_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": S4_9_POLICY,
            "model": self.model,
            "split": self.split,
            "metrics": self.metrics,
            "query_counts": self.query_counts,
            "diagnostics": self.diagnostics,
            "query_records": self.query_records,
        }


def _metadata_product_id(item: Mapping[str, Any]) -> str:
    product_id = str(item.get("product_id", item.get("product_group", ""))).strip()
    if not product_id:
        raise ValueError("Gallery metadata contains an empty product ID.")
    return product_id


def _metadata_image_id(item: Mapping[str, Any]) -> str:
    image_id = str(item.get("image_id", item.get("image", ""))).strip()
    if not image_id:
        raise ValueError("Gallery metadata contains an empty image ID.")
    return image_id


def _metadata_category(item: Mapping[str, Any]) -> str:
    return str(item.get("category", "")).strip()


def _rank_category_metrics(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not rows:
        raise ValueError("Cannot evaluate an empty query set.")

    def score(items: Sequence[dict[str, Any]]) -> dict[str, float]:
        first_category_rank = [r["first_category_rank"] for r in items if r["first_category_rank"] is not None]
        return {
            "top1": sum(r["top1_category_hit"] for r in items) / len(items),
            "top5": sum(r["top5_category_hit"] for r in items) / len(items),
            "top10": sum(r["top10_category_hit"] for r in items) / len(items),
            "mrr": sum(r["category_reciprocal_rank"] for r in items) / len(items),
            "mean_first_category_rank": (
                sum(first_category_rank) / len(first_category_rank)
                if first_category_rank else float("nan")
            ),
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return score(rows), {category: score(items) for category, items in sorted(grouped.items())}


def evaluate_split(
    extractor: EmbeddingExtractor,
    gallery: Gallery,
    queries: Sequence[Sample],
    *,
    model: str,
    split: str,
    k: int = S4_9_K,
    preprocessor: ImagePreprocessor | None = None,
) -> FinalEvaluationResult:
    """Evaluate unseen images against a frozen train gallery.

    The exact product ID is deliberately not required to occur in the gallery.
    The primary ground truth is the query's known category. Ranked product IDs
    and categories are retained so a later human-relevance pass can assess
    actual visual similarity without rerunning inference.
    """
    if k != S4_9_K:
        raise ValueError(f"S4.9 requires k={S4_9_K} for a shared ranking.")
    if not queries:
        raise ValueError("S4.9 requires at least one query.")
    preprocessor = preprocessor or ImagePreprocessor()
    engine = SimilaritySearchEngine(gallery.embeddings, gallery.metadata)
    ranker = ProductRanker()
    rows: list[dict[str, Any]] = []

    for sample in queries:
        image = load_rgb_image(sample.image_path)
        tensor = preprocessor(image)
        embedding = extractor.extract_one(tensor)
        ranked = ranker.rank(engine.search(embedding, k=k))
        candidates = [
            {
                "rank": int(candidate.rank),
                "product_id": candidate.product_id,
                "category": candidate.category,
                "similarity": float(candidate.similarity),
            }
            for candidate in ranked.candidates
        ]
        category = sample.category
        category_ranks = [index for index, candidate in enumerate(candidates, 1) if candidate["category"] == category]
        first_category_rank = category_ranks[0] if category_ranks else None
        rows.append(
            {
                "query_image_id": sample.image_id,
                "query_product_id": sample.group_id,
                "category": category,
                "top1_category_hit": int(bool(category_ranks and category_ranks[0] <= 1)),
                "top5_category_hit": int(bool(category_ranks and category_ranks[0] <= 5)),
                "top10_category_hit": int(bool(category_ranks and category_ranks[0] <= 10)),
                "first_category_rank": first_category_rank,
                "category_reciprocal_rank": 1.0 / first_category_rank if first_category_rank else 0.0,
                "ranked_candidates": candidates,
            }
        )

    metrics, per_category = _rank_category_metrics(rows)
    finite_mean = metrics["mean_first_category_rank"]
    if not math.isfinite(finite_mean):
        metrics["mean_first_category_rank"] = None
    return FinalEvaluationResult(
        model=model,
        split=split,
        metrics=metrics,
        query_counts={"total": len(rows), "valid": len(rows), "excluded": 0},
        diagnostics={
            "gallery_size": gallery.size,
            "gallery_product_groups": len({_metadata_product_id(item) for item in gallery.metadata}),
            "query_product_groups_in_gallery": len(
                {_metadata_product_id(item) for item in gallery.metadata}
                & {sample.group_id for sample in queries}
            ),
            "unseen_query_product_groups": len({sample.group_id for sample in queries}),
            "exact_product_ground_truth_available": False,
            "per_category": per_category,
        },
        query_records=rows,
    )


def make_robustness_variants(image: Image.Image) -> dict[str, Image.Image]:
    """Create deterministic, mild query perturbations for robustness testing."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    crop_w = max(32, int(round(width * 0.90)))
    crop_h = max(32, int(round(height * 0.90)))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return {
        "original": rgb.copy(),
        "rotation": rgb.rotate(12, resample=Image.Resampling.BILINEAR, expand=False),
        "brightness": ImageEnhance.Brightness(rgb).enhance(1.20),
        "center_crop": rgb.crop((left, top, left + crop_w, top + crop_h)),
        "resize": rgb.resize((max(32, width // 2), max(32, height // 2)), Image.Resampling.BILINEAR),
    }


def evaluate_robustness(
    extractor: EmbeddingExtractor,
    gallery: Gallery,
    queries: Sequence[Sample],
    *,
    model: str,
    max_queries: int | None = None,
    k: int = S4_9_K,
    preprocessor: ImagePreprocessor | None = None,
) -> dict[str, Any]:
    """Measure category retrieval under deterministic image perturbations."""
    if k != S4_9_K:
        raise ValueError(f"S4.9 requires k={S4_9_K} for robustness evaluation.")
    if not queries:
        raise ValueError("Robustness evaluation requires at least one query.")
    preprocessor = preprocessor or ImagePreprocessor()
    selected = list(queries[:max_queries] if max_queries else queries)
    engine = SimilaritySearchEngine(gallery.embeddings, gallery.metadata)
    ranker = ProductRanker()
    transform_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sample in selected:
        image = load_rgb_image(sample.image_path)
        for transform_name, variant in make_robustness_variants(image).items():
            embedding = extractor.extract_one(preprocessor(variant))
            ranked = ranker.rank(engine.search(embedding, k=k))
            categories = [candidate.category for candidate in ranked.candidates]
            first_rank = next((i for i, category in enumerate(categories, 1) if category == sample.category), None)
            transform_rows[transform_name].append(
                {
                    "query_image_id": sample.image_id,
                    "category": sample.category,
                    "top1_category_hit": int(bool(first_rank == 1)),
                    "top5_category_hit": int(bool(first_rank and first_rank <= 5)),
                    "top10_category_hit": int(bool(first_rank and first_rank <= 10)),
                    "first_category_rank": first_rank,
                    "category_reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                    "top1_product_id": ranked.candidates[0].product_id if ranked.candidates else None,
                }
            )

    aggregate: dict[str, Any] = {}
    for transform, rows in transform_rows.items():
        aggregate[transform] = {
            "num_queries": len(rows),
            "top1": sum(r["top1_category_hit"] for r in rows) / len(rows),
            "top5": sum(r["top5_category_hit"] for r in rows) / len(rows),
            "top10": sum(r["top10_category_hit"] for r in rows) / len(rows),
            "mrr": sum(r["category_reciprocal_rank"] for r in rows) / len(rows),
            "mean_first_category_rank": (
                sum(r["first_category_rank"] for r in rows if r["first_category_rank"] is not None)
                / max(1, sum(r["first_category_rank"] is not None for r in rows))
            ),
        }
    return {
        "policy": S4_9_POLICY,
        "model": model,
        "num_queries": len(selected),
        "transforms": list(S4_9_TRANSFORMS),
        "results": aggregate,
        "query_records": dict(transform_rows),
    }


def compare_models(results: Mapping[str, FinalEvaluationResult]) -> dict[str, Any]:
    """Compare frozen models on untouched test metrics without hidden tuning."""
    if not results:
        raise ValueError("At least one model result is required.")
    for name, result in results.items():
        if result.split != "test":
            raise ValueError(f"Final model selection must use test results; {name!r} is {result.split!r}.")
        for metric in ("top1", "top5", "top10", "mrr"):
            value = result.metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid test metric {metric!r} for model {name!r}.")
    selected = max(
        results,
        key=lambda name: (
            float(results[name].metrics["top1"]),
            float(results[name].metrics["top5"]),
            float(results[name].metrics["mrr"]),
        ),
    )
    return {
        "policy": S4_9_POLICY,
        "selected_model": selected,
        "selection_priority": ["test_top1_category", "test_top5_category", "test_mrr_category"],
        "selection_basis": "unseen-image category-aware retrieval against train gallery",
        "human_visual_relevance_required_for_exact_visual_search_claim": True,
        "models": {
            name: {"metrics": result.metrics, "query_counts": result.query_counts, "diagnostics": result.diagnostics}
            for name, result in results.items()
        },
    }
