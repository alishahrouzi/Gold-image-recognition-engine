"""Final unseen-image evaluation and robustness protocol (S4.9)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image, ImageEnhance

from data.loaders.image_loader import load_rgb_image
from data.preprocessing import ImagePreprocessor
from data.types import Sample
from retrieval.embedding import EmbeddingExtractor
from retrieval.gallery import Gallery
from retrieval.ranking import ProductRanker
from retrieval.search import SimilaritySearchEngine

S4_9_POLICY = "s4.9-final-unseen-image-evaluation-v2"
S4_9_K = 10
S4_9_TRANSFORMS = ("original", "rotation", "brightness", "center_crop", "resize")


@dataclass(frozen=True)
class FinalEvaluationResult:
    """Serializable final-evaluation result for one frozen checkpoint."""

    model: str
    split: str
    metrics: dict[str, float | None]
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


def _product_id(item: Mapping[str, Any]) -> str:
    value = str(item.get("product_id", item.get("product_group", ""))).strip()
    if not value:
        raise ValueError("Gallery metadata contains an empty product ID.")
    return value


def _category_metrics(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    if not rows:
        raise ValueError("Cannot evaluate an empty query set.")

    def score(items: Sequence[dict[str, Any]]) -> dict[str, float | None]:
        ranks = [r["first_category_rank"] for r in items if r["first_category_rank"] is not None]
        return {
            "top1": sum(r["top1_category_hit"] for r in items) / len(items),
            "top5": sum(r["top5_category_hit"] for r in items) / len(items),
            "top10": sum(r["top10_category_hit"] for r in items) / len(items),
            "mrr": sum(r["category_reciprocal_rank"] for r in items) / len(items),
            "mean_first_category_rank": sum(ranks) / len(ranks) if ranks else None,
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

    Dataset 1 valid/test groups are singleton groups, so the query's exact
    product ID is not expected in the train gallery. Automatic ground truth is
    the manifest category. Ranked candidates are retained for human relevance.
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
        embedding = extractor.extract_one(preprocessor(load_rgb_image(sample.image_path)))
        ranked = ranker.rank(engine.search(embedding, k=k))
        candidates = [
            {
                "rank": int(candidate.rank),
                "product_id": candidate.product_id,
                "category": candidate.category,
                "similarity": float(candidate.similarity),
                "matched_image_ids": list(candidate.matched_image_ids),
                "matched_image_paths": list(candidate.matched_image_paths),
            }
            for candidate in ranked.candidates
        ]
        first_category_rank = next((i for i, c in enumerate(candidates, 1) if c["category"] == sample.category), None)
        rows.append(
            {
                "query_image_id": sample.image_id,
                "query_product_id": sample.group_id,
                "category": sample.category,
                "top1_category_hit": int(first_category_rank == 1),
                "top5_category_hit": int(bool(first_category_rank and first_category_rank <= 5)),
                "top10_category_hit": int(bool(first_category_rank and first_category_rank <= 10)),
                "first_category_rank": first_category_rank,
                "category_reciprocal_rank": 1.0 / first_category_rank if first_category_rank else 0.0,
                "ranked_candidates": candidates,
            }
        )

    metrics, per_category = _category_metrics(rows)
    return FinalEvaluationResult(
        model=model,
        split=split,
        metrics=metrics,
        query_counts={"total": len(rows), "valid": len(rows), "excluded": 0},
        diagnostics={
            "gallery_size": gallery.size,
            "gallery_product_groups": len({_product_id(item) for item in gallery.metadata}),
            "query_product_groups_in_gallery": 0,
            "unseen_query_product_groups": len({sample.group_id for sample in queries}),
            "exact_product_ground_truth_available": False,
            "per_category": per_category,
        },
        query_records=rows,
    )


def make_robustness_variants(image: Image.Image) -> dict[str, Image.Image]:
    """Create deterministic, mild query perturbations."""
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
    """Measure category retrieval under deterministic perturbations."""
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
                    "top1_category_hit": int(first_rank == 1),
                    "top5_category_hit": int(bool(first_rank and first_rank <= 5)),
                    "top10_category_hit": int(bool(first_rank and first_rank <= 10)),
                    "first_category_rank": first_rank,
                    "category_reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                    "top1_product_id": ranked.candidates[0].product_id if ranked.candidates else None,
                }
            )

    aggregate: dict[str, Any] = {}
    for transform, rows in transform_rows.items():
        ranks = [r["first_category_rank"] for r in rows if r["first_category_rank"] is not None]
        aggregate[transform] = {
            "num_queries": len(rows),
            "top1": sum(r["top1_category_hit"] for r in rows) / len(rows),
            "top5": sum(r["top5_category_hit"] for r in rows) / len(rows),
            "top10": sum(r["top10_category_hit"] for r in rows) / len(rows),
            "mrr": sum(r["category_reciprocal_rank"] for r in rows) / len(rows),
            "mean_first_category_rank": sum(ranks) / len(ranks) if ranks else None,
        }
    return {
        "policy": S4_9_POLICY,
        "model": model,
        "num_queries": len(selected),
        "transforms": list(S4_9_TRANSFORMS),
        "results": aggregate,
        "query_records": dict(transform_rows),
    }


def select_model_from_validation(results: Mapping[str, FinalEvaluationResult]) -> str:
    """Select a final candidate using validation only; never use test metrics."""
    if not results:
        raise ValueError("At least one validation result is required.")
    for name, result in results.items():
        if result.split != "valid":
            raise ValueError(f"Model selection requires valid results; {name!r} is {result.split!r}.")
        for metric in ("top1", "top5", "mrr"):
            value = result.metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid validation metric {metric!r} for model {name!r}.")
    return max(
        results,
        key=lambda name: (
            float(results[name].metrics["top1"]),
            float(results[name].metrics["top5"]),
            float(results[name].metrics["mrr"]),
        ),
    )


def build_model_selection_report(
    validation_results: Mapping[str, FinalEvaluationResult],
    test_results: Mapping[str, FinalEvaluationResult],
    *,
    selected_model: str,
) -> dict[str, Any]:
    """Build an auditable report: validation selects; test confirms."""
    if selected_model not in validation_results or selected_model not in test_results:
        raise ValueError("Selected model must exist in both validation and test results.")
    if any(result.split != "valid" for result in validation_results.values()):
        raise ValueError("All validation results must have split='valid'.")
    if any(result.split != "test" for result in test_results.values()):
        raise ValueError("All test results must have split='test'.")
    return {
        "policy": S4_9_POLICY,
        "selected_model": selected_model,
        "selection_priority": ["valid_top1_category", "valid_top5_category", "valid_mrr_category"],
        "selection_basis": "development validation only; test is confirmation",
        "test_used_for_model_selection": False,
        "human_visual_relevance_required_for_exact_visual_search_claim": True,
        "validation_models": {
            name: {"metrics": result.metrics, "query_counts": result.query_counts, "diagnostics": result.diagnostics}
            for name, result in validation_results.items()
        },
        "test_models": {
            name: {"metrics": result.metrics, "query_counts": result.query_counts, "diagnostics": result.diagnostics}
            for name, result in test_results.items()
        },
        "final_test_model": {
            "model": selected_model,
            "metrics": test_results[selected_model].metrics,
            "query_counts": test_results[selected_model].query_counts,
            "diagnostics": test_results[selected_model].diagnostics,
        },
    }
