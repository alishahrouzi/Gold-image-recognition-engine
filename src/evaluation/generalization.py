"""S3.13 group-disjoint generalization and model-selection evaluation.

This module defines the experiment protocol only. It deliberately does not
change the runtime inference architecture or the existing S3.11 protocol.

The gate holds out product groups that contain at least two images, trains a
candidate model only on the remaining Dataset 1 train groups, then evaluates
queries from the held-out groups against a gallery containing all train images.
The query product therefore remains completely unseen during training while
still being represented by other held-out images in the runtime-like gallery.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Mapping, Sequence

from sklearn.model_selection import StratifiedGroupKFold

from data.pairs.generator import generate_positive_pairs, write_pairs_csv
from data.pairs.strategies import (
    CROSS_CATEGORY_NEGATIVE,
    RANDOM_NEGATIVE,
    SAME_CATEGORY_NEGATIVE,
    sample_negatives_by_strategy,
)
from data.pairs.types import PAIR_TYPE_NEGATIVE, PAIR_TYPE_POSITIVE, Pair, sort_pairs
from data.pairs.validation import validate_pairs
from data.types import Sample
from retrieval.gallery import Gallery
from retrieval.ranking import ProductRanker
from retrieval.search import SimilaritySearchEngine


S3_13_POLICY = "s3.13-group-disjoint-generalization-v1"
S3_13_K = 10
S3_13_METRICS = ("top1", "top5", "top10", "mrr")
S3_13_MODELS = (
    "s3.4_hybrid",
    "s3.5_random",
    "s3.5_same_category",
    "s3.5_cross_category",
)


@dataclass(frozen=True)
class GroupHoldoutSplit:
    """Deterministic product-group-disjoint split for S3.13."""

    train_samples: tuple[Sample, ...]
    holdout_samples: tuple[Sample, ...]
    train_group_ids: tuple[str, ...]
    holdout_group_ids: tuple[str, ...]
    eligible_holdout_group_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_images": len(self.train_samples),
            "holdout_images": len(self.holdout_samples),
            "train_groups": len(self.train_group_ids),
            "holdout_groups": len(self.holdout_group_ids),
            "eligible_holdout_groups": len(self.eligible_holdout_group_ids),
            "train_group_ids": list(self.train_group_ids),
            "holdout_group_ids": list(self.holdout_group_ids),
            "category_counts_train": dict(
                sorted(Counter(sample.category for sample in self.train_samples).items())
            ),
            "category_counts_holdout": dict(
                sorted(Counter(sample.category for sample in self.holdout_samples).items())
            ),
        }


def build_group_holdout(
    samples: Sequence[Sample],
    *,
    holdout_fraction: float = 0.20,
    seed: int = 42,
) -> GroupHoldoutSplit:
    """Create a stratified, group-disjoint holdout from Dataset 1 train groups.

    Only groups with at least two images are eligible for holdout because an
    unseen-product retrieval query needs another image of the same product in
    the gallery. Singleton groups remain in training and are valid negative
    distractors, but never become S3.13 query products.
    """
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5).")
    train_samples = tuple(sample for sample in samples if sample.split == "train")
    if not train_samples:
        raise ValueError("S3.13 requires Dataset 1 train samples.")

    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in train_samples:
        by_group[sample.group_id].append(sample)

    eligible_groups = {
        group_id: members
        for group_id, members in by_group.items()
        if len(members) >= 2
    }
    if len(eligible_groups) < 5:
        raise ValueError("S3.13 requires at least five multi-image product groups.")

    group_ids = sorted(eligible_groups)
    group_labels = [eligible_groups[group_id][0].category for group_id in group_ids]
    # Five folds gives an approximately 20% held-out group set while keeping
    # categories approximately stratified. The seed controls fold assignment.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    indices = list(range(len(group_ids)))
    fold_train, fold_holdout = next(
        splitter.split(indices, group_labels, groups=group_ids)
    )
    holdout_count = max(1, round(len(group_ids) * holdout_fraction))
    selected_holdout_indices = set(fold_holdout)
    if len(selected_holdout_indices) != holdout_count:
        rng = Random(seed)
        ordered = sorted(selected_holdout_indices)
        rng.shuffle(ordered)
        if len(ordered) > holdout_count:
            selected_holdout_indices = set(ordered[:holdout_count])
        else:
            remaining = [i for i in indices if i not in selected_holdout_indices]
            rng.shuffle(remaining)
            selected_holdout_indices.update(remaining[: holdout_count - len(ordered)])

    holdout_group_ids = {
        group_ids[index] for index in selected_holdout_indices
    }
    holdout_samples = tuple(
        sample for sample in train_samples if sample.group_id in holdout_group_ids
    )
    train_only_samples = tuple(
        sample for sample in train_samples if sample.group_id not in holdout_group_ids
    )

    train_groups = {sample.group_id for sample in train_only_samples}
    actual_holdout_groups = {sample.group_id for sample in holdout_samples}
    if train_groups & actual_holdout_groups:
        raise AssertionError("S3.13 group split leakage detected.")
    if any(
        sum(sample.group_id == group_id for sample in holdout_samples) < 2
        for group_id in actual_holdout_groups
    ):
        raise AssertionError("S3.13 selected a holdout group without two images.")

    return GroupHoldoutSplit(
        train_samples=tuple(sorted(train_only_samples, key=lambda item: item.image_id)),
        holdout_samples=tuple(sorted(holdout_samples, key=lambda item: item.image_id)),
        train_group_ids=tuple(sorted(train_groups)),
        holdout_group_ids=tuple(sorted(actual_holdout_groups)),
        eligible_holdout_group_ids=tuple(sorted(eligible_groups)),
    )


def build_gate_pairs(
    train_samples: Sequence[Sample],
    *,
    model_name: str,
    seed: int = 2026,
    negative_ratio: float = 1.0,
) -> tuple[tuple[Pair, ...], dict[str, Any]]:
    """Generate S3.13 training pairs for one candidate strategy.

    The hybrid strategy reproduces the S3.4 50/50 same-vs-cross-category
    negative policy. The other strategies reuse the exact S3.5 samplers.
    """
    if model_name not in S3_13_MODELS:
        raise ValueError(f"Unsupported S3.13 model: {model_name!r}.")
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be >= 0.")

    samples = list(train_samples)
    positives = generate_positive_pairs(samples)
    negative_count = int(round(len(positives) * negative_ratio))

    if model_name == "s3.4_hybrid":
        rng = Random(seed)
        same_count = int(round(negative_count * 0.5))
        cross_count = negative_count - same_count
        from data.pairs.sampler import sample_negative_pairs, occupied_keys

        negatives = sample_negative_pairs(
            samples,
            rng,
            n_same_category=same_count,
            n_cross_category=cross_count,
            occupied=occupied_keys(positives),
        )
    else:
        from data.pairs.sampler import occupied_keys

        negatives = sample_negatives_by_strategy(
            samples,
            Random(seed),
            strategy={
                "s3.5_random": RANDOM_NEGATIVE,
                "s3.5_same_category": SAME_CATEGORY_NEGATIVE,
                "s3.5_cross_category": CROSS_CATEGORY_NEGATIVE,
            }[model_name],
            count=negative_count,
            occupied=occupied_keys(positives),
        )

    pairs = sort_pairs([*positives, *negatives], ("train", "valid", "test"))
    checks = validate_pairs(pairs, samples)
    if not all(checks.values()):
        raise ValueError(f"S3.13 pair validation failed: {checks}")

    negative_types = Counter(pair.negative_type for pair in negatives)
    report = {
        "model": model_name,
        "policy": S3_13_POLICY,
        "seed": seed,
        "positive_negative_ratio": negative_ratio,
        "total_pairs": len(pairs),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "negative_counts": {
            "random": negative_types.get("random", 0),
            "same_category": negative_types.get("same_category", 0),
            "cross_category": negative_types.get("cross_category", 0),
        },
        "training_images": len(samples),
        "training_groups": len({sample.group_id for sample in samples}),
        "validation_results": {"passed": all(checks.values()), "checks": dict(checks)},
    }
    return pairs, report


def evaluate_unseen_groups(
    gallery: Gallery,
    *,
    holdout_group_ids: set[str] | frozenset[str],
    k: int = S3_13_K,
) -> dict[str, Any]:
    """Evaluate only queries whose product group was unseen during training."""
    if k != S3_13_K:
        raise ValueError(f"S3.13 requires k={S3_13_K} for one shared ranking.")
    if not holdout_group_ids:
        raise ValueError("holdout_group_ids must not be empty.")
    if gallery.size == 0:
        raise ValueError("S3.13 requires a non-empty gallery.")
    if not math.isfinite(float(gallery.embeddings.float().norm(dim=1).max().item())):
        raise ValueError("Gallery contains non-finite embeddings.")

    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(gallery.metadata):
        product_id = str(item.get("product_id", item.get("product_group", "")))
        if not product_id:
            raise ValueError(f"Gallery row {index} has no product ID.")
        group_indices[product_id].append(index)

    missing = sorted(group_id for group_id in holdout_group_ids if group_id not in group_indices)
    if missing:
        raise ValueError(f"Holdout groups are missing from the evaluation gallery: {missing[:10]}")

    search_engine = SimilaritySearchEngine(gallery.embeddings, gallery.metadata)
    ranker = ProductRanker()
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(gallery.metadata):
        product_id = str(item.get("product_id", item.get("product_group", "")))
        if product_id not in holdout_group_ids:
            continue
        image_id = str(item.get("image_id", item.get("image", "")))
        category = str(item.get("category") or "") or None
        positive_images = len(group_indices[product_id]) - 1
        if positive_images <= 0:
            raise ValueError(
                f"Holdout query {image_id!r} has no same-product gallery image."
            )

        result = search_engine.search(
            gallery.embeddings[index],
            query_id=image_id,
            k=k,
            exclude_image_id=image_id,
        )
        ranked = ranker.rank(result)
        ranked_ids = [candidate.product_id for candidate in ranked.candidates]
        first_rank = next(
            (rank for rank, candidate_id in enumerate(ranked_ids, start=1) if candidate_id == product_id),
            None,
        )
        rows.append(
            {
                "query_image_id": image_id,
                "query_product_id": product_id,
                "category": category,
                "first_positive_rank": first_rank,
                "top1_hit": int(product_id in ranked_ids[:1]),
                "top5_hit": int(product_id in ranked_ids[:5]),
                "top10_hit": int(product_id in ranked_ids[:10]),
                "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                "gallery_positive_images": positive_images,
                "ranked_product_ids": ranked_ids,
            }
        )

    if not rows:
        raise ValueError("S3.13 produced no held-out queries.")

    ranks = [row["first_positive_rank"] for row in rows if row["first_positive_rank"] is not None]
    metrics = {
        "top1": sum(row["top1_hit"] for row in rows) / len(rows),
        "top5": sum(row["top5_hit"] for row in rows) / len(rows),
        "top10": sum(row["top10_hit"] for row in rows) / len(rows),
        "mrr": sum(row["reciprocal_rank"] for row in rows) / len(rows),
    }

    per_category: dict[str, dict[str, float | int]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"] or "unknown"].append(row)
    for category, category_rows in sorted(grouped.items()):
        per_category[category] = {
            "num_queries": len(category_rows),
            "top1": sum(row["top1_hit"] for row in category_rows) / len(category_rows),
            "top5": sum(row["top5_hit"] for row in category_rows) / len(category_rows),
            "top10": sum(row["top10_hit"] for row in category_rows) / len(category_rows),
            "mrr": sum(row["reciprocal_rank"] for row in category_rows) / len(category_rows),
        }

    return {
        "metrics": metrics,
        "query_counts": {
            "total": len(rows),
            "valid": len(rows),
            "excluded": 0,
        },
        "diagnostics": {
            "gallery_size": gallery.size,
            "gallery_product_groups": len(group_indices),
            "holdout_product_groups": len(holdout_group_ids),
            "queries_with_at_least_one_positive": len(rows),
            "positive_outside_top10": sum(row["top10_hit"] == 0 for row in rows),
            "mean_first_positive_rank": sum(ranks) / len(ranks) if ranks else None,
            "median_first_positive_rank": statistics.median(ranks) if ranks else None,
            "min_first_positive_rank": min(ranks) if ranks else None,
            "max_first_positive_rank": max(ranks) if ranks else None,
            "self_match_exclusions": len(rows),
            "nan_inf_count": 0,
            "per_category": per_category,
        },
        "query_records": rows,
    }


def select_model(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Select the candidate by Top-1, then Top-5, then MRR."""
    if not results:
        raise ValueError("S3.13 requires at least one model result.")
    for model_name, payload in results.items():
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Missing metrics for model {model_name!r}.")
        for metric in S3_13_METRICS:
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid {metric} for model {model_name!r}.")
    return max(
        results,
        key=lambda name: (
            float(results[name]["metrics"]["top1"]),
            float(results[name]["metrics"]["top5"]),
            float(results[name]["metrics"]["mrr"]),
        ),
    )


def write_pairs(pairs: Sequence[Pair], output_path: str) -> None:
    """Persist S3.13 pair CSV using the existing S1.10 writer."""
    write_pairs_csv(pairs, output_path)
