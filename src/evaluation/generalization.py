"""S3.13 group-disjoint generalization and model-selection evaluation."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Mapping, Sequence

from sklearn.model_selection import StratifiedGroupKFold

from data.pairs.generator import generate_positive_pairs, write_pairs_csv
from data.pairs.strategies import CROSS_CATEGORY_NEGATIVE, RANDOM_NEGATIVE, SAME_CATEGORY_NEGATIVE, sample_negatives_by_strategy
from data.pairs.types import PAIR_TYPE_NEGATIVE, PAIR_TYPE_POSITIVE, Pair, sort_pairs
from data.pairs.validation import validate_pairs
from data.types import Sample
from retrieval.gallery import Gallery
from retrieval.ranking import ProductRanker
from retrieval.search import SimilaritySearchEngine

S3_13_POLICY = "s3.13-group-disjoint-generalization-v1"
S3_13_K = 10
S3_13_METRICS = ("top1", "top5", "top10", "mrr")
S3_13_MODELS = ("s3.4_hybrid", "s3.5_random", "s3.5_same_category", "s3.5_cross_category")

@dataclass(frozen=True)
class GroupHoldoutSplit:
    train_samples: tuple[Sample, ...]
    holdout_samples: tuple[Sample, ...]
    train_group_ids: tuple[str, ...]
    holdout_group_ids: tuple[str, ...]
    eligible_holdout_group_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_images": len(self.train_samples), "holdout_images": len(self.holdout_samples),
            "train_groups": len(self.train_group_ids), "holdout_groups": len(self.holdout_group_ids),
            "eligible_holdout_groups": len(self.eligible_holdout_group_ids),
            "train_group_ids": list(self.train_group_ids), "holdout_group_ids": list(self.holdout_group_ids),
            "category_counts_train": dict(sorted(Counter(s.category for s in self.train_samples).items())),
            "category_counts_holdout": dict(sorted(Counter(s.category for s in self.holdout_samples).items())),
        }

def build_group_holdout(samples: Sequence[Sample], *, holdout_fraction: float = 0.20, seed: int = 42) -> GroupHoldoutSplit:
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5).")
    train_samples = tuple(s for s in samples if s.split == "train")
    if not train_samples:
        raise ValueError("S3.13 requires Dataset 1 train samples.")
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in train_samples:
        by_group[sample.group_id].append(sample)
    eligible = {gid: members for gid, members in by_group.items() if len(members) >= 2}
    if len(eligible) < 5:
        raise ValueError("S3.13 requires at least five multi-image product groups.")
    group_ids = sorted(eligible)
    labels = [eligible[gid][0].category for gid in group_ids]
    class_counts = Counter(labels)
    min_class_groups = min(class_counts.values())
    if min_class_groups < 2:
        raise ValueError("S3.13 requires at least two eligible product groups per category.")
    n_splits = min(5, min_class_groups, len(group_ids))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, holdout_indices = next(splitter.split(group_ids, labels, groups=group_ids))
    target_count = max(1, round(len(group_ids) * holdout_fraction))
    selected = list(holdout_indices)
    rng = Random(seed)
    rng.shuffle(selected)
    selected = selected[:target_count]
    if len(selected) < target_count:
        remaining = [i for i in range(len(group_ids)) if i not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:target_count - len(selected)])
    holdout_ids = {group_ids[i] for i in selected}
    holdout_samples = tuple(s for s in train_samples if s.group_id in holdout_ids)
    train_only = tuple(s for s in train_samples if s.group_id not in holdout_ids)
    train_groups = {s.group_id for s in train_only}
    actual_holdout = {s.group_id for s in holdout_samples}
    if train_groups & actual_holdout:
        raise AssertionError("S3.13 group split leakage detected.")
    if any(sum(s.group_id == gid for s in holdout_samples) < 2 for gid in actual_holdout):
        raise AssertionError("S3.13 selected a holdout group without two images.")
    return GroupHoldoutSplit(tuple(sorted(train_only, key=lambda s: s.image_id)), tuple(sorted(holdout_samples, key=lambda s: s.image_id)), tuple(sorted(train_groups)), tuple(sorted(actual_holdout)), tuple(sorted(eligible)))

def build_gate_pairs(train_samples: Sequence[Sample], *, model_name: str, seed: int = 2026, negative_ratio: float = 1.0) -> tuple[tuple[Pair, ...], dict[str, Any]]:
    if model_name not in S3_13_MODELS:
        raise ValueError(f"Unsupported S3.13 model: {model_name!r}.")
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be >= 0.")
    samples = list(train_samples)
    positives = generate_positive_pairs(samples)
    negative_count = int(round(len(positives) * negative_ratio))
    from data.pairs.sampler import occupied_keys, sample_negative_pairs
    occupied = occupied_keys(positives)
    if model_name == "s3.4_hybrid":
        same_count = int(round(negative_count * 0.5))
        negatives = sample_negative_pairs(samples, Random(seed), n_same_category=same_count, n_cross_category=negative_count - same_count, occupied=occupied)
    else:
        strategy = {"s3.5_random": RANDOM_NEGATIVE, "s3.5_same_category": SAME_CATEGORY_NEGATIVE, "s3.5_cross_category": CROSS_CATEGORY_NEGATIVE}[model_name]
        negatives = sample_negatives_by_strategy(samples, Random(seed), strategy=strategy, count=negative_count, occupied=occupied)
    pairs = sort_pairs([*positives, *negatives], ("train", "valid", "test"))
    checks = validate_pairs(pairs, samples)
    if not all(checks.values()):
        raise ValueError(f"S3.13 pair validation failed: {checks}")
    counts = Counter(pair.negative_type for pair in negatives)
    return pairs, {"model": model_name, "policy": S3_13_POLICY, "seed": seed, "positive_negative_ratio": negative_ratio, "total_pairs": len(pairs), "positive_pairs": len(positives), "negative_pairs": len(negatives), "negative_counts": {"random": counts.get("random", 0), "same_category": counts.get("same_category", 0), "cross_category": counts.get("cross_category", 0)}, "training_images": len(samples), "training_groups": len({s.group_id for s in samples}), "validation_results": {"passed": all(checks.values()), "checks": dict(checks)}}

def evaluate_unseen_groups(gallery: Gallery, *, holdout_group_ids: set[str] | frozenset[str], k: int = S3_13_K) -> dict[str, Any]:
    if k != S3_13_K:
        raise ValueError(f"S3.13 requires k={S3_13_K} for one shared ranking.")
    if not holdout_group_ids or gallery.size == 0:
        raise ValueError("S3.13 requires holdout groups and a non-empty gallery.")
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(gallery.metadata):
        product_id = str(item.get("product_id", item.get("product_group", "")))
        if not product_id:
            raise ValueError(f"Gallery row {index} has no product ID.")
        group_indices[product_id].append(index)
    missing = sorted(g for g in holdout_group_ids if g not in group_indices)
    if missing:
        raise ValueError(f"Holdout groups are missing from the evaluation gallery: {missing[:10]}")
    engine, ranker = SimilaritySearchEngine(gallery.embeddings, gallery.metadata), ProductRanker()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(gallery.metadata):
        product_id = str(item.get("product_id", item.get("product_group", "")))
        if product_id not in holdout_group_ids:
            continue
        image_id = str(item.get("image_id", item.get("image", "")))
        if len(group_indices[product_id]) < 2:
            raise ValueError(f"Holdout query {image_id!r} has no same-product gallery image.")
        ranked = ranker.rank(engine.search(gallery.embeddings[index], query_id=image_id, k=k, exclude_image_id=image_id))
        ids = [c.product_id for c in ranked.candidates]
        rank = next((r for r, pid in enumerate(ids, 1) if pid == product_id), None)
        rows.append({"query_image_id": image_id, "query_product_id": product_id, "category": str(item.get("category") or "") or None, "first_positive_rank": rank, "top1_hit": int(product_id in ids[:1]), "top5_hit": int(product_id in ids[:5]), "top10_hit": int(product_id in ids[:10]), "reciprocal_rank": 1.0 / rank if rank else 0.0, "gallery_positive_images": len(group_indices[product_id]) - 1, "ranked_product_ids": ids})
    if not rows:
        raise ValueError("S3.13 produced no held-out queries.")
    ranks = [r["first_positive_rank"] for r in rows if r["first_positive_rank"] is not None]
    metrics = {m: sum(r[f"{m}_hit"] for r in rows) / len(rows) for m in ("top1", "top5", "top10")}
    metrics["mrr"] = sum(r["reciprocal_rank"] for r in rows) / len(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[row["category"] or "unknown"].append(row)
    per_category = {cat: {"num_queries": len(items), "top1": sum(r["top1_hit"] for r in items) / len(items), "top5": sum(r["top5_hit"] for r in items) / len(items), "top10": sum(r["top10_hit"] for r in items) / len(items), "mrr": sum(r["reciprocal_rank"] for r in items) / len(items)} for cat, items in sorted(grouped.items())}
    return {"metrics": metrics, "query_counts": {"total": len(rows), "valid": len(rows), "excluded": 0}, "diagnostics": {"gallery_size": gallery.size, "gallery_product_groups": len(group_indices), "holdout_product_groups": len(holdout_group_ids), "queries_with_at_least_one_positive": len(rows), "positive_outside_top10": sum(r["top10_hit"] == 0 for r in rows), "mean_first_positive_rank": sum(ranks) / len(ranks) if ranks else None, "median_first_positive_rank": statistics.median(ranks) if ranks else None, "min_first_positive_rank": min(ranks) if ranks else None, "max_first_positive_rank": max(ranks) if ranks else None, "self_match_exclusions": len(rows), "nan_inf_count": 0, "per_category": per_category}, "query_records": rows}

def select_model(results: Mapping[str, Mapping[str, Any]]) -> str:
    if not results: raise ValueError("S3.13 requires at least one model result.")
    for model_name, payload in results.items():
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping): raise ValueError(f"Missing metrics for model {model_name!r}.")
        for metric in S3_13_METRICS:
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)): raise ValueError(f"Invalid {metric} for model {model_name!r}.")
    return max(results, key=lambda name: (float(results[name]["metrics"]["top1"]), float(results[name]["metrics"]["top5"]), float(results[name]["metrics"]["mrr"])))

def write_pairs(pairs: Sequence[Pair], output_path: str) -> None:
    write_pairs_csv(pairs, output_path)
