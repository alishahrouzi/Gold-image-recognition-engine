"""S3.6 embedding-geometry evaluation utilities.

This module evaluates whether learned embeddings separate image pairs that
belong to the same product from images belonging to different products. It is
independent from training-pair sampling and from the S2.8 retrieval metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch import Tensor

from data.loaders.image_loader import load_rgb_image
from data.loaders.manifest import load_manifest
from data.preprocessing.pipeline import ImagePreprocessor
from data.types import Sample
from metric_learning.distance import CosineDistance, EuclideanDistance
from metric_learning.siamese import SiameseNetwork
from retrieval.embedding import load_siamese_embedding_model


@dataclass(frozen=True)
class EvaluationPair:
    """Independent S3.6 image pair with product-identity ground truth."""

    image_id_1: str
    image_id_2: str
    group_id_1: str
    group_id_2: str
    category_1: str
    category_2: str
    label: int  # 1=same product, 0=different product
    negative_type: str | None = None

    @property
    def same_product(self) -> bool:
        return self.label == 1


def _canonical_key(image_a: str, image_b: str) -> tuple[str, str]:
    if image_a == image_b:
        raise ValueError("Self-pairs are not valid evaluation pairs.")
    return tuple(sorted((image_a, image_b)))


def build_evaluation_pairs(
    samples: Sequence[Sample],
    *,
    seed: int = 2026,
    negative_ratio: float = 1.0,
) -> tuple[EvaluationPair, ...]:
    """Build an independent balanced pair set from one image split.

    Positive pairs are all unordered image pairs within the same product group.
    The negative pool is sampled deterministically from different product groups
    until ``negative_ratio * positives`` negatives are obtained. No training
    pair CSV is reused, so S3.6 remains independent from S3.4/S3.5 sampling.
    """
    if not samples:
        raise ValueError("At least one sample is required.")
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be > 0.")

    by_group: dict[str, list[Sample]] = {}
    for sample in samples:
        by_group.setdefault(sample.group_id, []).append(sample)

    positives: list[EvaluationPair] = []
    occupied: set[tuple[str, str]] = set()
    for group_id, group_samples in sorted(by_group.items()):
        for index, sample_a in enumerate(group_samples):
            for sample_b in group_samples[index + 1 :]:
                key = _canonical_key(sample_a.image_id, sample_b.image_id)
                occupied.add(key)
                positives.append(
                    EvaluationPair(
                        sample_a.image_id,
                        sample_b.image_id,
                        sample_a.group_id,
                        sample_b.group_id,
                        sample_a.category,
                        sample_b.category,
                        1,
                        None,
                    )
                )

    if not positives:
        raise ValueError("Evaluation split contains no same-product positive pairs.")

    target_negatives = max(1, int(round(len(positives) * negative_ratio)))
    rng = Random(seed)
    group_ids = sorted(by_group)
    if len(group_ids) < 2:
        raise ValueError("At least two product groups are required for negatives.")

    negatives: list[EvaluationPair] = []
    max_attempts = max(1000, target_negatives * 100)
    attempts = 0
    while len(negatives) < target_negatives and attempts < max_attempts:
        attempts += 1
        group_a, group_b = rng.sample(group_ids, 2)
        sample_a = rng.choice(by_group[group_a])
        sample_b = rng.choice(by_group[group_b])
        key = _canonical_key(sample_a.image_id, sample_b.image_id)
        if key in occupied:
            continue
        occupied.add(key)
        negative_type = "same_category" if sample_a.category == sample_b.category else "cross_category"
        negatives.append(
            EvaluationPair(
                sample_a.image_id,
                sample_b.image_id,
                sample_a.group_id,
                sample_b.group_id,
                sample_a.category,
                sample_b.category,
                0,
                negative_type,
            )
        )

    if len(negatives) != target_negatives:
        raise RuntimeError(
            f"Could only construct {len(negatives)} of {target_negatives} negative evaluation pairs."
        )
    return tuple(positives + negatives)


def _load_batch(sample: Sample, preprocessor: ImagePreprocessor) -> Tensor:
    image = load_rgb_image(sample.image_path)
    return preprocessor(image)


def extract_embeddings(
    model: SiameseNetwork,
    samples: Sequence[Sample],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    preprocessor: ImagePreprocessor | None = None,
) -> Mapping[str, Tensor]:
    """Extract deterministic, non-augmented embeddings for manifest samples."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    processor = preprocessor or ImagePreprocessor()
    resolved_device = torch.device(device)
    model = model.to(resolved_device).eval()
    result: dict[str, Tensor] = {}
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        batch = torch.stack([_load_batch(sample, processor) for sample in batch_samples])
        with torch.inference_mode():
            embeddings = model.encode(batch.to(resolved_device, non_blocking=True)).detach().cpu()
        for sample, embedding in zip(batch_samples, embeddings):
            result[sample.image_id] = embedding
    return result


def _distance_stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty distance set.")
    percentiles = np.percentile(array, [25, 50, 75, 90, 95])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p25": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p90": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "max": float(array.max()),
    }


def _best_threshold(y_true: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    """Find the accuracy-maximizing distance threshold, lower=positive."""
    thresholds = np.unique(distances)
    best_accuracy = -1.0
    best_threshold = float("inf")
    best_precision = best_recall = best_f1 = 0.0
    for threshold in thresholds:
        predicted = (distances <= threshold).astype(np.int64)
        accuracy = float(accuracy_score(y_true, predicted))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predicted, average="binary", zero_division=0
        )
        if accuracy > best_accuracy or (accuracy == best_accuracy and threshold < best_threshold):
            best_accuracy = accuracy
            best_threshold = float(threshold)
            best_precision = float(precision)
            best_recall = float(recall)
            best_f1 = float(f1)
    return {
        "threshold": best_threshold,
        "accuracy": best_accuracy,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
    }


def _eer(y_true: np.ndarray, distances: np.ndarray) -> tuple[float, float]:
    scores = -distances
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0), float(-thresholds[index])


def evaluate_pair_geometry(
    pairs: Sequence[EvaluationPair],
    embeddings: Mapping[str, Tensor],
) -> Mapping[str, object]:
    """Evaluate same/different-product separation using cosine distance."""
    if not pairs:
        raise ValueError("At least one evaluation pair is required.")
    cosine = CosineDistance()
    euclidean = EuclideanDistance()
    cosine_distances: list[float] = []
    euclidean_distances: list[float] = []
    labels: list[int] = []
    negative_types: dict[str, int] = {"same_category": 0, "cross_category": 0}

    for pair in pairs:
        try:
            a = embeddings[pair.image_id_1].unsqueeze(0)
            b = embeddings[pair.image_id_2].unsqueeze(0)
        except KeyError as exc:
            raise KeyError(f"Missing embedding for evaluation image: {exc.args[0]}") from exc
        cosine_distances.append(float(cosine.pairwise(a, b)[0]))
        euclidean_distances.append(float(euclidean.pairwise(a, b)[0]))
        labels.append(pair.label)
        if pair.negative_type is not None:
            negative_types[pair.negative_type] += 1

    y_true = np.asarray(labels, dtype=np.int64)
    distances = np.asarray(cosine_distances, dtype=np.float64)
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    positive_distances = distances[positive_mask]
    negative_distances = distances[negative_mask]

    result = {
        "pair_counts": {
            "total": int(len(pairs)),
            "same_product": int(positive_mask.sum()),
            "different_product": int(negative_mask.sum()),
            "different_product_same_category": int(negative_types["same_category"]),
            "different_product_cross_category": int(negative_types["cross_category"]),
        },
        "cosine_distance": {
            "same_product": _distance_stats(positive_distances),
            "different_product": _distance_stats(negative_distances),
            "separation_mean": float(negative_distances.mean() - positive_distances.mean()),
            "separation_median": float(np.median(negative_distances) - np.median(positive_distances)),
        },
        "euclidean_distance": {
            "same_product": _distance_stats(np.asarray(euclidean_distances)[positive_mask]),
            "different_product": _distance_stats(np.asarray(euclidean_distances)[negative_mask]),
        },
        "classification": {
            "roc_auc": float(roc_auc_score(y_true, -distances)),
            "pr_auc": float(average_precision_score(y_true, -distances)),
            "best_threshold": _best_threshold(y_true, distances),
            "eer": _eer(y_true, distances)[0],
            "eer_threshold": _eer(y_true, distances)[1],
        },
    }
    return result


def nearest_neighbor_accuracy(
    samples: Sequence[Sample],
    embeddings: Mapping[str, Tensor],
) -> Mapping[str, object]:
    """Leave-one-image-out nearest-neighbor product-identity sanity check."""
    if len(samples) < 2:
        raise ValueError("At least two samples are required.")
    ids = [sample.image_id for sample in samples]
    matrix = torch.stack([embeddings[image_id] for image_id in ids]).float()
    matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
    distances = torch.cdist(matrix, matrix, p=2)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.argmin(dim=1).tolist()
    correct = [samples[index].group_id == samples[nn_index].group_id for index, nn_index in enumerate(nearest)]
    return {
        "queries": len(samples),
        "correct_same_product": int(sum(correct)),
        "accuracy": float(np.mean(correct)),
        "self_exclusions": len(samples),
    }


def evaluate_model(
    checkpoint_path: str | Path,
    samples: Sequence[Sample],
    pairs: Sequence[EvaluationPair],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
) -> Mapping[str, object]:
    """Load one Siamese checkpoint and evaluate it under the S3.6 protocol."""
    extractor = load_siamese_embedding_model(checkpoint_path, device=device)
    embeddings: dict[str, Tensor] = {}
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        batch = torch.stack([_load_batch(sample, ImagePreprocessor()) for sample in batch_samples])
        batch_embeddings = extractor.extract(batch)
        for sample, embedding in zip(batch_samples, batch_embeddings):
            embeddings[sample.image_id] = embedding
    result = dict(evaluate_pair_geometry(pairs, embeddings))
    result["nearest_neighbor"] = dict(nearest_neighbor_accuracy(samples, embeddings))
    result["checkpoint"] = str(Path(checkpoint_path))
    return result


def load_evaluation_samples(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path,
    split: str = "train",
) -> tuple[Sample, ...]:
    """Load one manifest split with file validation for S3.6."""
    samples = load_manifest(manifest_path, dataset_root=dataset_root, validate_files=True)
    selected = tuple(sample for sample in samples if sample.split == split)
    if not selected:
        raise ValueError(f"Manifest contains no samples for split={split!r}.")
    return selected
