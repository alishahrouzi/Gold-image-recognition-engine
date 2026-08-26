"""Leakage-safe, deterministic pair generation from a Dataset 1 manifest.

Operates on Sample metadata only. Does not load image pixels, does not
modify the manifest, and does not implement hard-negative mining.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..constants import CATEGORY_TO_ID, SOURCE_DATASET1, SPLIT_ORDER
from ..errors import PairGenerationError
from ..types import Sample
from ..validation import validate_samples
from .config import PairGenerationConfig
from .sampler import (
    count_available_positive_pairs,
    generate_positive_pairs,
    occupied_keys,
    sample_negative_pairs,
    samples_by_split,
)
from .types import (
    NEGATIVE_TYPE_CROSS_CATEGORY,
    NEGATIVE_TYPE_SAME_CATEGORY,
    PAIR_CSV_FIELDS,
    PAIR_TYPE_NEGATIVE,
    PAIR_TYPE_POSITIVE,
    Pair,
    sort_pairs,
)
from .validation import validate_pairs

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PairGenerationResult:
    """Generated pairs plus the audit report that describes them."""

    pairs: tuple[Pair, ...]
    report: Dict[str, Any]
    config: PairGenerationConfig


def generate_pair_dataset(
    samples: Sequence[Sample],
    config: Optional[PairGenerationConfig] = None,
    *,
    dataset: str = SOURCE_DATASET1,
    manifest: Optional[PathLike] = None,
) -> PairGenerationResult:
    """Generate split-isolated pairs from validated manifest samples.

    Positives and negatives are produced independently per included split.
    A local ``random.Random(config.seed)`` is used for sampling; global
    random state is not modified.
    """
    config = config or PairGenerationConfig()
    if not samples:
        raise PairGenerationError("Cannot generate pairs from an empty sample list.")

    validate_samples(samples, validate_files=False)
    _assert_included_splits_present(samples, config)

    rng = Random(config.seed)
    grouped = samples_by_split(samples, config.included_splits)

    available_by_split: Dict[str, int] = {}
    positives_by_split: Dict[str, List[Pair]] = {}
    for split in config.included_splits:
        split_samples = grouped[split]
        available_by_split[split] = count_available_positive_pairs(split_samples)
        positives_by_split[split] = generate_positive_pairs(split_samples)

    selected_positives = _select_positives(positives_by_split, config, rng)
    selected_negatives = _sample_negatives_per_split(
        grouped,
        selected_positives,
        config,
        rng,
    )

    pairs = sort_pairs([*selected_positives, *selected_negatives], SPLIT_ORDER)
    checks = validate_pairs(pairs, samples)
    report = build_pair_generation_report(
        pairs,
        samples,
        config,
        dataset=dataset,
        manifest=manifest,
        available_positive_by_split=available_by_split,
        validation_checks=checks,
    )
    logger.info(
        "Generated %s pairs (%s positive, %s negative) seed=%s",
        len(pairs),
        report["positive_pairs"],
        report["negative_pairs"],
        config.seed,
    )
    return PairGenerationResult(pairs=pairs, report=report, config=config)


def build_pair_generation_report(
    pairs: Sequence[Pair],
    samples: Sequence[Sample],
    config: PairGenerationConfig,
    *,
    dataset: str,
    manifest: Optional[PathLike],
    available_positive_by_split: Mapping[str, int],
    validation_checks: Mapping[str, bool],
) -> Dict[str, Any]:
    """Build the JSON audit report. Counts are derived, not hard-coded."""
    positive_pairs = [pair for pair in pairs if pair.pair_type == PAIR_TYPE_POSITIVE]
    negative_pairs = [pair for pair in pairs if pair.pair_type == PAIR_TYPE_NEGATIVE]
    same_category = [
        pair for pair in negative_pairs if pair.negative_type == NEGATIVE_TYPE_SAME_CATEGORY
    ]
    cross_category = [
        pair
        for pair in negative_pairs
        if pair.negative_type == NEGATIVE_TYPE_CROSS_CATEGORY
    ]
    selected_positive = len(positive_pairs)
    selected_negative = len(negative_pairs)
    realized_ratio = (
        selected_negative / selected_positive if selected_positive else None
    )
    available_positive = sum(available_positive_by_split.get(split, 0) for split in config.included_splits)

    return {
        "dataset": dataset,
        "manifest": None if manifest is None else str(Path(manifest)),
        "seed": config.seed,
        "configuration": dict(config.as_loggable_dict()),
        "total_pairs": len(pairs),
        "positive_pairs": selected_positive,
        "negative_pairs": selected_negative,
        "available_positive_pairs": available_positive,
        "selected_positive_pairs": selected_positive,
        "selected_negative_pairs": selected_negative,
        "positive_negative_ratio": config.positive_negative_ratio,
        "realized_positive_negative_ratio": realized_ratio,
        "same_category_negative_ratio": config.same_category_negative_ratio,
        "same_category_negative_count": len(same_category),
        "cross_category_negative_count": len(cross_category),
        "per_split_statistics": _per_split_statistics(
            pairs, available_positive_by_split, config
        ),
        "per_category_statistics": _per_category_statistics(pairs),
        "sample_counts": {
            "images": len(samples),
            "groups": len({sample.group_id for sample in samples}),
        },
        "validation_results": {
            "passed": True,
            "checks": dict(validation_checks),
        },
    }


def write_pairs_csv(pairs: Sequence[Pair], output_path: PathLike) -> Path:
    """Write the pair dataset CSV. Does not modify the image manifest."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIR_CSV_FIELDS))
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.to_csv_row())
    logger.info("Wrote pair dataset: %s (%s rows)", path, len(pairs))
    return path


def write_pair_generation_report(report: Mapping[str, Any], output_path: PathLike) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Wrote pair generation report: %s", path)
    return path


def _assert_included_splits_present(
    samples: Sequence[Sample],
    config: PairGenerationConfig,
) -> None:
    present = {sample.split for sample in samples}
    missing = [split for split in config.included_splits if split not in present]
    if missing:
        raise PairGenerationError(
            f"Configured splits are missing from the manifest: {missing}."
        )


def _select_positives(
    positives_by_split: Mapping[str, List[Pair]],
    config: PairGenerationConfig,
    rng: Random,
) -> List[Pair]:
    ordered: List[Pair] = []
    for split in config.included_splits:
        ordered.extend(positives_by_split.get(split, []))
    cap = config.max_positive_pairs
    if cap is None or cap >= len(ordered):
        return ordered
    selected = list(rng.sample(ordered, cap))
    return list(sort_pairs(selected, SPLIT_ORDER))


def _sample_negatives_per_split(
    grouped: Mapping[str, Sequence[Sample]],
    selected_positives: Sequence[Pair],
    config: PairGenerationConfig,
    rng: Random,
) -> List[Pair]:
    positives_by_split: Dict[str, List[Pair]] = {split: [] for split in config.included_splits}
    for pair in selected_positives:
        positives_by_split[pair.split].append(pair)

    sampled: List[Pair] = []
    for split in config.included_splits:
        split_positives = positives_by_split[split]
        n_neg = _target_count(len(split_positives), config.positive_negative_ratio)
        n_same = _target_count(n_neg, config.same_category_negative_ratio)
        n_cross = n_neg - n_same
        occupied = occupied_keys(split_positives)
        sampled.extend(
            sample_negative_pairs(
                grouped[split],
                rng,
                n_same_category=n_same,
                n_cross_category=n_cross,
                occupied=occupied,
            )
        )

    cap = config.max_negative_pairs
    if cap is None or cap >= len(sampled):
        return sampled
    reduced = list(rng.sample(sampled, cap))
    return list(sort_pairs(reduced, SPLIT_ORDER))


def _target_count(reference: int, ratio: float) -> int:
    return int(round(reference * ratio))


def _per_split_statistics(
    pairs: Sequence[Pair],
    available_positive_by_split: Mapping[str, int],
    config: PairGenerationConfig,
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for split in config.included_splits:
        split_pairs = [pair for pair in pairs if pair.split == split]
        positives = [pair for pair in split_pairs if pair.pair_type == PAIR_TYPE_POSITIVE]
        negatives = [pair for pair in split_pairs if pair.pair_type == PAIR_TYPE_NEGATIVE]
        stats[split] = {
            "total_pairs": len(split_pairs),
            "positive_pairs": len(positives),
            "negative_pairs": len(negatives),
            "available_positive_pairs": available_positive_by_split.get(split, 0),
            "same_category_negative_count": sum(
                1
                for pair in negatives
                if pair.negative_type == NEGATIVE_TYPE_SAME_CATEGORY
            ),
            "cross_category_negative_count": sum(
                1
                for pair in negatives
                if pair.negative_type == NEGATIVE_TYPE_CROSS_CATEGORY
            ),
        }
    return stats


def _per_category_statistics(pairs: Sequence[Pair]) -> Dict[str, Dict[str, int]]:
    stats = {
        category: {
            "positive_pairs": 0,
            "same_category_negative_count": 0,
            "cross_category_negative_count": 0,
        }
        for category in CATEGORY_TO_ID
    }
    for pair in pairs:
        categories = {pair.category_1, pair.category_2}
        if pair.pair_type == PAIR_TYPE_POSITIVE:
            # Positives share group_id; Dataset 1 groups are single-category.
            stats[pair.category_1]["positive_pairs"] += 1
            continue
        if pair.negative_type == NEGATIVE_TYPE_SAME_CATEGORY:
            stats[pair.category_1]["same_category_negative_count"] += 1
            continue
        for category in categories:
            stats[category]["cross_category_negative_count"] += 1
    return stats
