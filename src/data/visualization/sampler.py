"""Deterministic visualization sampling. Does not modify global RNG state."""

from __future__ import annotations

from collections import defaultdict
from random import Random
from typing import Dict, List, Mapping, Sequence, Tuple

from ..errors import VisualizationError
from ..pairs.types import (
    NEGATIVE_TYPE_CROSS_CATEGORY,
    NEGATIVE_TYPE_SAME_CATEGORY,
    PAIR_TYPE_NEGATIVE,
    PAIR_TYPE_POSITIVE,
    Pair,
)
from ..types import Sample
from .config import VisualizationConfig
from .types import GroupPanel, PairPanel, VisualizationSelection


def groups_by_id(samples: Sequence[Sample], split: str) -> Dict[str, List[Sample]]:
    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.split == split:
            grouped[sample.group_id].append(sample)
    return {
        group_id: sorted(members, key=lambda item: item.image_id)
        for group_id, members in sorted(grouped.items())
    }


def sample_split_images(
    samples: Sequence[Sample],
    split: str,
    count: int,
    rng: Random,
) -> List[Sample]:
    pool = sorted(
        (sample for sample in samples if sample.split == split),
        key=lambda item: item.image_id,
    )
    return _take(pool, count, rng)


def sample_train_groups(
    samples: Sequence[Sample],
    count: int,
    rng: Random,
) -> List[GroupPanel]:
    """Sample train groups, preferring a mix of sizes 1 / 2 / 3+.

    ``count`` is the number of groups, not images. Every member of a
    selected group is included so multi-view products stay together.
    """
    grouped = groups_by_id(samples, "train")
    if count == 0 or not grouped:
        return []

    buckets: Dict[int, List[str]] = {1: [], 2: [], 3: []}
    for group_id, members in grouped.items():
        size_key = 3 if len(members) >= 3 else len(members)
        buckets[size_key].append(group_id)
    for size_key in buckets:
        rng.shuffle(buckets[size_key])

    selected_ids: List[str] = []
    while len(selected_ids) < count:
        progressed = False
        for size_key in (3, 2, 1):
            if buckets[size_key] and len(selected_ids) < count:
                selected_ids.append(buckets[size_key].pop(0))
                progressed = True
        if not progressed:
            break

    return [
        GroupPanel(
            group_id=group_id,
            category=grouped[group_id][0].category,
            split="train",
            samples=tuple(grouped[group_id]),
        )
        for group_id in selected_ids
    ]


def sample_pairs(
    pairs: Sequence[Pair],
    samples_by_image: Mapping[str, Sample],
    *,
    pair_type: str,
    count: int,
    rng: Random,
    negative_type: str | None = None,
) -> List[PairPanel]:
    pool = [
        pair
        for pair in pairs
        if pair.pair_type == pair_type
        and (negative_type is None or pair.negative_type == negative_type)
    ]
    pool = sorted(pool, key=lambda pair: pair.pair_id)
    selected = _take(pool, count, rng)
    return [_to_panel(pair, samples_by_image) for pair in selected]


def sample_negative_pairs_balanced(
    pairs: Sequence[Pair],
    samples_by_image: Mapping[str, Sample],
    count: int,
    rng: Random,
) -> Tuple[List[PairPanel], List[PairPanel]]:
    """Split the requested negative count across same- and cross-category.

    When one type is exhausted, remaining slots are filled from the other
    type so the total still approaches ``count``.
    """
    n_same = count // 2
    n_cross = count - n_same
    same = sample_pairs(
        pairs,
        samples_by_image,
        pair_type=PAIR_TYPE_NEGATIVE,
        count=n_same,
        rng=rng,
        negative_type=NEGATIVE_TYPE_SAME_CATEGORY,
    )
    cross = sample_pairs(
        pairs,
        samples_by_image,
        pair_type=PAIR_TYPE_NEGATIVE,
        count=n_cross,
        rng=rng,
        negative_type=NEGATIVE_TYPE_CROSS_CATEGORY,
    )
    remaining = count - len(same) - len(cross)
    if remaining > 0:
        used = {panel.pair.pair_id for panel in same + cross}
        leftover_same = [
            pair
            for pair in pairs
            if pair.pair_type == PAIR_TYPE_NEGATIVE
            and pair.negative_type == NEGATIVE_TYPE_SAME_CATEGORY
            and pair.pair_id not in used
        ]
        leftover_cross = [
            pair
            for pair in pairs
            if pair.pair_type == PAIR_TYPE_NEGATIVE
            and pair.negative_type == NEGATIVE_TYPE_CROSS_CATEGORY
            and pair.pair_id not in used
        ]
        extra_same = _take(sorted(leftover_same, key=lambda pair: pair.pair_id), remaining, rng)
        same.extend(_to_panel(pair, samples_by_image) for pair in extra_same)
        remaining = count - len(same) - len(cross)
        extra_cross = _take(sorted(leftover_cross, key=lambda pair: pair.pair_id), remaining, rng)
        cross.extend(_to_panel(pair, samples_by_image) for pair in extra_cross)
    return same, cross


def sample_augmentation_images(
    samples: Sequence[Sample],
    count: int,
    rng: Random,
) -> List[Sample]:
    """Sample train images (at most one per group) for before/after views."""
    grouped = groups_by_id(samples, "train")
    group_ids = list(grouped.keys())
    rng.shuffle(group_ids)
    selected: List[Sample] = []
    for group_id in group_ids:
        if len(selected) >= count:
            break
        selected.append(grouped[group_id][0])
    return selected


def build_selection(
    samples: Sequence[Sample],
    pairs: Sequence[Pair],
    config: VisualizationConfig,
    rng: Random,
) -> VisualizationSelection:
    by_image = _index_samples(samples)
    train_groups = sample_train_groups(samples, config.train_samples, rng)
    valid_samples = sample_split_images(samples, "valid", config.valid_samples, rng)
    test_samples = sample_split_images(samples, "test", config.test_samples, rng)
    positive_pairs = sample_pairs(
        pairs,
        by_image,
        pair_type=PAIR_TYPE_POSITIVE,
        count=config.positive_pairs,
        rng=rng,
    )
    same, cross = sample_negative_pairs_balanced(
        pairs, by_image, config.negative_pairs, rng
    )
    augmentation_samples = sample_augmentation_images(
        samples, config.augmentation_samples, rng
    )
    return VisualizationSelection(
        train_groups=train_groups,
        valid_samples=valid_samples,
        test_samples=test_samples,
        positive_pairs=positive_pairs,
        same_category_negatives=same,
        cross_category_negatives=cross,
        augmentation_samples=augmentation_samples,
    )


def _index_samples(samples: Sequence[Sample]) -> Dict[str, Sample]:
    by_image: Dict[str, Sample] = {}
    for sample in samples:
        if sample.image_id in by_image:
            raise VisualizationError(f"Duplicate image_id {sample.image_id!r}.")
        by_image[sample.image_id] = sample
    return by_image


def _to_panel(pair: Pair, samples_by_image: Mapping[str, Sample]) -> PairPanel:
    try:
        left = samples_by_image[pair.image_id_1]
        right = samples_by_image[pair.image_id_2]
    except KeyError as exc:
        missing = exc.args[0]
        raise VisualizationError(
            f"Pair {pair.pair_id} references missing image_id {missing!r}."
        ) from exc
    return PairPanel(pair=pair, sample_a=left, sample_b=right)


def _take(items: Sequence, count: int, rng: Random) -> list:
    if count < 0:
        raise VisualizationError("Sample count must be >= 0.")
    if count == 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    return rng.sample(list(items), count)
