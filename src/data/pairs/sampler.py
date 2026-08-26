"""Positive enumeration and group-aware random negative sampling.

Positives are exhaustive C(N, 2) within a split. Negatives are sampled;
they are never fully enumerated. Sampling uses a caller-supplied
``random.Random`` and does not touch global RNG state.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from random import Random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..errors import PairGenerationError
from ..types import Sample
from .types import (
    NEGATIVE_TYPE_CROSS_CATEGORY,
    NEGATIVE_TYPE_SAME_CATEGORY,
    PAIR_TYPE_NEGATIVE,
    PAIR_TYPE_POSITIVE,
    Pair,
    pair_from_samples,
)

UnorderedKey = Tuple[str, str]


class SplitIndex:
    """Per-split lookup tables for group-aware negative sampling."""

    def __init__(self, samples: Sequence[Sample]) -> None:
        if not samples:
            raise PairGenerationError("Cannot build a split index from an empty sample list.")
        splits = {sample.split for sample in samples}
        if len(splits) != 1:
            raise PairGenerationError(
                "SplitIndex requires samples from exactly one split, "
                f"got {sorted(splits)}."
            )
        self.split = next(iter(splits))
        self.samples: List[Sample] = sorted(samples, key=lambda item: item.image_id)
        self.by_image_id: Dict[str, Sample] = {}
        self.by_group: Dict[str, List[Sample]] = defaultdict(list)
        self.by_category: Dict[str, List[Sample]] = defaultdict(list)

        for sample in self.samples:
            if sample.image_id in self.by_image_id:
                raise PairGenerationError(f"Duplicate image_id {sample.image_id!r}.")
            self.by_image_id[sample.image_id] = sample
            self.by_group[sample.group_id].append(sample)
            self.by_category[sample.category].append(sample)

        self.by_group = {
            group_id: sorted(members, key=lambda item: item.image_id)
            for group_id, members in sorted(self.by_group.items())
        }
        self.by_category = {
            category: sorted(members, key=lambda item: item.image_id)
            for category, members in sorted(self.by_category.items())
        }
        groups_by_category: Dict[str, List[str]] = defaultdict(list)
        for group_id, members in self.by_group.items():
            categories = {member.category for member in members}
            if len(categories) != 1:
                # Identity is still group_id; mixed-category groups cannot
                # supply same-category negatives from a single group list.
                continue
            groups_by_category[next(iter(categories))].append(group_id)
        self.groups_by_category: Dict[str, List[str]] = {
            category: list(group_ids)
            for category, group_ids in sorted(groups_by_category.items())
        }

    @property
    def same_category_eligible_categories(self) -> List[str]:
        return [
            category
            for category, group_ids in self.groups_by_category.items()
            if len(group_ids) >= 2
        ]

    @property
    def cross_category_eligible_categories(self) -> List[str]:
        return [
            category
            for category, members in self.by_category.items()
            if members
        ]


def generate_positive_pairs(samples: Sequence[Sample]) -> List[Pair]:
    """Enumerate all unordered positives for one split (C(N, 2) per group)."""
    if not samples:
        return []
    index = SplitIndex(samples)
    pairs: List[Pair] = []
    for group_id, members in index.by_group.items():
        if len(members) < 2:
            continue
        splits = {member.split for member in members}
        if len(splits) != 1:
            raise PairGenerationError(
                f"Group {group_id!r} spans multiple splits inside pair generation."
            )
        for left, right in combinations(members, 2):
            pair = pair_from_samples(
                left,
                right,
                label=1,
                pair_type=PAIR_TYPE_POSITIVE,
                negative_type=None,
            )
            if pair.group_id_1 != pair.group_id_2:
                raise PairGenerationError(
                    f"Positive pair {pair.pair_id} has mismatched group IDs."
                )
            pairs.append(pair)
    return pairs


def count_available_positive_pairs(samples: Sequence[Sample]) -> int:
    """Return C(N, 2) positives available from groups with N >= 2."""
    sizes: Dict[str, int] = defaultdict(int)
    for sample in samples:
        sizes[sample.group_id] += 1
    return sum(size * (size - 1) // 2 for size in sizes.values() if size >= 2)


def sample_negative_pairs(
    samples: Sequence[Sample],
    rng: Random,
    *,
    n_same_category: int,
    n_cross_category: int,
    occupied: Optional[Set[UnorderedKey]] = None,
    max_attempts_per_pair: int = 10_000,
) -> List[Pair]:
    """Sample unique unordered negatives inside one split."""
    if n_same_category < 0 or n_cross_category < 0:
        raise PairGenerationError("Negative counts must be >= 0.")
    if n_same_category == 0 and n_cross_category == 0:
        return []
    if not samples:
        raise PairGenerationError("Cannot sample negatives from an empty split.")

    index = SplitIndex(samples)
    used: Set[UnorderedKey] = set(occupied or ())
    selected: List[Pair] = []

    selected.extend(
        _sample_typed_negatives(
            index,
            rng,
            count=n_same_category,
            negative_type=NEGATIVE_TYPE_SAME_CATEGORY,
            used=used,
            max_attempts_per_pair=max_attempts_per_pair,
        )
    )
    selected.extend(
        _sample_typed_negatives(
            index,
            rng,
            count=n_cross_category,
            negative_type=NEGATIVE_TYPE_CROSS_CATEGORY,
            used=used,
            max_attempts_per_pair=max_attempts_per_pair,
        )
    )
    return selected


def _sample_typed_negatives(
    index: SplitIndex,
    rng: Random,
    *,
    count: int,
    negative_type: str,
    used: Set[UnorderedKey],
    max_attempts_per_pair: int,
) -> List[Pair]:
    if count == 0:
        return []
    _assert_negative_pool_exists(index, negative_type)
    found: List[Pair] = []
    attempts = 0
    attempt_budget = max(count * max_attempts_per_pair, max_attempts_per_pair)
    while len(found) < count:
        attempts += 1
        if attempts > attempt_budget:
            raise PairGenerationError(
                f"Could not sample {count} {negative_type} negative pairs "
                f"in split {index.split!r} after {attempts - 1} attempts "
                f"({len(found)} unique pairs found)."
            )
        candidate = _draw_negative_candidate(index, rng, negative_type)
        if candidate is None:
            continue
        key = candidate.unordered_key()
        if key in used:
            continue
        used.add(key)
        found.append(candidate)
    return found


def _assert_negative_pool_exists(index: SplitIndex, negative_type: str) -> None:
    if negative_type == NEGATIVE_TYPE_SAME_CATEGORY:
        if not index.same_category_eligible_categories:
            raise PairGenerationError(
                f"Split {index.split!r} has no two groups in the same category; "
                "same-category negatives cannot be sampled."
            )
        return
    if negative_type == NEGATIVE_TYPE_CROSS_CATEGORY:
        if len(index.cross_category_eligible_categories) < 2:
            raise PairGenerationError(
                f"Split {index.split!r} does not contain two categories; "
                "cross-category negatives cannot be sampled."
            )
        return
    raise PairGenerationError(f"Unknown negative_type {negative_type!r}.")


def _draw_negative_candidate(
    index: SplitIndex,
    rng: Random,
    negative_type: str,
) -> Optional[Pair]:
    if negative_type == NEGATIVE_TYPE_SAME_CATEGORY:
        categories = index.same_category_eligible_categories
        category = rng.choice(categories)
        group_ids = index.groups_by_category[category]
        group_a, group_b = rng.sample(group_ids, 2)
        left = rng.choice(index.by_group[group_a])
        right = rng.choice(index.by_group[group_b])
    elif negative_type == NEGATIVE_TYPE_CROSS_CATEGORY:
        categories = index.cross_category_eligible_categories
        category_a, category_b = rng.sample(categories, 2)
        left = rng.choice(index.by_category[category_a])
        right = rng.choice(index.by_category[category_b])
    else:
        raise PairGenerationError(f"Unknown negative_type {negative_type!r}.")

    if left.group_id == right.group_id:
        return None
    pair = pair_from_samples(
        left,
        right,
        label=0,
        pair_type=PAIR_TYPE_NEGATIVE,
        negative_type=negative_type,
    )
    expected_same = left.category == right.category
    if negative_type == NEGATIVE_TYPE_SAME_CATEGORY and not expected_same:
        return None
    if negative_type == NEGATIVE_TYPE_CROSS_CATEGORY and expected_same:
        return None
    return pair


def occupied_keys(pairs: Iterable[Pair]) -> Set[UnorderedKey]:
    return {pair.unordered_key() for pair in pairs}


def samples_by_split(
    samples: Sequence[Sample],
    included_splits: Sequence[str],
) -> Mapping[str, List[Sample]]:
    grouped: Dict[str, List[Sample]] = {split: [] for split in included_splits}
    for sample in samples:
        if sample.split in grouped:
            grouped[sample.split].append(sample)
    return grouped
