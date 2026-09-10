"""Negative-pair sampling strategies for S3.5 experiments.

This module keeps S1.10 pair-generation ownership intact while exposing
explicit strategies for controlled sampling experiments.
"""

from __future__ import annotations

from random import Random
from typing import List, Optional, Set, Tuple

from ..errors import PairGenerationError
from ..types import Sample
from .generator import SplitIndex
from .types import (
    NEGATIVE_TYPE_CROSS_CATEGORY,
    NEGATIVE_TYPE_SAME_CATEGORY,
    PAIR_TYPE_NEGATIVE,
    Pair,
    pair_from_samples,
)

UnorderedKey = Tuple[str, str]

RANDOM_NEGATIVE = "random"
SAME_CATEGORY_NEGATIVE = NEGATIVE_TYPE_SAME_CATEGORY
CROSS_CATEGORY_NEGATIVE = NEGATIVE_TYPE_CROSS_CATEGORY

SUPPORTED_NEGATIVE_STRATEGIES = (
    RANDOM_NEGATIVE,
    SAME_CATEGORY_NEGATIVE,
    CROSS_CATEGORY_NEGATIVE,
)


def sample_random_negatives(
    samples: List[Sample],
    rng: Random,
    *,
    count: int,
    occupied: Optional[Set[UnorderedKey]] = None,
    max_attempts_per_pair: int = 10_000,
) -> List[Pair]:
    """Sample unique negatives uniformly from valid image pairs.

    Category is unconstrained, but product groups must differ. Existing
    occupied pairs (including positives) are excluded.
    """
    if count < 0:
        raise PairGenerationError("count must be >= 0.")
    if count == 0:
        return []
    if len(samples) < 2:
        raise PairGenerationError("At least two samples are required.")

    index = SplitIndex(samples)
    used = set(occupied or ())
    found: List[Pair] = []
    attempts = 0
    budget = max(count * max_attempts_per_pair, max_attempts_per_pair)
    while len(found) < count:
        attempts += 1
        if attempts > budget:
            raise PairGenerationError(
                f"Could not sample {count} random negative pairs after "
                f"{attempts - 1} attempts ({len(found)} found)."
            )
        left, right = rng.sample(index.samples, 2)
        if left.group_id == right.group_id:
            continue
        pair = pair_from_samples(
            left,
            right,
            label=0,
            pair_type=PAIR_TYPE_NEGATIVE,
            negative_type=RANDOM_NEGATIVE,
        )
        key = pair.unordered_key()
        if key in used:
            continue
        used.add(key)
        found.append(pair)
    return found


def sample_negatives_by_strategy(
    samples: List[Sample],
    rng: Random,
    *,
    strategy: str,
    count: int,
    occupied: Optional[Set[UnorderedKey]] = None,
    max_attempts_per_pair: int = 10_000,
) -> List[Pair]:
    """Dispatch one explicit S3.5 negative sampling strategy."""
    if strategy == RANDOM_NEGATIVE:
        return sample_random_negatives(
            samples, rng, count=count, occupied=occupied,
            max_attempts_per_pair=max_attempts_per_pair,
        )
    if strategy in (SAME_CATEGORY_NEGATIVE, CROSS_CATEGORY_NEGATIVE):
        from .sampler import _sample_typed_negatives

        index = SplitIndex(samples)
        return _sample_typed_negatives(
            index,
            rng,
            count=count,
            negative_type=strategy,
            used=set(occupied or ()),
            max_attempts_per_pair=max_attempts_per_pair,
        )
    raise PairGenerationError(
        f"Unknown negative sampling strategy {strategy!r}. "
        f"Supported: {SUPPORTED_NEGATIVE_STRATEGIES}."
    )
