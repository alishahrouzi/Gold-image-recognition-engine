from random import Random

import pytest

from data.constants import SOURCE_DATASET1
from data.errors import PairGenerationError
from data.pairs.strategies import (
    CROSS_CATEGORY_NEGATIVE,
    RANDOM_NEGATIVE,
    SAME_CATEGORY_NEGATIVE,
    sample_negatives_by_strategy,
)
from data.types import Sample


def _samples() -> list[Sample]:
    return [
        Sample(
            image_id="a1",
            image_path="a1.jpg",
            group_id="g1",
            category="Bracelet",
            category_id=0,
            split="train",
            source=SOURCE_DATASET1,
        ),
        Sample(
            image_id="a2",
            image_path="a2.jpg",
            group_id="g1",
            category="Bracelet",
            category_id=0,
            split="train",
            source=SOURCE_DATASET1,
        ),
        Sample(
            image_id="b1",
            image_path="b1.jpg",
            group_id="g2",
            category="Bracelet",
            category_id=0,
            split="train",
            source=SOURCE_DATASET1,
        ),
        Sample(
            image_id="c1",
            image_path="c1.jpg",
            group_id="g3",
            category="Earrings",
            category_id=1,
            split="train",
            source=SOURCE_DATASET1,
        ),
        Sample(
            image_id="d1",
            image_path="d1.jpg",
            group_id="g4",
            category="Necklace",
            category_id=2,
            split="train",
            source=SOURCE_DATASET1,
        ),
    ]


def test_random_negative_has_different_groups_and_explicit_type():
    pairs = sample_negatives_by_strategy(_samples(), Random(42), strategy=RANDOM_NEGATIVE, count=3)
    assert len(pairs) == 3
    assert all(pair.group_id_1 != pair.group_id_2 for pair in pairs)
    assert all(pair.negative_type == RANDOM_NEGATIVE for pair in pairs)
    assert all(pair.label == 0 for pair in pairs)


def test_same_category_strategy_is_category_constrained():
    pairs = sample_negatives_by_strategy(_samples(), Random(42), strategy=SAME_CATEGORY_NEGATIVE, count=2)
    assert len(pairs) == 2
    assert all(pair.category_1 == pair.category_2 for pair in pairs)
    assert all(pair.group_id_1 != pair.group_id_2 for pair in pairs)
    assert all(pair.negative_type == SAME_CATEGORY_NEGATIVE for pair in pairs)


def test_cross_category_strategy_is_category_constrained():
    pairs = sample_negatives_by_strategy(_samples(), Random(42), strategy=CROSS_CATEGORY_NEGATIVE, count=3)
    assert len(pairs) == 3
    assert all(pair.category_1 != pair.category_2 for pair in pairs)
    assert all(pair.group_id_1 != pair.group_id_2 for pair in pairs)
    assert all(pair.negative_type == CROSS_CATEGORY_NEGATIVE for pair in pairs)


def test_random_strategy_is_reproducible():
    left = sample_negatives_by_strategy(_samples(), Random(123), strategy=RANDOM_NEGATIVE, count=4)
    right = sample_negatives_by_strategy(_samples(), Random(123), strategy=RANDOM_NEGATIVE, count=4)
    assert [pair.pair_id for pair in left] == [pair.pair_id for pair in right]


def test_unknown_strategy_rejected():
    with pytest.raises(PairGenerationError):
        sample_negatives_by_strategy(_samples(), Random(42), strategy="unknown", count=1)
