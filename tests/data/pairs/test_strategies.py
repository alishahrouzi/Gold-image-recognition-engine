from random import Random

import pytest

from data.pairs.strategies import (
    CROSS_CATEGORY_NEGATIVE,
    RANDOM_NEGATIVE,
    SAME_CATEGORY_NEGATIVE,
    sample_negatives_by_strategy,
)
from data.pairs.types import Pair
from data.types import Sample


def _samples() -> list[Sample]:
    return [
        Sample("a1", "a1.jpg", "g1", 0, "Bracelet", "train"),
        Sample("a2", "a2.jpg", "g1", 0, "Bracelet", "train"),
        Sample("b1", "b1.jpg", "g2", 0, "Bracelet", "train"),
        Sample("c1", "c1.jpg", "g3", 1, "Earrings", "train"),
        Sample("d1", "d1.jpg", "g4", 2, "Necklace", "train"),
    ]


def test_random_negative_has_different_groups_and_explicit_type():
    pairs = sample_negatives_by_strategy(
        _samples(), Random(42), strategy=RANDOM_NEGATIVE, count=3
    )
    assert len(pairs) == 3
    assert all(pair.group_id_1 != pair.group_id_2 for pair in pairs)
    assert all(pair.negative_type == RANDOM_NEGATIVE for pair in pairs)
    assert all(pair.label == 0 for pair in pairs)


def test_same_category_strategy_is_category_constrained():
    pairs = sample_negatives_by_strategy(
        _samples(), Random(42), strategy=SAME_CATEGORY_NEGATIVE, count=3
    )
    assert len(pairs) == 3
    assert all(pair.category_1 == pair.category_2 for pair in pairs)
    assert all(pair.group_id_1 != pair.group_id_2 for pair in pairs)
    assert all(pair.negative_type == SAME_CATEGORY_NEGATIVE for pair in pairs)


def test_cross_category_strategy_is_category_constrained():
    pairs = sample_negatives_by_strategy(
        _samples(), Random(42), strategy=CROSS_CATEGORY_NEGATIVE, count=3
    )
    assert len(pairs) == 3
    assert all(pair.category_1 != pair.category_2 for pair in pairs)
    assert all(pair.group_id_1 != pair.group_id_2 for pair in pairs)
    assert all(pair.negative_type == CROSS_CATEGORY_NEGATIVE for pair in pairs)


def test_random_strategy_is_reproducible():
    left = sample_negatives_by_strategy(
        _samples(), Random(123), strategy=RANDOM_NEGATIVE, count=4
    )
    right = sample_negatives_by_strategy(
        _samples(), Random(123), strategy=RANDOM_NEGATIVE, count=4
    )
    assert [pair.pair_id for pair in left] == [pair.pair_id for pair in right]


def test_unknown_strategy_rejected():
    with pytest.raises(Exception):
        sample_negatives_by_strategy(
            _samples(), Random(42), strategy="unknown", count=1
        )
