"""Unit tests for src/evaluation/metrics.py.

Uses small synthetic `is_positive_ranked` boolean sequences -- no trained
model, embeddings, or dataset required.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import metrics  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Perfect ranking (positive at rank 1, all further positives right after)
# ---------------------------------------------------------------------------
def test_perfect_ranking():
    ranking = [True, True, False, False, False]
    assert metrics.hit_at_k(ranking, 1) == 1
    assert metrics.hit_at_k(ranking, 5) == 1
    assert metrics.precision_at_k(ranking, 1) == 1.0
    assert metrics.precision_at_k(ranking, 5) == pytest.approx(2 / 5)
    assert metrics.recall_at_k(ranking, 1, total_positives=2) == pytest.approx(0.5)
    assert metrics.recall_at_k(ranking, 5, total_positives=2) == pytest.approx(1.0)
    assert metrics.first_positive_rank(ranking) == 1
    assert metrics.reciprocal_rank(ranking) == 1.0


# ---------------------------------------------------------------------------
# 2. Completely incorrect ranking (no positive found within the ranking)
# ---------------------------------------------------------------------------
def test_completely_incorrect_ranking():
    ranking = [False, False, False, False, False]
    assert metrics.hit_at_k(ranking, 1) == 0
    assert metrics.hit_at_k(ranking, 5) == 0
    assert metrics.precision_at_k(ranking, 5) == 0.0
    # total_positives=1: the single positive exists somewhere in the full
    # gallery but was not retrieved within this ranked slice.
    assert metrics.recall_at_k(ranking, 5, total_positives=1) == 0.0
    assert metrics.first_positive_rank(ranking) is None
    assert metrics.reciprocal_rank(ranking) == 0.0


# ---------------------------------------------------------------------------
# 3. Positive at rank 1
# ---------------------------------------------------------------------------
def test_positive_at_rank_1():
    ranking = [True, False, False, False, False, False, False, False, False, False]
    assert metrics.first_positive_rank(ranking) == 1
    assert metrics.reciprocal_rank(ranking) == 1.0
    assert metrics.hit_at_k(ranking, 1) == 1
    assert metrics.hit_at_k(ranking, 5) == 1
    assert metrics.hit_at_k(ranking, 10) == 1


# ---------------------------------------------------------------------------
# 4. Positive at rank 5
# ---------------------------------------------------------------------------
def test_positive_at_rank_5():
    ranking = [False, False, False, False, True, False, False, False, False, False]
    assert metrics.first_positive_rank(ranking) == 5
    assert metrics.reciprocal_rank(ranking) == pytest.approx(1 / 5)
    assert metrics.hit_at_k(ranking, 1) == 0
    assert metrics.hit_at_k(ranking, 5) == 1
    assert metrics.hit_at_k(ranking, 10) == 1
    assert metrics.precision_at_k(ranking, 5) == pytest.approx(1 / 5)
    assert metrics.recall_at_k(ranking, 5, total_positives=1) == 1.0
    assert metrics.recall_at_k(ranking, 1, total_positives=1) == 0.0


# ---------------------------------------------------------------------------
# 5. Positive at rank 10
# ---------------------------------------------------------------------------
def test_positive_at_rank_10():
    ranking = [False] * 9 + [True]
    assert metrics.first_positive_rank(ranking) == 10
    assert metrics.reciprocal_rank(ranking) == pytest.approx(1 / 10)
    assert metrics.hit_at_k(ranking, 5) == 0
    assert metrics.hit_at_k(ranking, 10) == 1
    assert metrics.precision_at_k(ranking, 10) == pytest.approx(1 / 10)
    assert metrics.recall_at_k(ranking, 10, total_positives=1) == 1.0
    assert metrics.recall_at_k(ranking, 5, total_positives=1) == 0.0


# ---------------------------------------------------------------------------
# 6. Multiple positives
# ---------------------------------------------------------------------------
def test_multiple_positives():
    # Positives at rank 1 and rank 3, out of 3 total positives in the gallery.
    ranking = [True, False, True, False, False, False, False, False, False, False]
    assert metrics.first_positive_rank(ranking) == 1
    assert metrics.reciprocal_rank(ranking) == 1.0
    assert metrics.precision_at_k(ranking, 5) == pytest.approx(2 / 5)
    assert metrics.recall_at_k(ranking, 1, total_positives=3) == pytest.approx(1 / 3)
    assert metrics.recall_at_k(ranking, 5, total_positives=3) == pytest.approx(2 / 3)
    assert metrics.recall_at_k(ranking, 10, total_positives=3) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 7. No positives at all (query would be excluded upstream by the evaluator)
# ---------------------------------------------------------------------------
def test_no_positives():
    ranking = [False, False, False]
    assert metrics.first_positive_rank(ranking) is None
    assert metrics.reciprocal_rank(ranking) == 0.0
    assert metrics.hit_at_k(ranking, 1) == 0
    assert metrics.precision_at_k(ranking, 1) == 0.0
    # recall_at_k is undefined (and must not be called) when there are no
    # positives in the full gallery -- it must raise rather than silently
    # returning 0 or dividing by zero.
    with pytest.raises(ValueError):
        metrics.recall_at_k(ranking, 1, total_positives=0)


# ---------------------------------------------------------------------------
# Additional edge-case coverage
# ---------------------------------------------------------------------------
def test_mean_reciprocal_rank():
    assert metrics.mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert metrics.mean_reciprocal_rank([]) == 0.0


def test_invalid_k_raises():
    ranking = [True, False]
    with pytest.raises(ValueError):
        metrics.hit_at_k(ranking, 0)
    with pytest.raises(ValueError):
        metrics.precision_at_k(ranking, -1)
    with pytest.raises(ValueError):
        metrics.recall_at_k(ranking, 0, total_positives=1)


def test_none_ranking_raises():
    with pytest.raises(ValueError):
        metrics.hit_at_k(None, 1)
