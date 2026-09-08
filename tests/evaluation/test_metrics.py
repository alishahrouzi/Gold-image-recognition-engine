"""Unit tests for S2.8 Top-1 / Top-5 / Top-10 / MRR using product identity."""

from __future__ import annotations

import pytest

from evaluation import metrics
from evaluation.errors import EvaluationError
from evaluation.retrieval_evaluator import ranking_from_groups


def test_case1_positive_at_rank_1() -> None:
    ranking = ranking_from_groups(
        ["query-group", "other-a", "other-b", "other-c"],
        "query-group",
    )
    assert metrics.hit_at_k(ranking, 1) == 1
    assert metrics.hit_at_k(ranking, 5) == 1
    assert metrics.hit_at_k(ranking, 10) == 1
    assert metrics.reciprocal_rank(ranking) == 1.0


def test_case2_positive_at_rank_3() -> None:
    ranking = ranking_from_groups(
        ["neg-1", "neg-2", "query-group", "neg-4"],
        "query-group",
    )
    assert metrics.hit_at_k(ranking, 1) == 0
    assert metrics.hit_at_k(ranking, 5) == 1
    assert metrics.hit_at_k(ranking, 10) == 1
    assert metrics.reciprocal_rank(ranking) == pytest.approx(1 / 3)


def test_case3_positive_at_rank_7() -> None:
    ranking = ranking_from_groups(
        [f"neg-{index}" for index in range(6)] + ["query-group"],
        "query-group",
    )
    assert metrics.hit_at_k(ranking, 1) == 0
    assert metrics.hit_at_k(ranking, 5) == 0
    assert metrics.hit_at_k(ranking, 10) == 1
    assert metrics.reciprocal_rank(ranking) == pytest.approx(1 / 7)


def test_case4_no_positive() -> None:
    ranking = ranking_from_groups(["neg-1", "neg-2", "neg-3"], "query-group")
    assert metrics.hit_at_k(ranking, 1) == 0
    assert metrics.hit_at_k(ranking, 5) == 0
    assert metrics.hit_at_k(ranking, 10) == 0
    assert metrics.reciprocal_rank(ranking) == 0.0


def test_case5_multiple_images_of_same_product_count_once() -> None:
    ranking = ranking_from_groups(
        ["query-group", "query-group", "query-group", "neg-1"],
        "query-group",
    )
    # Three images of the query product collapse to one Positive; the
    # unrelated product remains one Negative. This must not become
    # [True, True, True, False].
    assert ranking == [True, False]
    assert ranking.count(True) == 1
    assert metrics.hit_at_k(ranking, 1) == 1
    assert metrics.reciprocal_rank(ranking) == 1.0
    assert metrics.mean_reciprocal_rank([metrics.reciprocal_rank(ranking)]) == 1.0


def test_case8_invalid_group_id_fails_loudly() -> None:
    with pytest.raises(EvaluationError, match="query_group_id"):
        ranking_from_groups(["neg-1"], "")
    with pytest.raises(EvaluationError, match="product_group"):
        ranking_from_groups([""], "query-group")


def test_case10_perfect_retrieval() -> None:
    rankings = [
        ranking_from_groups(["g1", "n1"], "g1"),
        ranking_from_groups(["g2", "n2"], "g2"),
        ranking_from_groups(["g3", "n3"], "g3"),
    ]
    assert all(metrics.hit_at_k(row, 1) == 1 for row in rankings)
    assert all(metrics.hit_at_k(row, 5) == 1 for row in rankings)
    assert all(metrics.hit_at_k(row, 10) == 1 for row in rankings)
    assert metrics.mean_reciprocal_rank(
        [metrics.reciprocal_rank(row) for row in rankings]
    ) == 1.0
