"""S2.8 evaluator tests over S2.7 RetrievalResult / TopKRetriever."""

from __future__ import annotations

import math

import pytest
import torch

from evaluation.errors import EvaluationError
from evaluation.retrieval_evaluator import (
    evaluate_leave_one_out_gallery,
    evaluate_retrieval,
    score_retrieval_result,
)
from src.retrieval.gallery import Gallery
from src.retrieval.result import RetrievalCandidate, RetrievalResult
from src.retrieval.topk import TopKRetriever


def _candidate(
    image_id: str,
    product_group: str,
    similarity: float,
    rank: int,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        image_id=image_id,
        product_group=product_group,
        category="Ring",
        similarity=similarity,
        rank=rank,
    )


def _result(query_id: str, groups: list[tuple[str, str, float]]) -> RetrievalResult:
    candidates = tuple(
        _candidate(image_id, group_id, score, rank)
        for rank, (image_id, group_id, score) in enumerate(groups, start=1)
    )
    return RetrievalResult(query_id=query_id, candidates=candidates)


def _score(result: RetrievalResult, query_group: str, gallery_positives: int = 1):
    return score_retrieval_result(
        result,
        query_group_id=query_group,
        query_image_id=result.query_id,
        split="train",
        gallery_split="train",
        requested_k=10,
        gallery_positive_images=gallery_positives,
        gallery_positive_products=1 if gallery_positives else 0,
        self_match_excluded=True,
    )


def test_case1_positive_at_rank_1() -> None:
    result = _result(
        "q1",
        [("img-pos", "g-q", 0.9), ("img-neg", "g-n", 0.1)],
    )
    record = _score(result, "g-q")
    assert (record.top1_hit, record.top5_hit, record.top10_hit, record.reciprocal_rank) == (
        1,
        1,
        1,
        1.0,
    )


def test_case2_positive_at_rank_3() -> None:
    groups = [(f"n{i}", f"neg-{i}", 1.0 - i * 0.05) for i in range(1, 3)]
    groups.append(("pos", "g-q", 0.4))
    record = _score(_result("q1", groups), "g-q")
    assert record.top1_hit == 0
    assert record.top5_hit == 1
    assert record.top10_hit == 1
    assert record.reciprocal_rank == pytest.approx(1 / 3)
    assert record.first_positive_rank == 3


def test_case3_positive_at_rank_7() -> None:
    groups = [(f"n{i}", f"neg-{i}", 1.0 - i * 0.05) for i in range(1, 7)]
    groups.append(("pos", "g-q", 0.2))
    record = _score(_result("q1", groups), "g-q")
    assert record.top1_hit == 0
    assert record.top5_hit == 0
    assert record.top10_hit == 1
    assert record.reciprocal_rank == pytest.approx(1 / 7)


def test_case4_no_positive() -> None:
    result = _result("q1", [("n1", "neg-a", 0.9), ("n2", "neg-b", 0.8)])
    record = _score(result, "g-q", gallery_positives=0)
    assert record.excluded is True
    assert record.top1_hit == record.top5_hit == record.top10_hit == 0
    assert record.reciprocal_rank == 0.0


def test_case5_multiple_images_same_product_count_as_one() -> None:
    result = _result(
        "q1",
        [
            ("img-a", "g-q", 0.99),
            ("img-b", "g-q", 0.98),
            ("img-c", "g-q", 0.97),
            ("neg", "g-n", 0.1),
        ],
    )
    record = _score(result, "g-q", gallery_positives=3)
    assert record.number_of_positive_candidates == 1
    assert record.first_positive_rank == 1
    assert record.reciprocal_rank == 1.0
    assert record.unique_products_in_ranking == 2


def test_case6_self_image_is_excluded_by_retriever() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ]
    )
    metadata = (
        {"image_id": "q", "product_group": "g-q", "category": "Ring"},
        {"image_id": "pos", "product_group": "g-q", "category": "Ring"},
        {"image_id": "neg", "product_group": "g-n", "category": "Ring"},
    )
    retriever = TopKRetriever(embeddings, metadata)
    result = retriever.retrieve(
        embeddings[0],
        query_id="q",
        k=10,
        exclude_image_id="q",
    )
    assert "q" not in [candidate.image_id for candidate in result.candidates]
    record = score_retrieval_result(
        result,
        query_group_id="g-q",
        query_image_id="q",
        split="train",
        gallery_split="train",
        gallery_positive_images=1,
        gallery_positive_products=1,
        self_match_excluded=True,
    )
    assert record.top1_hit == 1
    assert record.self_match_excluded is True


def test_case6_self_image_in_result_fails_loudly() -> None:
    result = _result(
        "q",
        [("q", "g-q", 1.0), ("pos", "g-q", 0.9)],
    )
    with pytest.raises(EvaluationError, match="self-image exclusion"):
        _score(result, "g-q")


def test_case7_nan_and_inf_scores_fail_loudly() -> None:
    nan_result = _result("q1", [("n1", "neg-a", math.nan)])
    inf_result = _result("q1", [("n1", "neg-a", math.inf)])
    with pytest.raises(EvaluationError, match="Non-finite"):
        _score(nan_result, "g-q")
    with pytest.raises(EvaluationError, match="Non-finite"):
        _score(inf_result, "g-q")


def test_case8_invalid_group_id_fails_loudly() -> None:
    result = _result("q1", [("n1", "neg-a", 0.5)])
    with pytest.raises(EvaluationError, match="query_group_id"):
        score_retrieval_result(
            result,
            query_group_id=" ",
            query_image_id="q1",
            split="train",
            gallery_split="train",
            gallery_positive_images=1,
            gallery_positive_products=1,
        )
    bad = RetrievalResult(
        query_id="q1",
        candidates=(_candidate("n1", "", 0.5, 1),),
    )
    with pytest.raises(EvaluationError, match="product_group"):
        _score(bad, "g-q")


def test_case9_rank_follows_retrieval_order() -> None:
    unordered = RetrievalResult(
        query_id="q1",
        candidates=(
            _candidate("low", "neg-a", 0.1, 1),
            _candidate("high", "g-q", 0.9, 2),
        ),
    )
    with pytest.raises(EvaluationError, match="non-increasing similarity"):
        _score(unordered, "g-q")

    wrong_rank = RetrievalResult(
        query_id="q1",
        candidates=(
            _candidate("a", "neg-a", 0.9, 2),
            _candidate("b", "g-q", 0.8, 1),
        ),
    )
    with pytest.raises(EvaluationError, match="sequential order"):
        _score(wrong_rank, "g-q")


def test_case10_perfect_retrieval_aggregate() -> None:
    records = [
        _score(_result("q1", [("p1", "g1", 0.9), ("n1", "n1", 0.1)]), "g1"),
        _score(_result("q2", [("p2", "g2", 0.8), ("n2", "n2", 0.1)]), "g2"),
    ]
    summary = evaluate_retrieval(records)
    assert summary.top1 == summary.top5 == summary.top10 == summary.mrr == 1.0


def test_leave_one_out_gallery_excludes_self_and_keeps_same_product() -> None:
    gallery = Gallery(
        embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
            ]
        ),
        metadata=(
            {"image_id": "a", "product_group": "g1", "category": "Ring", "split": "train"},
            {"image_id": "b", "product_group": "g1", "category": "Ring", "split": "train"},
            {"image_id": "c", "product_group": "g2", "category": "Necklace", "split": "train"},
        ),
    )
    summary = evaluate_leave_one_out_gallery(gallery, k=10, query_split="train", gallery_split="train")
    query_a = next(record for record in summary.query_records if record.query_id == "a")
    query_c = next(record for record in summary.query_records if record.query_id == "c")
    assert query_a.excluded is False
    assert query_a.top1_hit == 1
    assert query_c.excluded is True
    assert query_c.exclusion_reason == "no_positive_in_gallery"
    assert summary.diagnostics["self_match_exclusions"] == 3


def test_cross_split_protocol_is_rejected() -> None:
    gallery = Gallery(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        metadata=(
            {"image_id": "a", "product_group": "g1", "category": "Ring", "split": "train"},
            {"image_id": "b", "product_group": "g2", "category": "Ring", "split": "train"},
        ),
    )
    with pytest.raises(EvaluationError, match="query_split == gallery_split"):
        evaluate_leave_one_out_gallery(gallery, query_split="valid", gallery_split="train")
