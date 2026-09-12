"""Tests for S3.11 metric-learning retrieval evaluation."""

from __future__ import annotations

import pytest
import torch

from src.evaluation.errors import EvaluationError
from src.evaluation.metric_learning_retrieval import evaluate_gallery
from src.retrieval.gallery import Gallery


def _gallery() -> Gallery:
    # Two images per product. Query embeddings are close to their partner and
    # separated from the other product, so product-level retrieval is exact.
    metadata = (
        {"product_id": "p1", "category": "Ring", "image": "p1a.jpg", "image_id": "p1a", "product_group": "p1"},
        {"product_id": "p1", "category": "Ring", "image": "p1b.jpg", "image_id": "p1b", "product_group": "p1"},
        {"product_id": "p2", "category": "Bracelet", "image": "p2a.jpg", "image_id": "p2a", "product_group": "p2"},
        {"product_id": "p2", "category": "Bracelet", "image": "p2b.jpg", "image_id": "p2b", "product_group": "p2"},
    )
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=torch.float32,
    )
    return Gallery(embeddings=embeddings, metadata=metadata)


def test_evaluate_gallery_uses_product_level_leave_one_out() -> None:
    result = evaluate_gallery(_gallery())
    assert result.num_queries == 4
    assert result.num_valid_queries == 4
    assert result.num_excluded_queries == 0
    assert result.top1 == pytest.approx(1.0)
    assert result.top5 == pytest.approx(1.0)
    assert result.top10 == pytest.approx(1.0)
    assert result.precision_at_1 == pytest.approx(1.0)
    assert result.precision_at_5 == pytest.approx(0.2)
    assert result.precision_at_10 == pytest.approx(0.1)
    assert result.recall_at_1 == pytest.approx(1.0)
    assert result.recall_at_5 == pytest.approx(1.0)
    assert result.recall_at_10 == pytest.approx(1.0)
    assert result.mrr == pytest.approx(1.0)
    assert result.diagnostics["self_match_exclusions"] == 4


def test_singleton_product_queries_are_excluded() -> None:
    gallery = Gallery(
        embeddings=torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=torch.float32
        ),
        metadata=(
            {"product_id": "p1", "category": "Ring", "image": "a", "image_id": "a"},
            {"product_id": "p1", "category": "Ring", "image": "b", "image_id": "b"},
            {"product_id": "p2", "category": "Ring", "image": "c", "image_id": "c"},
        ),
    )
    result = evaluate_gallery(gallery)
    assert result.num_queries == 3
    assert result.num_valid_queries == 2
    assert result.num_excluded_queries == 1
    assert result.diagnostics["queries_with_zero_positives"] == 1


def test_k_must_be_ten_for_shared_evaluation_ranking() -> None:
    with pytest.raises(EvaluationError, match="k=10"):
        evaluate_gallery(_gallery(), k=5)


def test_invalid_gallery_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        Gallery(
            embeddings=torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
            metadata=(
                {"product_id": "p1", "category": "Ring", "image": "a", "image_id": "a"},
                {"product_id": "p1", "category": "Ring", "image": "b", "image_id": "b"},
            ),
        )
