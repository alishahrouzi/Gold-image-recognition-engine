from __future__ import annotations

import pytest
import torch

from metric_learning.distance import CosineDistance, EuclideanDistance


def test_cosine_distance_identical_orthogonal_and_opposite() -> None:
    metric = CosineDistance()
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    values = metric(a, b)
    assert torch.allclose(values, torch.tensor([0.0, 1.0, 2.0]), atol=1e-6)


def test_euclidean_distance_known_pairs() -> None:
    metric = EuclideanDistance()
    a = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    b = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    values = metric(a, b)
    assert torch.allclose(values, torch.tensor([0.0, 1.0, 2**0.5]), atol=1e-6)


def test_pairwise_shape_and_batch_support() -> None:
    a = torch.randn(16, 128)
    b = torch.randn(16, 128)
    assert CosineDistance()(a, b).shape == (16,)
    assert EuclideanDistance()(a, b).shape == (16,)


def test_matrix_shape_and_diagonal() -> None:
    x = torch.nn.functional.normalize(torch.randn(4, 128), p=2, dim=1)
    y = torch.nn.functional.normalize(torch.randn(7, 128), p=2, dim=1)
    assert CosineDistance().matrix(x, y).shape == (4, 7)
    assert EuclideanDistance().matrix(x, y).shape == (4, 7)

    self_distance = CosineDistance().matrix(x, x)
    self_euclidean = EuclideanDistance().matrix(x, x)
    assert torch.allclose(torch.diag(self_distance), torch.zeros(4), atol=1e-6)
    assert torch.allclose(torch.diag(self_euclidean), torch.zeros(4), atol=1e-6)


def test_normalized_embedding_geometry() -> None:
    x = torch.nn.functional.normalize(torch.randn(8, 128), p=2, dim=1)
    y = torch.nn.functional.normalize(torch.randn(8, 128), p=2, dim=1)
    cosine_distance = CosineDistance()(x, y)
    euclidean_distance = EuclideanDistance()(x, y)
    assert torch.allclose(
        euclidean_distance.square(), 2.0 * cosine_distance, atol=1e-5
    )


@pytest.mark.parametrize(
    "metric",
    [CosineDistance(), EuclideanDistance()],
)
def test_pairwise_rejects_invalid_shapes(metric) -> None:
    with pytest.raises(ValueError):
        metric(torch.randn(4, 8), torch.randn(4, 7))
    with pytest.raises(ValueError):
        metric(torch.randn(4, 8, 1), torch.randn(4, 8, 1))
    with pytest.raises(ValueError):
        metric(torch.empty(0, 8), torch.empty(0, 8))


@pytest.mark.parametrize(
    "metric",
    [CosineDistance(), EuclideanDistance()],
)
def test_matrix_rejects_invalid_shapes(metric) -> None:
    with pytest.raises(ValueError):
        metric.matrix(torch.randn(4, 8), torch.randn(7, 7))
    with pytest.raises(ValueError):
        metric.matrix(torch.randn(4, 8, 1), torch.randn(7, 8, 1))
    with pytest.raises(ValueError):
        metric.matrix(torch.empty(0, 8), torch.randn(7, 8))
