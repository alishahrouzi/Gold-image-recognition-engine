"""Tests for S3.3 metric-learning losses."""

import pytest
import torch

from src.metric_learning.losses import ContrastiveLoss


def test_positive_loss_matches_squared_euclidean_distance() -> None:
    loss = ContrastiveLoss(margin=1.0)
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.5, 0.0]])
    labels = torch.tensor([0])
    assert torch.allclose(loss(a, b, labels), torch.tensor(0.25))


def test_identical_positive_pair_has_zero_loss() -> None:
    loss = ContrastiveLoss()
    x = torch.randn(4, 128)
    labels = torch.zeros(4)
    assert torch.allclose(loss(x, x, labels), torch.tensor(0.0), atol=1e-7)


def test_negative_inside_margin_is_penalized() -> None:
    loss = ContrastiveLoss(margin=1.0)
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.5, 0.0]])
    labels = torch.tensor([1])
    assert torch.allclose(loss(a, b, labels), torch.tensor(0.25))


def test_negative_at_margin_has_zero_loss() -> None:
    loss = ContrastiveLoss(margin=1.0)
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 0.0]])
    labels = torch.tensor([1])
    assert torch.allclose(loss(a, b, labels), torch.tensor(0.0), atol=1e-7)


def test_negative_outside_margin_has_zero_loss() -> None:
    loss = ContrastiveLoss(margin=1.0)
    a = torch.tensor([[2.0, 0.0]])
    b = torch.tensor([[0.0, 0.0]])
    labels = torch.tensor([1])
    assert torch.allclose(loss(a, b, labels), torch.tensor(0.0), atol=1e-7)


def test_mixed_batch_is_mean_of_positive_and_negative_terms() -> None:
    loss = ContrastiveLoss(margin=1.0)
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[0.5, 0.0], [0.0, 0.0]])
    labels = torch.tensor([0, 1])
    assert torch.allclose(loss(a, b, labels), torch.tensor(0.125))


def test_loss_produces_gradients() -> None:
    loss = ContrastiveLoss()
    a = torch.randn(4, 128, requires_grad=True)
    b = torch.randn(4, 128, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    value = loss(a, b, labels)
    value.backward()
    assert value.ndim == 0
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()


@pytest.mark.parametrize("margin", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_margin_is_rejected(margin: float) -> None:
    with pytest.raises(ValueError, match="margin"):
        ContrastiveLoss(margin=margin)


def test_invalid_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="labels"):
        ContrastiveLoss()(torch.randn(2, 4), torch.randn(2, 4), torch.tensor([0, 2]))


def test_non_finite_embeddings_are_rejected() -> None:
    a = torch.tensor([[float("nan"), 0.0]])
    b = torch.tensor([[0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        ContrastiveLoss()(a, b, torch.tensor([0]))


def test_invalid_embedding_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        ContrastiveLoss()(torch.randn(2, 4), torch.randn(2, 5), torch.tensor([0, 1]))
