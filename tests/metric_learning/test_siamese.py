"""S3.1 Siamese architecture contract tests."""

from __future__ import annotations

import pytest
import torch

from models import CustomCNNEncoder, EmbeddingHead, EmbeddingHeadConfig, EncoderConfig
from metric_learning import SiameseNetwork

BATCH_SIZES = (1, 4, 16)
NORM_ATOL = 1e-5


def _images(batch_size: int) -> torch.Tensor:
    return torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)


def test_default_architecture_contract() -> None:
    model = SiameseNetwork()

    assert isinstance(model.backbone, torch.nn.Module)
    assert model.embedding_dim == 128
    assert model.feature_dim == 256
    assert model.backbone.encoder.feature_dim == 256
    assert model.backbone.head.embedding_dim == 128


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_forward_shapes_finite_and_unit_norm(batch_size: int) -> None:
    model = SiameseNetwork()
    model.eval()
    image_a = _images(batch_size)
    image_b = _images(batch_size)

    with torch.no_grad():
        embedding_a, embedding_b = model(image_a, image_b)

    assert embedding_a.shape == (batch_size, 128)
    assert embedding_b.shape == (batch_size, 128)
    assert embedding_a.dtype == torch.float32
    assert embedding_b.dtype == torch.float32
    assert torch.isfinite(embedding_a).all()
    assert torch.isfinite(embedding_b).all()
    assert torch.allclose(
        torch.linalg.vector_norm(embedding_a, ord=2, dim=1),
        torch.ones(batch_size),
        atol=NORM_ATOL,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(embedding_b, ord=2, dim=1),
        torch.ones(batch_size),
        atol=NORM_ATOL,
    )


def test_single_shared_backbone_instance() -> None:
    model = SiameseNetwork()

    assert model.backbone is model.backbone
    assert model.backbone.encoder is not model.backbone.head
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    assert len(parameter_ids) == len(set(parameter_ids))


def test_both_paths_use_the_same_parameters() -> None:
    model = SiameseNetwork()
    model.eval()
    image = _images(2)

    with torch.no_grad():
        embedding_a, embedding_b = model(image, image)

    assert torch.equal(embedding_a, embedding_b)


def test_forward_matches_independent_shared_backbone_calls() -> None:
    model = SiameseNetwork()
    model.eval()
    image_a = _images(3)
    image_b = _images(3)

    with torch.no_grad():
        expected_a = model.backbone(image_a)
        expected_b = model.backbone(image_b)
        actual_a, actual_b = model(image_a, image_b)

    assert torch.equal(actual_a, expected_a)
    assert torch.equal(actual_b, expected_b)


def test_gradients_accumulate_through_both_siamese_paths() -> None:
    model = SiameseNetwork()
    model.train()
    image_a = _images(1)
    image_b = _images(1)

    embedding_a, embedding_b = model(image_a, image_b)
    loss = embedding_a.sum() + embedding_b.sum()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_encode_matches_first_siamese_path() -> None:
    model = SiameseNetwork()
    model.eval()
    image = _images(2)
    other = _images(2)

    with torch.no_grad():
        encoded = model.encode(image)
        embedding_a, _ = model(image, other)

    assert torch.equal(encoded, embedding_a)


def test_custom_encoder_and_head_are_reused() -> None:
    encoder = CustomCNNEncoder(
        EncoderConfig(block_channels=(16, 32, 64, 128))
    )
    head = EmbeddingHead(
        config=EmbeddingHeadConfig(feature_dim=128, embedding_dim=128)
    )
    model = SiameseNetwork(encoder=encoder, head=head)

    assert model.backbone.encoder is encoder
    assert model.backbone.head is head
    assert model.feature_dim == 128
    assert model.embedding_dim == 128

    model.eval()
    with torch.no_grad():
        output_a, output_b = model(_images(2), _images(2))
    assert output_a.shape == (2, 128)
    assert output_b.shape == (2, 128)


def test_invalid_pair_input_is_rejected_by_shared_backbone() -> None:
    model = SiameseNetwork()
    valid = _images(2)
    invalid = torch.randn(2, 3, 128, 128, dtype=torch.float32)

    with pytest.raises(Exception):
        model(valid, invalid)


def test_cpu_execution() -> None:
    model = SiameseNetwork().cpu().eval()
    with torch.no_grad():
        embedding_a, embedding_b = model(_images(2), _images(2))
    assert embedding_a.device.type == "cpu"
    assert embedding_b.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_execution() -> None:
    model = SiameseNetwork().cuda().eval()
    with torch.no_grad():
        embedding_a, embedding_b = model(_images(2).cuda(), _images(2).cuda())
    assert embedding_a.device.type == "cuda"
    assert embedding_b.device.type == "cuda"
    assert torch.isfinite(embedding_a).all()
    assert torch.isfinite(embedding_b).all()
