"""Custom CNN encoder contract tests."""

from __future__ import annotations

import inspect

import pytest
import torch

from models import (
    CustomCNNEncoder,
    Encoder,
    EncoderConfig,
    EncoderInputError,
    count_parameters,
    estimate_parameter_bytes,
)

BATCH_SIZES = (1, 2, 8, 32)


def _random_batch(batch_size: int, config: EncoderConfig | None = None) -> torch.Tensor:
    config = config or EncoderConfig()
    return torch.randn(
        batch_size,
        config.input_channels,
        config.input_height,
        config.input_width,
        dtype=torch.float32,
    )


def test_encoder_instantiates() -> None:
    encoder = CustomCNNEncoder()
    assert isinstance(encoder, Encoder)
    assert isinstance(encoder, torch.nn.Module)
    assert encoder.embedding_dim == EncoderConfig().embedding_dim


def test_forward_output_shape_and_dtype() -> None:
    encoder = CustomCNNEncoder()
    encoder.eval()
    for batch_size in BATCH_SIZES:
        images = _random_batch(batch_size)
        embeddings = encoder(images)
        assert embeddings.shape == (batch_size, encoder.embedding_dim)
        assert embeddings.dtype == torch.float32
        assert embeddings.ndim == 2


def test_batch_dimension_preserved() -> None:
    encoder = CustomCNNEncoder()
    images = _random_batch(8)
    embeddings = encoder(images)
    assert embeddings.shape[0] == images.shape[0]


@pytest.mark.parametrize("embedding_dim", [64, 128, 256])
def test_configurable_embedding_dimension(embedding_dim: int) -> None:
    encoder = CustomCNNEncoder(EncoderConfig(embedding_dim=embedding_dim))
    embeddings = encoder(_random_batch(4, encoder.config))
    assert embeddings.shape == (4, embedding_dim)
    assert encoder.embedding_dim == embedding_dim


def test_cpu_execution() -> None:
    encoder = CustomCNNEncoder()
    images = _random_batch(2)
    embeddings = encoder(images)
    assert embeddings.device.type == "cpu"
    assert images.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_execution() -> None:
    device = torch.device("cuda")
    encoder = CustomCNNEncoder().to(device)
    images = _random_batch(2).to(device)
    embeddings = encoder(images)
    assert embeddings.device.type == "cuda"
    assert embeddings.shape == (2, encoder.embedding_dim)


def test_no_nan_or_inf() -> None:
    encoder = CustomCNNEncoder()
    embeddings = encoder(_random_batch(8))
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()


def test_eval_mode_forward_consistency() -> None:
    encoder = CustomCNNEncoder()
    encoder.eval()
    images = _random_batch(8)
    with torch.no_grad():
        first = encoder(images)
        second = encoder(images)
    assert torch.equal(first, second)


def test_trainable_parameters() -> None:
    encoder = CustomCNNEncoder()
    n_trainable = count_parameters(encoder, trainable_only=True)
    n_all = count_parameters(encoder, trainable_only=False)
    assert n_trainable > 0
    assert n_trainable == n_all
    assert any(parameter.requires_grad for parameter in encoder.parameters())
    assert estimate_parameter_bytes(encoder) > 0


@pytest.mark.parametrize(
    "shape",
    [
        (3, 224, 224),
        (8, 1, 224, 224),
        (8, 3, 128, 128),
        (8, 3, 224, 200),
        (0, 3, 224, 224),
        (8, 3, 224, 224, 1),
    ],
)
def test_invalid_input_shape(shape: tuple[int, ...]) -> None:
    encoder = CustomCNNEncoder()
    images = torch.randn(*shape, dtype=torch.float32)
    with pytest.raises(EncoderInputError):
        encoder(images)


def test_invalid_input_type_and_dtype() -> None:
    encoder = CustomCNNEncoder()
    with pytest.raises(EncoderInputError):
        encoder([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(EncoderInputError):
        encoder(torch.randint(0, 255, (2, 3, 224, 224)))


def test_does_not_accept_metadata_or_ids() -> None:
    signature = inspect.signature(CustomCNNEncoder.forward)
    assert list(signature.parameters) == ["self", "x"]
    encoder = CustomCNNEncoder()
    images = _random_batch(2)
    with pytest.raises(TypeError):
        encoder(images, group_id=["g1", "g2"])  # type: ignore[misc]
    with pytest.raises(TypeError):
        encoder(images, category_id=[0, 1])  # type: ignore[misc]


def test_does_not_modify_input_inplace() -> None:
    encoder = CustomCNNEncoder()
    images = _random_batch(4)
    original = images.clone()
    encoder(images)
    assert torch.equal(images, original)


def test_encode_alias_matches_forward() -> None:
    encoder = CustomCNNEncoder()
    encoder.eval()
    images = _random_batch(2)
    with torch.no_grad():
        assert torch.equal(encoder.encode(images), encoder.forward(images))


def test_no_hardcoded_cuda_in_construction() -> None:
    encoder = CustomCNNEncoder()
    for parameter in encoder.parameters():
        assert parameter.device.type == "cpu"
