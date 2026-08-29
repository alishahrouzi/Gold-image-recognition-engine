"""S2.3 EmbeddingHead contract tests."""

from __future__ import annotations

import inspect

import pytest
import torch

from models import (
    CustomCNNEncoder,
    EmbeddingHead,
    EmbeddingHeadConfig,
    EmbeddingHeadConfigError,
    EmbeddingHeadInputError,
    EncoderWithEmbeddingHead,
    count_parameters,
    estimate_parameter_bytes,
)

BATCH_SIZES = (1, 2, 8, 16, 32)
EMBEDDING_DIMS = (128, 256)
NORM_ATOL = 1e-5


def _features(batch_size: int, feature_dim: int = 256) -> torch.Tensor:
    return torch.randn(batch_size, feature_dim, dtype=torch.float32)


def test_head_instantiates_with_embedding_dim() -> None:
    head_128 = EmbeddingHead(embedding_dim=128)
    head_256 = EmbeddingHead(embedding_dim=256)
    assert head_128.embedding_dim == 128
    assert head_256.embedding_dim == 256
    assert head_128.feature_dim == 256
    assert isinstance(head_128, torch.nn.Module)


def test_head_accepts_config_object() -> None:
    config = EmbeddingHeadConfig(embedding_dim=256, feature_dim=256)
    head = EmbeddingHead(config)
    assert head.config is config
    head_kw = EmbeddingHead(config=config)
    assert head_kw.embedding_dim == 256


@pytest.mark.parametrize("embedding_dim", EMBEDDING_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_output_shape_dtype_finite_and_unit_norm(embedding_dim: int, batch_size: int) -> None:
    head = EmbeddingHead(embedding_dim=embedding_dim)
    head.eval()
    embeddings = head(_features(batch_size))
    assert embeddings.shape == (batch_size, embedding_dim)
    assert embeddings.dtype == torch.float32
    assert torch.isfinite(embeddings).all()
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=NORM_ATOL)


def test_zero_features_remain_finite() -> None:
    head = EmbeddingHead(embedding_dim=128)
    zeros = torch.zeros(4, 256, dtype=torch.float32)
    embeddings = head(zeros)
    assert torch.isfinite(embeddings).all()
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()


def test_cpu_execution() -> None:
    head = EmbeddingHead(embedding_dim=128)
    features = _features(2)
    embeddings = head(features)
    assert embeddings.device.type == "cpu"
    assert features.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_execution_and_cpu_shape_consistency() -> None:
    device = torch.device("cuda")
    head = EmbeddingHead(embedding_dim=128)
    features = _features(4)
    cpu_out = head(features)
    head_cuda = EmbeddingHead(embedding_dim=128).to(device)
    # Copy CPU head weights so this is a device check, not a random-init check.
    head_cuda.load_state_dict(head.state_dict())
    cuda_out = head_cuda(features.to(device))
    assert cuda_out.device.type == "cuda"
    assert cuda_out.shape == cpu_out.shape == (4, 128)
    assert torch.isfinite(cuda_out).all()
    norms = torch.linalg.vector_norm(cuda_out, ord=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=NORM_ATOL)
    assert torch.allclose(cuda_out.cpu(), cpu_out, atol=1e-5)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.randn(2, 3, 224, 224),
        torch.randn(8),
        torch.randn(2, 128),
        torch.randn(2, 255),
        torch.randn(2, 257),
        torch.randint(0, 255, (2, 256)),
    ],
)
def test_invalid_inputs_rejected(tensor: torch.Tensor) -> None:
    head = EmbeddingHead(embedding_dim=128)
    with pytest.raises(EmbeddingHeadInputError):
        head(tensor)


def test_invalid_input_type() -> None:
    head = EmbeddingHead(embedding_dim=128)
    with pytest.raises(EmbeddingHeadInputError):
        head([1.0] * 256)  # type: ignore[arg-type]


def test_empty_batch_rejected() -> None:
    head = EmbeddingHead(embedding_dim=128)
    with pytest.raises(EmbeddingHeadInputError):
        head(torch.zeros(0, 256, dtype=torch.float32))


def test_does_not_accept_metadata() -> None:
    signature = inspect.signature(EmbeddingHead.forward)
    assert list(signature.parameters) == ["self", "features"]
    head = EmbeddingHead(embedding_dim=128)
    with pytest.raises(TypeError):
        head(_features(2), group_id=["g1", "g2"])  # type: ignore[misc]


def test_does_not_modify_input_inplace() -> None:
    head = EmbeddingHead(embedding_dim=256)
    features = _features(4)
    original = features.clone()
    head(features)
    assert torch.equal(features, original)


def test_no_hardcoded_cuda_in_construction() -> None:
    head = EmbeddingHead(embedding_dim=128)
    for parameter in head.parameters():
        assert parameter.device.type == "cpu"


def test_single_linear_projection_only() -> None:
    head = EmbeddingHead(embedding_dim=128)
    linears = [module for module in head.modules() if isinstance(module, torch.nn.Linear)]
    assert len(linears) == 1
    assert linears[0].in_features == 256
    assert linears[0].out_features == 128
    forbidden = (
        torch.nn.BatchNorm1d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.Softmax,
        torch.nn.CrossEntropyLoss,
    )
    assert not any(isinstance(module, forbidden) for module in head.modules())


def test_eval_mode_forward_consistency() -> None:
    head = EmbeddingHead(embedding_dim=128)
    head.eval()
    features = _features(8)
    with torch.no_grad():
        first = head(features)
        second = head(features)
    assert torch.equal(first, second)


def test_parameter_counts() -> None:
    head_128 = EmbeddingHead(embedding_dim=128)
    head_256 = EmbeddingHead(embedding_dim=256)
    assert count_parameters(head_128, trainable_only=False) == 256 * 128 + 128
    assert count_parameters(head_256, trainable_only=False) == 256 * 256 + 256
    assert estimate_parameter_bytes(head_128) == count_parameters(head_128, trainable_only=False) * 4


def test_mismatched_feature_dim_rejected() -> None:
    head = EmbeddingHead(config=EmbeddingHeadConfig(feature_dim=128, embedding_dim=128))
    with pytest.raises(EmbeddingHeadInputError):
        head(_features(2, feature_dim=256))


def test_encoder_plus_head_composition() -> None:
    encoder = CustomCNNEncoder()
    head = EmbeddingHead(embedding_dim=128)
    model = EncoderWithEmbeddingHead(encoder, head)
    images = torch.randn(4, 3, 224, 224, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        features = model.encode_features(images)
        embeddings = model(images)
        embeddings_from_features = head(features)
    assert features.shape == (4, 256)
    assert embeddings.shape == (4, 128)
    assert torch.equal(embeddings, embeddings_from_features)
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=NORM_ATOL)


def test_encoder_head_feature_dim_mismatch() -> None:
    encoder = CustomCNNEncoder()
    head = EmbeddingHead(config=EmbeddingHeadConfig(feature_dim=128, embedding_dim=128))
    with pytest.raises(EmbeddingHeadConfigError):
        EncoderWithEmbeddingHead(encoder, head)
