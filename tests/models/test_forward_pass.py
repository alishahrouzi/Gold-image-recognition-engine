"""S2.4 forward-pass validation (preprocessor → encoder → embedding)."""

from __future__ import annotations

import pytest
import torch

from models import (
    CustomCNNEncoder,
    EmbeddingHead,
    ENCODER_FEATURE_DIM,
    EXPECTED_IMAGE_SHAPE,
    L2_ATOL,
    REQUIRED_BATCH_SIZES,
    collect_l2_stats,
    preprocessed_images_from_preprocessor,
    run_forward_pass,
    validate_embeddings,
    validate_encoder_features,
    validate_input_tensor,
)
from models.forward_pass import make_synthetic_images

EMBEDDING_DIMS = (128, 256)


@pytest.mark.parametrize("batch_size", REQUIRED_BATCH_SIZES)
def test_preprocessor_input_shape(batch_size: int) -> None:
    images = preprocessed_images_from_preprocessor(batch_size)
    assert images.shape == (batch_size, *EXPECTED_IMAGE_SHAPE)
    assert images.dtype == torch.float32
    assert validate_input_tensor(images, batch_size) == []


@pytest.mark.parametrize("batch_size", REQUIRED_BATCH_SIZES)
def test_encoder_output_shape_dtype_finite(batch_size: int) -> None:
    images = preprocessed_images_from_preprocessor(batch_size)
    encoder = CustomCNNEncoder()
    encoder.eval()
    with torch.no_grad():
        features = encoder.encode_features(images)
    assert features.ndim == 2
    assert features.shape == (batch_size, ENCODER_FEATURE_DIM)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()
    assert validate_encoder_features(features, batch_size) == []
    # Raw encoder features are not L2-validated; that rule applies to embeddings only.


@pytest.mark.parametrize("embedding_dim", EMBEDDING_DIMS)
@pytest.mark.parametrize("batch_size", REQUIRED_BATCH_SIZES)
def test_embedding_head_shape_and_l2(batch_size: int, embedding_dim: int) -> None:
    images = preprocessed_images_from_preprocessor(batch_size)
    encoder = CustomCNNEncoder()
    head = EmbeddingHead(embedding_dim=embedding_dim)
    record = run_forward_pass(
        images,
        encoder=encoder,
        head=head,
        device=torch.device("cpu"),
        split="preprocessor",
        source="image_preprocessor",
    )
    assert record["passed"], record["failures"]
    assert record["encoder_features"]["shape"] == [batch_size, 256]
    assert record["embedding"]["shape"] == [batch_size, embedding_dim]
    assert record["embedding"]["dtype"] == "torch.float32"
    assert not record["embedding"]["has_nan"]
    assert not record["embedding"]["has_inf"]
    assert record["embedding"]["max_l2_error"] < L2_ATOL


def test_cpu_path() -> None:
    images = preprocessed_images_from_preprocessor(8)
    record = run_forward_pass(
        images,
        encoder=CustomCNNEncoder(),
        head=EmbeddingHead(embedding_dim=128),
        device=torch.device("cpu"),
        split="preprocessor",
        source="image_preprocessor",
    )
    assert record["device"] == "cpu"
    assert record["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_path() -> None:
    device = torch.device("cuda")
    images = preprocessed_images_from_preprocessor(8)
    for dim in EMBEDDING_DIMS:
        record = run_forward_pass(
            images,
            encoder=CustomCNNEncoder(),
            head=EmbeddingHead(embedding_dim=dim),
            device=device,
            split="preprocessor",
            source="image_preprocessor",
        )
        assert record["passed"], record["failures"]
        assert record["device"].startswith("cuda")


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("zero", {"fill": 0.0}),
        ("constant", {"fill": 0.25}),
        ("tiny", {"scale": 1e-12}),
    ],
)
def test_finite_edge_inputs_do_not_produce_nan_inf(kind: str, kwargs: dict) -> None:
    images = make_synthetic_images(8, **kwargs)
    assert torch.isfinite(images).all()
    for dim in EMBEDDING_DIMS:
        record = run_forward_pass(
            images,
            encoder=CustomCNNEncoder(),
            head=EmbeddingHead(embedding_dim=dim),
            device=torch.device("cpu"),
            split=f"edge_{kind}",
            source="synthetic_finite",
        )
        assert record["passed"], record["failures"]
        assert not record["encoder_features"]["has_nan"]
        assert not record["encoder_features"]["has_inf"]
        assert not record["embedding"]["has_nan"]
        assert not record["embedding"]["has_inf"]


def test_l2_stats_and_validator() -> None:
    head = EmbeddingHead(embedding_dim=256)
    head.eval()
    with torch.no_grad():
        embeddings = head(torch.randn(4, 256))
    stats = collect_l2_stats(embeddings)
    assert stats["max_l2_error"] < L2_ATOL
    assert validate_embeddings(embeddings, 4, 256) == []
    bad = embeddings.clone()
    bad[0] = 0
    assert validate_embeddings(bad, 4, 256)


def test_input_and_encoder_validators_report_shape_errors() -> None:
    wrong = torch.zeros(2, 3, 64, 64)
    assert validate_input_tensor(wrong, 2)
    features = torch.zeros(2, 128)
    assert validate_encoder_features(features, 2)
