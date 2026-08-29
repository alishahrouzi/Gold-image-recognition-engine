"""S2.4 Dataset 1 integration: manifest → preprocess → encoder → embedding.

Skipped automatically when the Dataset 1 root is unavailable.
Override the root with env var ``ZARGAR_DATASET1_ROOT``.
Valid and test use deterministic preprocessing. Train uses role='train'
without an AugmentationConfig (augmentation off for deterministic checks).
"""

from __future__ import annotations

import pytest
import torch

from models import (
    CustomCNNEncoder,
    EmbeddingHead,
    L2_ATOL,
    REQUIRED_BATCH_SIZES,
    dataset1_available,
    load_dataset1_image_batch,
    run_forward_pass,
)

pytestmark = pytest.mark.skipif(
    not dataset1_available(),
    reason="Dataset 1 root or manifest not available",
)

SPLITS = ("train", "valid", "test")
EMBEDDING_DIMS = (128, 256)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("batch_size", REQUIRED_BATCH_SIZES)
@pytest.mark.parametrize("embedding_dim", EMBEDDING_DIMS)
def test_dataset1_forward_path(split: str, batch_size: int, embedding_dim: int) -> None:
    images = load_dataset1_image_batch(split, batch_size)
    assert images.shape == (batch_size, 3, 224, 224)
    assert images.dtype == torch.float32
    record = run_forward_pass(
        images,
        encoder=CustomCNNEncoder(),
        head=EmbeddingHead(embedding_dim=embedding_dim),
        device=torch.device("cpu"),
        split=split,
        source="dataset1",
    )
    assert record["passed"], record["failures"]
    assert record["encoder_features"]["shape"] == [batch_size, 256]
    assert record["embedding"]["shape"] == [batch_size, embedding_dim]
    assert not record["encoder_features"]["has_nan"]
    assert not record["encoder_features"]["has_inf"]
    assert not record["embedding"]["has_nan"]
    assert not record["embedding"]["has_inf"]
    assert record["embedding"]["max_l2_error"] < L2_ATOL
    assert record["sample_count"] == batch_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("split", SPLITS)
def test_dataset1_cuda_when_available(split: str) -> None:
    images = load_dataset1_image_batch(split, 8)
    record = run_forward_pass(
        images,
        encoder=CustomCNNEncoder(),
        head=EmbeddingHead(embedding_dim=256),
        device=torch.device("cuda"),
        split=split,
        source="dataset1",
    )
    assert record["passed"], record["failures"]
    assert record["device"].startswith("cuda")
