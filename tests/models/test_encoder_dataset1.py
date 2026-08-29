"""Dataset 1 → preprocessing → DataLoader → CustomCNNEncoder (no training).

Skipped automatically when the Dataset 1 root is unavailable.
Override the root with env var ``ZARGAR_DATASET1_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from data.collate import collate_preprocessed_samples
from data.datasets.unified_dataset import UnifiedDataset
from data.preprocessing import ImagePreprocessor, PreprocessedDataset
from models import CustomCNNEncoder, EmbeddingHead, EncoderWithEmbeddingHead

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
_LEGACY_DATASET_ROOT = Path(
    r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision"
)
_LOCAL_DATASET_ROOT = PROJECT_ROOT.parent / "dataset" / "ai-tool-pool-jewelry-vision"


def _dataset_root() -> Path:
    env = os.environ.get("ZARGAR_DATASET1_ROOT")
    if env:
        return Path(env)
    if _LEGACY_DATASET_ROOT.is_dir():
        return _LEGACY_DATASET_ROOT
    return _LOCAL_DATASET_ROOT


def _dataset_available() -> bool:
    return DEFAULT_MANIFEST.is_file() and (_dataset_root() / "train").is_dir()


pytestmark = pytest.mark.skipif(
    not _dataset_available(),
    reason="Dataset 1 root or manifest not available",
)


def test_dataset1_preprocessed_batch_through_encoder() -> None:
    dataset_root = _dataset_root()
    base = UnifiedDataset(
        DEFAULT_MANIFEST,
        dataset_root=dataset_root,
        split="valid",
        validate_files=False,
    )
    assert len(base) >= 4

    dataset = PreprocessedDataset(Subset(base, list(range(4))), ImagePreprocessor(), role="valid")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
    )
    batch = next(iter(loader))
    images = batch["image"]
    assert images.shape == (4, 3, 224, 224)
    assert images.dtype == torch.float32

    encoder = CustomCNNEncoder()
    encoder.eval()
    with torch.no_grad():
        features = encoder.encode_features(images)

    assert features.shape == (4, encoder.feature_dim)
    assert encoder.feature_dim == 256
    assert features.dtype == torch.float32
    assert features.shape[0] == images.shape[0]
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()
    assert "group_id" in batch
    assert "category_id" in batch
    assert encoder(images).shape[0] == len(batch["group_id"])


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("embedding_dim", [128, 256])
def test_dataset1_encoder_then_embedding_head(batch_size: int, embedding_dim: int) -> None:
    dataset_root = _dataset_root()
    base = UnifiedDataset(
        DEFAULT_MANIFEST,
        dataset_root=dataset_root,
        split="valid",
        validate_files=False,
    )
    assert len(base) >= batch_size
    dataset = PreprocessedDataset(
        Subset(base, list(range(batch_size))),
        ImagePreprocessor(),
        role="valid",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
    )
    batch = next(iter(loader))
    images = batch["image"]
    assert images.shape == (batch_size, 3, 224, 224)
    assert images.dtype == torch.float32

    encoder = CustomCNNEncoder()
    head = EmbeddingHead(embedding_dim=embedding_dim)
    model = EncoderWithEmbeddingHead(encoder, head)
    model.eval()
    with torch.no_grad():
        features = model.encode_features(images)
        embeddings = model(images)

    assert features.shape == (batch_size, 256)
    assert embeddings.shape == (batch_size, embedding_dim)
    assert embeddings.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert torch.isfinite(embeddings).all()
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dataset1_encoder_head_cuda() -> None:
    device = torch.device("cuda")
    dataset_root = _dataset_root()
    base = UnifiedDataset(
        DEFAULT_MANIFEST,
        dataset_root=dataset_root,
        split="valid",
        validate_files=False,
    )
    dataset = PreprocessedDataset(Subset(base, list(range(8))), ImagePreprocessor(), role="valid")
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
    )
    images = next(iter(loader))["image"]
    model = EncoderWithEmbeddingHead(CustomCNNEncoder(), EmbeddingHead(embedding_dim=128)).to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model(images.to(device))
    assert embeddings.device.type == "cuda"
    assert embeddings.shape == (8, 128)
    assert torch.isfinite(embeddings).all()
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
