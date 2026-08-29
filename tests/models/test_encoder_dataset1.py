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
from models import CustomCNNEncoder

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
        embeddings = encoder(images)

    assert embeddings.shape == (4, encoder.embedding_dim)
    assert embeddings.dtype == torch.float32
    assert embeddings.shape[0] == images.shape[0]
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()
    assert "group_id" in batch
    assert "category_id" in batch
    assert encoder(images).shape[0] == len(batch["group_id"])
