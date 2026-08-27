"""Encoder consumes existing preprocessing batches without metadata."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.collate import collate_preprocessed_samples
from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1
from data.datasets.unified_dataset import UnifiedDataset
from data.preprocessing import ImagePreprocessor, PreprocessedDataset
from models import CustomCNNEncoder
from tests.data.helpers import write_manifest, write_rgb_image


def test_encoder_accepts_preprocessed_dataloader_batch(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        image_path = write_rgb_image(tmp_path / f"{index}.jpg", size=(80, 60))
        rows.append(
            {
                "image_id": f"img_{index}",
                "image_path": str(image_path),
                "group_id": f"g_{index}",
                "category": "Ring",
                "category_id": CATEGORY_TO_ID["Ring"],
                "split": "valid",
                "source": SOURCE_DATASET1,
            }
        )
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    dataset = PreprocessedDataset(
        UnifiedDataset(manifest, split="valid"),
        ImagePreprocessor(),
        role="valid",
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_preprocessed_samples,
    )
    batch = next(iter(loader))
    images = batch["image"]
    assert images.shape == (4, 3, 224, 224)
    assert images.dtype == torch.float32

    encoder = CustomCNNEncoder()
    embeddings = encoder(images)
    assert embeddings.shape == (4, encoder.embedding_dim)
    assert embeddings.dtype == torch.float32
    # Metadata stays on the batch; the encoder never consumes it.
    assert "group_id" in batch
    assert "category_id" in batch
