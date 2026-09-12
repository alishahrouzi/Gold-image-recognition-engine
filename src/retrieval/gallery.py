"""Gallery construction and persistence for retrieval models.

The gallery is image-level: each row represents one catalog image and stores
its product identity, category, image path, and normalized embedding. Product-
level aggregation remains a downstream retrieval concern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.collate import collate_preprocessed_samples
from src.data.datasets import UnifiedDataset
from src.data.preprocessing import ImagePreprocessor, build_preprocessed_dataset
from src.training.seed import make_dataloader_generator, seed_worker

from .embedding import EmbeddingExtractor


@dataclass(frozen=True)
class Gallery:
    """Persistable image-level retrieval gallery."""

    embeddings: Tensor
    metadata: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("Gallery embeddings must have shape [N, D].")
        if self.embeddings.shape[0] != len(self.metadata):
            raise ValueError("Gallery embeddings and metadata must have the same number of rows.")
        if self.embeddings.numel() and not torch.isfinite(self.embeddings).all():
            raise ValueError("Gallery embeddings contain non-finite values.")
        required = {"product_id", "category", "image"}
        for index, item in enumerate(self.metadata):
            if not required.issubset(item):
                missing = sorted(required - set(item))
                raise ValueError(f"Gallery metadata row {index} is missing fields: {missing}.")

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def save(self, embedding_path: str | Path, metadata_path: str | Path) -> None:
        embedding_path = Path(embedding_path)
        metadata_path = Path(metadata_path)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.embeddings.cpu(), embedding_path)
        metadata_path.write_text(
            json.dumps(list(self.metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, embedding_path: str | Path, metadata_path: str | Path) -> "Gallery":
        embeddings = torch.load(Path(embedding_path), map_location="cpu", weights_only=True)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            raise ValueError("Gallery metadata must be a JSON list.")
        return cls(embeddings=embeddings.float(), metadata=tuple(metadata))


class GalleryBuilder:
    """Build a deterministic image-level gallery from an existing dataset split."""

    def __init__(self, extractor: EmbeddingExtractor) -> None:
        self.extractor = extractor

    def build(self, loader: DataLoader) -> Gallery:
        embeddings: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        for batch in loader:
            batch_embeddings = self.extractor.extract(batch["image"])
            embeddings.append(batch_embeddings)
            for i, image_id in enumerate(batch["image_id"]):
                product_id = batch["group_id"][i]
                image_path = batch["image_path"][i]
                metadata.append({
                    "product_id": product_id,
                    "category": batch["category"][i],
                    "image": image_path,
                    "image_id": image_id,
                    "product_group": product_id,
                    "image_path": image_path,
                    "category_id": int(batch["category_id"][i]),
                    "split": batch["split"][i],
                    "source": batch["source"][i],
                })
        if not embeddings:
            raise ValueError("Cannot build a gallery from an empty DataLoader.")
        return Gallery(torch.cat(embeddings, dim=0), tuple(metadata))


def build_gallery_loader(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    split: str = "train",
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create the existing project's preprocessing/DataLoader stack for retrieval."""
    raw = UnifiedDataset(manifest_path, dataset_root=dataset_root, split=split)
    data = build_preprocessed_dataset(raw, role="valid", preprocessor=ImagePreprocessor())
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(seed),
        collate_fn=collate_preprocessed_samples,
    )
