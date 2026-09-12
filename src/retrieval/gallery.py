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
    """Persistable image-level retrieval gallery.

    S3.7 metadata uses ``product_id``, ``category``, and ``image`` as its
    canonical fields. Legacy S2.7 metadata using ``product_group`` and
    ``image_path`` remains accepted for compatibility.
    """

    embeddings: Tensor
    metadata: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("Gallery embeddings must have shape [N, D].")
        if self.embeddings.shape[0] != len(self.metadata):
            raise ValueError("Gallery embeddings and metadata must have the same number of rows.")
        if self.embeddings.numel() and not torch.isfinite(self.embeddings).all():
            raise ValueError("Gallery embeddings contain non-finite values.")

        for index, item in enumerate(self.metadata):
            # S3.7 canonical contract.
            canonical_missing = {"product_id", "category", "image"} - set(item)
            if not canonical_missing:
                continue

            # S2.7 compatibility contract. Do not mutate metadata here: the
            # Gallery object should preserve exactly what its caller supplied.
            legacy_missing = {"product_group", "category"} - set(item)
            if not legacy_missing and ("image_path" in item or "image_id" in item):
                continue

            raise ValueError(
                f"Gallery metadata row {index} is missing fields: "
                f"{sorted(canonical_missing)}."
            )

    @staticmethod
    def normalize_metadata(metadata: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        """Normalize legacy S2.7 metadata to the S3.7 gallery contract.

        Canonical S3.7 rows are returned unchanged so save/load is lossless.
        Only rows that actually use the legacy ``product_group``/``image_path``
        shape receive compatibility-field normalization.
        """
        normalized: list[dict[str, Any]] = []
        for item in metadata:
            row = dict(item)
            is_canonical = {"product_id", "category", "image"}.issubset(row)
            if is_canonical:
                normalized.append(row)
                continue

            if "product_id" not in row and "product_group" in row:
                row["product_id"] = row["product_group"]
            if "image" not in row:
                row["image"] = row.get("image_path", row.get("image_id", ""))
            if "product_group" not in row and "product_id" in row:
                row["product_group"] = row["product_id"]
            if "image_path" not in row and "image" in row:
                row["image_path"] = row["image"]
            normalized.append(row)
        return tuple(normalized)

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
        return cls(embeddings=embeddings.float(), metadata=cls.normalize_metadata(metadata))


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
                metadata.append(
                    {
                        "product_id": product_id,
                        "category": batch["category"][i],
                        "image": image_path,
                        "image_id": image_id,
                        "product_group": product_id,
                        "image_path": image_path,
                        "category_id": int(batch["category_id"][i]),
                        "split": batch["split"][i],
                        "source": batch["source"][i],
                    }
                )
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
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(seed),
        collate_fn=collate_preprocessed_samples,
    )
