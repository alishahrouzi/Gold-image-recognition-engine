"""End-to-end S2.7 baseline retrieval pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor

from .embedding import EmbeddingExtractor
from .gallery import Gallery, GalleryBuilder, build_gallery_loader
from .result import RetrievalResult
from .topk import TopKRetriever


class BaselineRetrievalPipeline:
    """Query the S2.6 embedding space against a materialized gallery."""

    def __init__(self, extractor: EmbeddingExtractor, gallery: Gallery) -> None:
        self.extractor = extractor
        self.gallery = gallery
        self.retriever = TopKRetriever(gallery.embeddings, gallery.metadata)

    def query(
        self,
        image: Tensor,
        *,
        query_id: str = "query",
        k: int = 5,
        exclude_image_id: str | None = None,
    ) -> RetrievalResult:
        embedding = self.extractor.extract_one(image)
        return self.retriever.retrieve(
            embedding,
            query_id=query_id,
            k=k,
            exclude_image_id=exclude_image_id,
        )


def build_baseline_gallery(
    extractor: EmbeddingExtractor,
    manifest_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    split: str = "train",
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Gallery:
    """Build the S2.7 baseline gallery using the project's data pipeline."""
    loader = build_gallery_loader(
        manifest_path,
        dataset_root=dataset_root,
        split=split,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return GalleryBuilder(extractor).build(loader)
