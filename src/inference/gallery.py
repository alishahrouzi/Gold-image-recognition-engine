"""S4.3 runtime gallery loading for the functional MVP.

S3.7 already materializes image-level embeddings and metadata. S4.3 turns
those persisted artifacts into a validated in-memory Gallery plus the existing
S3.8 SimilaritySearchEngine. It does not rebuild embeddings and does not own
model/checkpoint loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from retrieval.gallery import Gallery
from retrieval.search import SimilaritySearchEngine


DEFAULT_SIAMESE_MODEL = "s3.5_random"
DEFAULT_GALLERY_ROOT = Path("experiments/retrieval/siamese")
DEFAULT_EMBEDDING_FILENAME = "gallery_embeddings.pt"
DEFAULT_METADATA_FILENAME = "gallery_metadata.json"
DEFAULT_EMBEDDING_DIM = 128


@dataclass(frozen=True)
class GalleryRuntimeConfig:
    """Filesystem and validation policy for one runtime gallery."""

    model_name: str = DEFAULT_SIAMESE_MODEL
    root: str | Path = DEFAULT_GALLERY_ROOT
    embedding_filename: str = DEFAULT_EMBEDDING_FILENAME
    metadata_filename: str = DEFAULT_METADATA_FILENAME
    expected_embedding_dim: int = DEFAULT_EMBEDDING_DIM
    require_unit_norm: bool = True
    norm_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if self.expected_embedding_dim < 1:
            raise ValueError("expected_embedding_dim must be positive.")
        if self.norm_tolerance < 0:
            raise ValueError("norm_tolerance must be non-negative.")
        if not self.embedding_filename or not self.metadata_filename:
            raise ValueError("Gallery filenames must be non-empty.")

    @property
    def directory(self) -> Path:
        return Path(self.root) / self.model_name

    @property
    def embedding_path(self) -> Path:
        return self.directory / self.embedding_filename

    @property
    def metadata_path(self) -> Path:
        return self.directory / self.metadata_filename


@dataclass(frozen=True)
class RuntimeGallery:
    """Validated in-memory gallery and its ready-to-use search engine."""

    gallery: Gallery
    search_engine: SimilaritySearchEngine
    model_name: str
    embedding_path: Path | None = None
    metadata_path: Path | None = None

    @property
    def size(self) -> int:
        return self.gallery.size

    @property
    def embedding_dim(self) -> int:
        return self.gallery.embedding_dim

    @property
    def product_count(self) -> int:
        return len(
            {
                str(item.get("product_id", item.get("product_group")))
                for item in self.gallery.metadata
            }
        )


class GalleryLoader:
    """Load and validate a persisted or in-memory retrieval gallery."""

    def __init__(self, config: GalleryRuntimeConfig | None = None) -> None:
        self.config = config or GalleryRuntimeConfig()

    def load(self) -> RuntimeGallery:
        """Load the configured S3.7 artifacts and build the runtime search engine."""
        embedding_path = self.config.embedding_path
        metadata_path = self.config.metadata_path
        if not embedding_path.is_file():
            raise FileNotFoundError(
                f"Gallery embeddings do not exist: {embedding_path}. "
                "Build the S3.7 gallery first."
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Gallery metadata does not exist: {metadata_path}. "
                "Build the S3.7 gallery first."
            )

        gallery = Gallery.load(embedding_path, metadata_path)
        return self.from_gallery(
            gallery,
            model_name=self.config.model_name,
            embedding_path=embedding_path,
            metadata_path=metadata_path,
        )

    def from_gallery(
        self,
        gallery: Gallery,
        *,
        model_name: str | None = None,
        embedding_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> RuntimeGallery:
        """Validate an already materialized gallery and keep it in memory."""
        self._validate_gallery(gallery)
        search_engine = SimilaritySearchEngine(gallery.embeddings, gallery.metadata)
        return RuntimeGallery(
            gallery=gallery,
            search_engine=search_engine,
            model_name=model_name or self.config.model_name,
            embedding_path=Path(embedding_path) if embedding_path is not None else None,
            metadata_path=Path(metadata_path) if metadata_path is not None else None,
        )

    def _validate_gallery(self, gallery: Gallery) -> None:
        if gallery.size == 0:
            raise ValueError("Runtime gallery cannot be empty.")
        if gallery.embedding_dim != self.config.expected_embedding_dim:
            raise ValueError(
                "Gallery embedding dimension does not match the runtime contract: "
                f"expected {self.config.expected_embedding_dim}, got {gallery.embedding_dim}."
            )
        if not torch.isfinite(gallery.embeddings).all():
            raise ValueError("Runtime gallery contains non-finite embeddings.")

        image_ids: set[str] = set()
        for index, item in enumerate(gallery.metadata):
            product_id = item.get("product_id", item.get("product_group"))
            category = item.get("category")
            image_id = item.get("image_id", item.get("image"))
            if product_id is None or not str(product_id).strip():
                raise ValueError(f"Gallery row {index} has an empty product_id.")
            if category is None or not str(category).strip():
                raise ValueError(f"Gallery row {index} has an empty category.")
            if image_id is None or not str(image_id).strip():
                raise ValueError(f"Gallery row {index} has an empty image_id.")
            normalized_image_id = str(image_id)
            if normalized_image_id in image_ids:
                raise ValueError(f"Gallery contains duplicate image_id: {normalized_image_id}")
            image_ids.add(normalized_image_id)

        if self.config.require_unit_norm:
            norms = torch.linalg.vector_norm(gallery.embeddings, dim=1)
            max_error = float(torch.abs(norms - 1.0).max().item())
            if max_error > self.config.norm_tolerance:
                raise ValueError(
                    "Runtime gallery embeddings are not L2-normalized within tolerance: "
                    f"max_norm_error={max_error:.6g}, "
                    f"tolerance={self.config.norm_tolerance}."
                )


def load_runtime_gallery(
    *,
    model_name: str = DEFAULT_SIAMESE_MODEL,
    root: str | Path = DEFAULT_GALLERY_ROOT,
    expected_embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> RuntimeGallery:
    """Convenience loader for the selected MVP gallery."""
    config = GalleryRuntimeConfig(
        model_name=model_name,
        root=root,
        expected_embedding_dim=expected_embedding_dim,
    )
    return GalleryLoader(config).load()
