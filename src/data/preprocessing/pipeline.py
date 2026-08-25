"""Reusable preprocessing pipeline for training, evaluation, and retrieval.

ImagePreprocessor can run on a standalone PIL image (query or gallery).
PreprocessedDataset wraps UnifiedDataset so DataLoader batching stays in collate.
"""

from __future__ import annotations

from typing import Optional

import torch
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset as TorchDataset

from ..errors import PreprocessingError
from ..types import DatasetItem
from .config import ImagePreprocessingConfig
from .transforms import (
    assert_finite,
    ensure_rgb,
    image_to_tensor,
    normalize_tensor,
    resize_image,
)


class ImagePreprocessor:
    """Deterministic RGB → resize → tensor → normalize pipeline.

    The same instance is used for train, valid, test, query, and gallery.
    Augmentation is out of scope for S1.8.
    """

    def __init__(self, config: Optional[ImagePreprocessingConfig] = None) -> None:
        self.config = config or ImagePreprocessingConfig()

    def __call__(self, image: PILImage) -> torch.Tensor:
        """Preprocess one loaded image.

        Returns:
            Tensor of shape ``[3, H, W]`` and dtype ``float32``.
        """
        return self.process(image, normalize=True)

    def to_float_tensor(self, image: PILImage) -> torch.Tensor:
        """RGB, resize, and convert to ``[0, 1]`` CHW without mean/std normalization."""
        return self.process(image, normalize=False)

    def process(self, image: PILImage, *, normalize: bool = True) -> torch.Tensor:
        rgb = ensure_rgb(image)
        resized = resize_image(
            rgb,
            self.config.image_size,
            self.config.resample,
        )
        if resized.mode != "RGB":
            raise PreprocessingError(
                f"Expected RGB after channel conversion, got mode {resized.mode!r}."
            )
        tensor = image_to_tensor(resized)
        expected = self.config.output_shape
        if tuple(tensor.shape) != expected:
            raise PreprocessingError(
                f"Unexpected tensor shape {tuple(tensor.shape)}; expected {expected}."
            )
        if tensor.dtype != torch.float32:
            raise PreprocessingError(
                f"Unexpected tensor dtype {tensor.dtype}; expected torch.float32."
            )
        if not normalize:
            return tensor
        normalized = normalize_tensor(tensor, self.config.mean, self.config.std)
        return assert_finite(normalized)


class PreprocessedDataset(TorchDataset):
    """Apply ImagePreprocessor to UnifiedDataset items.

    Ingestion stays responsible for locating samples and loading RGB images.
    This wrapper only converts each in-memory image to a tensor.
    Original files and the dataset split are never modified.
    """

    def __init__(
        self,
        dataset: TorchDataset,
        preprocessor: Optional[ImagePreprocessor] = None,
    ) -> None:
        self.dataset = dataset
        self.preprocessor = preprocessor or ImagePreprocessor()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DatasetItem:
        item = self.dataset[index]
        if not isinstance(item, DatasetItem):
            raise PreprocessingError(
                "PreprocessedDataset expects DatasetItem values from the wrapped dataset."
            )
        tensor = self.preprocessor(item.image)
        return DatasetItem(sample=item.sample, image=tensor)
