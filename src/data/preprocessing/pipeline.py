"""Reusable preprocessing pipeline for training, evaluation, and retrieval.

ImagePreprocessor is deterministic (RGB → resize → normalize). Training
augmentation is applied only by PreprocessedDataset when role='train'.
Query and gallery paths must call ImagePreprocessor directly (no augmentor).
"""

from __future__ import annotations

from typing import Optional

import torch
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset as TorchDataset

from ..errors import PreprocessingError
from ..types import DatasetItem
from .augmentation import (
    AugmentationConfig,
    TrainingAugmentor,
    augmentor_for_role,
    validate_pipeline_role,
)
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

    The same instance is used for train (after optional augmentation),
    valid, test, query, and gallery. This class never applies random
    augmentation.
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
    """Optional training augmentation, then ImagePreprocessor.

    Ingestion stays responsible for locating samples and loading RGB images.
    Original files are never modified.

    Augmentation runs only when ``role='train'`` and an enabled
    ``AugmentationConfig`` is provided. Roles ``valid``, ``test``,
    ``query``, and ``gallery`` are always deterministic.
    """

    def __init__(
        self,
        dataset: TorchDataset,
        preprocessor: Optional[ImagePreprocessor] = None,
        *,
        role: str = "valid",
        augmentation: Optional[AugmentationConfig] = None,
    ) -> None:
        self.dataset = dataset
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.role = validate_pipeline_role(role)
        self.augmentation = augmentation
        self.augmentor: Optional[TrainingAugmentor] = augmentor_for_role(
            self.role,
            augmentation,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DatasetItem:
        item = self.dataset[index]
        if not isinstance(item, DatasetItem):
            raise PreprocessingError(
                "PreprocessedDataset expects DatasetItem values from the wrapped dataset."
            )
        image = item.image
        if self.augmentor is not None:
            image = self.augmentor(image)
        tensor = self.preprocessor(image)
        return DatasetItem(sample=item.sample, image=tensor)


def build_preprocessed_dataset(
    dataset: TorchDataset,
    *,
    role: str,
    preprocessor: Optional[ImagePreprocessor] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> PreprocessedDataset:
    """Construct a split/role-aware view.

    Pass ``role='train'`` plus an enabled config to apply S1.9 augmentation.
    Query and gallery should use ``ImagePreprocessor`` on a loaded PIL image,
    or this helper with ``role='query'`` / ``role='gallery'`` and no enabled
    augmentation.
    """
    return PreprocessedDataset(
        dataset,
        preprocessor,
        role=role,
        augmentation=augmentation,
    )
