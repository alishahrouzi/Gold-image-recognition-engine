"""Deterministic image transforms used by ImagePreprocessor.

These functions never write files. RGB conversion reuses the ingestion
loader so palette / alpha handling stays consistent with S1.2.
"""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from torchvision.transforms import functional as F

from ..errors import PreprocessingError
from ..loaders.image_loader import to_rgb_image


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return an in-memory RGB copy. Does not modify the source file."""
    if not isinstance(image, Image.Image):
        raise PreprocessingError(
            f"Unsupported input type {type(image)!r}. Expected a PIL.Image.Image."
        )
    try:
        width, height = image.size
    except Exception as exc:
        raise PreprocessingError("Unable to read image size from the provided image.") from exc
    if width < 1 or height < 1:
        raise PreprocessingError(
            f"Image has invalid size {(width, height)}. Width and height must be positive."
        )
    try:
        return to_rgb_image(image)
    except Exception as exc:
        raise PreprocessingError("Unable to convert image to RGB.") from exc


def resize_image(
    image: Image.Image,
    image_size: int,
    resample: Image.Resampling,
) -> Image.Image:
    """Stretch-resize to ``image_size x image_size``. Aspect ratio is not preserved."""
    if image.size == (image_size, image_size):
        return image
    return image.resize((image_size, image_size), resample=resample)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to ``float32`` CHW tensor in ``[0, 1]``."""
    tensor = F.to_tensor(image)
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Apply ``(x - mean) / std`` per channel. Does not clamp the result."""
    return F.normalize(tensor, mean=list(mean), std=list(std))


def assert_finite(tensor: torch.Tensor) -> torch.Tensor:
    """Raise if normalization produced NaN or Inf. Errors are not suppressed."""
    if torch.isnan(tensor).any():
        raise PreprocessingError("Preprocessing produced NaN values.")
    if torch.isinf(tensor).any():
        raise PreprocessingError("Preprocessing produced Inf values.")
    return tensor
