"""Load image files as RGB PIL images.

This module does not resize, normalize, augment, or convert images to tensors.
Those steps belong to later preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image, UnidentifiedImageError

from ..errors import DatasetIngestionError

PathLike = Union[str, Path]

_RGB_BACKGROUND = (255, 255, 255)


def load_rgb_image(path: PathLike) -> Image.Image:
    """Open an image file and return a detached RGB ``PIL.Image``.

    Grayscale images are converted to RGB. RGBA / transparency is composited
    onto a white background so the result has no alpha channel.

    Args:
        path: Filesystem path to the image.

    Returns:
        A copy of the image in RGB mode.

    Raises:
        DatasetIngestionError: If the file is missing or cannot be decoded.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise DatasetIngestionError(f"Image file does not exist: {image_path}")

    try:
        with Image.open(image_path) as image:
            image.load()
            return _to_rgb(image)
    except DatasetIngestionError:
        raise
    except UnidentifiedImageError as exc:
        raise DatasetIngestionError(
            f"Unable to read image file (unidentified format): {image_path}"
        ) from exc
    except OSError as exc:
        raise DatasetIngestionError(
            f"Unable to read image file: {image_path}"
        ) from exc


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert a loaded PIL image to RGB without silent data loss of alpha."""
    if image.mode == "RGB":
        return image.copy()

    has_palette_transparency = image.mode == "P" and "transparency" in image.info
    if image.mode in {"RGBA", "LA"} or has_palette_transparency:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, _RGB_BACKGROUND)
        background.paste(rgba, mask=rgba.split()[-1])
        return background

    return image.convert("RGB")
