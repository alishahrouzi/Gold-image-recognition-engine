"""Explicit configuration for deterministic image preprocessing.

Values live here (or in a constructed ImagePreprocessingConfig) rather than
inside transform implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, Union

from PIL import Image

from ..errors import PreprocessingError

# Square CNN input size. Architecture.md leaves H/W experimental; 224 is the
# S1.8 MVP default because it is the standard ImageNet-compatible resolution
# and no encoder contract currently specifies another value.
DEFAULT_IMAGE_SIZE: int = 224

# ImageNet channel-wise statistics applied after scaling pixels to [0, 1].
# Chosen because no project contract defines jewelry-specific mean/std yet.
DEFAULT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Deterministic stretch-resize. Bilinear is the torchvision/ImageNet default.
DEFAULT_INTERPOLATION: str = "bilinear"

InterpolationName = Union[str, Image.Resampling]

_PIL_RESAMPLE: Mapping[str, Image.Resampling] = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def resolve_interpolation(interpolation: InterpolationName) -> Image.Resampling:
    """Map a config interpolation value to a PIL resampling filter."""
    if isinstance(interpolation, Image.Resampling):
        return interpolation
    if not isinstance(interpolation, str) or not interpolation.strip():
        raise PreprocessingError(
            "interpolation must be a non-empty string or PIL.Image.Resampling."
        )
    key = interpolation.strip().lower()
    if key not in _PIL_RESAMPLE:
        allowed = ", ".join(sorted(_PIL_RESAMPLE))
        raise PreprocessingError(
            f"Unsupported interpolation {interpolation!r}. Expected one of: {allowed}."
        )
    return _PIL_RESAMPLE[key]


def _as_rgb_stats(values: Sequence[float], field_name: str) -> Tuple[float, float, float]:
    try:
        triple = tuple(float(component) for component in values)
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(
            f"{field_name} must be a sequence of three numeric channel values."
        ) from exc
    if len(triple) != 3:
        raise PreprocessingError(
            f"{field_name} must have length 3 (RGB), got {len(triple)}."
        )
    if any(not _is_finite(component) for component in triple):
        raise PreprocessingError(f"{field_name} must contain finite values.")
    return triple  # type: ignore[return-value]


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    """Encoder-independent preprocessing hyperparameters.

    image_size
        Output height and width in pixels (square).
    mean / std
        Per-channel RGB statistics for ``(x - mean) / std`` after [0, 1]
        conversion. Defaults are ImageNet statistics.
    interpolation
        Resize filter: ``nearest``, ``bilinear``, ``bicubic``, or ``lanczos``.
        Resize stretches to ``(image_size, image_size)`` and does not
        letterbox or randomly crop.
    """

    image_size: int = DEFAULT_IMAGE_SIZE
    mean: Tuple[float, float, float] = DEFAULT_MEAN
    std: Tuple[float, float, float] = DEFAULT_STD
    interpolation: InterpolationName = DEFAULT_INTERPOLATION

    def __post_init__(self) -> None:
        if not isinstance(self.image_size, int) or isinstance(self.image_size, bool):
            raise PreprocessingError("image_size must be an integer.")
        if self.image_size < 1:
            raise PreprocessingError(
                f"image_size must be a positive integer, got {self.image_size}."
            )

        mean = _as_rgb_stats(self.mean, "mean")
        std = _as_rgb_stats(self.std, "std")
        if any(component == 0.0 for component in std):
            raise PreprocessingError(
                "std components must be non-zero to avoid division by zero."
            )
        interpolation = resolve_interpolation(self.interpolation)

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "interpolation", interpolation.name.lower())
        object.__setattr__(self, "_resample", interpolation)

    @property
    def resample(self) -> Image.Resampling:
        """PIL resampling filter corresponding to ``interpolation``."""
        return self._resample  # type: ignore[attr-defined]

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Single-image tensor layout ``(C, H, W)``."""
        return (3, self.image_size, self.image_size)
