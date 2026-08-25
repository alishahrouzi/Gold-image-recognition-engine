"""Identity-preserving training augmentation (Sprint S1.9).

Applied in memory on RGB PIL images, before ImagePreprocessor.
Never writes files. Never used for valid / test / query / gallery.

Random crop is implemented only as a conservative, disabled-by-default
experiment hook. S1.9 does not enable it.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

from PIL import Image
from torchvision.transforms import functional as F

from ..errors import AugmentationError
from .transforms import ensure_rgb

PIPELINE_ROLES: Tuple[str, ...] = ("train", "valid", "test", "query", "gallery")
DETERMINISTIC_ROLES: Tuple[str, ...] = ("valid", "test", "query", "gallery")
TRAINING_ROLE: str = "train"

# Jewelry-identity safety caps. Values above these require a new experiment
# contract; they are rejected rather than silently clipped.
MAX_ROTATION_DEGREES: float = 15.0
MAX_BRIGHTNESS_FACTOR: float = 0.25
MAX_CONTRAST_FACTOR: float = 0.25
MAX_SATURATION_FACTOR: float = 0.15
MAX_HUE_FACTOR: float = 0.04
MIN_RANDOM_CROP_SCALE: float = 0.85

DEFAULT_HORIZONTAL_FLIP_PROBABILITY: float = 0.5
DEFAULT_ROTATION_DEGREES: float = 10.0
DEFAULT_BRIGHTNESS_FACTOR: float = 0.15
DEFAULT_CONTRAST_FACTOR: float = 0.15
DEFAULT_SATURATION_FACTOR: float = 0.08
DEFAULT_HUE_FACTOR: float = 0.02
DEFAULT_RANDOM_CROP_SCALE: Tuple[float, float] = (0.90, 1.00)
DEFAULT_RANDOM_CROP_RATIO: Tuple[float, float] = (0.95, 1.05)

_ROTATION_FILL = (255, 255, 255)


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise AugmentationError(f"{field_name} must be a boolean.")
    return value


def _require_probability(value: Any, field_name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise AugmentationError(f"{field_name} must be numeric.") from exc
    if probability != probability or probability in {float("inf"), float("-inf")}:
        raise AugmentationError(f"{field_name} must be finite.")
    if probability < 0.0 or probability > 1.0:
        raise AugmentationError(f"{field_name} must be in [0, 1], got {probability}.")
    return probability


def _require_non_negative_factor(
    value: Any,
    field_name: str,
    *,
    maximum: float,
) -> float:
    try:
        factor = float(value)
    except (TypeError, ValueError) as exc:
        raise AugmentationError(f"{field_name} must be numeric.") from exc
    if factor != factor or factor in {float("inf"), float("-inf")}:
        raise AugmentationError(f"{field_name} must be finite.")
    if factor < 0.0:
        raise AugmentationError(f"{field_name} must be >= 0, got {factor}.")
    if factor > maximum:
        raise AugmentationError(
            f"{field_name}={factor} exceeds the identity-preserving cap "
            f"{maximum}. Aggressive jewelry color/geometry changes are out of scope."
        )
    return factor


def _require_pair(
    values: Sequence[float],
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> Tuple[float, float]:
    try:
        pair = tuple(float(component) for component in values)
    except (TypeError, ValueError) as exc:
        raise AugmentationError(
            f"{field_name} must be a sequence of two numeric values."
        ) from exc
    if len(pair) != 2:
        raise AugmentationError(f"{field_name} must have length 2, got {len(pair)}.")
    lo, hi = pair
    if any(component != component or component in {float("inf"), float("-inf")} for component in pair):
        raise AugmentationError(f"{field_name} must contain finite values.")
    if lo > hi:
        raise AugmentationError(f"{field_name} low value must be <= high value.")
    if lo < minimum or hi > maximum:
        raise AugmentationError(
            f"{field_name} must lie in [{minimum}, {maximum}], got {pair}."
        )
    return lo, hi


@dataclass(frozen=True)
class HorizontalFlipConfig:
    enabled: bool = True
    probability: float = DEFAULT_HORIZONTAL_FLIP_PROBABILITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "horizontal_flip.enabled"))
        object.__setattr__(
            self,
            "probability",
            _require_probability(self.probability, "horizontal_flip.probability"),
        )


@dataclass(frozen=True)
class RotationConfig:
    """Small in-plane rotation of ``±degrees`` (not 90/180)."""

    enabled: bool = True
    degrees: float = DEFAULT_ROTATION_DEGREES

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "rotation.enabled"))
        object.__setattr__(
            self,
            "degrees",
            _require_non_negative_factor(
                self.degrees,
                "rotation.degrees",
                maximum=MAX_ROTATION_DEGREES,
            ),
        )


@dataclass(frozen=True)
class BrightnessConfig:
    enabled: bool = True
    factor: float = DEFAULT_BRIGHTNESS_FACTOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "brightness.enabled"))
        object.__setattr__(
            self,
            "factor",
            _require_non_negative_factor(
                self.factor,
                "brightness.factor",
                maximum=MAX_BRIGHTNESS_FACTOR,
            ),
        )


@dataclass(frozen=True)
class ContrastConfig:
    enabled: bool = True
    factor: float = DEFAULT_CONTRAST_FACTOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "contrast.enabled"))
        object.__setattr__(
            self,
            "factor",
            _require_non_negative_factor(
                self.factor,
                "contrast.factor",
                maximum=MAX_CONTRAST_FACTOR,
            ),
        )


@dataclass(frozen=True)
class ColorConfig:
    """Conservative saturation / hue jitter. Gold identity must be preserved."""

    enabled: bool = True
    saturation: float = DEFAULT_SATURATION_FACTOR
    hue: float = DEFAULT_HUE_FACTOR

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "color.enabled"))
        object.__setattr__(
            self,
            "saturation",
            _require_non_negative_factor(
                self.saturation,
                "color.saturation",
                maximum=MAX_SATURATION_FACTOR,
            ),
        )
        object.__setattr__(
            self,
            "hue",
            _require_non_negative_factor(
                self.hue,
                "color.hue",
                maximum=MAX_HUE_FACTOR,
            ),
        )


@dataclass(frozen=True)
class RandomCropConfig:
    """Conservative crop hook. Disabled for the S1.9 baseline."""

    enabled: bool = False
    scale: Tuple[float, float] = DEFAULT_RANDOM_CROP_SCALE
    ratio: Tuple[float, float] = DEFAULT_RANDOM_CROP_RATIO

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "random_crop.enabled"))
        object.__setattr__(
            self,
            "scale",
            _require_pair(
                self.scale,
                "random_crop.scale",
                minimum=MIN_RANDOM_CROP_SCALE,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "ratio",
            _require_pair(
                self.ratio,
                "random_crop.ratio",
                minimum=0.90,
                maximum=1.10,
            ),
        )


@dataclass(frozen=True)
class AugmentationConfig:
    """Serializable training-augmentation hyperparameters.

    ``enabled=False`` is a no-op and is the safe setting for any non-train
    pipeline. ``seed`` is optional; ``None`` uses a non-reproducible local RNG.
    """

    enabled: bool = True
    seed: Optional[int] = None
    horizontal_flip: HorizontalFlipConfig = field(default_factory=HorizontalFlipConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    brightness: BrightnessConfig = field(default_factory=BrightnessConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    random_crop: RandomCropConfig = field(default_factory=RandomCropConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "enabled"))
        if self.seed is not None:
            if not isinstance(self.seed, int) or isinstance(self.seed, bool):
                raise AugmentationError("seed must be an integer or None.")

        if not isinstance(self.horizontal_flip, HorizontalFlipConfig):
            raise AugmentationError("horizontal_flip must be a HorizontalFlipConfig.")
        if not isinstance(self.rotation, RotationConfig):
            raise AugmentationError("rotation must be a RotationConfig.")
        if not isinstance(self.brightness, BrightnessConfig):
            raise AugmentationError("brightness must be a BrightnessConfig.")
        if not isinstance(self.contrast, ContrastConfig):
            raise AugmentationError("contrast must be a ContrastConfig.")
        if not isinstance(self.color, ColorConfig):
            raise AugmentationError("color must be a ColorConfig.")
        if not isinstance(self.random_crop, RandomCropConfig):
            raise AugmentationError("random_crop must be a RandomCropConfig.")

    @classmethod
    def disabled(cls) -> "AugmentationConfig":
        return cls(enabled=False)

    def as_loggable_dict(self) -> Mapping[str, Any]:
        """Nested dict for experiment logs. Contains no metric values."""
        payload = asdict(self)
        payload["policy"] = "s1.9-identity-preserving"
        payload["random_crop_default_disabled"] = not self.random_crop.enabled
        return payload


def validate_pipeline_role(role: str) -> str:
    if role not in PIPELINE_ROLES:
        allowed = ", ".join(PIPELINE_ROLES)
        raise AugmentationError(
            f"Unknown pipeline role {role!r}. Expected one of: {allowed}."
        )
    return role


class TrainingAugmentor:
    """Apply configured training augmentations to an in-memory RGB image.

    Uses a dedicated ``random.Random`` instance. It does not call
    ``random.seed`` / ``torch.manual_seed`` globally.
    """

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        *,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = config or AugmentationConfig()
        resolved_seed = seed if seed is not None else self.config.seed
        self.seed = resolved_seed
        self._rng = rng if rng is not None else random.Random(resolved_seed)

    def reseed(self, seed: Optional[int]) -> None:
        """Replace the local RNG. Used by tests and future DataLoader workers."""
        self.seed = seed
        self._rng = random.Random(seed)

    def worker_seed(self, worker_id: int, base_seed: Optional[int] = None) -> int:
        """Derive a per-worker seed. Does not claim bit-level determinism."""
        root = base_seed if base_seed is not None else self.seed
        if root is None:
            raise AugmentationError(
                "A base seed is required to derive DataLoader worker seeds."
            )
        return int(root) + int(worker_id)

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = ensure_rgb(image)
        if not self.config.enabled:
            return rgb
        return self._apply(rgb)

    def _apply(self, image: Image.Image) -> Image.Image:
        config = self.config
        if config.random_crop.enabled:
            image = self._random_crop(image)
        if config.horizontal_flip.enabled and self._rng.random() < config.horizontal_flip.probability:
            image = F.hflip(image)
        if config.rotation.enabled and config.rotation.degrees > 0.0:
            angle = self._rng.uniform(-config.rotation.degrees, config.rotation.degrees)
            if angle != 0.0:
                image = image.rotate(
                    angle,
                    resample=Image.Resampling.BILINEAR,
                    expand=False,
                    fillcolor=_ROTATION_FILL,
                )
        if config.brightness.enabled and config.brightness.factor > 0.0:
            brightness = self._sample_factor(config.brightness.factor)
            image = F.adjust_brightness(image, brightness)
        if config.contrast.enabled and config.contrast.factor > 0.0:
            contrast = self._sample_factor(config.contrast.factor)
            image = F.adjust_contrast(image, contrast)
        if config.color.enabled:
            if config.color.saturation > 0.0:
                saturation = self._sample_factor(config.color.saturation)
                image = F.adjust_saturation(image, saturation)
            if config.color.hue > 0.0:
                hue = self._rng.uniform(-config.color.hue, config.color.hue)
                if hue != 0.0:
                    image = F.adjust_hue(image, hue)
        return image

    def _sample_factor(self, amplitude: float) -> float:
        return self._rng.uniform(1.0 - amplitude, 1.0 + amplitude)

    def _random_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_h, crop_w = _sample_crop_size(
            height,
            width,
            scale=self.config.random_crop.scale,
            ratio=self.config.random_crop.ratio,
            rng=self._rng,
        )
        top = self._rng.randint(0, height - crop_h)
        left = self._rng.randint(0, width - crop_w)
        return F.crop(image, top, left, crop_h, crop_w)


def _sample_crop_size(
    height: int,
    width: int,
    *,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    rng: random.Random,
) -> Tuple[int, int]:
    area = height * width
    log_ratio_min, log_ratio_max = math.log(ratio[0]), math.log(ratio[1])
    for _ in range(10):
        target_area = area * rng.uniform(scale[0], scale[1])
        aspect = math.exp(rng.uniform(log_ratio_min, log_ratio_max))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 1 <= crop_w <= width and 1 <= crop_h <= height:
            return crop_h, crop_w
    crop_h = max(1, int(round(height * scale[0])))
    crop_w = max(1, int(round(width * scale[0])))
    return min(crop_h, height), min(crop_w, width)


def augmentor_for_role(
    role: str,
    config: Optional[AugmentationConfig],
) -> Optional[TrainingAugmentor]:
    """Return a TrainingAugmentor only for ``role='train'`` when enabled."""
    validate_pipeline_role(role)
    if role != TRAINING_ROLE:
        if config is not None and config.enabled:
            raise AugmentationError(
                "Training augmentation is only allowed for role='train'. "
                f"Received role={role!r}. Validation, test, query, and gallery "
                "must use deterministic preprocessing."
            )
        return None
    if config is None or not config.enabled:
        return None
    return TrainingAugmentor(config)
