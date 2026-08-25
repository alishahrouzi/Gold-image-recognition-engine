"""Image-level validation for Dataset 1 (Sprint S1.2).

This module inspects files referenced by the authoritative manifest. It does
not move, rename, delete, or rewrite images, and it does not modify the
manifest. Contract checks (IDs, groups, splits) remain in ``validation.py``.

Status semantics
----------------
VALID
    File exists, pixels decode, dimensions are positive, and the image is
    not near-uniform.
WARNING
    File is readable, but luminance statistics look near-uniform (near-black,
    near-white, or otherwise flat). This is not corruption.
INVALID
    Missing file, decode failure, or non-positive width/height.

Color-mode policy
-----------------
Non-RGB modes are reported, not rejected. RGBA, L, P, and similar modes are
VALID when they decode and convert to RGB in memory.

Resolution policy
-----------------
Width and height are recorded. There is no minimum-size cutoff (including
no 224 px rule). Observed Dataset 1 ranges are not used as constraints.

Abnormal-image policy
---------------------
Jewelry on a white or black background is expected and must not be flagged
only because many pixels are bright or dark. An image is flagged only when
luminance standard deviation is below ``LOW_STD_THRESHOLD`` (near-uniform
content). Near-black / near-white additionally require an extreme mean
and/or an extreme dark/bright pixel ratio.

Thresholds (8-bit luminance, 0--255)
------------------------------------
LOW_STD_THRESHOLD              12.0
NEAR_BLACK_MEAN_THRESHOLD      12.0
NEAR_WHITE_MEAN_THRESHOLD     243.0
DARK_PIXEL_VALUE               16
BRIGHT_PIXEL_VALUE            239
DARK_PIXEL_RATIO_THRESHOLD    0.995
BRIGHT_PIXEL_RATIO_THRESHOLD  0.995
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from .constants import CATEGORY_TO_ID, SOURCE_DATASET1, SPLIT_ORDER
from .loaders.image_loader import to_rgb_image
from .types import Sample

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

STATUS_VALID = "valid"
STATUS_WARNING = "warning"
STATUS_INVALID = "invalid"

REASON_MISSING_FILE = "missing_file"
REASON_DECODE_ERROR = "decode_error"
REASON_ZERO_DIMENSION = "zero_dimension"
REASON_UNEXPECTED_ERROR = "unexpected_error"
REASON_NEAR_BLACK = "near_black"
REASON_NEAR_WHITE = "near_white"
REASON_NEAR_UNIFORM = "near_uniform"

# Conservative flags for obviously flat frames, not for studio backgrounds.
LOW_STD_THRESHOLD = 12.0
NEAR_BLACK_MEAN_THRESHOLD = 12.0
NEAR_WHITE_MEAN_THRESHOLD = 243.0
DARK_PIXEL_VALUE = 16
BRIGHT_PIXEL_VALUE = 239
DARK_PIXEL_RATIO_THRESHOLD = 0.995
BRIGHT_PIXEL_RATIO_THRESHOLD = 0.995

COMMON_RESOLUTION_LIMIT = 15
_ERROR_MESSAGE_LIMIT = 300
_FLOAT_DIGITS = 4


@dataclass
class ImageQualityResult:
    """Per-image inspection. Does not include decoded pixel buffers."""

    image_id: str
    image_path: str
    group_id: str
    category: str
    category_id: int
    split: str
    source: str
    status: str
    readable: bool
    original_mode: Optional[str] = None
    rgb_convertible: Optional[bool] = None
    width: Optional[int] = None
    height: Optional[int] = None
    total_pixels: Optional[int] = None
    aspect_ratio: Optional[float] = None
    mean_intensity: Optional[float] = None
    std_intensity: Optional[float] = None
    min_intensity: Optional[float] = None
    max_intensity: Optional[float] = None
    dark_pixel_ratio: Optional[float] = None
    bright_pixel_ratio: Optional[float] = None
    abnormal: bool = False
    abnormal_reasons: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "group_id": self.group_id,
            "category": self.category,
            "category_id": self.category_id,
            "split": self.split,
            "source": self.source,
            "status": self.status,
            "readable": self.readable,
            "original_mode": self.original_mode,
            "rgb_convertible": self.rgb_convertible,
            "width": self.width,
            "height": self.height,
            "total_pixels": self.total_pixels,
            "aspect_ratio": self.aspect_ratio,
            "mean_intensity": self.mean_intensity,
            "std_intensity": self.std_intensity,
            "min_intensity": self.min_intensity,
            "max_intensity": self.max_intensity,
            "dark_pixel_ratio": self.dark_pixel_ratio,
            "bright_pixel_ratio": self.bright_pixel_ratio,
            "abnormal": self.abnormal,
            "abnormal_reasons": list(self.abnormal_reasons),
            "reason": self.reason,
            "error": self.error,
        }


def inspect_sample(sample: Sample) -> ImageQualityResult:
    """Inspect one manifest sample. Never writes to the image path."""
    path = Path(sample.image_path)
    base = ImageQualityResult(
        image_id=sample.image_id,
        image_path=_path_for_report(path),
        group_id=sample.group_id,
        category=sample.category,
        category_id=sample.category_id,
        split=sample.split,
        source=sample.source,
        status=STATUS_INVALID,
        readable=False,
    )
    if not path.is_file():
        base.reason = REASON_MISSING_FILE
        base.error = f"Image file does not exist: {path}"
        return base
    return _inspect_existing_file(base, path)


def inspect_samples(samples: Sequence[Sample]) -> List[ImageQualityResult]:
    """Inspect every sample in manifest order."""
    results: List[ImageQualityResult] = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        results.append(inspect_sample(sample))
        if index % 500 == 0 or index == total:
            logger.info("Image validation progress: %s / %s", index, total)
    return results


def build_image_validation_report(
    results: Sequence[ImageQualityResult],
    *,
    dataset_root: PathLike,
    manifest_path: PathLike,
    dataset: str = SOURCE_DATASET1,
) -> Dict[str, Any]:
    """Aggregate image-level results into the S1.2 JSON document."""
    readable = [item for item in results if item.readable]
    invalid = [item for item in results if item.status == STATUS_INVALID]
    warning = [item for item in results if item.status == STATUS_WARNING]
    valid = [item for item in results if item.status == STATUS_VALID]
    abnormal = [item for item in results if item.abnormal]
    rgb = [item for item in readable if item.original_mode == "RGB"]
    non_rgb = [item for item in readable if item.original_mode != "RGB"]
    rgb_convertible = [item for item in readable if item.rgb_convertible]

    mode_counts = Counter(item.original_mode for item in readable if item.original_mode)
    abnormal_counts = Counter(
        reason for item in abnormal for reason in item.abnormal_reasons
    )

    return {
        "dataset": dataset,
        "dataset_root": str(Path(dataset_root)),
        "manifest": str(Path(manifest_path)),
        "thresholds": {
            "low_std_threshold": LOW_STD_THRESHOLD,
            "near_black_mean_threshold": NEAR_BLACK_MEAN_THRESHOLD,
            "near_white_mean_threshold": NEAR_WHITE_MEAN_THRESHOLD,
            "dark_pixel_value": DARK_PIXEL_VALUE,
            "bright_pixel_value": BRIGHT_PIXEL_VALUE,
            "dark_pixel_ratio_threshold": DARK_PIXEL_RATIO_THRESHOLD,
            "bright_pixel_ratio_threshold": BRIGHT_PIXEL_RATIO_THRESHOLD,
        },
        "summary": {
            "total_images": len(results),
            "valid_images": len(valid),
            "warning_images": len(warning),
            "invalid_images": len(invalid),
            "readable_images": len(readable),
            "unreadable_images": len(invalid),
            "corrupted_images": sum(
                1 for item in invalid if item.reason == REASON_DECODE_ERROR
            ),
            "rgb_images": len(rgb),
            "non_rgb_images": len(non_rgb),
            "rgb_convertible_images": len(rgb_convertible),
            "abnormal_images": len(abnormal),
        },
        "resolution": _resolution_summary(readable),
        "color_modes": dict(sorted(mode_counts.items())),
        "abnormal_statistics": {
            "near_black": abnormal_counts.get(REASON_NEAR_BLACK, 0),
            "near_white": abnormal_counts.get(REASON_NEAR_WHITE, 0),
            "near_uniform": abnormal_counts.get(REASON_NEAR_UNIFORM, 0),
        },
        "split_statistics": _group_statistics(results, SPLIT_ORDER, key=lambda item: item.split),
        "category_statistics": _group_statistics(
            results,
            tuple(CATEGORY_TO_ID.keys()),
            key=lambda item: item.category,
        ),
        "images": [item.to_dict() for item in results],
    }


def write_image_validation_report(report: Dict[str, Any], output_path: PathLike) -> Path:
    """Write the JSON report. Does not modify images or the manifest."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Wrote image validation report: %s", path)
    return path


def _inspect_existing_file(base: ImageQualityResult, path: Path) -> ImageQualityResult:
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            image.load()
            return _inspect_loaded_image(base, image)
    except (UnidentifiedImageError, OSError) as exc:
        base.reason = REASON_DECODE_ERROR
        base.error = _short_error(exc)
        return base
    except Exception as exc:  # noqa: BLE001 — surface unexpected decode failures
        base.reason = REASON_UNEXPECTED_ERROR
        base.error = _short_error(exc)
        return base


def _inspect_loaded_image(base: ImageQualityResult, image: Image.Image) -> ImageQualityResult:
    width, height = image.size
    base.original_mode = image.mode
    base.width = width
    base.height = height
    base.rgb_convertible = _is_rgb_convertible(image)

    if width <= 0 or height <= 0:
        base.reason = REASON_ZERO_DIMENSION
        base.error = f"Non-positive image size: {width}x{height}"
        return base

    base.readable = True
    base.total_pixels = width * height
    base.aspect_ratio = _round_float(width / height)
    _apply_luminance_metrics(base, image)

    if base.abnormal:
        base.status = STATUS_WARNING
    else:
        base.status = STATUS_VALID
    return base


def _is_rgb_convertible(image: Image.Image) -> bool:
    try:
        converted = to_rgb_image(image)
        ok = converted.mode == "RGB" and converted.size == image.size
        if converted is not image:
            converted.close()
        return ok
    except Exception:  # noqa: BLE001 — conversion failure is a reported flag
        return False


def _apply_luminance_metrics(base: ImageQualityResult, image: Image.Image) -> None:
    luminance = image.convert("L")
    array = np.asarray(luminance, dtype=np.float32)
    if array.size == 0:
        base.readable = False
        base.status = STATUS_INVALID
        base.reason = REASON_ZERO_DIMENSION
        base.error = "Decoded image has no pixels."
        return

    mean = float(array.mean())
    std = float(array.std())
    base.mean_intensity = _round_float(mean)
    base.std_intensity = _round_float(std)
    base.min_intensity = _round_float(float(array.min()))
    base.max_intensity = _round_float(float(array.max()))
    base.dark_pixel_ratio = _round_float(float((array <= DARK_PIXEL_VALUE).mean()))
    base.bright_pixel_ratio = _round_float(float((array >= BRIGHT_PIXEL_VALUE).mean()))

    reasons = _abnormal_reasons(
        mean=mean,
        std=std,
        dark_ratio=float((array <= DARK_PIXEL_VALUE).mean()),
        bright_ratio=float((array >= BRIGHT_PIXEL_VALUE).mean()),
    )
    base.abnormal_reasons = reasons
    base.abnormal = bool(reasons)


def _abnormal_reasons(
    *,
    mean: float,
    std: float,
    dark_ratio: float,
    bright_ratio: float,
) -> List[str]:
    if std >= LOW_STD_THRESHOLD:
        return []

    reasons: List[str] = []
    near_black = mean <= NEAR_BLACK_MEAN_THRESHOLD or dark_ratio >= DARK_PIXEL_RATIO_THRESHOLD
    near_white = mean >= NEAR_WHITE_MEAN_THRESHOLD or bright_ratio >= BRIGHT_PIXEL_RATIO_THRESHOLD
    if near_black:
        reasons.append(REASON_NEAR_BLACK)
    if near_white:
        reasons.append(REASON_NEAR_WHITE)
    if not near_black and not near_white:
        reasons.append(REASON_NEAR_UNIFORM)
    return reasons


def _resolution_summary(readable: Sequence[ImageQualityResult]) -> Dict[str, Any]:
    widths = [item.width for item in readable if item.width is not None]
    heights = [item.height for item in readable if item.height is not None]
    if not widths or not heights:
        return {
            "min_width": None,
            "max_width": None,
            "min_height": None,
            "max_height": None,
            "avg_width": None,
            "avg_height": None,
            "common_resolutions": [],
        }

    pairs = Counter((item.width, item.height) for item in readable if item.width and item.height)
    common = [
        {"width": width, "height": height, "count": count}
        for (width, height), count in pairs.most_common(COMMON_RESOLUTION_LIMIT)
    ]
    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_width": _round_float(float(sum(widths) / len(widths))),
        "avg_height": _round_float(float(sum(heights) / len(heights))),
        "common_resolutions": common,
    }


def _group_statistics(
    results: Sequence[ImageQualityResult],
    labels: Sequence[str],
    *,
    key,
) -> Dict[str, Dict[str, int]]:
    buckets: Dict[str, List[ImageQualityResult]] = defaultdict(list)
    for item in results:
        buckets[key(item)].append(item)

    stats: Dict[str, Dict[str, int]] = {}
    for label in labels:
        items = buckets.get(label, [])
        stats[label] = {
            "total": len(items),
            "readable": sum(1 for item in items if item.readable),
            "unreadable": sum(1 for item in items if not item.readable),
            "abnormal": sum(1 for item in items if item.abnormal),
            "valid": sum(1 for item in items if item.status == STATUS_VALID),
            "warning": sum(1 for item in items if item.status == STATUS_WARNING),
            "invalid": sum(1 for item in items if item.status == STATUS_INVALID),
        }
    return stats


def _path_for_report(path: Path) -> str:
    return path.as_posix()


def _round_float(value: float) -> float:
    return round(value, _FLOAT_DIGITS)


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if len(text) > _ERROR_MESSAGE_LIMIT:
        return text[:_ERROR_MESSAGE_LIMIT] + "..."
    return text
