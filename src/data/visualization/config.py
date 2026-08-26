"""Configurable sampling for Dataset 1 visualization (S1.11).

Defaults are QA starting points. They do not change pair generation or
preprocessing contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..errors import VisualizationError

DEFAULT_VISUALIZATION_SEED: int = 2026
DEFAULT_TRAIN_SAMPLES: int = 20
DEFAULT_VALID_SAMPLES: int = 20
DEFAULT_TEST_SAMPLES: int = 20
DEFAULT_POSITIVE_PAIRS: int = 20
DEFAULT_NEGATIVE_PAIRS: int = 20
DEFAULT_AUGMENTATION_SAMPLES: int = 10

# In-memory display thumbnail only. Source files are never resized on disk.
DISPLAY_MAX_EDGE: int = 256


def _require_seed(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VisualizationError("seed must be an integer.")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VisualizationError(f"{field_name} must be an integer.")
    if value < 0:
        raise VisualizationError(f"{field_name} must be >= 0, got {value}.")
    return value


@dataclass(frozen=True)
class VisualizationConfig:
    """Sampling counts for the read-only visualization tool.

    train_samples
        Number of **groups** to visualize from train. Every image in a
        selected group is shown (group size 1, 2, or 3+).
    valid_samples / test_samples
        Number of **images** from that split (Dataset 1 valid/test groups
        are currently singletons).
    positive_pairs / negative_pairs
        Number of pair CSV rows to visualize. Negatives are split between
        ``same_category`` and ``cross_category`` when both exist.
    augmentation_samples
        Number of **train images** shown as original vs S1.9-augmented.
    """

    seed: int = DEFAULT_VISUALIZATION_SEED
    train_samples: int = DEFAULT_TRAIN_SAMPLES
    valid_samples: int = DEFAULT_VALID_SAMPLES
    test_samples: int = DEFAULT_TEST_SAMPLES
    positive_pairs: int = DEFAULT_POSITIVE_PAIRS
    negative_pairs: int = DEFAULT_NEGATIVE_PAIRS
    augmentation_samples: int = DEFAULT_AUGMENTATION_SAMPLES

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _require_seed(self.seed))
        object.__setattr__(
            self, "train_samples", _require_non_negative_int(self.train_samples, "train_samples")
        )
        object.__setattr__(
            self, "valid_samples", _require_non_negative_int(self.valid_samples, "valid_samples")
        )
        object.__setattr__(
            self, "test_samples", _require_non_negative_int(self.test_samples, "test_samples")
        )
        object.__setattr__(
            self,
            "positive_pairs",
            _require_non_negative_int(self.positive_pairs, "positive_pairs"),
        )
        object.__setattr__(
            self,
            "negative_pairs",
            _require_non_negative_int(self.negative_pairs, "negative_pairs"),
        )
        object.__setattr__(
            self,
            "augmentation_samples",
            _require_non_negative_int(self.augmentation_samples, "augmentation_samples"),
        )

    def as_loggable_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["policy"] = "s1.11-data-visualization"
        payload["train_samples_unit"] = "groups"
        payload["valid_samples_unit"] = "images"
        payload["test_samples_unit"] = "images"
        payload["augmentation_samples_unit"] = "train_images"
        payload["sampling"] = "local random.Random(seed); global RNG is not modified"
        return payload
