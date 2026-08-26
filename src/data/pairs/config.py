"""Configurable pair-generation hyperparameters (S1.10).

Defaults are starting points for later experiments, not frozen training values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Tuple

from ..constants import SPLIT_ORDER
from ..errors import PairGenerationError

DEFAULT_PAIR_SEED: int = 2026
DEFAULT_POSITIVE_NEGATIVE_RATIO: float = 1.0
DEFAULT_SAME_CATEGORY_NEGATIVE_RATIO: float = 0.5


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise PairGenerationError(f"{field_name} must be a boolean.")
    return value


def _require_seed(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PairGenerationError("seed must be an integer.")
    return value


def _require_non_negative_finite(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PairGenerationError(f"{field_name} must be numeric.") from exc
    if number != number or number in {float("inf"), float("-inf")}:
        raise PairGenerationError(f"{field_name} must be finite.")
    if number < 0.0:
        raise PairGenerationError(f"{field_name} must be >= 0, got {number}.")
    return number


def _require_unit_interval(value: Any, field_name: str) -> float:
    number = _require_non_negative_finite(value, field_name)
    if number > 1.0:
        raise PairGenerationError(f"{field_name} must be in [0, 1], got {number}.")
    return number


def _require_optional_positive_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise PairGenerationError(f"{field_name} must be an integer or None.")
    if value < 1:
        raise PairGenerationError(f"{field_name} must be >= 1 when set, got {value}.")
    return value


@dataclass(frozen=True)
class PairGenerationConfig:
    """Sampling configuration for leakage-safe pair generation.

    positive_negative_ratio
        Target negatives per selected positive (1.0 => approximately 1:1).
    same_category_negative_ratio
        Fraction of selected negatives that are same-category
        (0.5 => approximately 50% same-category / 50% cross-category).
    max_positive_pairs / max_negative_pairs
        Optional global caps after per-split generation. None means no cap.
    """

    seed: int = DEFAULT_PAIR_SEED
    positive_negative_ratio: float = DEFAULT_POSITIVE_NEGATIVE_RATIO
    same_category_negative_ratio: float = DEFAULT_SAME_CATEGORY_NEGATIVE_RATIO
    max_positive_pairs: Optional[int] = None
    max_negative_pairs: Optional[int] = None
    include_train: bool = True
    include_valid: bool = True
    include_test: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _require_seed(self.seed))
        object.__setattr__(
            self,
            "positive_negative_ratio",
            _require_non_negative_finite(
                self.positive_negative_ratio, "positive_negative_ratio"
            ),
        )
        object.__setattr__(
            self,
            "same_category_negative_ratio",
            _require_unit_interval(
                self.same_category_negative_ratio, "same_category_negative_ratio"
            ),
        )
        object.__setattr__(
            self,
            "max_positive_pairs",
            _require_optional_positive_int(self.max_positive_pairs, "max_positive_pairs"),
        )
        object.__setattr__(
            self,
            "max_negative_pairs",
            _require_optional_positive_int(self.max_negative_pairs, "max_negative_pairs"),
        )
        object.__setattr__(self, "include_train", _require_bool(self.include_train, "include_train"))
        object.__setattr__(self, "include_valid", _require_bool(self.include_valid, "include_valid"))
        object.__setattr__(self, "include_test", _require_bool(self.include_test, "include_test"))
        if not self.included_splits:
            raise PairGenerationError(
                "At least one of include_train, include_valid, include_test must be True."
            )

    @property
    def included_splits(self) -> Tuple[str, ...]:
        flags = {
            "train": self.include_train,
            "valid": self.include_valid,
            "test": self.include_test,
        }
        return tuple(split for split in SPLIT_ORDER if flags[split])

    def as_loggable_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["policy"] = "s1.10-group-aware-pairs"
        payload["included_splits"] = list(self.included_splits)
        return payload
