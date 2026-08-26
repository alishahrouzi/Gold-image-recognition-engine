"""Pair contract for S1.10.

Pairs are relationships between manifest images. They do not load pixels
and they do not replace the image-level manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from ..errors import PairGenerationError
from ..types import Sample

PAIR_TYPE_POSITIVE = "positive"
PAIR_TYPE_NEGATIVE = "negative"
NEGATIVE_TYPE_SAME_CATEGORY = "same_category"
NEGATIVE_TYPE_CROSS_CATEGORY = "cross_category"

PAIR_CSV_FIELDS: Tuple[str, ...] = (
    "pair_id",
    "image_id_1",
    "image_id_2",
    "group_id_1",
    "group_id_2",
    "category_id_1",
    "category_id_2",
    "category_1",
    "category_2",
    "split",
    "label",
    "pair_type",
    "negative_type",
)


def canonicalize_image_ids(image_id_a: str, image_id_b: str) -> Tuple[str, str]:
    """Return an unordered pair in lexicographic order.

    (A, B) and (B, A) canonicalize to the same tuple. Self-pairs are invalid.
    """
    if not image_id_a or not image_id_b:
        raise PairGenerationError("Pair image IDs must be non-empty.")
    if image_id_a == image_id_b:
        raise PairGenerationError(f"Self-pair is invalid: {image_id_a!r}.")
    if image_id_a < image_id_b:
        return image_id_a, image_id_b
    return image_id_b, image_id_a


def make_pair_id(image_id_a: str, image_id_b: str) -> str:
    """Deterministic pair_id from the canonical unordered image IDs."""
    left, right = canonicalize_image_ids(image_id_a, image_id_b)
    return f"{left}__{right}"


@dataclass(frozen=True)
class Pair:
    """One unordered image pair with product-identity labels."""

    pair_id: str
    image_id_1: str
    image_id_2: str
    group_id_1: str
    group_id_2: str
    category_id_1: int
    category_id_2: int
    category_1: str
    category_2: str
    split: str
    label: int
    pair_type: str
    negative_type: Optional[str]

    def unordered_key(self) -> Tuple[str, str]:
        return (self.image_id_1, self.image_id_2)

    def to_csv_row(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "image_id_1": self.image_id_1,
            "image_id_2": self.image_id_2,
            "group_id_1": self.group_id_1,
            "group_id_2": self.group_id_2,
            "category_id_1": str(self.category_id_1),
            "category_id_2": str(self.category_id_2),
            "category_1": self.category_1,
            "category_2": self.category_2,
            "split": self.split,
            "label": str(self.label),
            "pair_type": self.pair_type,
            "negative_type": "" if self.negative_type is None else self.negative_type,
        }


def pair_from_csv_row(row: Mapping[str, Optional[str]]) -> Pair:
    """Parse one pair CSV row into the S1.10 Pair contract. Does not write files."""
    missing = [name for name in PAIR_CSV_FIELDS if name not in row]
    if missing:
        raise PairGenerationError(
            f"Pair CSV row is missing required columns: {sorted(missing)}."
        )

    def _text(field_name: str) -> str:
        return (row.get(field_name) or "").strip()

    def _int(field_name: str) -> int:
        raw = _text(field_name)
        try:
            return int(raw)
        except ValueError as exc:
            raise PairGenerationError(
                f"Invalid {field_name} {raw!r} in pair CSV."
            ) from exc

    negative_raw = _text("negative_type")
    return Pair(
        pair_id=_text("pair_id"),
        image_id_1=_text("image_id_1"),
        image_id_2=_text("image_id_2"),
        group_id_1=_text("group_id_1"),
        group_id_2=_text("group_id_2"),
        category_id_1=_int("category_id_1"),
        category_id_2=_int("category_id_2"),
        category_1=_text("category_1"),
        category_2=_text("category_2"),
        split=_text("split"),
        label=_int("label"),
        pair_type=_text("pair_type"),
        negative_type=negative_raw or None,
    )


def pair_from_samples(
    sample_a: Sample,
    sample_b: Sample,
    *,
    label: int,
    pair_type: str,
    negative_type: Optional[str],
) -> Pair:
    """Build a canonical unordered Pair from two manifest samples."""
    if sample_a.split != sample_b.split:
        raise PairGenerationError(
            f"Cross-split pair is invalid: {sample_a.image_id!r} ({sample_a.split}) "
            f"<-> {sample_b.image_id!r} ({sample_b.split})."
        )
    left_id, right_id = canonicalize_image_ids(sample_a.image_id, sample_b.image_id)
    left = sample_a if sample_a.image_id == left_id else sample_b
    right = sample_b if left is sample_a else sample_a
    return Pair(
        pair_id=make_pair_id(left_id, right_id),
        image_id_1=left.image_id,
        image_id_2=right.image_id,
        group_id_1=left.group_id,
        group_id_2=right.group_id,
        category_id_1=left.category_id,
        category_id_2=right.category_id,
        category_1=left.category,
        category_2=right.category,
        split=left.split,
        label=label,
        pair_type=pair_type,
        negative_type=negative_type,
    )


def sort_pairs(pairs: Sequence[Pair], split_order: Sequence[str]) -> Tuple[Pair, ...]:
    """Stable output order: split order, then pair_id."""
    rank = {name: index for index, name in enumerate(split_order)}
    return tuple(
        sorted(
            pairs,
            key=lambda pair: (rank.get(pair.split, len(rank)), pair.pair_id),
        )
    )
