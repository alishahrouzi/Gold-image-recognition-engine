"""Unified sample contract for dataset ingestion.

Sample holds metadata only. It never opens or decodes image files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Union

from .constants import ALLOWED_SPLITS, CATEGORY_TO_ID, SOURCE_DATASET1, SPLIT_ORDER
from .errors import DatasetIngestionError

PathLike = Union[str, Path]


def _require_non_empty(value: str, field_name: str, image_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        if field_name == "group_id":
            raise DatasetIngestionError(f"Missing group_id for image {image_id!r}.")
        raise DatasetIngestionError(f"{field_name} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True)
class Sample:
    """One image record in the unified dataset contract.

    Fields are taken from the manifest. group_id is never inferred here.
    """

    image_id: str
    image_path: Path
    group_id: str
    category: str
    category_id: int
    split: str
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image_id = _require_non_empty(self.image_id, "image_id", "")
        object.__setattr__(self, "image_id", image_id)

        path = Path(self.image_path)
        if not str(self.image_path).strip() or path.name in {"", "."}:
            raise DatasetIngestionError(
                f"image_path is not a valid path for image {image_id!r}."
            )
        object.__setattr__(self, "image_path", path)

        group_id = _require_non_empty(self.group_id, "group_id", image_id)
        object.__setattr__(self, "group_id", group_id)

        category = _require_non_empty(self.category, "category", image_id)
        if category not in CATEGORY_TO_ID:
            raise DatasetIngestionError(
                f"Unknown category {category!r} for image {image_id!r}."
            )
        object.__setattr__(self, "category", category)

        if not isinstance(self.category_id, int) or isinstance(self.category_id, bool):
            raise DatasetIngestionError(
                f"category_id must be an integer for image {image_id!r}."
            )
        expected_id = CATEGORY_TO_ID[category]
        if self.category_id != expected_id:
            raise DatasetIngestionError(
                f"category_id {self.category_id} does not match category "
                f"{category!r} (expected {expected_id}) for image {image_id!r}."
            )

        split = _require_non_empty(self.split, "split", image_id)
        if split not in ALLOWED_SPLITS:
            allowed = "/".join(SPLIT_ORDER)
            raise DatasetIngestionError(
                f"Invalid split {split!r}. Expected {allowed}."
            )
        object.__setattr__(self, "split", split)

        source = _require_non_empty(self.source, "source", image_id)
        if source != SOURCE_DATASET1:
            raise DatasetIngestionError(
                f"Invalid source {source!r} for image {image_id!r}. "
                f"Expected {SOURCE_DATASET1!r}."
            )
        object.__setattr__(self, "source", source)

        metadata = dict(self.metadata) if self.metadata is not None else {}
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class DatasetItem:
    """Indexed dataset element: contract metadata plus an image payload.

    UnifiedDataset sets ``image`` to an RGB PIL Image. PreprocessedDataset
    replaces it with a float32 CHW tensor after deterministic preprocessing.
    """

    sample: Sample
    image: Any
