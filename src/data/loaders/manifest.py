"""Load and validate a dataset manifest.

The manifest is the source of truth. This module does not scan folders or
parse filenames to recover group_id.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..constants import MANIFEST_REQUIRED_FIELDS
from ..errors import DatasetIngestionError
from ..types import Sample
from ..validation import validate_samples

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def resolve_image_path(image_path: PathLike, dataset_root: Optional[PathLike]) -> Path:
    """Resolve a manifest path against ``dataset_root`` when it is relative."""
    path = Path(image_path)
    if path.is_absolute() or dataset_root is None:
        return path
    return Path(dataset_root) / path


def load_manifest(
    manifest_path: PathLike,
    *,
    dataset_root: Optional[PathLike] = None,
    validate_files: bool = True,
) -> List[Sample]:
    """Load Samples from a CSV manifest.

    Expected columns: image_id, image_path, group_id, category, category_id,
    split, source. Any additional columns are stored on Sample.metadata.

    Args:
        manifest_path: Path to the CSV manifest.
        dataset_root: Directory used to resolve relative image_path values.
        validate_files: If True, require every image file to exist.

    Returns:
        Validated Sample list in manifest row order.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        DatasetIngestionError: If the contract is violated.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = MANIFEST_REQUIRED_FIELDS - fieldnames
        if missing:
            raise DatasetIngestionError(
                f"Manifest is missing required columns: {sorted(missing)}."
            )

        extra_fields = [name for name in (reader.fieldnames or []) if name not in MANIFEST_REQUIRED_FIELDS]
        samples: List[Sample] = []
        for row_number, row in enumerate(reader, start=2):
            samples.append(_row_to_sample(row, extra_fields, dataset_root, row_number))

    if not samples:
        raise DatasetIngestionError(f"Manifest at {path} contains no rows.")

    validate_samples(samples, validate_files=validate_files)
    logger.info(
        "Loaded dataset manifest %s (%s samples)",
        path,
        len(samples),
    )
    return samples


def _row_to_sample(
    row: Mapping[str, Optional[str]],
    extra_fields: Sequence[str],
    dataset_root: Optional[PathLike],
    row_number: int,
) -> Sample:
    image_id = (row.get("image_id") or "").strip()
    raw_path = (row.get("image_path") or "").strip()
    group_id = (row.get("group_id") or "").strip()
    category = (row.get("category") or "").strip()
    split = (row.get("split") or "").strip()
    source = (row.get("source") or "").strip()
    category_id_raw = (row.get("category_id") or "").strip()

    try:
        category_id = int(category_id_raw)
    except ValueError as exc:
        raise DatasetIngestionError(
            f"Invalid category_id {category_id_raw!r} for image {image_id!r} "
            f"(manifest row {row_number})."
        ) from exc

    metadata: Dict[str, Any] = {}
    for field_name in extra_fields:
        value = row.get(field_name)
        if value not in (None, ""):
            metadata[field_name] = value

    return Sample(
        image_id=image_id,
        image_path=resolve_image_path(raw_path, dataset_root),
        group_id=group_id,
        category=category,
        category_id=category_id,
        split=split,
        source=source,
        metadata=metadata,
    )
