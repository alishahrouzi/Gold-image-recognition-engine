#!/usr/bin/env python3
"""Generate the Dataset 1 CSV manifest from the existing S0.3 group analysis.

This script is NOT part of the Dataset Loader. It reuses filename parsing
from data/analyze_groups.py once, then writes group_id into the manifest.
UnifiedDataset only reads that manifest and never parses filenames.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.constants import CATEGORY_TO_ID, SOURCE_DATASET1, SPLIT_ORDER  # noqa: E402
from data.errors import DatasetIngestionError  # noqa: E402

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "group_id",
    "category",
    "category_id",
    "split",
    "source",
]


def _load_analyze_groups() -> Any:
    """Import the S0.3 analyze_groups module from the project data/ folder."""
    module_path = PROJECT_ROOT / "data" / "analyze_groups.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"S0.3 group analysis script not found: {module_path}"
        )
    spec = importlib.util.spec_from_file_location("analyze_groups", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses inspect sys.modules[cls.__module__]; register before exec.
    sys.modules["analyze_groups"] = module
    spec.loader.exec_module(module)
    return module


def build_manifest_rows(dataset_root: Path) -> List[dict]:
    """Scan Dataset 1 via S0.3 and return manifest row dicts.

    group_id is taken from parse_dataset1_filename. Unparsed files fail the
    generator rather than inventing a group id.
    """
    analyze_groups = _load_analyze_groups()
    scan_result = analyze_groups.scan_dataset(dataset_root)
    if not scan_result.records:
        raise DatasetIngestionError(
            f"No images found under {dataset_root} "
            f"(checked splits: {', '.join(SPLIT_ORDER)})."
        )

    rows: List[dict] = []
    unparsed = [record for record in scan_result.records if not record.parsed]
    if unparsed:
        first = unparsed[0]
        raise DatasetIngestionError(
            f"{len(unparsed)} image(s) could not be parsed for group_id. "
            f"First failure: {first.path}"
        )

    ordered = sorted(
        scan_result.records,
        key=lambda record: (SPLIT_ORDER.index(record.split), record.category, record.path),
    )

    for index, record in enumerate(ordered, start=1):
        if record.category not in CATEGORY_TO_ID:
            raise DatasetIngestionError(
                f"Unknown category {record.category!r} for file {record.path}."
            )
        relative_path = Path(record.path).resolve().relative_to(dataset_root.resolve())
        rows.append(
            {
                "image_id": f"DS1_IMG_{index:06d}",
                "image_path": relative_path.as_posix(),
                "group_id": record.group_id,
                "category": record.category,
                "category_id": CATEGORY_TO_ID[record.category],
                "split": record.split,
                "source": SOURCE_DATASET1,
            }
        )
    return rows


def write_manifest(rows: List[dict], output_path: Path) -> None:
    """Write manifest rows to CSV. Does not modify dataset images."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Dataset 1 CSV manifest using S0.3 group_id parsing. "
            "Does not move, rename, or modify images."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to Dataset 1 (directory containing train/valid/test).",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"),
        help="Output CSV path (default: reports/dataset/dataset1_manifest.csv).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        logger.error("Dataset root does not exist or is not a directory: %s", dataset_root)
        return 2

    rows = build_manifest_rows(dataset_root)
    output_path = Path(args.output).expanduser().resolve()
    write_manifest(rows, output_path)
    logger.info("Wrote %s rows to %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
