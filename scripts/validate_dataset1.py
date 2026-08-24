#!/usr/bin/env python3
"""Validate Dataset 1 ingestion against the authoritative manifest.

Prints counts derived from the manifest. Expected production totals are
reported for comparison only and are not hard-coded in src/data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.datasets.unified_dataset import UnifiedDataset  # noqa: E402
from data.validation import validate_samples  # noqa: E402

logger = logging.getLogger(__name__)

# Reference totals from Dataset 1 QA. Used only by this script for comparison.
REFERENCE_TOTAL_SAMPLES = 4969
REFERENCE_SPLIT_COUNTS = {"train": 4328, "valid": 429, "test": 212}
REFERENCE_TOTAL_GROUPS = 2135


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Dataset 1 ingestion.")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"),
        help="Path to the Dataset 1 CSV manifest.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to Dataset 1 (directory containing train/valid/test).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    dataset = UnifiedDataset(
        manifest_path,
        dataset_root=dataset_root,
        validate_files=True,
    )
    report = validate_samples(dataset.samples, validate_files=True)

    print("=" * 60)
    print("Dataset 1 ingestion validation")
    print("=" * 60)
    print(f"Manifest:       {manifest_path}")
    print(f"Dataset root:   {dataset_root}")
    print(f"Total samples:  {report.total_samples}  (reference {REFERENCE_TOTAL_SAMPLES})")
    print(f"Total groups:   {report.total_groups}  (reference {REFERENCE_TOTAL_GROUPS})")
    print()
    print("Split counts:")
    for split, count in report.split_counts.items():
        reference = REFERENCE_SPLIT_COUNTS.get(split)
        extra = f"  (reference {reference})" if reference is not None else ""
        print(f"  {split:<6} {count}{extra}")
    print()
    print("Category counts:")
    for category, count in report.category_counts.items():
        print(f"  {category:<12} {count}")
    print()
    print("Groups per split:")
    for split, count in report.groups_per_split.items():
        print(f"  {split:<6} {count}")
    print("=" * 60)

    mismatches = []
    if report.total_samples != REFERENCE_TOTAL_SAMPLES:
        mismatches.append("total_samples")
    if report.total_groups != REFERENCE_TOTAL_GROUPS:
        mismatches.append("total_groups")
    for split, expected in REFERENCE_SPLIT_COUNTS.items():
        if report.split_counts.get(split) != expected:
            mismatches.append(f"split:{split}")
    if mismatches:
        logger.warning("Counts differ from reference QA totals: %s", ", ".join(mismatches))
        logger.warning("The manifest is authoritative; inspect the dataset if this is unexpected.")
    else:
        logger.info("Counts match the Dataset 1 QA reference totals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
