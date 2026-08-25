#!/usr/bin/env python3
"""Validate Dataset 1 ingestion and image quality (S1.1 + S1.2).

Reads the authoritative manifest. Does not modify images or the manifest.
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

from data.image_quality import (  # noqa: E402
    build_image_validation_report,
    inspect_samples,
    write_image_validation_report,
)
from data.loaders.manifest import load_manifest  # noqa: E402
from data.validation import validate_samples  # noqa: E402

logger = logging.getLogger(__name__)

# Reference totals from Dataset 1 QA. Used only by this script for comparison.
REFERENCE_TOTAL_SAMPLES = 4969
REFERENCE_SPLIT_COUNTS = {"train": 4328, "valid": 429, "test": 212}
REFERENCE_TOTAL_GROUPS = 2135

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_IMAGE_REPORT = PROJECT_ROOT / "reports" / "dataset" / "dataset_validation_report.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Dataset 1 manifest contract and image files."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the Dataset 1 CSV manifest.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to Dataset 1 (directory containing train/valid/test).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_IMAGE_REPORT),
        help="JSON path for the image-level validation report.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Run only S1.1 contract/count checks (do not decode images).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    # File existence is reported per image in S1.2 rather than aborting here.
    samples = load_manifest(
        manifest_path,
        dataset_root=dataset_root,
        validate_files=False,
    )
    report = validate_samples(samples, validate_files=False)

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

    if args.skip_images:
        return 0

    results = inspect_samples(samples)
    image_report = build_image_validation_report(
        results,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
    )
    output_path = Path(args.output).expanduser().resolve()
    write_image_validation_report(image_report, output_path)

    summary = image_report["summary"]
    resolution = image_report["resolution"]
    print()
    print("=" * 60)
    print("Dataset 1 image validation")
    print("=" * 60)
    print(f"Report:         {output_path}")
    print(f"Valid:          {summary['valid_images']}")
    print(f"Warning:        {summary['warning_images']}")
    print(f"Invalid:        {summary['invalid_images']}")
    print(f"Readable:       {summary['readable_images']}")
    print(f"Corrupted:      {summary['corrupted_images']}")
    print(f"Abnormal:       {summary['abnormal_images']}")
    print(f"RGB:            {summary['rgb_images']}")
    print(f"Non-RGB:        {summary['non_rgb_images']}")
    print(f"RGB convertible:{summary['rgb_convertible_images']}")
    print()
    print("Resolution:")
    print(f"  width  {resolution['min_width']} .. {resolution['max_width']}  avg {resolution['avg_width']}")
    print(f"  height {resolution['min_height']} .. {resolution['max_height']}  avg {resolution['avg_height']}")
    print()
    print("Color modes:")
    for mode, count in image_report["color_modes"].items():
        print(f"  {mode:<8} {count}")
    print()
    print("Split statistics:")
    for split, stats in image_report["split_statistics"].items():
        print(
            f"  {split:<6} total={stats['total']} readable={stats['readable']} "
            f"unreadable={stats['unreadable']} abnormal={stats['abnormal']}"
        )
    print()
    print("Category statistics:")
    for category, stats in image_report["category_statistics"].items():
        print(
            f"  {category:<12} total={stats['total']} readable={stats['readable']} "
            f"unreadable={stats['unreadable']} abnormal={stats['abnormal']}"
        )
    print("=" * 60)
    if summary["invalid_images"]:
        logger.warning(
            "Image validation found %s invalid file(s). See %s",
            summary["invalid_images"],
            output_path,
        )
    else:
        logger.info("No invalid/unreadable images were found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
