#!/usr/bin/env python3
"""Audit Dataset 1 cleaning state (S1.4).

Non-destructive by design: never deletes, moves, renames, or rewrites images,
and never modifies the authoritative manifest.
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

from data.cleaning import (  # noqa: E402
    CleaningAuditConfig,
    audit_dataset_cleaning,
    write_cleaning_report,
)

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "dataset" / "dataset_cleaning_report.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S1.4 Dataset Cleaning audit. Verifies the cleaned Dataset 1 "
            "baseline. Never modifies images or the manifest."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to Dataset 1 (directory containing train/valid/test).",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the Dataset 1 CSV manifest.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON path for the cleaning audit report.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip S1.2 image inspection (contract/filesystem/groups only).",
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Skip S1.3 duplicate detection.",
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=5,
        help="Near-duplicate Hamming threshold (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    config = CleaningAuditConfig(
        mode="audit",
        run_image_inspection=not args.skip_images,
        run_duplicate_detection=not args.skip_duplicates,
        near_duplicate_threshold=args.near_threshold,
    )
    report = audit_dataset_cleaning(
        args.dataset_root,
        args.manifest,
        config=config,
    )
    output_path = Path(args.output).expanduser().resolve()
    write_cleaning_report(report, output_path)

    current = report["current_state"]
    reqs = report["requirements"]
    print("=" * 60)
    print("S1.4 Dataset Cleaning audit")
    print("=" * 60)
    print(f"Mode:          {report['mode']} (destructive={report['destructive_operations']})")
    print(f"Report:        {output_path}")
    print(f"Status:        {report['status']}")
    print(f"Images:        {current['images']}")
    print(f"Filesystem:    {current['filesystem_images']}")
    print(f"Readable:      {current['readable']}")
    print(f"Corrupted:     {current['corrupted']}")
    print(f"Warnings:      {current['warnings']}")
    print(f"Invalid:       {current['invalid']}")
    print(f"Exact dups:    {current['exact_duplicates']}")
    print(f"Perceptual:    {current['perceptual_duplicates']}")
    print(f"Near pairs:    {current['near_duplicate_pairs']}")
    print(f"Groups:        {current['groups']}")
    print(f"Cross-split:   {current['cross_split_groups']}")
    print()
    print("Requirements:")
    for key in (
        "corrupted_images",
        "duplicate_removal",
        "format_standardization",
        "metadata",
        "invalid_data",
    ):
        item = reqs[key]
        print(f"  {item['requirement']:<28} {item['status']}")
        print(f"    evidence: {item['evidence']}")
        print(f"    action:   {item['action']}")
    if report["unexpected_dataset_state"]:
        print()
        print("Unexpected dataset state:")
        for item in report["unexpected_dataset_state"]:
            print(f"  [{item['check']}] {item['message']}")
    print("=" * 60)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
