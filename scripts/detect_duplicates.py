#!/usr/bin/env python3
"""Detect exact, perceptual, and near duplicates (S1.3).

Read-only: does not modify dataset files or manifests.
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

from data.duplicates import (  # noqa: E402
    DEFAULT_EXACT_HASH,
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    DEFAULT_PERCEPTUAL_HASH,
    DuplicateDetectionConfig,
    detect_duplicates,
    write_duplicate_report,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "dataset" / "duplicate_report.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect exact, perceptual, and near-duplicate images. "
            "Reports only; never modifies images or manifests."
        )
    )
    parser.add_argument(
        "--dataset-a",
        required=True,
        help="Root directory of the primary dataset (Dataset 1 for the MVP).",
    )
    parser.add_argument(
        "--dataset-b",
        default=None,
        help="Optional second dataset root for cross-dataset comparison.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON report path (default: reports/dataset/duplicate_report.json).",
    )
    parser.add_argument(
        "--manifest-a",
        default=None,
        help="Optional CSV manifest for dataset A (read-only; adds image_id/split).",
    )
    parser.add_argument(
        "--manifest-b",
        default=None,
        help="Optional CSV manifest for dataset B (read-only).",
    )
    parser.add_argument(
        "--exact-hash",
        default=DEFAULT_EXACT_HASH,
        choices=["sha256", "sha1", "md5"],
        help=f"File content hash (default: {DEFAULT_EXACT_HASH}).",
    )
    parser.add_argument(
        "--perceptual-hash",
        default=DEFAULT_PERCEPTUAL_HASH,
        choices=["phash", "dhash", "ahash"],
        help=f"Perceptual hash (default: {DEFAULT_PERCEPTUAL_HASH}).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
        help=(
            "Max Hamming distance for near-duplicates on a 64-bit hash "
            f"(default: {DEFAULT_NEAR_DUPLICATE_THRESHOLD}; lower is stricter). "
            "Distance 0 is perceptual identity and is reported separately."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Skip an image if its candidate set exceeds this size (0 = unlimited).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Stop after this many Hamming comparisons (0 = unlimited).",
    )
    parser.add_argument(
        "--comparison-mode",
        default="both",
        choices=["intra", "cross", "both"],
        help="intra = within each dataset; cross = A↔B only; both = default.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="Image file extensions to include.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    config = DuplicateDetectionConfig(
        image_extensions=tuple(args.extensions),
        exact_hash=args.exact_hash,
        perceptual_hash=args.perceptual_hash,
        threshold=args.threshold,
        max_candidates=args.max_candidates,
        max_pairs=args.max_pairs,
        comparison_mode=args.comparison_mode,
    )
    report = detect_duplicates(
        args.dataset_a,
        args.dataset_b,
        config=config,
        manifest_a=args.manifest_a,
        manifest_b=args.manifest_b,
    )
    output_path = Path(args.output).expanduser()
    write_duplicate_report(report, output_path)

    summary = report["summary"]
    search = summary["near_duplicate_search"]
    print("=" * 60)
    print("S1.3 duplicate detection")
    print("=" * 60)
    print(f"Report:                    {output_path}")
    print(f"Total images:              {summary['total_images']}")
    print(f"Exact duplicate groups:    {summary['exact_duplicate_groups']}")
    print(f"Perceptual duplicate groups:{summary['perceptual_duplicate_groups']}")
    print(f"Near-duplicate pairs:      {summary['near_duplicate_pairs']}")
    print(f"Cross-dataset matches:     {summary['cross_dataset_matches']}")
    print(f"Near-duplicate search:     {search['status']}")
    print(f"  comparisons completed:   {search['comparisons_completed']}")
    print(f"  comparisons skipped:     {search['comparisons_skipped']}")
    if search["skip_reasons"]:
        print(f"  skip reasons:            {', '.join(search['skip_reasons'])}")
    if report["limitations"]:
        print("Limitations:")
        for item in report["limitations"]:
            print(f"  [{item['code']}] {item['message']}")
    print("=" * 60)
    if search["status"] != "complete":
        logger.warning(
            "Near-duplicate search was incomplete. Do not treat the near-duplicate "
            "count as evidence that none exist."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
