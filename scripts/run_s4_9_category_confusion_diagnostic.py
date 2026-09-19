"""Run the frozen S4.9 category-confusion diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.s4_9_category_confusion import (  # noqa: E402
    DEFAULT_ANNOTATIONS,
    DEFAULT_OUTPUT,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.annotations, args.output)
    print(json.dumps({
        "model": report["protocol"]["model"],
        "queries": report["protocol"]["queries"],
        "judgments": report["protocol"]["judgments"],
        "top1_mismatch_rate": report["top1"]["category_mismatch_rate"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
