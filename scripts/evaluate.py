"""CLI entry point for running the product-retrieval evaluation.

This script only wires things together:
    1. Load a dataset manifest CSV.
    2. Dynamically load an EmbeddingModel implementation (not bundled here
       -- the actual CNN/embedding model is connected later).
    3. Run RetrievalEvaluator.
    4. Save a JSON report and a human-readable text report.

Example:
    python scripts/evaluate.py \\
        --manifest data/manifest.csv \\
        --model-module my_project.models \\
        --model-class MyCnnEmbeddingModel \\
        --output-dir reports
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import List

# Make `src/` importable without requiring the package to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.evaluation.evaluator import (  # noqa: E402
    EvaluationReport,
    ManifestRecord,
    RetrievalEvaluator,
)

REQUIRED_MANIFEST_COLUMNS = {"image_id", "image_path", "group_id", "category", "split"}


def load_manifest(manifest_path: str) -> List[ManifestRecord]:
    """Load a dataset manifest CSV into a list of ManifestRecord.

    Expected columns: image_id, image_path, group_id, category, split[, source]

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If required columns are missing or the manifest is empty.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    records: List[ManifestRecord] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_MANIFEST_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        for row in reader:
            records.append(
                ManifestRecord(
                    image_id=row["image_id"],
                    image_path=row["image_path"],
                    group_id=row["group_id"],
                    category=row["category"],
                    split=row["split"],
                    source=row.get("source") or None,
                )
            )

    if not records:
        raise ValueError(f"Manifest at {manifest_path} contains no rows.")
    return records


def load_embedding_model(model_module: str, model_class: str):
    """Dynamically import and instantiate the embedding model.

    Kept dynamic so this evaluation script has zero dependency on any
    specific model implementation -- only the module/class name is
    required. The model must implement EmbeddingModel.encode(image_path).
    """
    module = importlib.import_module(model_module)
    cls = getattr(module, model_class)
    return cls()


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_report_text(report: EvaluationReport) -> str:
    """Render the human-readable report matching the project spec's format."""
    lines = [
        "Evaluation Results",
        "==================",
        "",
        "Queries:",
        f"    Total: {report.total_queries}",
        f"    Valid: {report.valid_queries}",
        f"    Excluded: {report.excluded_queries}",
        "",
        "Overall:",
        "",
    ]

    o = report.overall
    lines += [
        f"Top-1:        {_pct(o.hit_at_1)}",
        f"Top-5:        {_pct(o.hit_at_5)}",
        f"Top-10:       {_pct(o.hit_at_10)}",
        "",
        f"Precision@1:  {_pct(o.precision_at_1)}",
        f"Precision@5:  {_pct(o.precision_at_5)}",
        f"Precision@10: {_pct(o.precision_at_10)}",
        "",
        f"Recall@1:     {_pct(o.recall_at_1)}",
        f"Recall@5:     {_pct(o.recall_at_5)}",
        f"Recall@10:    {_pct(o.recall_at_10)}",
        "",
        f"MRR:          {_pct(o.mrr)}",
        "",
        "Per-category results:",
        "",
    ]

    for category, m in sorted(report.per_category.items()):
        lines += [
            category,
            f"    Queries:      {m.num_queries}",
            f"    Top-1:        {_pct(m.hit_at_1)}",
            f"    Top-5:        {_pct(m.hit_at_5)}",
            f"    Top-10:       {_pct(m.hit_at_10)}",
            f"    Precision@1:  {_pct(m.precision_at_1)}",
            f"    Precision@5:  {_pct(m.precision_at_5)}",
            f"    Precision@10: {_pct(m.precision_at_10)}",
            f"    Recall@1:     {_pct(m.recall_at_1)}",
            f"    Recall@5:     {_pct(m.recall_at_5)}",
            f"    Recall@10:    {_pct(m.recall_at_10)}",
            f"    MRR:          {_pct(m.mrr)}",
            "",
        ]

    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run product-level image retrieval evaluation on a manifest split."
    )
    parser.add_argument(
        "--manifest", required=True, help="Path to the dataset manifest CSV."
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Manifest split to evaluate (default: test, the official split).",
    )
    parser.add_argument(
        "--model-module",
        required=True,
        help="Python module containing the EmbeddingModel implementation, "
        "e.g. 'my_project.models'.",
    )
    parser.add_argument(
        "--model-class",
        required=True,
        help="Class name implementing EmbeddingModel.encode(image_path).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write the JSON and text reports to (default: reports).",
    )
    parser.add_argument(
        "--report-name",
        default="evaluation_report",
        help="Base filename (without extension) for the reports.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    records = load_manifest(args.manifest)
    model = load_embedding_model(args.model_module, args.model_class)

    evaluator = RetrievalEvaluator(records=records, embedding_model=model, split=args.split)
    report = evaluator.evaluate()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{args.report_name}.json"
    text_path = output_dir / f"{args.report_name}.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    report_text = format_report_text(report)
    with text_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved JSON report to: {json_path}")
    print(f"Saved text report to: {text_path}")


if __name__ == "__main__":
    main()
