"""Run S2.9 error analysis over the existing S2.7 gallery.

This is analysis-only: it loads the persisted S2.7 gallery, reuses the existing
TopKRetriever and S2.8 scoring contract, and writes an evidence report.
It does not train, rebuild embeddings, modify the manifest, or touch test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders.manifest import load_manifest
from src.evaluation.error_analysis import (
    build_error_record,
    build_summary,
    category_confusion,
    category_summary,
    select_representative_examples,
)
from src.evaluation.errors import EvaluationError
from src.evaluation.leakage import DATASET1_EXPECTED, manifest_sha256, verify_dataset1_contract
from src.evaluation.result import EVALUATION_K
from src.evaluation.retrieval_evaluator import score_retrieval_result
from src.retrieval.gallery import Gallery
from src.retrieval.topk import TopKRetriever

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_GALLERY_DIR = PROJECT_ROOT / "experiments" / "retrieval" / "baseline"
DEFAULT_SOURCE_REPORT = PROJECT_ROOT / "reports" / "evaluation" / "s2.8_baseline_evaluation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "evaluation" / "s2.9_baseline_error_analysis.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run S2.9 baseline retrieval error analysis.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--gallery-dir", default=str(DEFAULT_GALLERY_DIR))
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=EVALUATION_K)
    return parser


def _sample_path_map(samples: list[Any]) -> dict[str, str]:
    return {sample.image_id: str(sample.image_path) for sample in samples}


def _group_indices(metadata: tuple[dict[str, Any], ...]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[str(item["product_group"])].append(index)
    return groups


def _load_source_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"S2.8 source report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "EXP-0002" or payload.get("task") != "S2.8":
        raise EvaluationError("The supplied source report is not EXP-0002 / S2.8.")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    if args.k != EVALUATION_K:
        raise EvaluationError(f"S2.9 uses the S2.8 single ranking with k={EVALUATION_K}.")
    if args.per_group < 1:
        raise EvaluationError("--per-group must be positive.")
    if args.per_category < 1:
        raise EvaluationError("--per-category must be positive.")

    manifest = Path(args.manifest)
    gallery_dir = Path(args.gallery_dir)
    source_report_path = Path(args.source_report)
    output = Path(args.output)
    embeddings_path = gallery_dir / "gallery_embeddings.pt"
    metadata_path = gallery_dir / "gallery_metadata.json"

    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not embeddings_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "S2.7 gallery artifact not found. Expected gallery_embeddings.pt and gallery_metadata.json."
        )

    source_report = _load_source_report(source_report_path)
    samples = load_manifest(manifest, dataset_root=args.dataset_root, validate_files=False)
    contract = verify_dataset1_contract(samples, validate_files=False)
    manifest_hash = manifest_sha256(manifest)
    recorded_hash = source_report["dataset"]["manifest_sha256"]
    if manifest_hash != recorded_hash:
        raise EvaluationError(
            "Current Dataset 1 manifest hash does not match EXP-0002; S2.9 must analyze the same dataset version."
        )
    path_by_id = _sample_path_map(samples)

    gallery = Gallery.load(embeddings_path, metadata_path)
    if gallery.size != DATASET1_EXPECTED["train_images"]:
        raise EvaluationError(
            f"Expected {DATASET1_EXPECTED['train_images']} train gallery images, got {gallery.size}."
        )
    if gallery.embedding_dim != int(source_report["model"]["embedding_dim"]):
        raise EvaluationError("S2.7 gallery embedding dimension does not match EXP-0002.")

    unexpected_splits = sorted(
        {
            str(item.get("split", ""))
            for item in gallery.metadata
            if item.get("split") not in (None, "train", "")
        }
    )
    if unexpected_splits:
        raise EvaluationError(f"S2.9 requires the train gallery; found splits: {unexpected_splits}.")

    retriever = TopKRetriever(gallery.embeddings, gallery.metadata)
    group_indices = _group_indices(gallery.metadata)
    records = []

    for index, item in enumerate(gallery.metadata):
        query_id = str(item["image_id"])
        query_group = str(item["product_group"])
        positive_images = len(group_indices[query_group]) - 1
        positive_products = 1 if positive_images > 0 else 0
        result = retriever.retrieve(
            gallery.embeddings[index],
            query_id=query_id,
            k=args.k,
            exclude_image_id=query_id,
        )
        scored = score_retrieval_result(
            result,
            query_group_id=query_group,
            query_image_id=query_id,
            split="train",
            gallery_split="train",
            requested_k=args.k,
            gallery_positive_images=positive_images,
            gallery_positive_products=positive_products,
            category=str(item.get("category") or "") or None,
            self_match_excluded=True,
        )
        candidate_paths = {
            candidate.image_id: path_by_id.get(candidate.image_id, "")
            for candidate in result.candidates
        }
        records.append(
            build_error_record(
                result,
                scored,
                query_image_path=path_by_id.get(query_id),
                candidate_paths=candidate_paths,
            )
        )

    summary = build_summary(records)
    expected_queries = int(source_report["diagnostics"]["number_of_queries"])
    if summary["number_of_queries"] != expected_queries:
        raise EvaluationError(
            f"S2.9 analyzed {summary['number_of_queries']} queries, expected EXP-0002 count {expected_queries}."
        )
    if summary["valid_queries"] != int(source_report["diagnostics"]["queries_with_at_least_one_positive"]):
        raise EvaluationError("S2.9 valid-query count does not match EXP-0002.")

    report = {
        "experiment_id": "EXP-0003",
        "task": "S2.9",
        "policy": "s2.9-baseline-error-analysis-v2",
        "status": "COMPLETED",
        "source_experiment": "EXP-0002",
        "dataset": {
            "dataset_source": "dataset1",
            "manifest": str(manifest),
            "manifest_sha256": manifest_hash,
            "query_split": "train",
            "gallery_split": "train",
            "query_protocol": "leave-one-image-out",
            "total_images": DATASET1_EXPECTED["total_images"],
            "total_groups": DATASET1_EXPECTED["total_groups"],
            "gallery_count": gallery.size,
            "contract": contract,
        },
        "model": {
            "checkpoint": source_report["checkpoint"]["checkpoint_path"],
            "embedding_dimension": gallery.embedding_dim,
        },
        "retrieval": {
            "similarity": source_report["retrieval"]["similarity_method"],
            "k": args.k,
            "self_image_exclusion": "exclude_image_id (S2.7 TopKRetriever)",
            "product_positive_definition": "candidate.product_group == query.product_group",
        },
        "baseline_metrics": source_report["metrics"],
        "summary": summary,
        "per_category": category_summary(records),
        "category_confusion": category_confusion(records),
        "representative_examples": select_representative_examples(
            records,
            per_group=args.per_group,
            per_category=args.per_category,
            seed=args.seed,
        ),
        "manual_visual_review": {
            "angle_analysis": {
                "status": "manual_review_required",
                "note": "No angle annotation is inferred automatically from retrieval output.",
            },
            "background_analysis": {
                "status": "manual_review_required",
                "note": "No background cause is inferred automatically from retrieval output.",
            },
        },
        "findings": [
            "Automated analysis identifies retrieval and category failure patterns only.",
            "Category-stratified error samples are included to prevent dominant categories from being underrepresented during review.",
            "Top-10 image candidates are additionally collapsed into product evidence without claiming a true Top-10 product retrieval.",
            "Observed similarity margins are included only when a positive product is present in the retrieved image candidates.",
            "Visual causes such as viewpoint or background require inspection of representative image pairs.",
        ],
        "hypotheses_for_s3": [],
        "protocol_notes": [
            "Test split was not used.",
            "Model was not retrained.",
            "S2.7 retrieval behavior was not modified.",
            "S2.8 aggregate baseline metrics were not changed.",
            "top10_product_evidence describes products represented in the S2.7 Top-10 image results; it is not a separate Top-10 product retrieval run.",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote S2.9 report: {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
