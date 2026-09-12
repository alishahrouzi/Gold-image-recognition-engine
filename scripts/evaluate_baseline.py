"""S2.8 baseline retrieval evaluation over the existing S2.7 pipeline.

This script is evaluation-only. It does not train, modify the checkpoint,
or write to the Dataset 1 manifest or source images.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch

from src.data.loaders.manifest import load_manifest
from src.evaluation.errors import EvaluationError
from src.evaluation.leakage import (
    DATASET1_EXPECTED,
    assert_manifest_unchanged,
    assert_same_split_protocol,
    manifest_sha256,
    verify_dataset1_contract,
)
from src.evaluation.result import EVALUATION_K
from src.evaluation.retrieval_evaluator import evaluate_leave_one_out_gallery
from src.models.config import ARCHITECTURE_ID
from src.retrieval.embedding import load_baseline_embedding_model
from src.retrieval.gallery import Gallery, GalleryBuilder, build_gallery_loader

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "experiments" / "baseline" / "checkpoints" / "best.pt"
DEFAULT_GALLERY_DIR = PROJECT_ROOT / "experiments" / "retrieval" / "baseline"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation"
_LOCAL_DATASET_ROOT = PROJECT_ROOT.parent / "dataset" / "ai-tool-pool-jewelry-vision"
_LEGACY_DATASET_ROOT = Path(
    r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run S2.8 baseline product-retrieval evaluation.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gallery-dir", default=str(DEFAULT_GALLERY_DIR))
    parser.add_argument(
        "--rebuild-gallery",
        action="store_true",
        help="Rebuild the train gallery from the checkpoint instead of loading the S2.7 artifact.",
    )
    parser.add_argument("--query-split", default="train", choices=("train", "valid", "test"))
    parser.add_argument("--gallery-split", default="train", choices=("train", "valid", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--k", type=int, default=EVALUATION_K)
    return parser


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def resolve_dataset_root(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("ZARGAR_DATASET1_ROOT")
    if env:
        return Path(env)
    if _LEGACY_DATASET_ROOT.is_dir():
        return _LEGACY_DATASET_ROOT
    if _LOCAL_DATASET_ROOT.is_dir():
        return _LOCAL_DATASET_ROOT
    return None


def git_metadata() -> dict[str, str]:
    def _run(args: list[str]) -> str:
        completed = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def hardware_metadata(device: str) -> dict:
    gpu_name = None
    cuda_version = None
    peak_vram_mb = None
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        if device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "device": device,
        "platform": platform.platform(),
        "peak_vram_mb": peak_vram_mb,
    }


def load_checkpoint_metadata(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise EvaluationError("Invalid S2.6 checkpoint: missing model_state_dict.")
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_identifier": checkpoint.name,
        "checkpoint_version": payload.get("checkpoint_version"),
        "epoch": payload.get("epoch"),
        "best_metric": payload.get("best_metric"),
        "config": config,
    }


def load_or_build_gallery(
    *,
    rebuild: bool,
    gallery_dir: Path,
    extractor,
    manifest: Path,
    dataset_root: Path | None,
    split: str,
    batch_size: int,
    seed: int,
    num_workers: int,
    device: str,
) -> tuple[Gallery, float, str]:
    embeddings_path = gallery_dir / "gallery_embeddings.pt"
    metadata_path = gallery_dir / "gallery_metadata.json"
    if not rebuild and embeddings_path.is_file() and metadata_path.is_file():
        started = time.perf_counter()
        gallery = Gallery.load(embeddings_path, metadata_path)
        duration = time.perf_counter() - started
        return gallery, duration, "loaded_s2.7_artifact"

    if dataset_root is None:
        raise EvaluationError(
            "Gallery rebuild requires --dataset-root (or ZARGAR_DATASET1_ROOT / the local Dataset 1 root)."
        )
    started = time.perf_counter()
    loader = build_gallery_loader(
        manifest,
        dataset_root=dataset_root,
        split=split,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )
    gallery = GalleryBuilder(extractor).build(loader)
    duration = time.perf_counter() - started
    return gallery, duration, "rebuilt_from_checkpoint"


def format_markdown(report: dict) -> str:
    metrics = report["metrics"]
    diagnostics = report["diagnostics"]
    runtime = report["runtime"]
    dataset = report["dataset"]
    lines = [
        "# S2.8 Baseline Evaluation",
        "",
        "This report is the **real Dataset 1** baseline evaluation, not the unit-test suite.",
        "",
        "## Experiment metadata",
        "",
        f"- experiment_id: `{report['experiment_id']}`",
        f"- task: `{report['task']}`",
        f"- timestamp: `{report['timestamp']}`",
        f"- git commit: `{report['git_commit']}`",
        f"- git branch: `{report['git_branch']}`",
        f"- checkpoint: `{report['checkpoint']['checkpoint_path']}`",
        f"- embedding dimension: `{report['model']['embedding_dim']}`",
        f"- seed: `{report['seed']}`",
        f"- device: `{report['hardware']['device']}`",
        f"- GPU: `{report['hardware']['gpu_name']}`",
        "",
        "## Dataset",
        "",
        f"- source: `{dataset['dataset_source']}`",
        f"- manifest: `{dataset['manifest']}`",
        f"- query split: `{dataset['query_split']}`",
        f"- gallery split: `{dataset['gallery_split']}`",
        f"- query count: `{dataset['query_count']}`",
        f"- gallery count: `{dataset['gallery_count']}`",
        f"- group count: `{dataset['group_count']}`",
        f"- valid queries (at least one same-product gallery image): `{dataset['valid_query_count']}`",
        f"- excluded singleton queries: `{dataset['excluded_query_count']}`",
        "",
        "## Retrieval",
        "",
        f"- similarity: `{report['retrieval']['similarity_method']}`",
        f"- K: `{report['retrieval']['k']}`",
        f"- self-image exclusion: `{report['retrieval']['self_image_exclusion']}`",
        f"- product-level policy: `{report['retrieval']['product_level_aggregation']}`",
        "",
        "## Metrics (valid queries only)",
        "",
        f"- Top-1: **{metrics['top1']:.6f}**",
        f"- Top-5: **{metrics['top5']:.6f}**",
        f"- Top-10: **{metrics['top10']:.6f}**",
        f"- MRR: **{metrics['mrr']:.6f}**",
        "",
        "## Diagnostics",
        "",
        f"- queries with at least one positive: `{diagnostics['queries_with_at_least_one_positive']}`",
        f"- queries with zero positives: `{diagnostics['queries_with_zero_positives']}`",
        f"- mean first-positive rank: `{diagnostics['mean_first_positive_rank']}`",
        f"- median first-positive rank: `{diagnostics['median_first_positive_rank']}`",
        f"- min/max first-positive rank: `{diagnostics['min_first_positive_rank']}` / `{diagnostics['max_first_positive_rank']}`",
        f"- self-match exclusions: `{diagnostics['self_match_exclusions']}`",
        f"- short rankings: `{diagnostics['short_ranking_queries']}`",
        f"- NaN/Inf count: `{diagnostics['nan_inf_count']}`",
        "",
        "## Runtime",
        "",
        f"- embedding extraction / gallery load: `{runtime['embedding_or_gallery_seconds']:.3f}` s",
        f"- gallery construction source: `{runtime['gallery_source']}`",
        f"- retrieval/evaluation: `{runtime['retrieval_evaluation_seconds']:.3f}` s",
        f"- total: `{runtime['total_seconds']:.3f}` s",
        f"- peak VRAM (MiB): `{report['hardware']['peak_vram_mb']}`",
        "",
        "## Decision",
        "",
        report["decision"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    assert_same_split_protocol(args.query_split, args.gallery_split)
    if args.query_split != "train" or args.gallery_split != "train":
        raise EvaluationError(
            "S2.8 baseline evaluation uses the S2.7 train gallery with train "
            "leave-one-image-out queries. valid/test are one image per group "
            "and are not used for this baseline."
        )
    if args.k != EVALUATION_K:
        raise EvaluationError(f"S2.8 requires k={EVALUATION_K} so Top-1/5/10 and MRR share one ranking.")

    manifest = Path(args.manifest)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    gallery_dir = Path(args.gallery_dir)
    dataset_root = resolve_dataset_root(args.dataset_root)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    started_total = time.perf_counter()
    manifest_hash = manifest_sha256(manifest)
    samples = load_manifest(manifest, dataset_root=dataset_root, validate_files=False)
    dataset_contract = verify_dataset1_contract(samples, validate_files=False)

    device = resolve_device(args.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    checkpoint_meta = load_checkpoint_metadata(checkpoint)
    config_dim = checkpoint_meta["config"].get("embedding_dim")
    if config_dim is not None and int(config_dim) != args.embedding_dim:
        raise EvaluationError(
            f"Checkpoint embedding_dim={config_dim} does not match --embedding-dim={args.embedding_dim}."
        )

    extractor = load_baseline_embedding_model(
        checkpoint,
        embedding_dim=args.embedding_dim,
        device=device,
    )
    extractor.model.eval()

    gallery, gallery_seconds, gallery_source = load_or_build_gallery(
        rebuild=args.rebuild_gallery,
        gallery_dir=gallery_dir,
        extractor=extractor,
        manifest=manifest,
        dataset_root=dataset_root,
        split="train",
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
    )
    if gallery.size != DATASET1_EXPECTED["train_images"]:
        raise EvaluationError(
            f"Unexpected gallery size {gallery.size}; expected {DATASET1_EXPECTED['train_images']}."
        )
    if gallery.embedding_dim != args.embedding_dim:
        raise EvaluationError(
            f"Gallery embedding_dim={gallery.embedding_dim} does not match {args.embedding_dim}."
        )

    retrieval_started = time.perf_counter()
    metrics_result = evaluate_leave_one_out_gallery(
        gallery,
        k=args.k,
        query_split=args.query_split,
        gallery_split=args.gallery_split,
    )
    retrieval_seconds = time.perf_counter() - retrieval_started
    total_seconds = time.perf_counter() - started_total

    assert_manifest_unchanged(manifest, manifest_hash)
    hardware = hardware_metadata(device)
    git = git_metadata()
    groups = {item["product_group"] for item in gallery.metadata}

    report = {
        "experiment_id": "EXP-0002",
        "task": "S2.8",
        "policy": "s2.8-baseline-evaluation-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["git_commit"],
        "git_branch": git["git_branch"],
        "seed": args.seed,
        "hardware": hardware,
        "checkpoint": checkpoint_meta,
        "model": {
            "encoder_architecture": ARCHITECTURE_ID,
            "embedding_dimension": args.embedding_dim,
            "embedding_dim": args.embedding_dim,
            "checkpoint": str(checkpoint),
        },
        "dataset": {
            "dataset_source": "dataset1",
            "manifest": str(manifest),
            "manifest_sha256": manifest_hash,
            "total_images": DATASET1_EXPECTED["total_images"],
            "total_groups": DATASET1_EXPECTED["total_groups"],
            "query_split": args.query_split,
            "gallery_split": args.gallery_split,
            "query_count": metrics_result.num_queries,
            "valid_query_count": metrics_result.num_valid_queries,
            "excluded_query_count": metrics_result.num_excluded_queries,
            "gallery_count": gallery.size,
            "group_count": len(groups),
            "evaluated_groups": metrics_result.diagnostics.get("evaluated_groups"),
            "contract": dataset_contract,
            "augmentation": "disabled (gallery/query use PreprocessedDataset role=valid)",
        },
        "retrieval": {
            "similarity_method": "cosine",
            "k": args.k,
            "self_image_exclusion": "exclude_image_id (S2.7 TopKRetriever)",
            "product_level_aggregation": (
                "S2.7 ranked image candidates collapsed to first-seen product_group; "
                "positive iff candidate.product_group == query.group_id"
            ),
            "gallery_source": gallery_source,
        },
        "metrics": {
            "top1": metrics_result.top1,
            "top5": metrics_result.top5,
            "top10": metrics_result.top10,
            "mrr": metrics_result.mrr,
        },
        "diagnostics": metrics_result.diagnostics,
        "runtime": {
            "embedding_or_gallery_seconds": gallery_seconds,
            "gallery_source": gallery_source,
            "retrieval_evaluation_seconds": retrieval_seconds,
            "total_seconds": total_seconds,
        },
        "decision": (
            "BASELINE. These metrics evaluate the S2.6 category-trained embedding "
            "with S2.7 cosine Top-K on Dataset 1 train leave-one-image-out. "
            "They are not a 128D vs 256D decision and not a test-set final score."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "s2.8_baseline_evaluation.json"
    md_path = output_dir / "s2.8_baseline_evaluation.md"
    queries_path = output_dir / "s2.8_baseline_query_records.jsonl"

    serializable = dict(report)
    json_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(format_markdown(report), encoding="utf-8")
    with queries_path.open("w", encoding="utf-8") as handle:
        for record in metrics_result.query_records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    print(format_markdown(report))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Per-query records: {queries_path}")


if __name__ == "__main__":
    main()
