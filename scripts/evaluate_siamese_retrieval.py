"""S3.11 metric-learning retrieval evaluation over S3.7 galleries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.evaluation.metric_learning_retrieval import evaluate_gallery, result_to_dict
from src.retrieval.gallery import Gallery

DEFAULT_ROOT = PROJECT_ROOT / "experiments" / "retrieval" / "siamese"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "evaluation" / "s3.11_retrieval_evaluation.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "reports" / "evaluation" / "s3.11_retrieval_evaluation.md"
MODEL_DIRS = {
    "s3.4_hybrid": DEFAULT_ROOT / "s3.4_hybrid",
    "s3.5_random": DEFAULT_ROOT / "s3.5_random",
    "s3.5_same_category": DEFAULT_ROOT / "s3.5_same_category",
    "s3.5_cross_category": DEFAULT_ROOT / "s3.5_cross_category",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run S3.11 product-level retrieval evaluation on persisted Siamese galleries."
    )
    parser.add_argument(
        "--model",
        choices=("all", *MODEL_DIRS.keys()),
        default="all",
        help="Evaluate one model or all four S3.4/S3.5 strategy galleries.",
    )
    parser.add_argument("--gallery-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--k", type=int, default=10)
    return parser


def git_metadata() -> dict[str, str]:
    def run(args: list[str]) -> str:
        completed = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def load_gallery(directory: Path) -> Gallery:
    embeddings = directory / "gallery_embeddings.pt"
    metadata = directory / "gallery_metadata.json"
    if not embeddings.is_file():
        raise FileNotFoundError(f"Gallery embeddings not found: {embeddings}")
    if not metadata.is_file():
        raise FileNotFoundError(f"Gallery metadata not found: {metadata}")
    return Gallery.load(embeddings, metadata)


def markdown_report(report: dict) -> str:
    lines = [
        "# S3.11 — Retrieval Evaluation",
        "",
        "Evaluation of the metric-learning product retrieval path using the persisted S3.7 image-level galleries.",
        "",
        "## Protocol",
        "",
        f"- policy: `{report['policy']}`",
        f"- experiment_id: `{report['experiment_id']}`",
        f"- timestamp: `{report['timestamp']}`",
        f"- git commit: `{report['git']['commit']}`",
        f"- git branch: `{report['git']['branch']}`",
        f"- ranking depth: `K={report['k']}`",
        "- query protocol: leave-one-image-out over the train gallery",
        "- product aggregation: maximum image-to-query cosine similarity per product",
        "- ranking: S3.9 descending raw similarity with deterministic product-id tie-break",
        "- self image: excluded",
        "",
        "## Metrics",
        "",
        "| Model | Top-1 | Top-5 | Top-10 | Precision@1 | Precision@5 | Precision@10 | Recall@1 | Recall@5 | Recall@10 | MRR | Valid Queries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in report["models"].items():
        m = payload["metrics"]
        lines.append(
            f"| {name} | {m['top1']:.6f} | {m['top5']:.6f} | {m['top10']:.6f} | "
            f"{m['precision_at_1']:.6f} | {m['precision_at_5']:.6f} | {m['precision_at_10']:.6f} | "
            f"{m['recall_at_1']:.6f} | {m['recall_at_5']:.6f} | {m['recall_at_10']:.6f} | "
            f"{m['mrr']:.6f} | {payload['query_counts']['valid']} |"
        )

    lines += ["", "## Diagnostics", ""]
    for name, payload in report["models"].items():
        d = payload["diagnostics"]
        lines += [
            f"### {name}",
            "",
            f"- gallery size: `{payload['gallery']['size']}`",
            f"- product count: `{payload['gallery']['product_count']}`",
            f"- embedding dimension: `{payload['gallery']['embedding_dim']}`",
            f"- valid queries: `{payload['query_counts']['valid']}`",
            f"- excluded singleton queries: `{payload['query_counts']['excluded']}`",
            f"- positive outside Top-10: `{d['positive_outside_top10']}`",
            f"- mean first-positive rank: `{d['mean_first_positive_rank']}`",
            f"- median first-positive rank: `{d['median_first_positive_rank']}`",
            f"- self-match exclusions: `{d['self_match_exclusions']}`",
            "",
        ]
        lines.append("Per-category:")
        lines.append("")
        lines.append("| Category | Queries | Top-1 | Top-5 | Top-10 | Precision@5 | Recall@5 | MRR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for category, row in d["per_category"].items():
            lines.append(
                f"| {category} | {row['num_queries']} | {row['top1']:.6f} | {row['top5']:.6f} | "
                f"{row['top10']:.6f} | {row['precision_at_5']:.6f} | {row['recall_at_5']:.6f} | {row['mrr']:.6f} |"
            )
        lines.append("")

    lines += [
        "## Metric definitions",
        "",
        "- **Top-K:** fraction of valid queries whose true product appears within the first K ranked products.",
        "- **Precision@K:** retrieved relevant products divided by K. Because each query has exactly one relevant product in the leave-one-out product-identity protocol, a hit contributes `1/K`.",
        "- **Recall@K:** retrieved relevant products divided by all relevant products. Here there is exactly one relevant product, so it equals the Top-K hit indicator.",
        "- **MRR:** mean reciprocal rank of the first occurrence of the true product; zero when it is outside the evaluated ranking.",
        "",
        "## Scope boundary",
        "",
        "S3.11 evaluates retrieval. It does not compare against the S2.7 baseline or select a final Siamese model; that comparison belongs to S3.12.",
        "",
        "Policy identifier: `s3.11-product-retrieval-evaluation-v1`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    if args.k != 10:
        raise ValueError("S3.11 requires K=10 so Top-1/5/10 and MRR share one ranking.")

    gallery_root = Path(args.gallery_root)
    output = Path(args.output)
    markdown_output = Path(args.markdown_output)
    selected = MODEL_DIRS if args.model == "all" else {args.model: MODEL_DIRS[args.model]}

    models: dict[str, dict] = {}
    gallery_sizes: set[int] = set()
    embedding_dims: set[int] = set()
    product_counts: set[int] = set()

    for name, relative_default in selected.items():
        directory = gallery_root / relative_default.name
        gallery = load_gallery(directory)
        result = evaluate_gallery(gallery, k=args.k)
        payload = result_to_dict(result)
        product_count = len({_product_id(item) for item in gallery.metadata})
        payload["gallery"] = {
            "directory": str(directory),
            "size": gallery.size,
            "embedding_dim": gallery.embedding_dim,
            "product_count": product_count,
        }
        models[name] = payload
        gallery_sizes.add(gallery.size)
        embedding_dims.add(gallery.embedding_dim)
        product_counts.add(product_count)

    git = git_metadata()
    report = {
        "experiment_id": "EXP-0004",
        "task": "S3.11",
        "policy": "s3.11-product-retrieval-evaluation-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "k": args.k,
        "models": models,
        "cross_model_consistency": {
            "all_gallery_sizes_match": len(gallery_sizes) <= 1,
            "all_embedding_dims_match": len(embedding_dims) <= 1,
            "all_product_counts_match": len(product_counts) <= 1,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_output.write_text(markdown_report(report), encoding="utf-8")

    print("S3.11 Retrieval Evaluation")
    for name, payload in models.items():
        m = payload["metrics"]
        print(
            f"{name}: Top1={m['top1']:.6f} Top5={m['top5']:.6f} "
            f"Top10={m['top10']:.6f} P@5={m['precision_at_5']:.6f} "
            f"R@5={m['recall_at_5']:.6f} MRR={m['mrr']:.6f}"
        )
    print(f"report: {output}")
    print(f"markdown: {markdown_output}")


def _product_id(item: dict) -> str:
    value = item.get("product_id", item.get("product_group"))
    if value is None or not str(value).strip():
        raise ValueError("Gallery metadata contains an empty product ID.")
    return str(value)


if __name__ == "__main__":
    main()
