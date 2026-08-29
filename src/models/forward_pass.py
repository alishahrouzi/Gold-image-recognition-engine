"""S2.4 forward-pass validation: preprocessed tensor → encoder → embedding.

Verifies shapes, dtype, finite values, and L2 unit norms on embeddings.
Does not train, rank, retrieve, or choose between 128-D and 256-D.
Raw encoder features are checked for numerical validity only (not L2).
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .config import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    SUPPORTED_EMBEDDING_DIMS,
)
from .embedding_head import EmbeddingHead
from .encoder import CustomCNNEncoder
from .errors import ForwardPassValidationError

PathLike = Union[str, Path]

L2_ATOL: float = 1e-5
REQUIRED_BATCH_SIZES: Tuple[int, ...] = (1, 8, 32)
OPTIONAL_BATCH_SIZES: Tuple[int, ...] = (64,)
EXPECTED_IMAGE_SHAPE: Tuple[int, int, int] = (
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
)
ENCODER_FEATURE_DIM: int = DEFAULT_FEATURE_DIM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset" / "dataset1_manifest.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "benchmark" / "forward_pass"
_LEGACY_DATASET_ROOT = Path(
    r"e:\Privat File\Projects\Zargar Interview\dataset\ai-tool-pool-jewelry-vision"
)
_LOCAL_DATASET_ROOT = PROJECT_ROOT.parent / "dataset" / "ai-tool-pool-jewelry-vision"
DATASET1_SPLITS: Tuple[str, ...] = ("train", "valid", "test")


def dataset1_root() -> Path:
    env = os.environ.get("ZARGAR_DATASET1_ROOT")
    if env:
        return Path(env)
    if _LEGACY_DATASET_ROOT.is_dir():
        return _LEGACY_DATASET_ROOT
    return _LOCAL_DATASET_ROOT


def dataset1_available(
    manifest_path: PathLike = DEFAULT_MANIFEST,
    root: Optional[PathLike] = None,
) -> bool:
    dataset_root = Path(root) if root is not None else dataset1_root()
    return Path(manifest_path).is_file() and (dataset_root / "train").is_dir()


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "vram_total_mib": None,
        "detected_device": str(detect_device()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_total_mib"] = props.total_memory / (1024 * 1024)
    return info


def _as_float(value: Tensor) -> float:
    return float(value.detach().float().cpu().item())


def collect_tensor_stats(tensor: Tensor, *, include_nan_inf: bool = False) -> Dict[str, Any]:
    """Min/max/mean/std over finite values; optional NaN/Inf flags."""
    detached = tensor.detach()
    payload: Dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
    }
    if include_nan_inf:
        payload["has_nan"] = bool(torch.isnan(detached).any().item())
        payload["has_inf"] = bool(torch.isinf(detached).any().item())
    finite = detached.float()
    if include_nan_inf:
        mask = torch.isfinite(finite)
        if not bool(mask.any().item()):
            return payload
        finite = finite[mask]
    elif not torch.isfinite(finite).all():
        mask = torch.isfinite(finite)
        if not bool(mask.any().item()):
            return payload
        finite = finite[mask]
    payload["min"] = _as_float(finite.min())
    payload["max"] = _as_float(finite.max())
    payload["mean"] = _as_float(finite.mean())
    payload["std"] = _as_float(finite.std(unbiased=False)) if finite.numel() > 1 else 0.0
    return payload


def collect_l2_stats(embeddings: Tensor) -> Dict[str, float]:
    norms = torch.linalg.vector_norm(embeddings.detach().float(), ord=2, dim=1)
    errors = (norms - 1.0).abs()
    return {
        "l2_norm_min": _as_float(norms.min()),
        "l2_norm_max": _as_float(norms.max()),
        "l2_norm_mean": _as_float(norms.mean()),
        "max_l2_error": _as_float(errors.max()),
    }


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def validate_input_tensor(images: Tensor, batch_size: int) -> List[str]:
    failures: List[str] = []
    expected = (batch_size, *EXPECTED_IMAGE_SHAPE)
    if images.ndim != 4:
        failures.append(f"input rank {images.ndim} != 4")
    if tuple(images.shape) != expected:
        failures.append(f"input shape {tuple(images.shape)} != {expected}")
    if images.shape[0] != batch_size:
        failures.append(f"input batch {images.shape[0]} != {batch_size}")
    if images.dtype != torch.float32:
        failures.append(f"input dtype {_dtype_name(images.dtype)} != torch.float32")
    return failures


def validate_encoder_features(features: Tensor, batch_size: int) -> List[str]:
    """Raw GAP features: shape/dtype/finite only. No L2 requirement."""
    failures: List[str] = []
    expected = (batch_size, ENCODER_FEATURE_DIM)
    if features.ndim != 2:
        failures.append(f"encoder rank {features.ndim} != 2")
    if tuple(features.shape) != expected:
        failures.append(f"encoder shape {tuple(features.shape)} != {expected}")
    if features.shape[0] != batch_size:
        failures.append(f"encoder batch {features.shape[0]} != {batch_size}")
    if features.shape[1] != ENCODER_FEATURE_DIM:
        failures.append(f"encoder feature_dim {features.shape[1]} != {ENCODER_FEATURE_DIM}")
    if features.dtype != torch.float32:
        failures.append(f"encoder dtype {_dtype_name(features.dtype)} != torch.float32")
    if bool(torch.isnan(features).any().item()):
        failures.append("encoder features contain NaN")
    if bool(torch.isinf(features).any().item()):
        failures.append("encoder features contain Inf")
    if not bool(torch.isfinite(features).all().item()):
        failures.append("encoder features are not all finite")
    return failures


def validate_embeddings(
    embeddings: Tensor,
    batch_size: int,
    embedding_dim: int,
    *,
    atol: float = L2_ATOL,
) -> List[str]:
    failures: List[str] = []
    expected = (batch_size, embedding_dim)
    if embeddings.ndim != 2:
        failures.append(f"embedding rank {embeddings.ndim} != 2")
    if tuple(embeddings.shape) != expected:
        failures.append(f"embedding shape {tuple(embeddings.shape)} != {expected}")
    if embeddings.shape[0] != batch_size:
        failures.append(f"embedding batch {embeddings.shape[0]} != {batch_size}")
    if embeddings.shape[1] != embedding_dim:
        failures.append(f"embedding dim {embeddings.shape[1]} != {embedding_dim}")
    if embeddings.dtype != torch.float32:
        failures.append(f"embedding dtype {_dtype_name(embeddings.dtype)} != torch.float32")
    if bool(torch.isnan(embeddings).any().item()):
        failures.append("embeddings contain NaN")
    if bool(torch.isinf(embeddings).any().item()):
        failures.append("embeddings contain Inf")
    if not bool(torch.isfinite(embeddings).all().item()):
        failures.append("embeddings are not all finite")
    if embeddings.ndim == 2 and embeddings.shape[0] > 0:
        norms = torch.linalg.vector_norm(embeddings.float(), ord=2, dim=1)
        if not bool(torch.allclose(norms, torch.ones_like(norms), atol=atol)):
            max_err = _as_float((norms - 1.0).abs().max())
            failures.append(f"embedding L2 norms not within atol={atol} of 1 (max_error={max_err})")
    return failures


def load_dataset1_image_batch(
    split: str,
    batch_size: int,
    *,
    manifest_path: PathLike = DEFAULT_MANIFEST,
    root: Optional[PathLike] = None,
    validate_files: bool = False,
) -> Tensor:
    """Manifest → UnifiedDataset → PreprocessedDataset → DataLoader → batch['image'].

    Valid and test use the deterministic preprocessor path (no augmentation).
    Train uses role='train' without an AugmentationConfig so augmentation stays off
    for deterministic S2.4 checks.
    """
    if split not in DATASET1_SPLITS:
        raise ForwardPassValidationError(f"Unsupported split {split!r}.")
    from data.collate import collate_preprocessed_samples
    from data.datasets.unified_dataset import UnifiedDataset
    from data.preprocessing import ImagePreprocessor, PreprocessedDataset

    dataset_root = Path(root) if root is not None else dataset1_root()
    base = UnifiedDataset(
        manifest_path,
        dataset_root=dataset_root,
        split=split,
        validate_files=validate_files,
    )
    if len(base) < batch_size:
        raise ForwardPassValidationError(
            f"Split {split!r} has {len(base)} images; need batch_size={batch_size}."
        )
    dataset = PreprocessedDataset(
        Subset(base, list(range(batch_size))),
        ImagePreprocessor(),
        role=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_preprocessed_samples,
    )
    return next(iter(loader))["image"]


@torch.no_grad()
def run_forward_pass(
    images: Tensor,
    *,
    encoder: CustomCNNEncoder,
    head: EmbeddingHead,
    device: torch.device,
    split: str,
    source: str,
    atol: float = L2_ATOL,
) -> Dict[str, Any]:
    """Run encoder then head on a caller-placed or CPU batch; move tensors in this harness."""
    if head.embedding_dim not in SUPPORTED_EMBEDDING_DIMS:
        raise ForwardPassValidationError(
            f"embedding_dim must be one of {list(SUPPORTED_EMBEDDING_DIMS)}."
        )
    batch_size = int(images.shape[0]) if images.ndim >= 1 else 0
    encoder.eval()
    head.eval()
    encoder_dev = encoder.to(device)
    head_dev = head.to(device)
    images_dev = images.to(device)

    input_failures = validate_input_tensor(images_dev, batch_size)
    features = encoder_dev.encode_features(images_dev)
    encoder_failures = validate_encoder_features(features, batch_size)
    embeddings = head_dev(features)
    embedding_failures = validate_embeddings(
        embeddings,
        batch_size,
        head_dev.embedding_dim,
        atol=atol,
    )
    failures = input_failures + encoder_failures + embedding_failures

    input_stats = collect_tensor_stats(images_dev)
    encoder_stats = collect_tensor_stats(features, include_nan_inf=True)
    embedding_stats = collect_tensor_stats(embeddings, include_nan_inf=True)
    embedding_stats.update(collect_l2_stats(embeddings))

    return {
        "device": str(device),
        "batch_size": batch_size,
        "embedding_dim": head_dev.embedding_dim,
        "split": split,
        "sample_count": batch_size,
        "source": source,
        "passed": len(failures) == 0,
        "failures": failures,
        "input": input_stats,
        "encoder_features": encoder_stats,
        "embedding": embedding_stats,
    }


def make_synthetic_images(
    batch_size: int,
    *,
    fill: Optional[float] = None,
    scale: float = 1.0,
) -> Tensor:
    """Finite synthetic NCHW tensors for numerical edge cases (not a second preprocessor)."""
    shape = (batch_size, *EXPECTED_IMAGE_SHAPE)
    if fill is not None:
        return torch.full(shape, fill, dtype=torch.float32)
    return torch.randn(shape, dtype=torch.float32) * scale


def preprocessed_images_from_preprocessor(batch_size: int) -> Tensor:
    """Stack ImagePreprocessor outputs. Does not resize/normalize outside that pipeline."""
    from PIL import Image

    from data.preprocessing import ImagePreprocessor

    preprocessor = ImagePreprocessor()
    tensors = [
        preprocessor(Image.new("RGB", (80 + (index % 17), 60 + (index % 11)), (12, 40, 90)))
        for index in range(batch_size)
    ]
    return torch.stack(tensors, dim=0)


def run_validation_suite(
    *,
    batch_sizes: Sequence[int] = REQUIRED_BATCH_SIZES,
    embedding_dims: Sequence[int] = SUPPORTED_EMBEDDING_DIMS,
    include_dataset1: bool = True,
    include_optional_batch64: bool = True,
    include_edge_cases: bool = True,
    devices: Optional[Sequence[torch.device]] = None,
    atol: float = L2_ATOL,
) -> Dict[str, Any]:
    if devices is None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

    runs: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for device in devices:
        encoder = CustomCNNEncoder()
        heads = {dim: EmbeddingHead(embedding_dim=dim) for dim in embedding_dims}

        for batch_size in batch_sizes:
            images = preprocessed_images_from_preprocessor(batch_size)
            for dim in embedding_dims:
                runs.append(
                    run_forward_pass(
                        images,
                        encoder=encoder,
                        head=heads[dim],
                        device=device,
                        split="preprocessor",
                        source="image_preprocessor",
                        atol=atol,
                    )
                )

        if include_edge_cases:
            edge_cases: Iterable[Tuple[str, Tensor]] = (
                ("edge_zero", make_synthetic_images(8, fill=0.0)),
                ("edge_constant", make_synthetic_images(8, fill=0.25)),
                ("edge_tiny", make_synthetic_images(8, scale=1e-12)),
            )
            for split, images in edge_cases:
                for dim in embedding_dims:
                    runs.append(
                        run_forward_pass(
                            images,
                            encoder=encoder,
                            head=heads[dim],
                            device=device,
                            split=split,
                            source="synthetic_finite",
                            atol=atol,
                        )
                    )

        if include_optional_batch64:
            try:
                images = preprocessed_images_from_preprocessor(64)
                for dim in embedding_dims:
                    runs.append(
                        run_forward_pass(
                            images,
                            encoder=encoder,
                            head=heads[dim],
                            device=device,
                            split="preprocessor",
                            source="image_preprocessor_optional_b64",
                            atol=atol,
                        )
                    )
            except torch.cuda.OutOfMemoryError:
                warnings.append(f"optional batch_size=64 skipped on {device} (CUDA OOM)")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    warnings.append(f"optional batch_size=64 skipped on {device} ({exc})")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

        if include_dataset1:
            if not dataset1_available():
                warnings.append("Dataset 1 skipped (root or manifest unavailable)")
            else:
                for split in DATASET1_SPLITS:
                    for batch_size in batch_sizes:
                        try:
                            images = load_dataset1_image_batch(split, batch_size)
                        except ForwardPassValidationError as exc:
                            warnings.append(str(exc))
                            continue
                        for dim in embedding_dims:
                            runs.append(
                                run_forward_pass(
                                    images,
                                    encoder=encoder,
                                    head=heads[dim],
                                    device=device,
                                    split=split,
                                    source="dataset1",
                                    atol=atol,
                                )
                            )

    failed = [row for row in runs if not row["passed"]]
    status = "FAIL" if failed else "PASS"
    return {
        "milestone": "S2.4",
        "status": status,
        "l2_atol": atol,
        "hardware": hardware_info(),
        "required_batch_sizes": list(batch_sizes),
        "embedding_dims": list(embedding_dims),
        "notes": [
            "Forward-pass validation only. No training, loss, optimizer, or retrieval.",
            "Raw encoder features are not L2-normalized; embeddings are L2-normalized.",
            "Neither 128-D nor 256-D is declared the retrieval winner.",
            "Dataset 2 unused. Manifest and source images were not modified.",
            "CUDA absence is not a model failure; CUDA rows are omitted when unavailable.",
        ],
        "warnings": warnings,
        "failure_count": len(failed),
        "run_count": len(runs),
        "runs": runs,
    }


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    hw = payload["hardware"]
    lines = [
        "# S2.4 Forward Pass Validation",
        "",
        "Untrained model. **No retrieval metrics** (no Top-1 / Top-5 / Recall@K / MRR).",
        "This report records structural and numerical correctness of:",
        "",
        "```",
        "Preprocessed Tensor [B, 3, 224, 224]",
        "        ↓",
        "CustomCNNEncoder",
        "        ↓",
        "Raw Features [B, 256]   (not L2-normalized)",
        "        ↓",
        "EmbeddingHead",
        "        ↓",
        "L2-normalized Embedding [B, D],  D ∈ {128, 256}",
        "```",
        "",
        f"**Overall status: {payload['status']}**",
        "",
        f"L2 tolerance: `abs(norm - 1) < {payload['l2_atol']}` (floating-point, not exact equality).",
        "",
        "## Hardware",
        "",
        f"- Python: {hw['python']}",
        f"- PyTorch: {hw['pytorch']}",
        f"- Platform: {hw['platform']}",
        f"- Detected device: {hw['detected_device']}",
        f"- CUDA available: {hw['cuda_available']}",
        f"- GPU: {hw['gpu_name']}",
        f"- VRAM total (MiB): {hw['vram_total_mib']}",
        "",
        "## Tested configurations",
        "",
        f"- Batch sizes: {payload['required_batch_sizes']}",
        f"- Embedding dims: {payload['embedding_dims']}",
        f"- Runs: {payload['run_count']}",
        f"- Failures: {payload['failure_count']}",
        "",
        "## Warnings",
        "",
    ]
    if payload["warnings"]:
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    runs = list(payload["runs"])
    devices = sorted({str(row["device"]) for row in runs})
    dataset1 = [row for row in runs if row.get("source") == "dataset1"]
    splits = sorted({str(row["split"]) for row in dataset1})
    max_l2 = max((float(row["embedding"]["max_l2_error"]) for row in runs), default=0.0)
    lines.extend(
        [
            "",
            "## Numerical checks",
            "",
            "- Input: `[B, 3, 224, 224]`, `float32`, via `ImagePreprocessor` / `PreprocessedDataset`.",
            "- Encoder features: `[B, 256]`, `float32`, finite (no NaN / Inf). Not L2-normalized.",
            "- Embeddings: `[B, 128]` and `[B, 256]`, `float32`, finite.",
            f"- L2: `max_l2_error` across all runs = `{max_l2:.2e}` (atol `{payload['l2_atol']}`).",
            "",
            "## CPU / CUDA",
            "",
            f"- Devices exercised: {devices}",
        ]
    )
    if not hw["cuda_available"]:
        lines.append("- CUDA-specific rows were skipped because CUDA is unavailable (not a model failure).")
    else:
        lines.append("- CPU and CUDA both ran; device placement is in the harness, not inside the modules.")
    lines.extend(
        [
            "",
            "## Dataset 1 integration",
            "",
        ]
    )
    if dataset1:
        lines.append(
            f"- {len(dataset1)} Dataset 1 runs; splits={splits}; "
            "path: `dataset1_manifest.csv` → `UnifiedDataset` → `PreprocessedDataset` → "
            "`DataLoader` → `batch['image']` → encoder → embedding head."
        )
        lines.append("- train / valid / test used; valid and test unaugmented; train smoke without `AugmentationConfig`.")
        lines.append("- Dataset 2 unused. Manifest and source images were not modified.")
    else:
        lines.append("- No Dataset 1 runs in this report (root or manifest unavailable, or skipped).")
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| status | device | batch | D | split | source | in_shape | feat_shape | emb_shape | "
            "feat finite | emb finite | L2 min | L2 max | L2 mean | max L2 err |",
            "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["runs"]:
        feat = row["encoder_features"]
        emb = row["embedding"]
        feat_ok = (not feat.get("has_nan")) and (not feat.get("has_inf"))
        emb_ok = (not emb.get("has_nan")) and (not emb.get("has_inf"))
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"| {mark} | {row['device']} | {row['batch_size']} | {row['embedding_dim']} | "
            f"{row['split']} | {row['source']} | {row['input']['shape']} | {feat['shape']} | "
            f"{emb['shape']} | {feat_ok} | {emb_ok} | {emb['l2_norm_min']:.8f} | "
            f"{emb['l2_norm_max']:.8f} | {emb['l2_norm_mean']:.8f} | {emb['max_l2_error']:.2e} |"
        )
    failed = [row for row in payload["runs"] if not row["passed"]]
    lines.extend(["", "## Failures", ""])
    if failed:
        for row in failed:
            lines.append(
                f"- device={row['device']} batch={row['batch_size']} D={row['embedding_dim']} "
                f"split={row['split']}: {row['failures']}"
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            f"Final status: **{payload['status']}**",
            "",
        ]
    )
    return "\n".join(lines)


def write_forward_pass_reports(
    payload: Mapping[str, Any],
    report_dir: PathLike = DEFAULT_REPORT_DIR,
) -> Tuple[Path, Path]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "s2.4_forward_pass_report.json"
    md_path = report_dir / "s2.4_forward_pass_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return json_path, md_path
