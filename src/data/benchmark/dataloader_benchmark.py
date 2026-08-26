"""Empirical DataLoader benchmark for the Dataset 1 pipeline (S1.12).

Two stages:
    dataloader        CPU loading, RGB conversion, optional train
                      augmentation, preprocessing, collation. No GPU copy.
    dataloader_gpu    Same pipeline plus CPU → GPU tensor transfer.
                      No encoder. No optimizer. No training.

Reuses UnifiedDataset, ImagePreprocessor, PreprocessedDataset,
TrainingAugmentor (S1.9), and collate_preprocessed_samples.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from ..collate import collate_preprocessed_samples
from ..constants import ALLOWED_SPLITS, SPLIT_ORDER
from ..datasets.unified_dataset import UnifiedDataset
from ..preprocessing import (
    AugmentationConfig,
    ImagePreprocessingConfig,
    ImagePreprocessor,
    build_preprocessed_dataset,
)
from ..preprocessing.augmentation import DETERMINISTIC_ROLES, TRAINING_ROLE

PathLike = Union[str, Path]

BENCHMARK_NAME = "S1.12 DataLoader benchmark"
BENCHMARK_VERSION = "s1.12"
DEFAULT_SEED = 2026
DEFAULT_WARMUP_BATCHES = 5
DEFAULT_MEASUREMENT_BATCHES = 30
DEFAULT_BATCH_SIZES: Tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_NUM_WORKERS: Tuple[int, ...] = (0, 2, 4)

GPU_UTILIZATION_UNAVAILABLE_REASON = (
    "GPU utilization is not reported because short CPU→GPU copies do not "
    "produce a reliable utilization sample from this process. "
    "torch.cuda.utilization() requires CUPTI and is often unavailable; "
    "instant NVML/nvidia-smi snapshots during a few dozen transfers are "
    "noisy and would be misleading. gpu_utilization_percent is therefore null."
)

CONFIGURATION_RESULT_FIELDS: Tuple[str, ...] = (
    "split",
    "stage",
    "batch_size",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "status",
    "error",
    "augmentation_applied",
    "warmup_batches_used",
    "measurement_batches_used",
    "images_measured",
    "first_batch_ms",
    "mean_batch_ms",
    "median_batch_ms",
    "p95_batch_ms",
    "batches_per_sec",
    "images_per_sec",
    "ram_before_mb",
    "ram_after_warmup_mb",
    "ram_peak_mb",
    "ram_delta_mb",
    "gpu_allocated_mb",
    "gpu_reserved_mb",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "gpu_utilization_percent",
)

_NULL_METRIC_FIELDS = (
    "first_batch_ms",
    "mean_batch_ms",
    "median_batch_ms",
    "p95_batch_ms",
    "batches_per_sec",
    "images_per_sec",
    "ram_after_warmup_mb",
    "ram_peak_mb",
    "ram_delta_mb",
    "gpu_allocated_mb",
    "gpu_reserved_mb",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "gpu_utilization_percent",
)


class DataLoaderBenchmarkError(ValueError):
    """Invalid benchmark configuration or unsafe DataLoader arguments."""


@dataclass(frozen=True)
class DataLoaderBenchmarkConfig:
    """One DataLoader setting to measure.

    persistent_workers is only valid when num_workers > 0.
    """

    split: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    stage: str = "dataloader"

    def __post_init__(self) -> None:
        validate_benchmark_config(
            split=self.split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            stage=self.stage,
        )


def validate_benchmark_config(
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    stage: str = "dataloader",
) -> None:
    """Reject invalid DataLoader arguments before constructing a loader."""
    if split not in ALLOWED_SPLITS:
        allowed = "/".join(SPLIT_ORDER)
        raise DataLoaderBenchmarkError(
            f"Invalid split {split!r}. Expected {allowed}."
        )
    if stage not in {"dataloader", "dataloader_gpu"}:
        raise DataLoaderBenchmarkError(
            f"Invalid stage {stage!r}. Expected 'dataloader' or 'dataloader_gpu'."
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise DataLoaderBenchmarkError(
            f"batch_size must be a positive integer, got {batch_size!r}."
        )
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise DataLoaderBenchmarkError(
            f"num_workers must be an integer >= 0, got {num_workers!r}."
        )
    if not isinstance(pin_memory, bool):
        raise DataLoaderBenchmarkError("pin_memory must be a boolean.")
    if not isinstance(persistent_workers, bool):
        raise DataLoaderBenchmarkError("persistent_workers must be a boolean.")
    if persistent_workers and num_workers == 0:
        raise DataLoaderBenchmarkError(
            "persistent_workers=True requires num_workers > 0."
        )


def non_blocking_transfer(pin_memory: bool, device_type: str) -> bool:
    """Match tensor.to(..., non_blocking=) to pin_memory on CUDA."""
    return bool(pin_memory) and device_type == "cuda"


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation percentile. ``q`` is in [0, 100]."""
    if not values:
        raise DataLoaderBenchmarkError("Cannot compute a percentile of an empty series.")
    if q < 0.0 or q > 100.0:
        raise DataLoaderBenchmarkError(f"percentile q must be in [0, 100], got {q}.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def compute_latency_metrics(
    latencies_s: Sequence[float],
    *,
    images_measured: int,
) -> Dict[str, float]:
    """Throughput and latency summary for measured batches only."""
    if not latencies_s:
        raise DataLoaderBenchmarkError("No measured batch latencies.")
    if images_measured < 1:
        raise DataLoaderBenchmarkError("images_measured must be >= 1.")
    total_s = float(sum(latencies_s))
    if total_s <= 0.0:
        raise DataLoaderBenchmarkError("Measured duration must be positive.")
    mean_s = total_s / len(latencies_s)
    return {
        "mean_batch_ms": mean_s * 1000.0,
        "median_batch_ms": percentile(latencies_s, 50.0) * 1000.0,
        "p95_batch_ms": percentile(latencies_s, 95.0) * 1000.0,
        "batches_per_sec": len(latencies_s) / total_s,
        "images_per_sec": images_measured / total_s,
    }


def cap_batch_counts(
    dataset_len: int,
    batch_size: int,
    *,
    warmup_batches: int,
    measurement_batches: int,
) -> Tuple[int, int]:
    """Cap warmup/measurement so they fit in one epoch without wrapping."""
    if dataset_len < 1:
        raise DataLoaderBenchmarkError("Dataset is empty.")
    if batch_size < 1:
        raise DataLoaderBenchmarkError(
            f"batch_size must be a positive integer, got {batch_size!r}."
        )
    if warmup_batches < 0 or measurement_batches < 1:
        raise DataLoaderBenchmarkError(
            "warmup_batches must be >= 0 and measurement_batches must be >= 1."
        )
    available = math.ceil(dataset_len / batch_size)
    if available < 1:
        raise DataLoaderBenchmarkError("No batches available for this configuration.")
    if available == 1:
        return 0, 1
    warmup = min(int(warmup_batches), available - 1)
    remaining = available - warmup
    measured = min(int(measurement_batches), remaining)
    return warmup, measured


def process_rss_mb() -> float:
    """Current process RSS in MiB. Not OS-wide memory."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        if sys.platform == "win32":
            return _windows_rss_mb()
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB; macOS: bytes.
        if sys.platform == "darwin":
            return float(usage) / (1024.0 * 1024.0)
        return float(usage) / 1024.0


def _windows_rss_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise DataLoaderBenchmarkError("Unable to read process RSS on Windows.")
    return float(counters.WorkingSetSize) / (1024.0 * 1024.0)


def collect_hardware_snapshot(device: torch.device) -> Dict[str, Any]:
    ram_total_mb: Optional[float] = None
    try:
        import psutil

        ram_total_mb = float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
    except Exception:
        ram_total_mb = None

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    vram_total_mb = None
    cuda_version = None
    if cuda_available:
        gpu_index = device.index if device.type == "cuda" and device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(gpu_index)
        vram_total_mb = float(torch.cuda.get_device_properties(gpu_index).total_memory) / (
            1024.0 * 1024.0
        )
        cuda_version = torch.version.cuda

    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "vram_total_mb": _round_optional(vram_total_mb),
        "ram_total_mb": _round_optional(ram_total_mb),
        "process_rss_mb": _round_optional(process_rss_mb()),
        "gpu_utilization_measurable": False,
        "gpu_utilization_note": GPU_UTILIZATION_UNAVAILABLE_REASON,
    }


def configuration_result(
    *,
    split: str,
    stage: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    status: str,
    error: Optional[str] = None,
    augmentation_applied: bool = False,
    warmup_batches_used: int = 0,
    measurement_batches_used: int = 0,
    images_measured: int = 0,
    first_batch_ms: Optional[float] = None,
    mean_batch_ms: Optional[float] = None,
    median_batch_ms: Optional[float] = None,
    p95_batch_ms: Optional[float] = None,
    batches_per_sec: Optional[float] = None,
    images_per_sec: Optional[float] = None,
    ram_before_mb: Optional[float] = None,
    ram_after_warmup_mb: Optional[float] = None,
    ram_peak_mb: Optional[float] = None,
    ram_delta_mb: Optional[float] = None,
    gpu_allocated_mb: Optional[float] = None,
    gpu_reserved_mb: Optional[float] = None,
    gpu_peak_allocated_mb: Optional[float] = None,
    gpu_peak_reserved_mb: Optional[float] = None,
    gpu_utilization_percent: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a configuration row with a fixed key order."""
    payload = {
        "split": split,
        "stage": stage,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "status": status,
        "error": error,
        "augmentation_applied": augmentation_applied,
        "warmup_batches_used": warmup_batches_used,
        "measurement_batches_used": measurement_batches_used,
        "images_measured": images_measured,
        "first_batch_ms": _round_optional(first_batch_ms),
        "mean_batch_ms": _round_optional(mean_batch_ms),
        "median_batch_ms": _round_optional(median_batch_ms),
        "p95_batch_ms": _round_optional(p95_batch_ms),
        "batches_per_sec": _round_optional(batches_per_sec),
        "images_per_sec": _round_optional(images_per_sec),
        "ram_before_mb": _round_optional(ram_before_mb),
        "ram_after_warmup_mb": _round_optional(ram_after_warmup_mb),
        "ram_peak_mb": _round_optional(ram_peak_mb),
        "ram_delta_mb": _round_optional(ram_delta_mb),
        "gpu_allocated_mb": _round_optional(gpu_allocated_mb),
        "gpu_reserved_mb": _round_optional(gpu_reserved_mb),
        "gpu_peak_allocated_mb": _round_optional(gpu_peak_allocated_mb),
        "gpu_peak_reserved_mb": _round_optional(gpu_peak_reserved_mb),
        "gpu_utilization_percent": _round_optional(gpu_utilization_percent),
    }
    if tuple(payload.keys()) != CONFIGURATION_RESULT_FIELDS:
        raise DataLoaderBenchmarkError("Configuration result schema drifted.")
    return payload


def build_practical_configuration_matrix(
    *,
    splits: Sequence[str] = ("train", "valid", "test"),
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    num_workers: Sequence[int] = DEFAULT_NUM_WORKERS,
    include_gpu: bool = True,
) -> List[DataLoaderBenchmarkConfig]:
    """Practical (non-Cartesian) matrix covering batch size, workers, and pin_memory."""
    configs: List[DataLoaderBenchmarkConfig] = []
    stages = ("dataloader", "dataloader_gpu") if include_gpu else ("dataloader",)

    def add(**kwargs: Any) -> None:
        for stage in stages:
            item = DataLoaderBenchmarkConfig(stage=stage, **kwargs)
            if item not in configs:
                configs.append(item)

    if "train" in splits:
        for batch_size in batch_sizes:
            for workers in num_workers:
                add(
                    split="train",
                    batch_size=int(batch_size),
                    num_workers=int(workers),
                    pin_memory=False,
                    persistent_workers=False,
                )
        pin_batches = [size for size in batch_sizes if size in {16, 32}] or list(batch_sizes)[:1]
        pin_workers = [workers for workers in num_workers if workers in {0, 2}]
        if not pin_workers:
            pin_workers = [int(num_workers[0])]
        for batch_size in pin_batches:
            for workers in pin_workers:
                add(
                    split="train",
                    batch_size=int(batch_size),
                    num_workers=int(workers),
                    pin_memory=True,
                    persistent_workers=False,
                )
        persistent_workers = [workers for workers in num_workers if workers > 0]
        if persistent_workers:
            persist_batch = 16 if 16 in batch_sizes else int(batch_sizes[0])
            add(
                split="train",
                batch_size=persist_batch,
                num_workers=int(persistent_workers[0]),
                pin_memory=True if include_gpu else False,
                persistent_workers=True,
            )

    eval_batch = 16 if 16 in batch_sizes else int(batch_sizes[0])
    eval_workers = 0 if 0 in num_workers else int(num_workers[0])
    for split in splits:
        if split in {"valid", "test"}:
            add(
                split=split,
                batch_size=eval_batch,
                num_workers=eval_workers,
                pin_memory=False,
                persistent_workers=False,
            )
    return configs


def build_smoke_configuration_matrix(*, include_gpu: bool = True) -> List[DataLoaderBenchmarkConfig]:
    stages = ("dataloader", "dataloader_gpu") if include_gpu else ("dataloader",)
    configs: List[DataLoaderBenchmarkConfig] = []
    for stage in stages:
        configs.append(
            DataLoaderBenchmarkConfig(
                split="train",
                batch_size=8,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                stage=stage,
            )
        )
    configs.append(
        DataLoaderBenchmarkConfig(
            split="valid",
            batch_size=8,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            stage="dataloader",
        )
    )
    return configs


def _dataloader_worker_init(worker_id: int) -> None:
    """Reseed S1.9 augmentor per worker when present. Safe for valid/test."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    augmentor = getattr(dataset, "augmentor", None)
    if augmentor is None:
        return
    base = augmentor.seed if augmentor.seed is not None else 0
    augmentor.reseed(augmentor.worker_seed(worker_id, base))


def _round_optional(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _is_cuda_oom(exc: BaseException) -> bool:
    if type(exc).__name__ in {"OutOfMemoryError", "cudaErrorMemoryAllocation"}:
        return True
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message)


def _bytes_to_mb(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def _read_cuda_memory(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
            "gpu_peak_allocated_mb": None,
            "gpu_peak_reserved_mb": None,
        }
    index = int(_indexed_cuda_device(device).index)
    return {
        "gpu_allocated_mb": _bytes_to_mb(torch.cuda.memory_allocated(index)),
        "gpu_reserved_mb": _bytes_to_mb(torch.cuda.memory_reserved(index)),
        "gpu_peak_allocated_mb": _bytes_to_mb(torch.cuda.max_memory_allocated(index)),
        "gpu_peak_reserved_mb": _bytes_to_mb(torch.cuda.max_memory_reserved(index)),
    }


def _cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _shutdown_loader(loader: Optional[DataLoader]) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
    try:
        del iterator
    except Exception:
        pass
    try:
        del loader
    except Exception:
        pass
    gc.collect()


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    pin_memory: bool,
) -> Any:
    images = batch["image"]
    if device.type != "cuda":
        return images
    return images.to(device, non_blocking=non_blocking_transfer(pin_memory, device.type))


def _run_one_configuration(
    dataset,
    config: DataLoaderBenchmarkConfig,
    *,
    device: torch.device,
    warmup_batches: int,
    measurement_batches: int,
    seed: int,
) -> Dict[str, Any]:
    augmentation_applied = dataset.augmentor is not None
    ram_before = process_rss_mb()
    gpu_stats = {
        "gpu_allocated_mb": None,
        "gpu_reserved_mb": None,
        "gpu_peak_allocated_mb": None,
        "gpu_peak_reserved_mb": None,
        "gpu_utilization_percent": None,
    }

    use_gpu = config.stage == "dataloader_gpu"
    if use_gpu and device.type != "cuda":
        return configuration_result(
            split=config.split,
            stage=config.stage,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            status="SKIPPED",
            error="CUDA device is not available for dataloader_gpu stage.",
            augmentation_applied=augmentation_applied,
            ram_before_mb=ram_before,
        )

    warmup_used, measure_used = cap_batch_counts(
        len(dataset),
        config.batch_size,
        warmup_batches=warmup_batches,
        measurement_batches=measurement_batches,
    )

    loader_kwargs: Dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "collate_fn": collate_preprocessed_samples,
        "drop_last": False,
        "generator": torch.Generator().manual_seed(seed),
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["worker_init_fn"] = _dataloader_worker_init
        loader_kwargs["prefetch_factor"] = 2

    loader: Optional[DataLoader] = None
    held: List[Any] = []
    try:
        if use_gpu:
            device = _indexed_cuda_device(device)
            torch.cuda.set_device(device)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        loader = DataLoader(dataset, **loader_kwargs)
        iterator = iter(loader)

        first_batch_s: Optional[float] = None
        warmup_latencies: List[float] = []
        measure_latencies: List[float] = []
        images_measured = 0
        ram_peak = ram_before
        ram_after_warmup = ram_before

        total_needed = warmup_used + measure_used
        for index in range(total_needed):
            if use_gpu:
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            batch = next(iterator)
            gpu_tensor = None
            if use_gpu:
                gpu_tensor = _move_batch_to_device(
                    batch, device, pin_memory=config.pin_memory
                )
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

            if index == 0:
                first_batch_s = elapsed
            if index < warmup_used:
                warmup_latencies.append(elapsed)
                if index == warmup_used - 1 or warmup_used == 0:
                    ram_after_warmup = process_rss_mb()
            else:
                measure_latencies.append(elapsed)
                images_measured += int(batch["image"].shape[0])

            ram_peak = max(ram_peak, process_rss_mb())
            held.clear()
            held.append(gpu_tensor)
            del batch
            del gpu_tensor

        if warmup_used == 0:
            ram_after_warmup = process_rss_mb()
        ram_peak = max(ram_peak, process_rss_mb())

        if use_gpu:
            gpu_stats.update(_read_cuda_memory(device))
            gpu_stats["gpu_utilization_percent"] = None

        if not measure_latencies:
            raise DataLoaderBenchmarkError("Measurement phase produced no batches.")

        metrics = compute_latency_metrics(
            measure_latencies, images_measured=images_measured
        )
        return configuration_result(
            split=config.split,
            stage=config.stage,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            status="PASS",
            error=None,
            augmentation_applied=augmentation_applied,
            warmup_batches_used=warmup_used,
            measurement_batches_used=len(measure_latencies),
            images_measured=images_measured,
            first_batch_ms=(first_batch_s or 0.0) * 1000.0,
            ram_before_mb=ram_before,
            ram_after_warmup_mb=ram_after_warmup,
            ram_peak_mb=ram_peak,
            ram_delta_mb=ram_peak - ram_before,
            **metrics,
            **gpu_stats,
        )
    except StopIteration:
        return configuration_result(
            split=config.split,
            stage=config.stage,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            status="FAIL",
            error="DataLoader exhausted before the requested warmup/measurement batches.",
            augmentation_applied=augmentation_applied,
            ram_before_mb=ram_before,
            **gpu_stats,
        )
    except Exception as exc:
        status = "OOM" if (use_gpu and _is_cuda_oom(exc)) else "FAIL"
        if status == "OOM":
            _cleanup_cuda(device)
        return configuration_result(
            split=config.split,
            stage=config.stage,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            status=status,
            error=f"{type(exc).__name__}: {exc}",
            augmentation_applied=augmentation_applied,
            ram_before_mb=ram_before,
            **gpu_stats,
        )
    finally:
        held.clear()
        _shutdown_loader(loader)
        if use_gpu:
            _cleanup_cuda(device)


def _verify_split_augmentation(dataset, split: str) -> None:
    if split == TRAINING_ROLE:
        if dataset.role != TRAINING_ROLE or dataset.augmentor is None:
            raise DataLoaderBenchmarkError(
                "Train benchmark must attach the existing S1.9 TrainingAugmentor."
            )
        return
    if split in DETERMINISTIC_ROLES or split in {"valid", "test"}:
        if dataset.augmentor is not None:
            raise DataLoaderBenchmarkError(
                f"{split} must remain deterministic; a TrainingAugmentor was attached."
            )
        if dataset.role != split:
            raise DataLoaderBenchmarkError(
                f"Expected PreprocessedDataset role={split!r}, got {dataset.role!r}."
            )


def _preprocessing_snapshot(config: ImagePreprocessingConfig) -> Dict[str, Any]:
    return {
        "image_size": config.image_size,
        "mean": list(config.mean),
        "std": list(config.std),
        "interpolation": config.interpolation,
        "output_shape": list(config.output_shape),
        "contract": "RGB → resize 224×224 → ImageNet normalize → float32 [3, 224, 224]",
    }


def _indexed_cuda_device(device: torch.device) -> torch.device:
    """PyTorch 2.6+ CUDA APIs require an explicit device index."""
    if device.type != "cuda":
        return device
    if device.index is not None:
        return device
    if not torch.cuda.is_available():
        return device
    return torch.device("cuda", int(torch.cuda.current_device()))


def resolve_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized in {"auto", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if normalized == "cuda":
            raise DataLoaderBenchmarkError("CUDA was requested but is not available.")
        return torch.device("cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    return _indexed_cuda_device(torch.device(device_arg))


def build_recommendation(
    configurations: Sequence[Mapping[str, Any]],
    *,
    cuda_available: bool,
) -> Dict[str, Any]:
    """Data-loading recommendation. Not a final training batch size."""
    passed = [row for row in configurations if row.get("status") == "PASS"]
    oom = [row for row in configurations if row.get("status") == "OOM"]
    failed = [row for row in configurations if row.get("status") == "FAIL"]

    train_gpu = [
        row
        for row in passed
        if row["split"] == "train" and row["stage"] == "dataloader_gpu"
    ]
    train_cpu = [
        row for row in passed if row["split"] == "train" and row["stage"] == "dataloader"
    ]
    pool = train_gpu or train_cpu or list(passed)

    def _cfg_id(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "split": row["split"],
            "stage": row["stage"],
            "batch_size": row["batch_size"],
            "num_workers": row["num_workers"],
            "pin_memory": row["pin_memory"],
            "persistent_workers": row["persistent_workers"],
            "images_per_sec": row.get("images_per_sec"),
            "mean_batch_ms": row.get("mean_batch_ms"),
            "p95_batch_ms": row.get("p95_batch_ms"),
            "ram_peak_mb": row.get("ram_peak_mb"),
            "gpu_peak_allocated_mb": row.get("gpu_peak_allocated_mb"),
        }

    empty = {
        "largest_successful_dataloader_batch": None,
        "fastest_configuration": None,
        "best_throughput_configuration": None,
        "most_memory_efficient_configuration": None,
        "stable_configuration": None,
        "recommended_dataloader": None,
        "recommended_dataloader_batch_size": None,
        "dataloader_safe_batch_size_note": (
            "This is a DataLoader-safe batch size, not the final model-training "
            "batch size. Re-evaluate after the Encoder and optimizer are introduced."
        ),
        "oom_configurations": [_cfg_id(row) for row in oom],
        "failed_configurations": [
            {**_cfg_id(row), "error": row.get("error")} for row in failed
        ],
        "selection_pool": "none",
    }
    if not pool:
        return empty

    def _throughput(row: Mapping[str, Any]) -> float:
        return float(row.get("images_per_sec") or 0.0)

    largest = max(pool, key=lambda row: int(row["batch_size"]))
    fastest = max(pool, key=_throughput)
    memory_key = lambda row: (
        float(row.get("ram_peak_mb") or 0.0),
        float(row.get("gpu_peak_allocated_mb") or 0.0),
        -_throughput(row),
    )
    efficient = min(pool, key=memory_key)

    stable_candidates = []
    for row in pool:
        mean_ms = row.get("mean_batch_ms")
        p95_ms = row.get("p95_batch_ms")
        if mean_ms is None or p95_ms is None or mean_ms <= 0:
            continue
        if p95_ms > 3.0 * mean_ms:
            continue
        if cuda_available and row["stage"] == "dataloader_gpu":
            stable_candidates.append(row)
        elif not cuda_available and row["stage"] == "dataloader":
            stable_candidates.append(row)
    if not stable_candidates:
        stable_candidates = list(pool)

    def _stable_score(row: Mapping[str, Any]) -> Tuple[float, int, int, float]:
        workers_bonus = 1 if row["num_workers"] == 2 else (0 if row["num_workers"] == 0 else -1)
        pin_bonus = 1 if (cuda_available and row["pin_memory"]) else 0
        persist_penalty = 0 if not row["persistent_workers"] else 1
        return (_throughput(row), workers_bonus, pin_bonus, -persist_penalty)

    stable = max(stable_candidates, key=_stable_score)

    recommended = stable
    recommended_batch = int(recommended["batch_size"])

    note = (
        "This is a DataLoader-safe batch size, not the final model-training "
        "batch size. The largest successful DataLoader batch only means the "
        "CPU/GPU copy of preprocessed tensors fit; an Encoder, loss, and "
        "optimizer will consume additional VRAM and must be re-benchmarked "
        "before choosing a training batch size."
    )
    return {
        "largest_successful_dataloader_batch": int(largest["batch_size"]),
        "fastest_configuration": _cfg_id(fastest),
        "best_throughput_configuration": _cfg_id(fastest),
        "most_memory_efficient_configuration": _cfg_id(efficient),
        "stable_configuration": _cfg_id(stable),
        "recommended_dataloader": _cfg_id(recommended),
        "recommended_dataloader_batch_size": recommended_batch,
        "dataloader_safe_batch_size_note": note,
        "oom_configurations": [_cfg_id(row) for row in oom],
        "failed_configurations": [
            {**_cfg_id(row), "error": row.get("error")} for row in failed
        ],
        "selection_pool": "train_dataloader_gpu" if train_gpu else (
            "train_dataloader" if train_cpu else "all_pass"
        ),
    }


def run_dataloader_benchmark(
    *,
    manifest_path: PathLike,
    dataset_root: PathLike,
    configurations: Sequence[DataLoaderBenchmarkConfig],
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    measurement_batches: int = DEFAULT_MEASUREMENT_BATCHES,
    seed: int = DEFAULT_SEED,
    device: Union[str, torch.device] = "auto",
    preprocessing_config: Optional[ImagePreprocessingConfig] = None,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Dict[str, Any]:
    """Run the configuration matrix. Does not write files or mutate Dataset 1."""
    if warmup_batches < 0 or measurement_batches < 1:
        raise DataLoaderBenchmarkError(
            "warmup_batches must be >= 0 and measurement_batches must be >= 1."
        )
    if not configurations:
        raise DataLoaderBenchmarkError("No benchmark configurations were provided.")

    resolved_device = (
        _indexed_cuda_device(device)
        if isinstance(device, torch.device)
        else resolve_device(str(device))
    )
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    prep_config = preprocessing_config or ImagePreprocessingConfig()
    preprocessor = ImagePreprocessor(prep_config)
    train_aug = augmentation_config or AugmentationConfig(seed=seed)

    hardware = collect_hardware_snapshot(resolved_device)
    timestamp = datetime.now(timezone.utc).isoformat()

    splits_needed = []
    for item in configurations:
        if item.split not in splits_needed:
            splits_needed.append(item.split)

    datasets = {}
    split_sizes: Dict[str, int] = {}
    for split in splits_needed:
        base = UnifiedDataset(
            manifest,
            dataset_root=root,
            split=split,
            validate_files=True,
        )
        if split == TRAINING_ROLE:
            view = build_preprocessed_dataset(
                base,
                role=TRAINING_ROLE,
                preprocessor=preprocessor,
                augmentation=train_aug,
            )
        else:
            view = build_preprocessed_dataset(
                base,
                role=split,
                preprocessor=preprocessor,
                augmentation=None,
            )
        _verify_split_augmentation(view, split)
        datasets[split] = view
        split_sizes[split] = len(view)

    results: List[Dict[str, Any]] = []
    logger = logging.getLogger(__name__)
    for index, config in enumerate(configurations, start=1):
        logger.info(
            "Config %s/%s: split=%s stage=%s batch=%s workers=%s pin=%s persistent=%s",
            index,
            len(configurations),
            config.split,
            config.stage,
            config.batch_size,
            config.num_workers,
            config.pin_memory,
            config.persistent_workers,
        )
        if config.stage == "dataloader_gpu" and resolved_device.type != "cuda":
            results.append(
                configuration_result(
                    split=config.split,
                    stage=config.stage,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory,
                    persistent_workers=config.persistent_workers,
                    status="SKIPPED",
                    error="CUDA device is not available.",
                    augmentation_applied=config.split == TRAINING_ROLE,
                    ram_before_mb=process_rss_mb(),
                )
            )
            continue
        results.append(
            _run_one_configuration(
                datasets[config.split],
                config,
                device=resolved_device,
                warmup_batches=warmup_batches,
                measurement_batches=measurement_batches,
                seed=seed,
            )
        )

    recommendation = build_recommendation(
        results, cuda_available=resolved_device.type == "cuda"
    )
    return {
        "benchmark": {
            "name": BENCHMARK_NAME,
            "version": BENCHMARK_VERSION,
            "timestamp_utc": timestamp,
            "warmup_batches_requested": warmup_batches,
            "measurement_batches_requested": measurement_batches,
            "seed": seed,
            "device": str(resolved_device),
            "stages": ["dataloader", "dataloader_gpu"],
            "notes": [
                "dataloader stage does not copy tensors to GPU.",
                "dataloader_gpu stage copies collated batches to GPU only; no Encoder.",
                GPU_UTILIZATION_UNAVAILABLE_REASON,
            ],
        },
        "hardware": hardware,
        "dataset": {
            "name": "dataset1",
            "path": str(root),
            "manifest": str(manifest),
            "split_sizes": split_sizes,
            "splits_benchmarked": splits_needed,
        },
        "preprocessing": _preprocessing_snapshot(prep_config),
        "augmentation": {
            "implementation": "data.preprocessing.augmentation.TrainingAugmentor",
            "train": dict(train_aug.as_loggable_dict()),
            "valid": {"enabled": False, "deterministic": True},
            "test": {"enabled": False, "deterministic": True},
        },
        "configurations": results,
        "recommendation": recommendation,
    }


def write_dataloader_benchmark_json(report: Mapping[str, Any], output_path: PathLike) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def write_dataloader_benchmark_markdown(
    report: Mapping[str, Any], output_path: PathLike
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown(report), encoding="utf-8")
    return path


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    return "\n".join([header_line, sep, *body]) if body else header_line + "\n" + sep


def _md_cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _render_markdown(report: Mapping[str, Any]) -> str:
    hardware = report["hardware"]
    dataset = report["dataset"]
    benchmark = report["benchmark"]
    preprocessing = report["preprocessing"]
    augmentation = report["augmentation"]
    configurations = list(report["configurations"])
    recommendation = report["recommendation"]

    passed = [row for row in configurations if row["status"] == "PASS"]
    oom = [row for row in configurations if row["status"] == "OOM"]
    failed = [row for row in configurations if row["status"] == "FAIL"]

    rec_dl = recommendation.get("recommended_dataloader") or {}
    lines = [
        "# Dataset 1 DataLoader Benchmark (S1.12)",
        "",
        "This report measures the **data pipeline only** (load → optional train "
        "augmentation → preprocess → collate → optional GPU copy). It does not "
        "run an Encoder or a training loop.",
        "",
        "## 1. Hardware",
        "",
        _md_table(
            ["Field", "Value"],
            [
                ("Python", hardware.get("python_version")),
                ("PyTorch", hardware.get("pytorch_version")),
                ("Platform", hardware.get("platform")),
                ("CUDA available", hardware.get("cuda_available")),
                ("CUDA version", hardware.get("cuda_version")),
                ("GPU", hardware.get("gpu_name")),
                ("VRAM total (MiB)", hardware.get("vram_total_mb")),
                ("System RAM total (MiB)", hardware.get("ram_total_mb")),
                ("Process RSS at start (MiB)", hardware.get("process_rss_mb")),
                ("GPU utilization measurable", hardware.get("gpu_utilization_measurable")),
            ],
        ),
        "",
        str(hardware.get("gpu_utilization_note") or GPU_UTILIZATION_UNAVAILABLE_REASON),
        "",
        "Process RAM below is RSS for this Python process, not OS-wide memory.",
        "",
        "## 2. Dataset",
        "",
        _md_table(
            ["Field", "Value"],
            [
                ("Name", dataset.get("name")),
                ("Dataset path", dataset.get("path")),
                ("Manifest", dataset.get("manifest")),
                ("Splits", ", ".join(dataset.get("splits_benchmarked") or [])),
                ("Split sizes", json.dumps(dataset.get("split_sizes") or {})),
            ],
        ),
        "",
        "The existing Dataset 1 manifest is the source of truth. Source images "
        "and the manifest are not modified.",
        "",
        "## 3. Benchmark methodology",
        "",
        "- High-resolution monotonic clock (`time.perf_counter`).",
        f"- Requested warmup batches: {benchmark.get('warmup_batches_requested')}.",
        f"- Requested measurement batches: {benchmark.get('measurement_batches_requested')}.",
        "- Warmup latencies are discarded (first-batch latency is still recorded).",
        "- Measurement count is capped per configuration if a split is too small.",
        "- Train uses the existing S1.9 `TrainingAugmentor` defaults.",
        "- Valid and test use `ImagePreprocessor` only (no random augmentation).",
        "- `dataloader` stage never copies tensors to GPU.",
        "- `dataloader_gpu` stage copies `batch['image']` with `tensor.to(device, non_blocking=pin_memory)` after CUDA synchronize.",
        "- No Encoder, loss, optimizer, or backward pass.",
        f"- Seed: {benchmark.get('seed')}.",
        f"- Device: {benchmark.get('device')}.",
        "",
        "Preprocessing contract:",
        "",
        f"- {preprocessing.get('contract')}",
        f"- interpolation: `{preprocessing.get('interpolation')}`",
        f"- mean: `{preprocessing.get('mean')}`",
        f"- std: `{preprocessing.get('std')}`",
        "",
        "Train augmentation: enabled via S1.9 defaults "
        f"(seed={augmentation.get('train', {}).get('seed')}). "
        "Valid/test: deterministic, augmentation disabled.",
        "",
        "## 4. Configuration matrix",
        "",
        _md_table(
            [
                "split",
                "stage",
                "batch_size",
                "num_workers",
                "pin_memory",
                "persistent_workers",
                "status",
            ],
            (
                (
                    row["split"],
                    row["stage"],
                    row["batch_size"],
                    row["num_workers"],
                    row["pin_memory"],
                    row["persistent_workers"],
                    row["status"],
                )
                for row in configurations
            ),
        ),
        "",
        "## 5. CPU / RAM results",
        "",
        _md_table(
            [
                "split",
                "stage",
                "batch",
                "workers",
                "pin",
                "status",
                "RAM before (MiB)",
                "RAM after warmup (MiB)",
                "RAM peak (MiB)",
                "RAM delta (MiB)",
            ],
            (
                (
                    row["split"],
                    row["stage"],
                    row["batch_size"],
                    row["num_workers"],
                    row["pin_memory"],
                    row["status"],
                    row["ram_before_mb"],
                    row["ram_after_warmup_mb"],
                    row["ram_peak_mb"],
                    row["ram_delta_mb"],
                )
                for row in configurations
            ),
        ),
        "",
        "## 6. GPU / VRAM results",
        "",
        _md_table(
            [
                "split",
                "stage",
                "batch",
                "workers",
                "pin",
                "status",
                "alloc (MiB)",
                "reserved (MiB)",
                "peak alloc (MiB)",
                "peak reserved (MiB)",
                "GPU util %",
            ],
            (
                (
                    row["split"],
                    row["stage"],
                    row["batch_size"],
                    row["num_workers"],
                    row["pin_memory"],
                    row["status"],
                    row["gpu_allocated_mb"],
                    row["gpu_reserved_mb"],
                    row["gpu_peak_allocated_mb"],
                    row["gpu_peak_reserved_mb"],
                    row["gpu_utilization_percent"],
                )
                for row in configurations
                if row["stage"] == "dataloader_gpu" or row["gpu_allocated_mb"] is not None
            ),
        ),
        "",
        "## 7. Throughput comparison",
        "",
        _md_table(
            [
                "split",
                "stage",
                "batch",
                "workers",
                "pin",
                "persistent",
                "first ms",
                "mean ms",
                "median ms",
                "p95 ms",
                "batches/s",
                "images/s",
            ],
            (
                (
                    row["split"],
                    row["stage"],
                    row["batch_size"],
                    row["num_workers"],
                    row["pin_memory"],
                    row["persistent_workers"],
                    row["first_batch_ms"],
                    row["mean_batch_ms"],
                    row["median_batch_ms"],
                    row["p95_batch_ms"],
                    row["batches_per_sec"],
                    row["images_per_sec"],
                )
                for row in passed
            ),
        ),
        "",
        "## 8. OOM configurations",
        "",
    ]
    if oom:
        lines.append(
            _md_table(
                ["split", "stage", "batch_size", "num_workers", "pin_memory", "error"],
                (
                    (
                        row["split"],
                        row["stage"],
                        row["batch_size"],
                        row["num_workers"],
                        row["pin_memory"],
                        row["error"],
                    )
                    for row in oom
                ),
            )
        )
    else:
        lines.append("None. No CUDA OOM was recorded for the DataLoader / transfer stages.")
    lines.extend(
        [
            "",
            "Failed (non-OOM) configurations: "
            + ("none" if not failed else str(len(failed))),
            "",
        ]
    )
    if failed:
        lines.append(
            _md_table(
                ["split", "stage", "batch_size", "workers", "pin", "persistent", "error"],
                (
                    (
                        row["split"],
                        row["stage"],
                        row["batch_size"],
                        row["num_workers"],
                        row["pin_memory"],
                        row["persistent_workers"],
                        row["error"],
                    )
                    for row in failed
                ),
            )
        )
        lines.append("")

    lines.extend(
        [
            "## 9. Recommended DataLoader configuration",
            "",
            _md_table(
                ["Field", "Value"],
                [
                    (
                        "Largest successful DataLoader batch",
                        recommendation.get("largest_successful_dataloader_batch"),
                    ),
                    (
                        "Recommended DataLoader batch size",
                        recommendation.get("recommended_dataloader_batch_size"),
                    ),
                    ("Selection pool", recommendation.get("selection_pool")),
                    ("Recommended split", rec_dl.get("split")),
                    ("Recommended stage", rec_dl.get("stage")),
                    ("Recommended num_workers", rec_dl.get("num_workers")),
                    ("Recommended pin_memory", rec_dl.get("pin_memory")),
                    ("Recommended persistent_workers", rec_dl.get("persistent_workers")),
                    ("Recommended images/s", rec_dl.get("images_per_sec")),
                ],
            ),
            "",
            recommendation.get("dataloader_safe_batch_size_note"),
            "",
            "## 10. Limitations",
            "",
            "- Throughput is for the data pipeline, not training step time.",
            "- GPU utilization is not claimed (see hardware note).",
            "- Worker processes on Windows use spawn; first-batch latency includes worker start.",
            "- Valid/test use a smaller matrix than train.",
            "- A 4 GB GPU will be dominated by the Encoder later; tensor-only VRAM is not representative.",
            "- Persistent workers are sampled, not fully crossed with every batch size.",
            "",
            "## 11. Important note about final training batch size",
            "",
            "**This is a DataLoader-safe batch size, not the final model-training batch size.**",
            "",
            "After the Encoder and optimizer are introduced, re-run a training-step "
            "VRAM benchmark. Do not treat the largest successful DataLoader batch "
            "as the training batch size.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
