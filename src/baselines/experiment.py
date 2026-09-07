"""End-to-end S2.6 baseline experiment orchestration."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_preprocessed_samples
from src.data.datasets import UnifiedDataset
from src.data.preprocessing import AugmentationConfig, ImagePreprocessor, build_preprocessed_dataset
from src.data.constants import CATEGORY_TO_ID
from src.models.embedding_head import EmbeddingHead, EmbeddingHeadConfig, CustomCNNEncoder, EncoderWithEmbeddingHead
from src.models.config import EncoderConfig
from src.training.seed import make_dataloader_generator
from src.training.trainer import Trainer

from .classifier import BaselineClassifier
from .config import BaselineConfig
from .metrics import classification_metrics
from .objective import classification_step


def _make_loader(dataset: Any, *, batch_size: int, shuffle: bool, seed: int, workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=__import__("src.training.seed", fromlist=["seed_worker"]).seed_worker,
        generator=make_dataloader_generator(seed),
        collate_fn=collate_preprocessed_samples,
    )


def _build_model(config: BaselineConfig) -> BaselineClassifier:
    encoder = CustomCNNEncoder(EncoderConfig())
    head = EmbeddingHead(EmbeddingHeadConfig(
        feature_dim=encoder.feature_dim,
        embedding_dim=config.embedding_dim,
    ))
    retrieval_model = EncoderWithEmbeddingHead(encoder, head)
    return BaselineClassifier(retrieval_model, num_classes=config.num_classes)


def _hardware_metadata(device: torch.device) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        payload.update({
            "gpu_name": props.name,
            "gpu_total_memory_mb": round(props.total_memory / (1024 ** 2), 2),
            "gpu_index": index,
        })
    return payload


def run_baseline(
    manifest_path: str | Path,
    *,
    dataset_root: Optional[str | Path] = None,
    config: Optional[BaselineConfig] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> Mapping[str, Any]:
    """Train S2.6 on Dataset 1 train/valid splits and write a benchmark report.

    Validation is used for model selection and category accuracy only. The
    test split is deliberately never loaded, because Dataset 1 validation has
    one image per product group and therefore cannot provide same-product
    retrieval positives for Recall@K.
    """
    cfg = config or BaselineConfig()
    manifest = Path(manifest_path)
    root = Path(dataset_root) if dataset_root is not None else None
    run_dir = Path(cfg.checkpoint_dir).parent
    run_dir.mkdir(parents=True, exist_ok=True)

    train_raw = UnifiedDataset(manifest, dataset_root=root, split="train")
    valid_raw = UnifiedDataset(manifest, dataset_root=root, split="valid")
    preprocessor = ImagePreprocessor()
    train_data = build_preprocessed_dataset(
        train_raw,
        role="train",
        preprocessor=preprocessor,
        augmentation=augmentation or AugmentationConfig(seed=cfg.seed),
    )
    valid_data = build_preprocessed_dataset(
        valid_raw,
        role="valid",
        preprocessor=preprocessor,
    )

    train_loader = _make_loader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=cfg.seed,
        workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    valid_loader = _make_loader(
        valid_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        seed=cfg.seed + 1,
        workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    model = _build_model(cfg)
    training_cfg = cfg.to_training_config()
    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        config=training_cfg,
        train_step=classification_step,
        valid_step=classification_step,
    )
    started = time.perf_counter()
    summary = dict(trainer.fit())
    wall_seconds = time.perf_counter() - started
    metrics = classification_metrics(model, valid_loader)

    report = {
        "experiment": "S2.6-baseline",
        "policy": "s2.6-category-baseline-v1",
        "status": summary["status"],
        "objective": "category_cross_entropy",
        "dataset": {
            "source": "dataset1",
            "train_samples": len(train_raw),
            "train_groups": len({sample.group_id for sample in train_raw.samples}),
            "valid_samples": len(valid_raw),
            "valid_groups": len({sample.group_id for sample in valid_raw.samples}),
            "test_used": False,
        },
        "model": {
            "architecture": "custom-cnn-v1 + embedding-head-v1 + baseline-category-classifier",
            "embedding_dim": cfg.embedding_dim,
            "num_classes": cfg.num_classes,
            "category_to_id": dict(CATEGORY_TO_ID),
        },
        "config": dict(cfg.as_loggable_dict()),
        "hardware": dict(_hardware_metadata(trainer.device)),
        "training": {
            "completed_epoch": summary["completed_epoch"],
            "best_val_loss": summary["best_val_loss"],
            "stopped_early": summary["stopped_early"],
            "duration_seconds": summary["duration_seconds"],
            "wall_clock_seconds": wall_seconds,
            "history": summary["history"],
        },
        "validation_metrics": dict(metrics),
        "checkpoint_last": summary["checkpoint_last"],
        "checkpoint_best": summary["checkpoint_best"],
        "manifest": str(manifest),
    }
    report_path = run_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report
