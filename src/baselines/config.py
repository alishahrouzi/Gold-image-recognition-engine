"""Configuration for the S2.6 category-supervised baseline experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.training.config import TrainingConfig


@dataclass(frozen=True)
class BaselineConfig:
    """S2.6 baseline settings.

    The baseline uses category supervision only as a reference objective. It
    does not introduce product-pair or contrastive semantics; those belong to
    S3. The production retrieval architecture remains Encoder + EmbeddingHead.
    """

    embedding_dim: int = 128
    num_classes: int = 5
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = True
    deterministic: bool = True
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    save_best: bool = True
    checkpoint_dir: str = "experiments/baseline/checkpoints"

    def __post_init__(self) -> None:
        if self.embedding_dim not in (128, 256):
            raise ValueError("embedding_dim must be 128 or 256.")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("batch_size/epochs must be positive and num_workers non-negative.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive.")

    def to_training_config(self) -> TrainingConfig:
        """Map shared optimization settings into the stable S2.5 contract."""
        return TrainingConfig(
            seed=self.seed,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            # Kept for S2.5 compatibility; the actual baseline objective is injected.
            loss_name="contrastive",
            optimizer_name="adamw",
            scheduler_name="cosine",
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
            save_best=self.save_best,
            early_stopping_enabled=self.early_stopping_enabled,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            monitor="val_loss",
            deterministic=self.deterministic,
        )

    def as_loggable_dict(self) -> Mapping[str, Any]:
        payload = dict(asdict(self))
        payload["objective"] = "category_cross_entropy"
        payload["policy"] = "s2.6-category-baseline-v1"
        return payload
