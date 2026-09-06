"""High-level S2.5 trainer orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from .checkpoint import CheckpointManager
from .config import TrainingConfig
from .early_stopping import EarlyStopping
from .errors import TrainingError
from .logging import TrainingLogger
from .loop import run_training_epoch
from .optimizer import build_optimizer, current_learning_rates
from .scheduler import build_scheduler
from .seed import set_seed
from .validation import StepFunction, run_validation_epoch


def resolve_device(config: TrainingConfig) -> torch.device:
    """Resolve the configured device and fail clearly when CUDA is unavailable."""
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise TrainingError("TrainingConfig requested CUDA, but CUDA is not available.")
    return torch.device(config.device)


class Trainer:
    """Coordinate seed, optimization, validation, scheduling, checkpointing, and logging.

    The trainer deliberately does not own the metric-learning loss or pair
    semantics. Those belong to the Sprint 3 metric-learning layer and are
    supplied through ``step_fn``. This keeps S2.5 infrastructure reusable and
    prevents training infrastructure from leaking retrieval-specific logic
    into the model or data layers.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        valid_loader: Any,
        *,
        config: Optional[TrainingConfig] = None,
        train_step: StepFunction,
        valid_step: Optional[StepFunction] = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_step = train_step
        self.valid_step = valid_step or train_step
        self.device = resolve_device(self.config)

        set_seed(self.config.seed, deterministic=self.config.deterministic)
        self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, self.config)
        self.scheduler = build_scheduler(self.optimizer, self.config)
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode="min",
        )

        self.checkpoints = CheckpointManager(self.config.checkpoint_dir)
        log_dir = Path(self.config.checkpoint_dir) / "logs"
        self.logger = TrainingLogger(log_dir, config=self.config.as_loggable_dict())
        self.history: list[Mapping[str, Any]] = []
        self.best_metric: Optional[float] = None
        self.start_epoch = 1

        if self.config.resume_from is not None:
            self._resume(self.config.resume_from)

    def _resume(self, path: str) -> None:
        payload = self.checkpoints.load(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            restore_rng=True,
            map_location=self.device,
        )
        completed_epoch = int(payload.get("epoch", 0))
        if completed_epoch >= self.config.epochs:
            raise TrainingError(
                f"Checkpoint epoch {completed_epoch} is already at or beyond configured "
                f"epochs={self.config.epochs}."
            )
        self.start_epoch = completed_epoch + 1
        raw_history = payload.get("history", [])
        self.history = [dict(item) for item in raw_history]
        raw_best = payload.get("best_metric")
        self.best_metric = None if raw_best is None else float(raw_best)
        early_state = payload.get("early_stopping")
        if isinstance(early_state, Mapping):
            self.early_stopping.load_state_dict(early_state)
        self.logger.log("resume", checkpoint=str(path), epoch=completed_epoch)

    def fit(self) -> Mapping[str, Any]:
        """Train until configured epochs or early stopping."""
        started = time.perf_counter()
        self.logger.log(
            "start",
            device=str(self.device),
            start_epoch=self.start_epoch,
            configured_epochs=self.config.epochs,
        )

        stopped_early = False
        completed_epoch = self.start_epoch - 1

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            epoch_started = time.perf_counter()
            train_result = run_training_epoch(
                self.model,
                self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                step_fn=self.train_step,
            )
            valid_result = run_validation_epoch(
                self.model,
                self.valid_loader,
                device=self.device,
                step_fn=self.valid_step,
            )

            val_loss = valid_result.loss
            improved = self.best_metric is None or (
                val_loss < self.best_metric - self.config.early_stopping_min_delta
            )
            if improved:
                self.best_metric = val_loss

            if self.scheduler is not None:
                self.scheduler.step()

            learning_rate = current_learning_rates(self.optimizer)[0]
            duration = time.perf_counter() - epoch_started
            record = {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "train_batches": train_result.batches,
                "train_samples": train_result.samples,
                "valid_batches": valid_result.batches,
                "valid_samples": valid_result.samples,
                "duration_seconds": duration,
            }
            self.history.append(record)
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_result.loss,
                val_loss=val_loss,
                learning_rate=learning_rate,
                duration_seconds=duration,
            )

            self.checkpoints.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_metric=self.best_metric,
                history=self.history,
                config=self.config.as_loggable_dict(),
                early_stopping=self.early_stopping.state_dict(),
            )
            if improved and self.config.save_best:
                self.checkpoints.save_best(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    history=self.history,
                    config=self.config.as_loggable_dict(),
                    early_stopping=self.early_stopping.state_dict(),
                )

            completed_epoch = epoch
            if self.config.early_stopping_enabled and self.early_stopping.step(val_loss):
                stopped_early = True
                self.logger.log("early_stopping", epoch=epoch, best_metric=self.best_metric)
                break

        total_duration = time.perf_counter() - started
        summary = {
            "status": "completed",
            "completed_epoch": completed_epoch,
            "best_val_loss": self.best_metric,
            "stopped_early": stopped_early,
            "duration_seconds": total_duration,
            "checkpoint_last": str(self.checkpoints.last_path),
            "checkpoint_best": str(self.checkpoints.best_path) if self.config.save_best else None,
            "device": str(self.device),
            "history": self.history,
        }
        self.logger.write_summary(summary)
        self.logger.log("finish", **{key: value for key, value in summary.items() if key != "history"})
        return summary
