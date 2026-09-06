"""Checkpoint persistence and resume support for S2.5."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn, optim

from .errors import TrainingError
from .seed import capture_rng_state, restore_rng_state

CHECKPOINT_VERSION = "s2.5-checkpoint-v1"


class CheckpointManager:
    """Save and restore complete training state without overwriting history."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler.LRScheduler],
        epoch: int,
        best_metric: Optional[float],
        history: list[Mapping[str, Any]],
        config: Mapping[str, Any],
        early_stopping: Optional[Mapping[str, Any]] = None,
        path: Optional[str | Path] = None,
    ) -> Path:
        """Persist model, optimizer, scheduler, RNG, and loop state atomically."""
        destination = Path(path) if path is not None else self.last_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")

        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "epoch": int(epoch),
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "history": list(history),
            "config": dict(config),
            "early_stopping": None if early_stopping is None else dict(early_stopping),
            "rng_state": dict(capture_rng_state()),
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        return destination

    def save_best(self, **kwargs: Any) -> Path:
        """Save the current best model/training state to ``best.pt``."""
        kwargs["path"] = self.best_path
        return self.save(**kwargs)

    def load(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        restore_rng: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> Mapping[str, Any]:
        """Restore a checkpoint and return its loop metadata.

        Full training checkpoints are trusted project artifacts; ``weights_only``
        is explicitly disabled because optimizer/scheduler/RNG objects are part
        of the resume contract.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(payload, Mapping):
            raise TrainingError("Checkpoint payload must be a mapping.")
        if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise TrainingError(
                "Unsupported checkpoint version: "
                f"{payload.get('checkpoint_version')!r}."
            )

        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if restore_rng and payload.get("rng_state") is not None:
            restore_rng_state(payload["rng_state"])  # type: ignore[arg-type]
        return payload
