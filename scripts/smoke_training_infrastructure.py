"""Smoke-test the S2.5 training infrastructure with a synthetic regression task.

This script validates loop/optimizer/scheduler/checkpoint/logging mechanics
without claiming to validate the project's metric-learning objective. The
actual Gold baseline training remains an S2.6/S3 concern.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import Trainer, TrainingConfig


def step(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    features, targets = batch
    return torch.nn.functional.mse_loss(model(features), targets)


def main() -> None:
    torch.manual_seed(7)
    features = torch.randn(32, 2)
    targets = features @ torch.tensor([[2.0], [-1.0]]) + 0.5
    loader = DataLoader(TensorDataset(features, targets), batch_size=8, shuffle=False)

    with tempfile.TemporaryDirectory(prefix="gold_s2_5_smoke_") as directory:
        config = TrainingConfig(
            epochs=2,
            scheduler_name="cosine",
            checkpoint_dir=directory,
            early_stopping_enabled=False,
        )
        trainer = Trainer(
            nn.Linear(2, 1),
            loader,
            loader,
            config=config,
            train_step=step,
            valid_step=step,
        )
        summary = trainer.fit()
        assert summary["completed_epoch"] == 2
        assert Path(summary["checkpoint_last"]).is_file()
        assert Path(summary["checkpoint_best"]).is_file()
        print("S2.5 training infrastructure smoke test: PASS")


if __name__ == "__main__":
    main()
