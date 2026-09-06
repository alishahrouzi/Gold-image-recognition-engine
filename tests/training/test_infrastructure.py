from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.checkpoint import CheckpointManager
from src.training.config import TrainingConfig
from src.training.early_stopping import EarlyStopping
from src.training.errors import TrainingConfigError, TrainingError
from src.training.loop import run_training_epoch
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.seed import capture_rng_state, restore_rng_state, set_seed
from src.training.trainer import Trainer, resolve_device
from src.training.validation import run_validation_epoch


def test_config_contains_early_stopping_contract() -> None:
    config = TrainingConfig()
    assert config.early_stopping_enabled is True
    assert config.early_stopping_patience == 3
    assert config.monitor == "val_loss"
    assert config.deterministic is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"early_stopping_patience": 0},
        {"early_stopping_patience": -1},
        {"early_stopping_patience": True},
        {"early_stopping_min_delta": -1},
        {"monitor": "accuracy"},
        {"deterministic": 1},
    ],
)
def test_invalid_training_controls_are_rejected(kwargs: dict) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(**kwargs)


def test_seed_reproduces_python_numpy_and_torch() -> None:
    set_seed(123, deterministic=False)
    first = (np.random.rand(), torch.rand(3))
    set_seed(123, deterministic=False)
    second = (np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_rng_state_can_be_restored() -> None:
    set_seed(77, deterministic=False)
    _ = torch.rand(4)
    state = capture_rng_state()
    expected = torch.rand(4)
    restore_rng_state(state)
    actual = torch.rand(4)
    assert torch.equal(expected, actual)


def test_optimizer_uses_trainable_parameters_only() -> None:
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    optimizer = build_optimizer(model, TrainingConfig(scheduler_name="none"))
    optimized_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized_ids
    assert all(parameter.requires_grad for group in optimizer.param_groups for parameter in group["params"])


def test_optimizer_rejects_model_without_trainable_parameters() -> None:
    model = nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.requires_grad = False
    with pytest.raises(TrainingError):
        build_optimizer(model, TrainingConfig(scheduler_name="none"))


def test_cosine_scheduler_uses_configured_epochs() -> None:
    model = nn.Linear(2, 1)
    config = TrainingConfig(epochs=4)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    assert scheduler is not None
    assert scheduler.T_max == 4


def test_none_scheduler_is_supported() -> None:
    model = nn.Linear(2, 1)
    config = TrainingConfig(scheduler_name="none")
    optimizer = build_optimizer(model, config)
    assert build_scheduler(optimizer, config) is None


def test_early_stopping_tracks_patience() -> None:
    stopping = EarlyStopping(patience=2)
    assert stopping.step(1.0) is False
    assert stopping.step(1.1) is False
    assert stopping.step(1.2) is True
    assert stopping.best == 1.0
    assert stopping.bad_epochs == 2


def test_early_stopping_resets_on_improvement() -> None:
    stopping = EarlyStopping(patience=2)
    stopping.step(1.0)
    stopping.step(1.1)
    assert stopping.step(0.9) is False
    assert stopping.bad_epochs == 0
    assert stopping.best == 0.9


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    config = TrainingConfig(checkpoint_dir=tmp_path.as_posix(), scheduler_name="none")
    optimizer = build_optimizer(model, config)
    manager = CheckpointManager(tmp_path)

    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=2,
        best_metric=0.25,
        history=[{"epoch": 1, "val_loss": 0.5}, {"epoch": 2, "val_loss": 0.25}],
        config=config.as_loggable_dict(),
    )
    assert path == manager.last_path
    assert path.is_file()

    restored_model = nn.Linear(2, 1)
    restored_optimizer = build_optimizer(restored_model, config)
    payload = manager.load(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        map_location="cpu",
    )
    assert payload["epoch"] == 2
    assert payload["best_metric"] == 0.25
    for left, right in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(left, right)


def _mse_step(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    features, targets = batch
    predictions = model(features)
    return torch.nn.functional.mse_loss(predictions, targets)


def test_training_epoch_updates_parameters() -> None:
    model = nn.Linear(2, 1)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = build_optimizer(model, TrainingConfig(scheduler_name="none"))
    dataset = TensorDataset(torch.randn(8, 2), torch.randn(8, 1))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    result = run_training_epoch(
        model,
        loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        step_fn=_mse_step,
    )
    assert result.batches == 2
    assert result.samples == 8
    assert result.loss >= 0
    assert any(not torch.equal(before, after) for before, after in zip(initial, model.parameters()))


def test_validation_epoch_does_not_update_parameters() -> None:
    model = nn.Linear(2, 1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    dataset = TensorDataset(torch.randn(8, 2), torch.randn(8, 1))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    result = run_validation_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        step_fn=_mse_step,
    )
    assert result.batches == 2
    assert result.samples == 8
    assert result.loss >= 0
    for old, new in zip(before, model.parameters()):
        assert torch.equal(old, new)


def test_training_rejects_non_scalar_loss() -> None:
    model = nn.Linear(2, 1)
    optimizer = build_optimizer(model, TrainingConfig(scheduler_name="none"))
    loader = DataLoader(TensorDataset(torch.randn(2, 2), torch.randn(2, 1)), batch_size=2)

    def bad_step(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return model(batch[0])

    with pytest.raises(TrainingError):
        run_training_epoch(
            model,
            loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            step_fn=bad_step,
        )


def test_resolve_auto_device() -> None:
    device = resolve_device(TrainingConfig(device="auto"))
    assert device.type in {"cpu", "cuda"}


def test_trainer_runs_and_writes_checkpoints_and_logs(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    features = torch.randn(12, 2)
    targets = torch.randn(12, 1)
    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    config = TrainingConfig(
        epochs=2,
        scheduler_name="none",
        early_stopping_enabled=False,
        checkpoint_dir=tmp_path.as_posix(),
        deterministic=True,
    )

    trainer = Trainer(
        model,
        loader,
        loader,
        config=config,
        train_step=_mse_step,
        valid_step=_mse_step,
    )
    summary = trainer.fit()

    assert summary["completed_epoch"] == 2
    assert Path(summary["checkpoint_last"]).is_file()
    assert Path(summary["checkpoint_best"]).is_file()
    assert (tmp_path / "logs" / "training_events.jsonl").is_file()
    assert (tmp_path / "logs" / "training_summary.json").is_file()
