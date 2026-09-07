"""Contract tests for S2.5 TrainingConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import (
    ALLOWED_DEVICES,
    ALLOWED_LOSSES,
    ALLOWED_OPTIMIZERS,
    ALLOWED_SCHEDULERS,
    SUPPORTED_EMBEDDING_DIMS,
    TrainingConfig,
    TrainingConfigError,
)


def test_default_config_is_valid() -> None:
    config = TrainingConfig()
    assert config.embedding_dim == 128
    assert config.batch_size == 16
    assert config.epochs == 10
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.loss_name == "contrastive"
    assert config.margin == pytest.approx(1.0)
    assert config.optimizer_name == "adamw"
    assert config.scheduler_name == "cosine"
    assert config.num_workers == 0
    assert config.pin_memory is True
    assert config.device == "auto"
    assert config.checkpoint_dir == "checkpoints"
    assert config.save_best is True
    assert config.resume_from is None


def test_config_is_immutable() -> None:
    config = TrainingConfig()
    with pytest.raises(AttributeError):
        config.batch_size = 32  # type: ignore[misc]


@pytest.mark.parametrize("embedding_dim", SUPPORTED_EMBEDDING_DIMS)
def test_supported_embedding_dimensions(embedding_dim: int) -> None:
    assert TrainingConfig(embedding_dim=embedding_dim).embedding_dim == embedding_dim


def test_unsupported_embedding_dimension_rejected() -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(embedding_dim=64)


@pytest.mark.parametrize("field", ["batch_size", "epochs"])
def test_positive_integer_fields_reject_zero_and_negative(field: str) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(**{field: 0})
    with pytest.raises(TrainingConfigError):
        TrainingConfig(**{field: -1})


@pytest.mark.parametrize("field", ["batch_size", "epochs", "num_workers"])
def test_integer_fields_reject_boolean(field: str) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(**{field: True})


@pytest.mark.parametrize("learning_rate", [0, -1, float("inf"), float("nan")])
def test_learning_rate_must_be_positive_finite(learning_rate: float) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(learning_rate=learning_rate)


@pytest.mark.parametrize("weight_decay", [-1, float("inf"), float("nan")])
def test_weight_decay_must_be_non_negative_finite(weight_decay: float) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(weight_decay=weight_decay)


def test_zero_weight_decay_is_valid() -> None:
    assert TrainingConfig(weight_decay=0).weight_decay == 0.0


@pytest.mark.parametrize("margin", [0, -1, float("inf"), float("nan")])
def test_margin_must_be_positive_finite(margin: float) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(margin=margin)


@pytest.mark.parametrize("device", ALLOWED_DEVICES)
def test_supported_devices(device: str) -> None:
    assert TrainingConfig(device=device).device == device


def test_device_is_case_insensitive() -> None:
    assert TrainingConfig(device=" CUDA ").device == "cuda"


@pytest.mark.parametrize("device", ["gpu", "mps", "", "cuda:0"])
def test_invalid_device_rejected(device: str) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(device=device)


@pytest.mark.parametrize("field, values", [
    ("loss_name", ["triplet", "", 1]),
    ("optimizer_name", ["adam", "", 1]),
    ("scheduler_name", ["step", "", 1]),
])
def test_invalid_choices_rejected(field: str, values: list[object]) -> None:
    for value in values:
        with pytest.raises(TrainingConfigError):
            TrainingConfig(**{field: value})


@pytest.mark.parametrize("field", ["pin_memory", "save_best"])
def test_boolean_fields_require_bool(field: str) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(**{field: 1})


def test_num_workers_accepts_zero() -> None:
    assert TrainingConfig(num_workers=0).num_workers == 0


def test_checkpoint_path_accepts_path_object() -> None:
    config = TrainingConfig(checkpoint_dir=Path("runs/checkpoints"))
    assert config.checkpoint_dir == "runs/checkpoints"


def test_resume_from_accepts_path_and_none() -> None:
    assert TrainingConfig(resume_from=Path("runs/best.pt")).resume_from == "runs/best.pt"
    assert TrainingConfig(resume_from=None).resume_from is None


@pytest.mark.parametrize("value", ["", "   ", 123])
def test_invalid_checkpoint_dir_rejected(value: object) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(checkpoint_dir=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", "   ", 123])
def test_invalid_resume_path_rejected(value: object) -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(resume_from=value)  # type: ignore[arg-type]


def test_seed_must_be_integer() -> None:
    with pytest.raises(TrainingConfigError):
        TrainingConfig(seed=1.5)  # type: ignore[arg-type]


def test_as_loggable_dict_contains_only_training_concerns() -> None:
    payload = TrainingConfig().as_loggable_dict()
    assert payload["embedding_dim"] == 128
    assert payload["loss_name"] == "contrastive"
    assert payload["policy"] == "s2.5-training-infrastructure-v1"
    assert payload["supported_embedding_dims"] == [128, 256]
    assert "block_channels" not in payload
    assert "input_height" not in payload


def test_as_loggable_dict_is_safe_to_serialize() -> None:
    payload = TrainingConfig().as_loggable_dict()
    assert all(isinstance(key, str) for key in payload)
    assert all(not isinstance(value, Path) for value in payload.values())
