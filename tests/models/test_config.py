"""EncoderConfig validation tests."""

from __future__ import annotations

import pytest

from models import EncoderConfig, EncoderConfigError
from models.config import (
    DEFAULT_BLOCK_CHANNELS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
)


def test_default_config() -> None:
    config = EncoderConfig()
    assert config.input_channels == DEFAULT_INPUT_CHANNELS
    assert config.embedding_dim == DEFAULT_EMBEDDING_DIM
    assert config.input_height == DEFAULT_INPUT_HEIGHT
    assert config.input_width == DEFAULT_INPUT_WIDTH
    assert config.block_channels == DEFAULT_BLOCK_CHANNELS
    assert config.input_shape == (3, 224, 224)


@pytest.mark.parametrize("embedding_dim", [64, 128, 256])
def test_configurable_embedding_dim(embedding_dim: int) -> None:
    config = EncoderConfig(embedding_dim=embedding_dim)
    assert config.embedding_dim == embedding_dim


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dim": 0},
        {"embedding_dim": -1},
        {"input_channels": 0},
        {"input_height": 0},
        {"input_width": -8},
        {"block_channels": ()},
        {"block_channels": (32, 0, 64)},
    ],
)
def test_invalid_positive_fields(kwargs: dict) -> None:
    with pytest.raises(EncoderConfigError):
        EncoderConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dim": 1.5},
        {"embedding_dim": True},
        {"input_channels": "3"},
        {"block_channels": "32,64"},
    ],
)
def test_invalid_types(kwargs: dict) -> None:
    with pytest.raises(EncoderConfigError):
        EncoderConfig(**kwargs)


def test_as_loggable_dict() -> None:
    payload = EncoderConfig(embedding_dim=64).as_loggable_dict()
    assert payload["embedding_dim"] == 64
    assert payload["policy"] == "s2.1-custom-cnn-encoder"
    assert payload["block_channels"] == [32, 64, 128, 256]
