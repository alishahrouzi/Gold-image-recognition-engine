"""EncoderConfig validation tests."""

from __future__ import annotations

import pytest

from models import (
    ARCHITECTURE_ID,
    ARCHITECTURE_POLICY,
    EMBEDDING_HEAD_POLICY,
    SUPPORTED_EMBEDDING_DIMS,
    EmbeddingHeadConfig,
    EmbeddingHeadConfigError,
    EncoderConfig,
    EncoderConfigError,
)
from models.config import (
    DEFAULT_ACTIVATION,
    DEFAULT_BLOCK_CHANNELS,
    DEFAULT_CONVS_PER_STAGE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_NORMALIZATION,
)


def test_default_config() -> None:
    config = EncoderConfig()
    assert config.input_channels == DEFAULT_INPUT_CHANNELS
    assert config.embedding_dim == DEFAULT_EMBEDDING_DIM
    assert config.input_height == DEFAULT_INPUT_HEIGHT
    assert config.input_width == DEFAULT_INPUT_WIDTH
    assert config.block_channels == DEFAULT_BLOCK_CHANNELS
    assert config.convs_per_stage == DEFAULT_CONVS_PER_STAGE
    assert config.kernel_size == DEFAULT_KERNEL_SIZE
    assert config.activation == DEFAULT_ACTIVATION
    assert config.normalization == DEFAULT_NORMALIZATION
    assert config.downsample == "max_pool"
    assert config.projection_dropout == 0.0
    assert config.input_shape == (3, 224, 224)
    assert config.number_of_stages == 4
    assert config.feature_dim == 256
    assert DEFAULT_EMBEDDING_DIM == 128


@pytest.mark.parametrize("embedding_dim", [64, 128, 256])
def test_configurable_embedding_dim(embedding_dim: int) -> None:
    config = EncoderConfig(embedding_dim=embedding_dim)
    assert config.embedding_dim == embedding_dim


@pytest.mark.parametrize("channels", [(32,), (16, 32, 64), (32, 64, 128, 256, 256)])
def test_channel_configuration(channels: tuple[int, ...]) -> None:
    config = EncoderConfig(block_channels=channels)
    assert config.block_channels == channels
    assert config.number_of_stages == len(channels)


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
        {"convs_per_stage": 0},
        {"kernel_size": 2},
        {"kernel_size": 4},
        {"activation": "swish"},
        {"normalization": "layer"},
        {"downsample": "stride"},
        {"projection_dropout": 1.0},
        {"projection_dropout": -0.1},
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
        {"activation": ""},
        {"normalization": 1},
    ],
)
def test_invalid_types(kwargs: dict) -> None:
    with pytest.raises(EncoderConfigError):
        EncoderConfig(**kwargs)


def test_as_loggable_dict() -> None:
    payload = EncoderConfig(embedding_dim=64).as_loggable_dict()
    assert payload["embedding_dim"] == 64
    assert payload["policy"] == ARCHITECTURE_POLICY
    assert payload["architecture_id"] == ARCHITECTURE_ID
    assert payload["architecture_id"] == "custom-cnn-v1"
    assert payload["policy"] == "s2.2-custom-cnn-v1"
    assert payload["block_channels"] == [32, 64, 128, 256]
    assert payload["feature_dim"] == 256
    assert payload["number_of_stages"] == 4
    assert payload["activation"] == "relu"
    assert payload["normalization"] == "batch"


def test_embedding_head_config_accepts_supported_dims() -> None:
    for dim in SUPPORTED_EMBEDDING_DIMS:
        config = EmbeddingHeadConfig(embedding_dim=dim)
        assert config.embedding_dim == dim
        assert config.feature_dim == 256
        assert config.l2_eps > 0
        payload = config.as_loggable_dict()
        assert payload["policy"] == EMBEDDING_HEAD_POLICY
        assert payload["embedding_dim"] == dim


def test_embedding_head_config_rejects_invalid_dims() -> None:
    for dim in (0, -1, 64, 127, 512, 1024):
        with pytest.raises(EmbeddingHeadConfigError):
            EmbeddingHeadConfig(embedding_dim=dim)
    with pytest.raises(EmbeddingHeadConfigError):
        EmbeddingHeadConfig(feature_dim=0)
    with pytest.raises(EmbeddingHeadConfigError):
        EmbeddingHeadConfig(feature_dim=-8)
    with pytest.raises(EmbeddingHeadConfigError):
        EmbeddingHeadConfig(l2_eps=0.0)
    with pytest.raises(EmbeddingHeadConfigError):
        EmbeddingHeadConfig(l2_eps=-1e-6)
    with pytest.raises(EmbeddingHeadConfigError):
        EmbeddingHeadConfig(l2_eps=float("inf"))
    with pytest.raises(EmbeddingHeadConfigError):
        EmbeddingHeadConfig(embedding_dim=True)  # type: ignore[arg-type]
    with pytest.raises(EmbeddingHeadConfigError):
        EmbeddingHeadConfig(feature_dim="256")  # type: ignore[arg-type]
