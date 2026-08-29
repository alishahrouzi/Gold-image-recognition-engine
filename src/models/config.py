"""Encoder and embedding-head configuration (no training / retrieval settings)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, Tuple

from .errors import EmbeddingHeadConfigError, EncoderConfigError, ModelError

DEFAULT_INPUT_CHANNELS: int = 3
DEFAULT_EMBEDDING_DIM: int = 128
DEFAULT_INPUT_HEIGHT: int = 224
DEFAULT_INPUT_WIDTH: int = 224
DEFAULT_BLOCK_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256)
DEFAULT_CONVS_PER_STAGE: int = 2
DEFAULT_KERNEL_SIZE: int = 3
DEFAULT_ACTIVATION: str = "relu"
DEFAULT_NORMALIZATION: str = "batch"
DEFAULT_DOWNSAMPLE: str = "max_pool"
DEFAULT_PROJECTION_DROPOUT: float = 0.0
DEFAULT_FEATURE_DIM: int = 256
DEFAULT_HEAD_EMBEDDING_DIM: int = 128
DEFAULT_L2_EPS: float = 1e-12
SUPPORTED_EMBEDDING_DIMS: Tuple[int, ...] = (128, 256)
ARCHITECTURE_ID: str = "custom-cnn-v1"
ARCHITECTURE_POLICY: str = "s2.2-custom-cnn-v1"
EMBEDDING_HEAD_POLICY: str = "s2.3-embedding-head-v1"

ALLOWED_ACTIVATIONS: Tuple[str, ...] = ("relu", "leaky_relu", "gelu")
ALLOWED_NORMALIZATIONS: Tuple[str, ...] = ("batch", "none")
ALLOWED_DOWNSAMPLES: Tuple[str, ...] = ("max_pool",)


def _require_positive_int(
    value: Any,
    field_name: str,
    error_cls: type[ModelError] = EncoderConfigError,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise error_cls(f"{field_name} must be an integer.")
    if value < 1:
        raise error_cls(f"{field_name} must be a positive integer, got {value}.")
    return value


def _require_block_channels(values: Sequence[int]) -> Tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EncoderConfigError("block_channels must be a sequence of positive integers.")
    if len(values) < 1:
        raise EncoderConfigError("block_channels must contain at least one stage.")
    channels = tuple(_require_positive_int(item, "block_channels item") for item in values)
    return channels


def _require_choice(value: Any, field_name: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EncoderConfigError(f"{field_name} must be a non-empty string.")
    key = value.strip().lower()
    if key not in allowed:
        raise EncoderConfigError(
            f"{field_name} must be one of {list(allowed)}, got {value!r}."
        )
    return key


def _require_dropout(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EncoderConfigError(f"{field_name} must be a number in [0, 1).")
    dropout = float(value)
    if dropout < 0.0 or dropout >= 1.0:
        raise EncoderConfigError(f"{field_name} must be in [0, 1), got {dropout}.")
    return dropout


def _require_odd_kernel(value: Any, field_name: str) -> int:
    kernel = _require_positive_int(value, field_name)
    if kernel % 2 == 0:
        raise EncoderConfigError(f"{field_name} must be odd so padding preserves spatial size.")
    return kernel


@dataclass(frozen=True)
class EncoderConfig:
    """Hyperparameters for Custom CNN v1.

    Training loop, optimizer, loss, and retrieval settings do not belong here.
    After S2.3, Custom CNN v1 emits a raw pooled vector of width
    ``feature_dim`` (last ``block_channels`` entry, default 256). Linear
    projection and L2 normalization live on ``EmbeddingHead``, not here.

    ``embedding_dim`` remains a validated positive integer on this dataclass
    for S2.1/S2.2 config compatibility and experiment logs. The CNN no longer
    projects to it; use ``EmbeddingHeadConfig.embedding_dim`` for the retrieval
    embedding width (128 or 256).
    """

    input_channels: int = DEFAULT_INPUT_CHANNELS
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    input_height: int = DEFAULT_INPUT_HEIGHT
    input_width: int = DEFAULT_INPUT_WIDTH
    block_channels: Tuple[int, ...] = DEFAULT_BLOCK_CHANNELS
    convs_per_stage: int = DEFAULT_CONVS_PER_STAGE
    kernel_size: int = DEFAULT_KERNEL_SIZE
    activation: str = DEFAULT_ACTIVATION
    normalization: str = DEFAULT_NORMALIZATION
    downsample: str = DEFAULT_DOWNSAMPLE
    projection_dropout: float = DEFAULT_PROJECTION_DROPOUT

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "input_channels", _require_positive_int(self.input_channels, "input_channels")
        )
        object.__setattr__(
            self, "embedding_dim", _require_positive_int(self.embedding_dim, "embedding_dim")
        )
        object.__setattr__(
            self, "input_height", _require_positive_int(self.input_height, "input_height")
        )
        object.__setattr__(
            self, "input_width", _require_positive_int(self.input_width, "input_width")
        )
        object.__setattr__(self, "block_channels", _require_block_channels(self.block_channels))
        object.__setattr__(
            self,
            "convs_per_stage",
            _require_positive_int(self.convs_per_stage, "convs_per_stage"),
        )
        object.__setattr__(
            self, "kernel_size", _require_odd_kernel(self.kernel_size, "kernel_size")
        )
        object.__setattr__(
            self,
            "activation",
            _require_choice(self.activation, "activation", ALLOWED_ACTIVATIONS),
        )
        object.__setattr__(
            self,
            "normalization",
            _require_choice(self.normalization, "normalization", ALLOWED_NORMALIZATIONS),
        )
        object.__setattr__(
            self,
            "downsample",
            _require_choice(self.downsample, "downsample", ALLOWED_DOWNSAMPLES),
        )
        object.__setattr__(
            self,
            "projection_dropout",
            _require_dropout(self.projection_dropout, "projection_dropout"),
        )

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Single-image tensor layout ``(C, H, W)`` expected by the encoder."""
        return (self.input_channels, self.input_height, self.input_width)

    @property
    def number_of_stages(self) -> int:
        """Convolutional stages after the stem (length of ``block_channels``)."""
        return len(self.block_channels)

    @property
    def feature_dim(self) -> int:
        """Raw GAP feature width (last stage channels). Default 256."""
        return self.block_channels[-1]

    def as_loggable_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["block_channels"] = list(self.block_channels)
        payload["number_of_stages"] = self.number_of_stages
        payload["feature_dim"] = self.feature_dim
        payload["architecture_id"] = ARCHITECTURE_ID
        payload["policy"] = ARCHITECTURE_POLICY
        return payload


def _require_supported_embedding_dim(value: Any) -> int:
    dim = _require_positive_int(value, "embedding_dim", EmbeddingHeadConfigError)
    if dim not in SUPPORTED_EMBEDDING_DIMS:
        raise EmbeddingHeadConfigError(
            f"embedding_dim must be one of {list(SUPPORTED_EMBEDDING_DIMS)}, got {dim}."
        )
    return dim


def _require_positive_eps(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmbeddingHeadConfigError(f"{field_name} must be a positive finite number.")
    eps = float(value)
    if eps <= 0.0 or eps != eps or eps == float("inf"):
        raise EmbeddingHeadConfigError(f"{field_name} must be a positive finite number, got {eps}.")
    return eps


@dataclass(frozen=True)
class EmbeddingHeadConfig:
    """S2.3 embedding head: Linear(feature_dim → embedding_dim) then L2 normalize.

    Training, loss, temperature, and retrieval settings do not belong here.
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    embedding_dim: int = DEFAULT_HEAD_EMBEDDING_DIM
    l2_eps: float = DEFAULT_L2_EPS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_dim",
            _require_positive_int(self.feature_dim, "feature_dim", EmbeddingHeadConfigError),
        )
        object.__setattr__(self, "embedding_dim", _require_supported_embedding_dim(self.embedding_dim))
        object.__setattr__(self, "l2_eps", _require_positive_eps(self.l2_eps, "l2_eps"))

    def as_loggable_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["supported_embedding_dims"] = list(SUPPORTED_EMBEDDING_DIMS)
        payload["policy"] = EMBEDDING_HEAD_POLICY
        return payload
