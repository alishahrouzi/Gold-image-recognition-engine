"""Encoder configuration, independent of training and retrieval settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, Tuple

from .errors import EncoderConfigError

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
ARCHITECTURE_ID: str = "custom-cnn-v1"
ARCHITECTURE_POLICY: str = "s2.2-custom-cnn-v1"

ALLOWED_ACTIVATIONS: Tuple[str, ...] = ("relu", "leaky_relu", "gelu")
ALLOWED_NORMALIZATIONS: Tuple[str, ...] = ("batch", "none")
ALLOWED_DOWNSAMPLES: Tuple[str, ...] = ("max_pool",)


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EncoderConfigError(f"{field_name} must be an integer.")
    if value < 1:
        raise EncoderConfigError(f"{field_name} must be a positive integer, got {value}.")
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
    ``embedding_dim`` is the encoder output width ``D`` and must stay
    configurable for later embedding-head experiments.

    Default ``embedding_dim`` remains 128: Dataset 1 has ~2135 products and
    ~4969 images, cosine retrieval cost scales with D, and a later embedding
    head (S2.3) can still expand or re-project without changing the backbone.
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

    def as_loggable_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["block_channels"] = list(self.block_channels)
        payload["number_of_stages"] = self.number_of_stages
        payload["architecture_id"] = ARCHITECTURE_ID
        payload["policy"] = ARCHITECTURE_POLICY
        return payload
