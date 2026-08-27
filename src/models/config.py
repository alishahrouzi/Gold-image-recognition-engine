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


@dataclass(frozen=True)
class EncoderConfig:
    """Hyperparameters for a custom CNN encoder.

    Training loop, optimizer, loss, and retrieval settings do not belong here.
    ``embedding_dim`` is the encoder output width ``D`` and must stay
    configurable for later embedding-head experiments.
    """

    input_channels: int = DEFAULT_INPUT_CHANNELS
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    input_height: int = DEFAULT_INPUT_HEIGHT
    input_width: int = DEFAULT_INPUT_WIDTH
    block_channels: Tuple[int, ...] = DEFAULT_BLOCK_CHANNELS

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

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Single-image tensor layout ``(C, H, W)`` expected by the encoder."""
        return (self.input_channels, self.input_height, self.input_width)

    def as_loggable_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["block_channels"] = list(self.block_channels)
        payload["policy"] = "s2.1-custom-cnn-encoder"
        return payload
