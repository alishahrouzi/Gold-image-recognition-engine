"""Model contracts: encoder (image → features) and embedding head (features → L2 embedding).

The encoder consumes only an image tensor. It does not load files, apply
preprocessing or augmentation, read metadata (``group_id``, ``category_id``,
``image_id``), compute similarity, or rank products.

The embedding head consumes only a feature tensor. It does not load images,
run the CNN, compute similarity, rank products, or apply a training loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from .config import EmbeddingHeadConfig, EncoderConfig
from .errors import EmbeddingHeadInputError, EncoderInputError


def validate_encoder_input(x: Tensor, config: EncoderConfig) -> Tensor:
    """Fail loud on contract violations. Never silently reshape."""
    if not isinstance(x, Tensor):
        raise EncoderInputError(
            f"Encoder input must be a torch.Tensor, got {type(x).__name__}."
        )
    if x.ndim != 4:
        raise EncoderInputError(
            f"Encoder input must have rank 4 [B, C, H, W], got shape {tuple(x.shape)}."
        )
    batch, channels, height, width = x.shape
    if batch < 1:
        raise EncoderInputError(f"Encoder batch size must be >= 1, got {batch}.")
    if channels != config.input_channels:
        raise EncoderInputError(
            f"Encoder expects {config.input_channels} channels, got {channels}."
        )
    if height != config.input_height or width != config.input_width:
        raise EncoderInputError(
            "Encoder expects spatial size "
            f"[{config.input_height}, {config.input_width}], "
            f"got [{height}, {width}]."
        )
    if not x.is_floating_point():
        raise EncoderInputError(
            f"Encoder input dtype must be a floating type, got {x.dtype}."
        )
    return x


def validate_embedding_head_input(x: Tensor, config: EmbeddingHeadConfig) -> Tensor:
    """Fail loud on contract violations. Never silently reshape, pad, or truncate."""
    if not isinstance(x, Tensor):
        raise EmbeddingHeadInputError(
            f"EmbeddingHead input must be a torch.Tensor, got {type(x).__name__}."
        )
    if x.ndim != 2:
        raise EmbeddingHeadInputError(
            f"EmbeddingHead input must have rank 2 [B, feature_dim], got shape {tuple(x.shape)}."
        )
    batch, features = x.shape
    if batch < 1:
        raise EmbeddingHeadInputError(f"EmbeddingHead batch size must be >= 1, got {batch}.")
    if features != config.feature_dim:
        raise EmbeddingHeadInputError(
            f"EmbeddingHead expects feature_dim={config.feature_dim}, got {features}."
        )
    if not x.is_floating_point():
        raise EmbeddingHeadInputError(
            f"EmbeddingHead input dtype must be a floating type, got {x.dtype}."
        )
    return x


class Encoder(nn.Module, ABC):
    """PyTorch encoder: ``Tensor[B, 3, H, W]`` → ``Tensor[B, C]`` raw features.

    ``C`` is ``feature_dim`` (GAP width). Batch dimension 0 is preserved.
    Output is unnormalized and is not the retrieval embedding; projection
    and L2 live on ``EmbeddingHead``.

    Subclasses must not call ``.cuda()`` internally. Device placement belongs
    to the training / inference caller via ``model.to(device)``.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Width of the encoder output (raw feature dim ``C``)."""

    @property
    def feature_dim(self) -> int:
        """Alias for the raw GAP width (same as ``embedding_dim`` after S2.3)."""
        return self.embedding_dim

    @property
    @abstractmethod
    def config(self) -> EncoderConfig:
        """Validated encoder configuration."""

    def encode(self, x: Tensor) -> Tensor:
        """Architecture 17.2 alias for ``forward`` (image tensor only)."""
        return self.forward(x)

    def encode_features(self, x: Tensor) -> Tensor:
        """Image tensor → raw GAP features. Same as ``forward`` for Custom CNN v1."""
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
