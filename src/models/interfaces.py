"""Encoder contract: preprocessed image tensors → embeddings.

The encoder consumes only an image tensor. It does not load files, apply
preprocessing or augmentation, read metadata (``group_id``, ``category_id``,
``image_id``), compute similarity, or rank products.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from .config import EncoderConfig
from .errors import EncoderInputError


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


class Encoder(nn.Module, ABC):
    """PyTorch encoder: ``Tensor[B, 3, H, W]`` → ``Tensor[B, D]``.

    ``D`` is ``embedding_dim``. Batch dimension 0 is preserved. Output is a
    2-D embedding tensor suitable for later heads (projection, L2, similarity)
    without changing the data or preprocessing contracts.

    Subclasses must not call ``.cuda()`` internally. Device placement belongs
    to the training / inference caller via ``model.to(device)``.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Width of the embedding dimension ``D``."""

    @property
    @abstractmethod
    def config(self) -> EncoderConfig:
        """Validated encoder configuration."""

    def encode(self, x: Tensor) -> Tensor:
        """Architecture 17.2 alias for ``forward`` (image tensor only)."""
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
