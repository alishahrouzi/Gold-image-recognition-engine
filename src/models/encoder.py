"""Minimal custom CNN encoder baseline (no pretrained weights)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .config import EncoderConfig
from .interfaces import Encoder, validate_encoder_input


class _ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → optional spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int, *, downsample: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=False)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.downsample(x)


class CustomCNNEncoder(Encoder):
    """From-scratch CNN: image tensor → unnormalized embedding.

    Architecture (S2.1 baseline):

        Conv blocks (Conv → BN → ReLU → MaxPool)
            → AdaptiveAvgPool2d(1)
            → Flatten
            → Linear projection to ``embedding_dim``

    Adaptive pooling avoids a hard-coded flattened spatial size. There is no
    classification head, similarity head, or L2 normalization.
    """

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        super().__init__()
        self._config = config or EncoderConfig()
        stages = []
        in_channels = self._config.input_channels
        for out_channels in self._config.block_channels:
            stages.append(_ConvBlock(in_channels, out_channels, downsample=True))
            in_channels = out_channels
        self.backbone = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(in_channels, self._config.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self._config.embedding_dim

    @property
    def config(self) -> EncoderConfig:
        return self._config

    def forward(self, x: Tensor) -> Tensor:
        validate_encoder_input(x, self._config)
        features = self.backbone(x)
        pooled = self.pool(features)
        flat = torch.flatten(pooled, 1)
        return self.projection(flat)


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Number of parameters (trainable by default)."""
    if trainable_only:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in module.parameters())


def estimate_parameter_bytes(module: nn.Module) -> int:
    """Storage size of parameters in bytes (excludes optimizer state)."""
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
