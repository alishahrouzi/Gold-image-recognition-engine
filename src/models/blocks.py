"""Small CNN building blocks for Custom CNN v1.

These modules are project-specific helpers, not a generic network framework.
"""

from __future__ import annotations

from torch import Tensor, nn

from .config import EncoderConfig
from .errors import EncoderConfigError


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise EncoderConfigError(f"Unsupported activation {name!r}.")


def build_normalization(name: str, num_channels: int) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm2d(num_channels)
    if name == "none":
        return nn.Identity()
    raise EncoderConfigError(f"Unsupported normalization {name!r}.")


def build_downsample(name: str) -> nn.Module:
    if name == "max_pool":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    raise EncoderConfigError(f"Unsupported downsample {name!r}.")


class ConvBlock(nn.Module):
    """Conv2d → normalization → activation. Spatial size is preserved."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        activation: str,
        normalization: str,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm = build_normalization(normalization, out_channels)
        self.activation = build_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class CNNStage(nn.Module):
    """Repeated ConvBlocks at one resolution, then optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        convs: int,
        kernel_size: int,
        activation: str,
        normalization: str,
        downsample: str,
        apply_downsample: bool,
    ) -> None:
        super().__init__()
        if convs < 1:
            raise EncoderConfigError("Each stage must contain at least one convolution.")
        blocks = []
        current = in_channels
        for _ in range(convs):
            blocks.append(
                ConvBlock(
                    current,
                    out_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    normalization=normalization,
                )
            )
            current = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.downsample = build_downsample(downsample) if apply_downsample else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(self.blocks(x))


class Stem(nn.Module):
    """Single full-resolution convolution. Does not downsample."""

    def __init__(self, in_channels: int, out_channels: int, config: EncoderConfig) -> None:
        super().__init__()
        self.block = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=config.kernel_size,
            activation=config.activation,
            normalization=config.normalization,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
