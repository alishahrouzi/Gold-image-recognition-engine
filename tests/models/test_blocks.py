"""Building-block tests for Custom CNN v1."""

from __future__ import annotations

import torch

from models.blocks import CNNStage, ConvBlock, Stem
from models.config import EncoderConfig


def test_conv_block_preserves_spatial_size() -> None:
    block = ConvBlock(3, 8, kernel_size=3, activation="relu", normalization="batch")
    output = block(torch.randn(2, 3, 40, 40))
    assert output.shape == (2, 8, 40, 40)


def test_stem_does_not_downsample() -> None:
    stem = Stem(3, 32, EncoderConfig())
    output = stem(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 32, 224, 224)


def test_stage_optional_downsample() -> None:
    pooled = CNNStage(
        8,
        16,
        convs=2,
        kernel_size=3,
        activation="relu",
        normalization="batch",
        downsample="max_pool",
        apply_downsample=True,
    )
    kept = CNNStage(
        8,
        16,
        convs=2,
        kernel_size=3,
        activation="relu",
        normalization="none",
        downsample="max_pool",
        apply_downsample=False,
    )
    images = torch.randn(2, 8, 32, 32)
    assert pooled(images).shape == (2, 16, 16, 16)
    assert kept(images).shape == (2, 16, 32, 32)
