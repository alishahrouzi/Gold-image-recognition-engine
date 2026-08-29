"""Custom CNN v1 encoder (no pretrained weights)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .blocks import CNNStage, Stem
from .config import EncoderConfig
from .interfaces import Encoder, validate_encoder_input

ShapeTrace = List[Tuple[str, Tuple[int, ...]]]


class CustomCNNEncoder(Encoder):
    """From-scratch CNN v1: image tensor → unnormalized embedding.

    Architecture (S2.2 Custom CNN v1)::

        Stem (Conv → BN → ReLU, no downsample)
            → Stage 1..N-1 (2× ConvBlock, then MaxPool)
            → Stage N (2× ConvBlock, no pool)
            → AdaptiveAvgPool2d(1)
            → Flatten
            → optional dropout
            → Linear projection to ``embedding_dim``

    Default spatial path at 224×224 with four stages and three 2× pools:

        224 (input)
            → 224 (stem, 32)
            → 112 (stage 1, 32)
            → 56  (stage 2, 64)
            → 28  (stage 3, 128)
            → 28  (stage 4, 256)
            → 1×1 GAP
            → D

    There is no classification head, similarity head, or L2 normalization.
    """

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        super().__init__()
        self._config = config or EncoderConfig()
        channels = self._config.block_channels
        stem_channels = channels[0]
        self.stem = Stem(self._config.input_channels, stem_channels, self._config)

        stages = []
        in_channels = stem_channels
        last_index = len(channels) - 1
        for index, out_channels in enumerate(channels):
            stages.append(
                CNNStage(
                    in_channels,
                    out_channels,
                    convs=self._config.convs_per_stage,
                    kernel_size=self._config.kernel_size,
                    activation=self._config.activation,
                    normalization=self._config.normalization,
                    downsample=self._config.downsample,
                    apply_downsample=index != last_index,
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        dropout = self._config.projection_dropout
        self.projection_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.projection = nn.Linear(in_channels, self._config.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self._config.embedding_dim

    @property
    def config(self) -> EncoderConfig:
        return self._config

    def forward(self, x: Tensor) -> Tensor:
        validate_encoder_input(x, self._config)
        features = self.stem(x)
        for stage in self.stages:
            features = stage(features)
        pooled = self.pool(features)
        flat = self.flatten(pooled)
        return self.projection(self.projection_dropout(flat))


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Number of parameters (trainable by default)."""
    if trainable_only:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in module.parameters())


def estimate_parameter_bytes(module: nn.Module) -> int:
    """Storage size of parameters in bytes (excludes optimizer state)."""
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def trace_encoder_shapes(
    encoder: CustomCNNEncoder,
    *,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> ShapeTrace:
    """Record tensor shapes through stem, stages, GAP, and projection."""
    config = encoder.config
    device = device or next(encoder.parameters()).device
    images = torch.zeros(
        batch_size,
        config.input_channels,
        config.input_height,
        config.input_width,
        dtype=torch.float32,
        device=device,
    )
    encoder.eval()
    steps: ShapeTrace = [("input", tuple(images.shape))]
    with torch.no_grad():
        features = encoder.stem(images)
        steps.append(("stem", tuple(features.shape)))
        for index, stage in enumerate(encoder.stages, start=1):
            features = stage(features)
            steps.append((f"stage_{index}", tuple(features.shape)))
        pooled = encoder.pool(features)
        steps.append(("global_average_pool", tuple(pooled.shape)))
        flat = encoder.flatten(pooled)
        steps.append(("flatten", tuple(flat.shape)))
        dropped = encoder.projection_dropout(flat)
        embedding = encoder.projection(dropped)
        steps.append(("projection", tuple(embedding.shape)))
    return steps


def summarize_encoder(encoder: CustomCNNEncoder) -> List[Tuple[str, str, int]]:
    """Programmatic summary rows: (name, role, parameter count)."""
    rows: List[Tuple[str, str, int]] = [
        ("stem", "Conv-BN-Act, full resolution", count_parameters(encoder.stem, trainable_only=False)),
    ]
    last_index = len(encoder.stages)
    for index, stage in enumerate(encoder.stages, start=1):
        down = "MaxPool 2x" if index < last_index else "no downsample"
        rows.append(
            (
                f"stage_{index}",
                f"{encoder.config.convs_per_stage}x ConvBlock, {down}",
                count_parameters(stage, trainable_only=False),
            )
        )
    rows.append(("global_average_pool", "AdaptiveAvgPool2d(1)", 0))
    rows.append(("flatten", "N/A", 0))
    if not isinstance(encoder.projection_dropout, nn.Identity):
        rows.append(("projection_dropout", "Dropout", 0))
    rows.append(
        (
            "projection",
            f"Linear -> {encoder.embedding_dim}",
            count_parameters(encoder.projection, trainable_only=False),
        )
    )
    return rows


def _channels_from_shape(shape: Tuple[int, ...]) -> str:
    if len(shape) == 4:
        return str(shape[1])
    if len(shape) == 2:
        return str(shape[1])
    return "—"


def format_encoder_summary(
    encoder: CustomCNNEncoder,
    shape_trace: Optional[Sequence[Tuple[str, Tuple[int, ...]]]] = None,
) -> str:
    """Human-readable table generated from the live module (does not go stale)."""
    trace = list(shape_trace) if shape_trace is not None else trace_encoder_shapes(encoder)
    previous_by_name: dict[str, Tuple[int, ...]] = {}
    for index, (name, _shape) in enumerate(trace):
        if index == 0:
            continue
        previous_by_name[name] = trace[index - 1][1]
    shape_by_name = {name: shape for name, shape in trace}
    header = (
        f"{'Layer / Stage':<22} {'Input Shape':<22} {'Output Shape':<22} "
        f"{'Channels':<10} {'Params':>8}"
    )
    lines = [header, "-" * len(header)]
    for name, _role, n_params in summarize_encoder(encoder):
        out_shape = shape_by_name.get(name)
        in_shape = previous_by_name.get(name)
        in_text = str(list(in_shape)) if in_shape else "—"
        out_text = str(list(out_shape)) if out_shape else "—"
        channels = _channels_from_shape(out_shape) if out_shape else "—"
        lines.append(
            f"{name:<22} {in_text:<22} {out_text:<22} {channels:<10} {n_params:>8}"
        )
    lines.append("-" * len(header))
    total = count_parameters(encoder, trainable_only=False)
    lines.append(f"{'total':<22} {'':<22} {'':<22} {'':<10} {total:>8}")
    return "\n".join(lines)
