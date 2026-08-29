"""Custom CNN encoder module (S2.2).

Preprocessed image tensor → embedding. No training loop, loss, similarity,
ranking, or pretrained backbones.
"""

from .config import (
    ARCHITECTURE_ID,
    ARCHITECTURE_POLICY,
    DEFAULT_ACTIVATION,
    DEFAULT_BLOCK_CHANNELS,
    DEFAULT_CONVS_PER_STAGE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_NORMALIZATION,
    EncoderConfig,
)
from .encoder import (
    CustomCNNEncoder,
    count_parameters,
    estimate_parameter_bytes,
    format_encoder_summary,
    summarize_encoder,
    trace_encoder_shapes,
)
from .errors import EncoderConfigError, EncoderInputError, ModelError
from .interfaces import Encoder, validate_encoder_input

__all__ = [
    "ARCHITECTURE_ID",
    "ARCHITECTURE_POLICY",
    "DEFAULT_ACTIVATION",
    "DEFAULT_BLOCK_CHANNELS",
    "DEFAULT_CONVS_PER_STAGE",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_INPUT_CHANNELS",
    "DEFAULT_INPUT_HEIGHT",
    "DEFAULT_INPUT_WIDTH",
    "DEFAULT_KERNEL_SIZE",
    "DEFAULT_NORMALIZATION",
    "CustomCNNEncoder",
    "Encoder",
    "EncoderConfig",
    "EncoderConfigError",
    "EncoderInputError",
    "ModelError",
    "count_parameters",
    "estimate_parameter_bytes",
    "format_encoder_summary",
    "summarize_encoder",
    "trace_encoder_shapes",
    "validate_encoder_input",
]
