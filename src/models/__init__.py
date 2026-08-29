"""Custom CNN encoder and S2.3 embedding head.

Preprocessed image tensor → raw features → L2-normalized embedding.
No training loop, loss, similarity, ranking, or pretrained backbones.
"""

from .config import (
    ARCHITECTURE_ID,
    ARCHITECTURE_POLICY,
    DEFAULT_ACTIVATION,
    DEFAULT_BLOCK_CHANNELS,
    DEFAULT_CONVS_PER_STAGE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_FEATURE_DIM,
    DEFAULT_HEAD_EMBEDDING_DIM,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_NORMALIZATION,
    EMBEDDING_HEAD_POLICY,
    SUPPORTED_EMBEDDING_DIMS,
    EmbeddingHeadConfig,
    EncoderConfig,
)
from .embedding_head import EmbeddingHead, EncoderWithEmbeddingHead
from .encoder import (
    CustomCNNEncoder,
    count_parameters,
    estimate_parameter_bytes,
    format_encoder_summary,
    summarize_encoder,
    trace_encoder_shapes,
)
from .errors import (
    EmbeddingHeadConfigError,
    EmbeddingHeadInputError,
    EncoderConfigError,
    EncoderInputError,
    ModelError,
)
from .interfaces import Encoder, validate_embedding_head_input, validate_encoder_input

__all__ = [
    "ARCHITECTURE_ID",
    "ARCHITECTURE_POLICY",
    "DEFAULT_ACTIVATION",
    "DEFAULT_BLOCK_CHANNELS",
    "DEFAULT_CONVS_PER_STAGE",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_HEAD_EMBEDDING_DIM",
    "DEFAULT_INPUT_CHANNELS",
    "DEFAULT_INPUT_HEIGHT",
    "DEFAULT_INPUT_WIDTH",
    "DEFAULT_KERNEL_SIZE",
    "DEFAULT_NORMALIZATION",
    "EMBEDDING_HEAD_POLICY",
    "SUPPORTED_EMBEDDING_DIMS",
    "CustomCNNEncoder",
    "EmbeddingHead",
    "EmbeddingHeadConfig",
    "EmbeddingHeadConfigError",
    "EmbeddingHeadInputError",
    "Encoder",
    "EncoderConfig",
    "EncoderConfigError",
    "EncoderInputError",
    "EncoderWithEmbeddingHead",
    "ModelError",
    "count_parameters",
    "estimate_parameter_bytes",
    "format_encoder_summary",
    "summarize_encoder",
    "trace_encoder_shapes",
    "validate_embedding_head_input",
    "validate_encoder_input",
]
