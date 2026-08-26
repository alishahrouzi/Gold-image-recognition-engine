"""Group-aware pair generation (S1.10).

Manifest metadata → unordered positive/negative pairs with split isolation.
Does not load pixels, train models, or implement evaluation retrieval.
"""

from .config import (
    DEFAULT_PAIR_SEED,
    DEFAULT_POSITIVE_NEGATIVE_RATIO,
    DEFAULT_SAME_CATEGORY_NEGATIVE_RATIO,
    PairGenerationConfig,
)
from .errors import PairGenerationError
from .generator import (
    PairGenerationResult,
    build_pair_generation_report,
    generate_pair_dataset,
    load_pairs_csv,
    write_pair_generation_report,
    write_pairs_csv,
)
from .sampler import count_available_positive_pairs, generate_positive_pairs, sample_negative_pairs
from .types import (
    NEGATIVE_TYPE_CROSS_CATEGORY,
    NEGATIVE_TYPE_SAME_CATEGORY,
    PAIR_CSV_FIELDS,
    PAIR_TYPE_NEGATIVE,
    PAIR_TYPE_POSITIVE,
    Pair,
    canonicalize_image_ids,
    make_pair_id,
    pair_from_csv_row,
    pair_from_samples,
)
from .validation import validate_pairs

__all__ = [
    "DEFAULT_PAIR_SEED",
    "DEFAULT_POSITIVE_NEGATIVE_RATIO",
    "DEFAULT_SAME_CATEGORY_NEGATIVE_RATIO",
    "NEGATIVE_TYPE_CROSS_CATEGORY",
    "NEGATIVE_TYPE_SAME_CATEGORY",
    "PAIR_CSV_FIELDS",
    "PAIR_TYPE_NEGATIVE",
    "PAIR_TYPE_POSITIVE",
    "Pair",
    "PairGenerationConfig",
    "PairGenerationError",
    "PairGenerationResult",
    "build_pair_generation_report",
    "canonicalize_image_ids",
    "count_available_positive_pairs",
    "generate_pair_dataset",
    "generate_positive_pairs",
    "load_pairs_csv",
    "make_pair_id",
    "pair_from_csv_row",
    "pair_from_samples",
    "sample_negative_pairs",
    "validate_pairs",
    "write_pair_generation_report",
    "write_pairs_csv",
]
