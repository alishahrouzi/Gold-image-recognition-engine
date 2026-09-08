"""S3 metric-learning building blocks.

The package contains loss-agnostic components for learning product-level
visual similarity. S3.1 introduces the shared-backbone Siamese architecture.
"""

from .siamese import SiameseNetwork

__all__ = ["SiameseNetwork"]
