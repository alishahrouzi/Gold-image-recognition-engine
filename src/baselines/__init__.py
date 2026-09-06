"""S2.6 category-supervised baseline training and evaluation."""

from .classifier import BaselineClassifier
from .config import BaselineConfig
from .experiment import run_baseline
from .objective import classification_step
from .metrics import classification_metrics

__all__ = [
    "BaselineClassifier",
    "BaselineConfig",
    "classification_metrics",
    "classification_step",
    "run_baseline",
]
