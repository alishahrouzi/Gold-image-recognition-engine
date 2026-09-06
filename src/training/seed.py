"""Reproducibility helpers for S2.5 training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    """Serializable RNG state bundle used by checkpoints."""

    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: Optional[list[torch.Tensor]]


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic execution.

    Deterministic execution is intentionally configurable because deterministic
    CUDA kernels can be slower. The default follows the project's reproducible
    experiment policy.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)


def capture_rng_state() -> Mapping[str, object]:
    """Capture process RNG state for exact training resume."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]

    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]
