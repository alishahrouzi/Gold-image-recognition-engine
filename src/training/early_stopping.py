"""Early-stopping policy for S2.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class EarlyStopping:
    """Track validation improvement and decide when training should stop."""

    patience: int = 3
    min_delta: float = 0.0
    mode: str = "min"

    best: Optional[float] = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.patience, int) or isinstance(self.patience, bool) or self.patience < 1:
            raise ValueError("patience must be a positive integer.")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0.")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")

    def step(self, value: float) -> bool:
        """Update the monitor and return True when patience is exhausted."""
        value = float(value)
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False

        if self._is_improvement(value):
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.best = None if state.get("best") is None else float(state["best"])
        self.bad_epochs = int(state.get("bad_epochs", 0))
