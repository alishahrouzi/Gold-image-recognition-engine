"""Structured training logging for S2.5."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Write append-only epoch events plus a final summary JSON file."""

    def __init__(
        self,
        directory: str | Path,
        *,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "training_events.jsonl"
        self.summary_path = self.directory / "training_summary.json"
        self.config = dict(config or {})

    def log(self, event: str, **payload: Any) -> None:
        """Append one JSON event without rewriting previous events."""
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        logger.info("training.%s %s", event, payload)

    def log_epoch(self, *, epoch: int, train_loss: float, val_loss: float, learning_rate: float, duration_seconds: float) -> None:
        self.log(
            "epoch",
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            learning_rate=float(learning_rate),
            duration_seconds=float(duration_seconds),
        )

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        """Write the final run summary atomically enough for a single process."""
        payload = {
            "config": self.config,
            **dict(summary),
        }
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        return self.summary_path
