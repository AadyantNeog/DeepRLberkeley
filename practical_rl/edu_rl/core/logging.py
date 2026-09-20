"""Minimal experiment logging to CSV and, when installed, TensorBoard."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Write scalar dictionaries in a format that remains easy to inspect."""

    def __init__(self, directory: str | Path, config: Any | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, float | int]] = []
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.directory / "tensorboard")
        except ImportError:
            # CSV remains the dependency-free source of truth.
            pass
        if config is not None:
            config_value = getattr(config, "__dict__", config)
            with (self.directory / "config.json").open("w", encoding="utf-8") as file:
                json.dump(config_value, file, indent=2, default=str)

    def log(self, step: int, **metrics: float) -> None:
        row: dict[str, float | int] = {"step": step, **metrics}
        self.rows.append(row)
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.rows:
            fieldnames = sorted({key for row in self.rows for key in row})
            with (self.directory / "metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
        if self.writer is not None:
            self.writer.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

