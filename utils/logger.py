from __future__ import annotations

from typing import Dict

from rich.console import Console


class Logger:
    """Simple Rich logger."""

    def __init__(self) -> None:
        self.console = Console()

    def log_step(self, epoch: int, epochs: int, step: int, total: int, loss_dict: Dict) -> None:
        loss_str = ", ".join([f"{k}: {self._to_float(v):.4f}" for k, v in loss_dict.items()])
        self.console.log(f"Epoch {epoch}/{epochs} | Step {step}/{total} | {loss_str}")

    def log_epoch(self, epoch: int, epochs: int, metrics: Dict[str, float]) -> None:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.console.log(f"Epoch {epoch}/{epochs} | {metrics_str}")

    @staticmethod
    def _to_float(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        return float(value)
