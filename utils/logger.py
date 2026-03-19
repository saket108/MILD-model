from __future__ import annotations

from typing import Dict

from rich.console import Console
from rich.rule import Rule
from rich.table import Table


class Logger:
    """Compact training logger with YOLO-style progress rows."""

    def __init__(self) -> None:
        self.console = Console(soft_wrap=True)

    def log_epoch_start(
        self,
        epoch: int,
        epochs: int,
        total_steps: int,
        batch_size: int | None,
        image_size: int,
    ) -> None:
        self.console.print(Rule(f"[bold cyan]Epoch {epoch}/{epochs}[/bold cyan]"))
        meta = f"steps={total_steps} | batch={batch_size or '?'} | size={image_size}"
        self.console.print(meta, style="dim")
        header = (
            f"{'Epoch':<8}"
            f"{'GPU_mem':<10}"
            f"{'box':<10}"
            f"{'giou':<10}"
            f"{'cls':<10}"
            f"{'sev':<10}"
            f"{'Inst':<8}"
            f"{'Size':<8}"
            f"{'LR':<12}"
            f"{'Step':<14}"
            f"{'ETA':<10}"
        )
        self.console.print(header, style="bold")

    def log_step(
        self,
        epoch: int,
        epochs: int,
        step: int,
        total: int,
        loss_dict: Dict,
        lr: float,
        num_instances: int,
        image_size: int,
        eta_seconds: float | None,
        gpu_mem_gb: float | None,
    ) -> None:
        row = (
            f"{epoch}/{epochs:<6}"
            f"{self._fmt_gpu_mem(gpu_mem_gb):<10}"
            f"{self._fmt_metric(loss_dict.get('loss_bbox')):<10}"
            f"{self._fmt_metric(loss_dict.get('loss_giou')):<10}"
            f"{self._fmt_metric(loss_dict.get('loss_cls')):<10}"
            f"{self._fmt_metric(loss_dict.get('loss_severity')):<10}"
            f"{num_instances:<8}"
            f"{image_size:<8}"
            f"{self._fmt_lr(lr):<12}"
            f"{step}/{total:<9}"
            f"{self._fmt_eta(eta_seconds):<10}"
        )
        self.console.print(row)

    def log_epoch_metrics(self, stage: str, epoch: int, epochs: int, metrics: Dict[str, float]) -> None:
        table = Table(title=f"{stage.capitalize()} Summary {epoch}/{epochs}", show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key, value in metrics.items():
            table.add_row(key, self._fmt_metric(value))
        self.console.print(table)

    def log_best(self, metric_key: str, value: float) -> None:
        self.console.print(f"[green]Best {metric_key}[/green]: {value:.4f}")

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return float("nan")
        if hasattr(value, "detach"):
            value = value.detach()
        return float(value)

    @classmethod
    def _fmt_metric(cls, value) -> str:
        numeric = cls._to_float(value)
        if numeric != numeric:
            return "-"
        return f"{numeric:.4f}"

    @staticmethod
    def _fmt_lr(value: float) -> str:
        return f"{value:.2e}"

    @staticmethod
    def _fmt_gpu_mem(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.2f}G"

    @staticmethod
    def _fmt_eta(seconds: float | None) -> str:
        if seconds is None:
            return "-"
        total_seconds = max(int(seconds), 0)
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
