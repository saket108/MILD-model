from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable

import torch

from losses.box_losses import box_xyxy_to_cxcywh
from utils.checkpoint import save_checkpoint
from utils.logger import Logger


class Trainer:
    """Basic training loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        loss_fn,
        device: torch.device,
        save_dir: str | Path,
        logger: Logger | None = None,
        evaluator=None,
        best_metric_key: str = "map50",
        use_amp: bool = False,
        grad_clip_norm: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or Logger()
        self.evaluator = evaluator
        self.best_metric_key = best_metric_key
        self.best_metric = float("-inf")
        self.use_amp = use_amp and device.type == "cuda"
        self.grad_clip_norm = grad_clip_norm
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_one_epoch(self, dataloader: Iterable, epoch: int, epochs: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        loss_sums: Dict[str, float] = {}
        start_time = time.perf_counter()
        total_steps = len(dataloader)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for step, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(self.device)
            prompts = batch["prompt"]
            metrics = batch.get("metrics")
            severity = batch.get("severity")
            if step == 1:
                self.logger.log_epoch_start(
                    epoch=epoch,
                    epochs=epochs,
                    total_steps=total_steps,
                    batch_size=getattr(dataloader, "batch_size", None),
                    image_size=int(images.shape[-1]),
                )

            targets = []
            _, _, h, w = images.shape
            scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
            for i, (boxes, labels) in enumerate(zip(batch["boxes"], batch["labels"])):
                boxes = boxes.to(self.device)
                labels = labels.to(self.device)
                if boxes.numel() > 0:
                    boxes = boxes / scale
                    boxes = box_xyxy_to_cxcywh(boxes)
                sev = None
                if severity is not None:
                    sev = severity[i].to(self.device)
                targets.append({"boxes": boxes, "labels": labels, "severity": sev})

            metrics_tensor = metrics.to(self.device) if metrics is not None else None
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images, prompts, metrics_tensor)
                    loss, loss_dict = self.loss_fn(outputs, targets)
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, prompts, metrics_tensor)
                loss, loss_dict = self.loss_fn(outputs, targets)
                loss.backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            total_loss += float(loss.item())
            for key, value in loss_dict.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + self._to_float(value)

            should_log = step == 1 or step % 10 == 0 or step == total_steps
            if should_log:
                avg_loss_dict = {key: total / step for key, total in loss_sums.items()}
                elapsed = time.perf_counter() - start_time
                eta_seconds = (elapsed / step) * max(total_steps - step, 0) if step > 0 else None
                gpu_mem = None
                if self.device.type == "cuda":
                    gpu_mem = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
                num_instances = sum(int(labels.shape[0]) for labels in batch["labels"])
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_step(
                    epoch=epoch,
                    epochs=epochs,
                    step=step,
                    total=total_steps,
                    loss_dict=avg_loss_dict,
                    lr=lr,
                    num_instances=num_instances,
                    image_size=int(images.shape[-1]),
                    eta_seconds=eta_seconds,
                    gpu_mem_gb=gpu_mem,
                )

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(len(dataloader), 1)
        metrics = {"loss": avg_loss}
        metrics.update({key: total / max(total_steps, 1) for key, total in loss_sums.items()})
        return metrics

    def fit(self, train_loader: Iterable, epochs: int, val_loader: Iterable | None = None) -> None:
        history = []
        for epoch in range(1, epochs + 1):
            metrics = self.train_one_epoch(train_loader, epoch, epochs)
            self.logger.log_epoch_metrics("train", epoch, epochs, metrics)
            save_checkpoint(self.save_dir / "last.pt", self.model, self.optimizer, self.scheduler, epoch, metrics)

            if self.evaluator is not None and val_loader is not None:
                val_metrics = self.evaluator(self.model, val_loader, self.device)
                self.logger.log_epoch_metrics("val", epoch, epochs, val_metrics)
                score = val_metrics.get(self.best_metric_key)
                if score is not None and score > self.best_metric:
                    self.best_metric = score
                    save_checkpoint(
                        self.save_dir / "best.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        val_metrics,
                    )
                    self.logger.log_best(self.best_metric_key, score)
                history.append({"epoch": epoch, "train": metrics, "val": val_metrics})
            else:
                history.append({"epoch": epoch, "train": metrics})

            metrics_path = self.save_dir / "metrics.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    @staticmethod
    def _to_float(value) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        return float(value)
