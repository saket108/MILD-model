from __future__ import annotations

import json
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
        for step, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(self.device)
            prompts = batch["prompt"]
            metrics = batch.get("metrics")
            severity = batch.get("severity")

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
            if step % 10 == 0:
                self.logger.log_step(epoch, epochs, step, len(dataloader), loss_dict)

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(len(dataloader), 1)
        return {"loss": avg_loss}

    def fit(self, train_loader: Iterable, epochs: int, val_loader: Iterable | None = None) -> None:
        history = []
        for epoch in range(1, epochs + 1):
            metrics = self.train_one_epoch(train_loader, epoch, epochs)
            self.logger.log_epoch(epoch, epochs, metrics)
            save_checkpoint(self.save_dir / "last.pt", self.model, self.optimizer, self.scheduler, epoch, metrics)

            if self.evaluator is not None and val_loader is not None:
                val_metrics = self.evaluator(self.model, val_loader, self.device)
                self.logger.log_epoch(epoch, epochs, val_metrics)
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
                history.append({"epoch": epoch, "train": metrics, "val": val_metrics})
            else:
                history.append({"epoch": epoch, "train": metrics})

            metrics_path = self.save_dir / "metrics.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
