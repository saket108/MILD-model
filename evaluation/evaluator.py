from __future__ import annotations

from typing import Iterable

import torch

from evaluation.metrics import DetectionMetrics
from losses.box_losses import box_cxcywh_to_xyxy


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: Iterable, device: torch.device) -> dict:
    model.eval()
    metrics = DetectionMetrics()
    dataset = getattr(dataloader, "dataset", None)
    class_name_map = getattr(dataset, "id_to_label", None)

    for batch in dataloader:
        images = batch["image"].to(device)
        prompts = batch["prompt"]
        batch_metrics = batch.get("metrics")
        targets = {
            "boxes": batch["boxes"],
            "labels": batch["labels"],
            "severity": batch.get("severity"),
            "image_id": batch["image_id"],
        }

        metrics_tensor = batch_metrics.to(device) if batch_metrics is not None else None
        outputs = model(images, prompts, metrics_tensor)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        pred_severity = outputs.get("pred_severity")

        scores, labels = pred_logits.sigmoid().max(dim=-1)
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes).cpu().numpy()
        pred_severity_np = pred_severity.cpu().numpy() if pred_severity is not None else None

        for i in range(len(images)):
            _, _, h, w = images.shape
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)
            target_boxes = targets["boxes"][i].cpu() / scale

            metrics.update(
                image_id=targets["image_id"][i],
                pred_boxes=pred_boxes_xyxy[i],
                pred_scores=scores[i].cpu().numpy(),
                pred_labels=labels[i].cpu().numpy(),
                pred_severity=pred_severity_np[i] if pred_severity_np is not None else None,
                target_boxes=target_boxes.numpy(),
                target_labels=targets["labels"][i].cpu().numpy(),
                target_severity=targets.get("severity", [None])[i].cpu().numpy()
                if targets.get("severity") is not None
                else None,
            )

    return metrics.compute(class_name_map=class_name_map)
