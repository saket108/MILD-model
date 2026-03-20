from __future__ import annotations

from typing import Iterable

import torch

from evaluation.metrics import DetectionMetrics
from utils.detection_postprocess import postprocess_detections


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    score_threshold: float = 0.0,
    nms_iou: float | None = None,
    top_k: int | None = None,
    max_detections: int | None = None,
) -> dict:
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
        detections = postprocess_detections(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            pred_severity=pred_severity,
            score_threshold=score_threshold,
            nms_iou=nms_iou,
            top_k=top_k,
            max_detections=max_detections,
        )

        for i in range(len(images)):
            _, _, h, w = images.shape
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)
            target_boxes = targets["boxes"][i].cpu() / scale
            pred = detections[i]
            pred_severity_np = pred["severity"].cpu().numpy() if pred["severity"] is not None else None

            metrics.update(
                image_id=targets["image_id"][i],
                pred_boxes=pred["boxes"].cpu().numpy(),
                pred_scores=pred["scores"].cpu().numpy(),
                pred_labels=pred["labels"].cpu().numpy(),
                pred_severity=pred_severity_np,
                target_boxes=target_boxes.numpy(),
                target_labels=targets["labels"][i].cpu().numpy(),
                target_severity=targets.get("severity", [None])[i].cpu().numpy()
                if targets.get("severity") is not None
                else None,
            )

    return metrics.compute(class_name_map=class_name_map)
