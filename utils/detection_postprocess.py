from __future__ import annotations

from typing import Dict, List

import torch

from losses.box_losses import box_cxcywh_to_xyxy


def _box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=box.dtype, device=box.device)

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: List[int] = []

    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break

        remaining = order[1:]
        ious = _box_iou(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


def _class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    keep_parts = []
    for class_id in torch.unique(labels).tolist():
        class_mask = labels == int(class_id)
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
        class_keep = _nms(boxes[class_indices], scores[class_indices], iou_threshold)
        keep_parts.append(class_indices[class_keep])

    if not keep_parts:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    keep = torch.cat(keep_parts)
    return keep[scores[keep].argsort(descending=True)]


def _empty_result(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_severity: torch.Tensor | None,
) -> Dict[str, torch.Tensor | None]:
    severity = None
    if pred_severity is not None:
        severity = torch.empty((0,), dtype=pred_severity.dtype, device=pred_severity.device)
    return {
        "boxes": torch.empty((0, 4), dtype=pred_boxes.dtype, device=pred_boxes.device),
        "scores": torch.empty((0,), dtype=pred_logits.dtype, device=pred_logits.device),
        "labels": torch.empty((0,), dtype=torch.long, device=pred_logits.device),
        "severity": severity,
    }


def postprocess_detections(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_severity: torch.Tensor | None = None,
    score_threshold: float = 0.0,
    nms_iou: float | None = None,
    top_k: int | None = None,
    max_detections: int | None = None,
) -> List[Dict[str, torch.Tensor | None]]:
    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    scores, labels = pred_logits.sigmoid().max(dim=-1)

    results = []
    for idx in range(pred_logits.shape[0]):
        image_boxes = boxes_xyxy[idx]
        image_scores = scores[idx]
        image_labels = labels[idx]
        image_severity = pred_severity[idx] if pred_severity is not None else None

        keep = (image_boxes[:, 2] > image_boxes[:, 0]) & (image_boxes[:, 3] > image_boxes[:, 1])
        keep_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if keep_indices.numel() == 0:
            results.append(_empty_result(pred_boxes, pred_logits, pred_severity))
            continue

        keep_indices = keep_indices[image_scores[keep_indices].argsort(descending=True)]

        if score_threshold > 0:
            keep_indices = keep_indices[image_scores[keep_indices] >= score_threshold]
        if top_k is not None and top_k > 0:
            keep_indices = keep_indices[:top_k]
        if keep_indices.numel() == 0:
            results.append(_empty_result(pred_boxes, pred_logits, pred_severity))
            continue

        image_boxes = image_boxes[keep_indices]
        image_scores = image_scores[keep_indices]
        image_labels = image_labels[keep_indices]
        if image_severity is not None:
            image_severity = image_severity[keep_indices]

        if nms_iou is not None and nms_iou > 0:
            nms_keep = _class_aware_nms(image_boxes, image_scores, image_labels, float(nms_iou))
            image_boxes = image_boxes[nms_keep]
            image_scores = image_scores[nms_keep]
            image_labels = image_labels[nms_keep]
            if image_severity is not None:
                image_severity = image_severity[nms_keep]

        if max_detections is not None and max_detections > 0:
            image_boxes = image_boxes[:max_detections]
            image_scores = image_scores[:max_detections]
            image_labels = image_labels[:max_detections]
            if image_severity is not None:
                image_severity = image_severity[:max_detections]

        results.append(
            {
                "boxes": image_boxes,
                "scores": image_scores,
                "labels": image_labels,
                "severity": image_severity,
            }
        )

    return results
