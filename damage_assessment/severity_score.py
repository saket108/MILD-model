from __future__ import annotations

from typing import Dict

import torch

from damage_assessment.adas_metrics import compute_metrics, compute_score


def severity_from_cxcywh(
    boxes_cxcywh: torch.Tensor,
    weights: Dict[str, float] | None = None,
    category: str | int | None = None,
    zone: str | None = None,
    class_ranks: Dict | None = None,
    zone_weights: Dict | None = None,
    default_rank: float = 1.0,
    default_zone_weight: float = 1.0,
) -> torch.Tensor:
    """Compute ADAS severity from normalized cx,cy,w,h boxes."""
    if boxes_cxcywh.numel() == 0:
        return boxes_cxcywh.new_zeros((0,))

    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    scores = []
    for i in range(boxes_cxcywh.shape[0]):
        area, elongation, edge = compute_metrics(float(cx[i]), float(cy[i]), float(w[i]), float(h[i]))
        scores.append(
            compute_score(
                area,
                elongation,
                edge,
                weights=weights,
                category=category,
                zone=zone,
                class_ranks=class_ranks,
                zone_weights=zone_weights,
                default_rank=default_rank,
                default_zone_weight=default_zone_weight,
            )
        )
    return torch.tensor(scores, device=boxes_cxcywh.device, dtype=boxes_cxcywh.dtype)


def severity_from_xyxy(
    boxes_xyxy: torch.Tensor,
    img_w: float,
    img_h: float,
    weights: Dict[str, float] | None = None,
    category: str | int | None = None,
    zone: str | None = None,
    class_ranks: Dict | None = None,
    zone_weights: Dict | None = None,
    default_rank: float = 1.0,
    default_zone_weight: float = 1.0,
) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((0,))
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
    return severity_from_cxcywh(
        boxes_cxcywh,
        weights=weights,
        category=category,
        zone=zone,
        class_ranks=class_ranks,
        zone_weights=zone_weights,
        default_rank=default_rank,
        default_zone_weight=default_zone_weight,
    )
