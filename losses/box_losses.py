from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from torchvision.ops import generalized_box_iou
except Exception as exc:  # pragma: no cover
    generalized_box_iou = None
    _torchvision_error = exc


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def l1_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_boxes, target_boxes, reduction="none").sum(-1).mean()


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    if generalized_box_iou is None:
        raise ImportError(f"torchvision.ops.generalized_box_iou is required: {_torchvision_error}")
    giou = generalized_box_iou(pred_boxes, target_boxes)
    loss = 1.0 - torch.diag(giou)
    return loss.mean()
