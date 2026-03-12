from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from losses.box_losses import box_cxcywh_to_xyxy

try:
    from torchvision.ops import generalized_box_iou
except Exception as exc:  # pragma: no cover
    generalized_box_iou = None
    _torchvision_error = exc

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


class HungarianMatcher(nn.Module):
    """Assigns predictions to targets using a cost matrix."""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 1.0, cost_giou: float = 1.0) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if generalized_box_iou is None:
            raise ImportError(f"torchvision.ops.generalized_box_iou is required: {_torchvision_error}")

        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].sigmoid()
        out_bbox = outputs["pred_boxes"]

        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]
            if tgt_bbox.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox[b])
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()

            if linear_sum_assignment is not None:
                row_ind, col_ind = linear_sum_assignment(C)
                indices.append(
                    (
                        torch.as_tensor(row_ind, dtype=torch.long),
                        torch.as_tensor(col_ind, dtype=torch.long),
                    )
                )
            else:
                # Greedy fallback if scipy is unavailable
                cost = C.clone()
                matched_pred = []
                matched_tgt = []
                for _ in range(min(cost.shape[0], cost.shape[1])):
                    min_idx = torch.argmin(cost)
                    i = int(min_idx // cost.shape[1])
                    j = int(min_idx % cost.shape[1])
                    matched_pred.append(i)
                    matched_tgt.append(j)
                    cost[i, :] = float("inf")
                    cost[:, j] = float("inf")
                indices.append(
                    (
                        torch.as_tensor(matched_pred, dtype=torch.long),
                        torch.as_tensor(matched_tgt, dtype=torch.long),
                    )
                )

        return indices
