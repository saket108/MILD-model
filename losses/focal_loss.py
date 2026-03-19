from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int | float = 1,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss on logits.

    targets should be one-hot encoded with the same shape as logits.
    """
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    num_boxes = max(float(num_boxes), 1.0)
    return loss.mean(dim=-1).sum() / num_boxes
