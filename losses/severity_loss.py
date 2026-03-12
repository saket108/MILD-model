from __future__ import annotations

import torch
import torch.nn.functional as F


def severity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred, target, reduction="mean")
