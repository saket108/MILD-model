from __future__ import annotations

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    total_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    sched_cfg = cfg.get("scheduler", cfg)
    sched_type = sched_cfg.get("type", sched_cfg.get("scheduler", "cosine"))
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if sched_type in (None, "none"):
        return None
    raise ValueError(f"Unsupported scheduler type: {sched_type}")
