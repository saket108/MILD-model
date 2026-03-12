from __future__ import annotations

import torch


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", cfg)
    lr = opt_cfg.get("lr", opt_cfg.get("learning_rate", 1e-4))
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    opt_type = opt_cfg.get("type", "adamw").lower()

    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer type: {opt_type}")
