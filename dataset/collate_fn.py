from __future__ import annotations

import torch


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "severity": [b["severity"] for b in batch],
        "metrics": torch.stack([b["metrics"] for b in batch], dim=0),
        "image_id": [b["image_id"] for b in batch],
    }
