from __future__ import annotations

import torch


def resolve_device(device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.lower() in ("auto", "cuda_if_available"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
