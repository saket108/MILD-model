from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import WeightedRandomSampler


def _align_counts(counts: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = counts.to(dtype=torch.float32)
    if counts.numel() == num_classes:
        return counts
    if counts.numel() > num_classes:
        return counts[:num_classes]
    aligned = torch.zeros(num_classes, dtype=torch.float32)
    aligned[: counts.numel()] = counts
    return aligned


def _inverse_power_weights(
    counts: torch.Tensor,
    power: float = 0.5,
    max_weight: float | None = None,
) -> torch.Tensor:
    counts = counts.to(dtype=torch.float32)
    weights = torch.ones_like(counts, dtype=torch.float32)
    positive = counts > 0
    if positive.any():
        reference = counts[positive].max()
        weights[positive] = torch.pow(reference / counts[positive], float(power))
    weights = torch.clamp(weights, min=1.0)
    if max_weight is not None and max_weight > 0:
        weights = torch.clamp(weights, max=float(max_weight))
    return weights


def build_class_weights(
    dataset,
    num_classes: int,
    power: float = 0.5,
    max_weight: float | None = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = _align_counts(dataset.get_class_instance_counts(), num_classes)
    weights = _inverse_power_weights(counts, power=power, max_weight=max_weight)
    return weights, counts


def build_sample_weights(
    dataset,
    num_classes: int,
    power: float = 0.5,
    mode: str = "max",
    max_weight: float | None = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_counts = _align_counts(dataset.get_class_image_counts(), num_classes)
    class_weights = _inverse_power_weights(image_counts, power=power, max_weight=max_weight)
    sample_weights = []

    for label_ids in dataset.get_label_id_lists():
        unique_ids = sorted({label_id for label_id in label_ids if 0 <= label_id < num_classes})
        if not unique_ids:
            sample_weights.append(1.0)
            continue
        label_weights = class_weights[unique_ids]
        if mode == "mean":
            sample_weight = float(label_weights.mean())
        else:
            sample_weight = float(label_weights.max())
        sample_weights.append(sample_weight)

    weights = torch.as_tensor(sample_weights, dtype=torch.double)
    if weights.numel() == 0:
        return weights, image_counts, class_weights
    mean_weight = weights.mean().item()
    if mean_weight > 0:
        weights = weights / mean_weight
    return weights, image_counts, class_weights


def build_weighted_sampler(sample_weights: torch.Tensor) -> WeightedRandomSampler:
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def format_named_values(names: dict[int, str], values: Iterable[float]) -> str:
    parts = []
    for idx, value in enumerate(values):
        name = names.get(idx, str(idx))
        if isinstance(value, torch.Tensor):
            value = float(value.item())
        if float(value).is_integer():
            parts.append(f"{name}={int(value)}")
        else:
            parts.append(f"{name}={float(value):.2f}")
    return ", ".join(parts)
