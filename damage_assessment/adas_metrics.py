from __future__ import annotations

from typing import Dict, Tuple


DEFAULT_WEIGHTS = {"area": 0.6, "elongation": 0.25, "edge": 0.15}


def compute_metrics(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float]:
    area = w * h
    min_side = max(min(w, h), 1e-8)
    elongation = max(w, h) / min_side
    edge = 1.0 - min(xc, yc, 1 - xc, 1 - yc)
    return area, elongation, edge


def compute_structural_weight(
    category: str | int | None,
    class_ranks: Dict | None = None,
    default_rank: float = 1.0,
) -> float:
    if category is None:
        return 1.0
    class_ranks = class_ranks or {}
    max_rank = max([float(v) for v in class_ranks.values()], default=float(default_rank))

    key = category
    if isinstance(category, str):
        key = category.lower()

    rank = class_ranks.get(key, default_rank)
    if max_rank <= 0:
        return 1.0
    return float(rank) / float(max_rank)


def compute_zone_weight(zone: str | None, zone_weights: Dict | None = None, default: float = 1.0) -> float:
    if zone is None:
        return float(default)
    zone_weights = zone_weights or {}
    key = zone.lower() if isinstance(zone, str) else zone
    return float(zone_weights.get(key, default))


def compute_score(
    area: float,
    elongation: float,
    edge: float,
    weights: Dict[str, float] | None = None,
    category: str | int | None = None,
    zone: str | None = None,
    class_ranks: Dict | None = None,
    zone_weights: Dict | None = None,
    default_rank: float = 1.0,
    default_zone_weight: float = 1.0,
) -> float:
    weights = weights or DEFAULT_WEIGHTS
    base = weights["area"] * area + weights["elongation"] * elongation + weights["edge"] * edge
    structural = compute_structural_weight(category, class_ranks, default_rank)
    zone_weight = compute_zone_weight(zone, zone_weights, default_zone_weight)
    return base * structural * zone_weight
