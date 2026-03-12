from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _lower_keys(mapping: Dict[Any, Any]) -> Dict[Any, Any]:
    lowered = {}
    for k, v in mapping.items():
        if isinstance(k, str):
            lowered[k.lower()] = v
        else:
            lowered[k] = v
    return lowered


def load_adas_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {
            "weights": {"area": 0.6, "elongation": 0.25, "edge": 0.15},
            "class_ranks": {},
            "zone_weights": {},
            "defaults": {"rank": 1.0, "zone_weight": 1.0},
        }

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    weights = data.get("weights", {})
    class_ranks = _lower_keys(data.get("class_ranks", {}))
    zone_weights = _lower_keys(data.get("zone_weights", {}))
    defaults = data.get("defaults", {})
    return {
        "weights": {
            "area": float(weights.get("area", 0.6)),
            "elongation": float(weights.get("elongation", 0.25)),
            "edge": float(weights.get("edge", 0.15)),
        },
        "class_ranks": class_ranks,
        "zone_weights": zone_weights,
        "defaults": {
            "rank": float(defaults.get("rank", 1.0)),
            "zone_weight": float(defaults.get("zone_weight", 1.0)),
        },
    }
