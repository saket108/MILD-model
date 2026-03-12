from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import yaml


@dataclass
class SeverityThresholds:
    p50: float
    p85: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SeverityThresholds":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(p50=float(data.get("p50", 0.15)), p85=float(data.get("p85", 0.90)))

    def classify(self, score: float) -> str:
        if score < self.p50:
            return "low"
        if score < self.p85:
            return "moderate"
        return "high"

    def classify_many(self, scores: Iterable[float]) -> List[str]:
        return [self.classify(score) for score in scores]
