from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def draw_boxes(
    image: np.ndarray,
    boxes: Iterable[Iterable[float]],
    labels: Iterable[str] | None = None,
    scores: Iterable[float] | None = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    output = image.copy()
    labels_list = list(labels) if labels is not None else None
    scores_list = list(scores) if scores is not None else None

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = None
        if labels_list is not None:
            text = labels_list[i]
        if scores_list is not None:
            score_text = f"{scores_list[i]:.2f}"
            text = f"{text} {score_text}" if text else score_text
        if text:
            cv2.putText(output, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output
