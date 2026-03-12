from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xa = np.maximum(x11, x21.T)
    ya = np.maximum(y11, y21.T)
    xb = np.minimum(x12, x22.T)
    yb = np.minimum(y12, y22.T)

    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2.T - inter
    return inter / np.maximum(union, 1e-6)


@dataclass
class _Prediction:
    image_id: str | int
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    severity: np.ndarray | None = None


@dataclass
class _Target:
    image_id: str | int
    boxes: np.ndarray
    labels: np.ndarray
    severity: np.ndarray | None = None


class DetectionMetrics:
    """Compute precision, recall, F1, and mAP for detections."""

    def __init__(self, iou_thresholds: List[float] | None = None) -> None:
        self.iou_thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]
        self.preds: List[_Prediction] = []
        self.targets: List[_Target] = []
        self.has_severity = False

    def update(
        self,
        image_id: int,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        pred_severity: np.ndarray | None,
        target_boxes: np.ndarray,
        target_labels: np.ndarray,
        target_severity: np.ndarray | None,
    ) -> None:
        if pred_severity is not None and target_severity is not None:
            self.has_severity = True
        self.preds.append(_Prediction(image_id, pred_boxes, pred_scores, pred_labels, pred_severity))
        self.targets.append(_Target(image_id, target_boxes, target_labels, target_severity))

    def _evaluate_threshold(self, iou_thresh: float) -> Tuple[float, float, float]:
        tps, fps, fns = 0, 0, 0
        for pred, target in zip(self.preds, self.targets):
            used = set()
            for pb, pl in zip(pred.boxes, pred.labels):
                gt_mask = target.labels == pl
                gt_boxes = target.boxes[gt_mask]
                if gt_boxes.size == 0:
                    fps += 1
                    continue
                ious = _box_iou(pb[None, :], gt_boxes)[0]
                max_iou = float(ious.max()) if ious.size else 0.0
                max_idx = int(ious.argmax()) if ious.size else -1
                if max_iou >= iou_thresh and max_idx not in used:
                    tps += 1
                    used.add(max_idx)
                else:
                    fps += 1
            fns += max(0, len(target.boxes) - len(used))

        precision = tps / max(tps + fps, 1)
        recall = tps / max(tps + fns, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        return precision, recall, f1

    def _average_precision(self, iou_thresh: float) -> float:
        ap_list = []
        classes = set()
        for tgt in self.targets:
            classes.update(tgt.labels.tolist())

        for cls in classes:
            cls_preds = []
            cls_gts = {}
            for pred in self.preds:
                mask = pred.labels == cls
                for box, score in zip(pred.boxes[mask], pred.scores[mask]):
                    cls_preds.append((pred.image_id, score, box))
            for tgt in self.targets:
                mask = tgt.labels == cls
                cls_gts[tgt.image_id] = tgt.boxes[mask]

            if not cls_preds:
                continue

            cls_preds.sort(key=lambda x: x[1], reverse=True)
            tp = np.zeros(len(cls_preds), dtype=np.float32)
            fp = np.zeros(len(cls_preds), dtype=np.float32)
            matched = {img_id: set() for img_id in cls_gts.keys()}

            for i, (img_id, score, box) in enumerate(cls_preds):
                gt_boxes = cls_gts.get(img_id, np.zeros((0, 4), dtype=np.float32))
                if gt_boxes.size == 0:
                    fp[i] = 1
                    continue
                ious = _box_iou(box[None, :], gt_boxes)[0]
                max_iou = float(ious.max()) if ious.size else 0.0
                max_idx = int(ious.argmax()) if ious.size else -1
                if max_iou >= iou_thresh and max_idx not in matched[img_id]:
                    tp[i] = 1
                    matched[img_id].add(max_idx)
                else:
                    fp[i] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / max(sum(len(v) for v in cls_gts.values()), 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

            recall_points = np.linspace(0, 1, 101)
            precisions = []
            for r in recall_points:
                p = prec[rec >= r].max() if np.any(rec >= r) else 0
                precisions.append(p)
            ap_list.append(float(np.mean(precisions)))

        if not ap_list:
            return 0.0
        return float(np.mean(ap_list))

    def _severity_errors(self, iou_thresh: float = 0.5) -> Tuple[float, float]:
        errors: List[float] = []
        for pred, target in zip(self.preds, self.targets):
            if pred.severity is None or target.severity is None:
                continue
            used = set()
            for pb, pl, ps in zip(pred.boxes, pred.labels, pred.severity):
                gt_mask = target.labels == pl
                gt_boxes = target.boxes[gt_mask]
                gt_sev = target.severity[gt_mask] if target.severity is not None else None
                if gt_sev is None or gt_boxes.size == 0:
                    continue
                ious = _box_iou(pb[None, :], gt_boxes)[0]
                max_iou = float(ious.max()) if ious.size else 0.0
                max_idx = int(ious.argmax()) if ious.size else -1
                if max_iou >= iou_thresh and max_idx not in used:
                    errors.append(float(ps - gt_sev[max_idx]))
                    used.add(max_idx)
        if not errors:
            return float("nan"), float("nan")
        err = np.asarray(errors, dtype=np.float32)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        return mae, rmse

    def compute(self) -> Dict[str, float]:
        precision, recall, f1 = self._evaluate_threshold(0.5)
        map50 = self._average_precision(0.5)
        map5095 = float(np.mean([self._average_precision(t) for t in self.iou_thresholds]))
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": map50,
            "map50_95": map5095,
        }
        if self.has_severity:
            mae, rmse = self._severity_errors(0.5)
            results["severity_mae"] = mae
            results["severity_rmse"] = rmse
        return results
