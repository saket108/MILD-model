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
    """Compute overall and per-class detection metrics."""

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

    def _class_ids(self, class_name_map: Dict[int, str] | None = None) -> List[int]:
        classes = set()
        for pred in self.preds:
            classes.update(int(label) for label in pred.labels.tolist())
        for target in self.targets:
            classes.update(int(label) for label in target.labels.tolist())
        if class_name_map:
            classes.update(int(label) for label in class_name_map.keys())
        return sorted(classes)

    def _class_label(self, class_id: int, class_name_map: Dict[int, str] | None) -> str:
        if class_name_map and class_id in class_name_map:
            return class_name_map[class_id]
        return f"class_{class_id}"

    def _match_counts_for_class(self, class_id: int, iou_thresh: float) -> Dict[str, int]:
        tp = 0
        fp = 0
        fn = 0
        images = 0
        instances = 0

        for pred, target in zip(self.preds, self.targets):
            pred_mask = pred.labels == class_id
            target_mask = target.labels == class_id
            pred_boxes = pred.boxes[pred_mask]
            pred_scores = pred.scores[pred_mask]
            gt_boxes = target.boxes[target_mask]

            if gt_boxes.size > 0:
                images += 1
                instances += len(gt_boxes)

            if pred_boxes.size == 0 and gt_boxes.size == 0:
                continue

            used = set()
            order = np.argsort(pred_scores)[::-1] if pred_scores.size else np.empty((0,), dtype=np.int64)
            for pred_idx in order.tolist():
                box = pred_boxes[pred_idx]
                if gt_boxes.size == 0:
                    fp += 1
                    continue
                ious = _box_iou(box[None, :], gt_boxes)[0]
                max_iou = float(ious.max()) if ious.size else 0.0
                max_idx = int(ious.argmax()) if ious.size else -1
                if max_iou >= iou_thresh and max_idx not in used:
                    tp += 1
                    used.add(max_idx)
                else:
                    fp += 1
            fn += max(0, len(gt_boxes) - len(used))

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "images": images,
            "instances": instances,
        }

    def _average_precision_for_class(self, class_id: int, iou_thresh: float) -> float:
        cls_preds = []
        cls_gts: Dict[str | int, np.ndarray] = {}

        for pred in self.preds:
            mask = pred.labels == class_id
            for box, score in zip(pred.boxes[mask], pred.scores[mask]):
                cls_preds.append((pred.image_id, float(score), box))

        for target in self.targets:
            mask = target.labels == class_id
            cls_gts[target.image_id] = target.boxes[mask]

        gt_count = sum(len(boxes) for boxes in cls_gts.values())
        if gt_count == 0:
            return 0.0
        if not cls_preds:
            return 0.0

        cls_preds.sort(key=lambda item: item[1], reverse=True)
        tp = np.zeros(len(cls_preds), dtype=np.float32)
        fp = np.zeros(len(cls_preds), dtype=np.float32)
        matched = {img_id: set() for img_id in cls_gts.keys()}

        for idx, (img_id, _, box) in enumerate(cls_preds):
            gt_boxes = cls_gts.get(img_id, np.zeros((0, 4), dtype=np.float32))
            if gt_boxes.size == 0:
                fp[idx] = 1
                continue
            ious = _box_iou(box[None, :], gt_boxes)[0]
            max_iou = float(ious.max()) if ious.size else 0.0
            max_idx = int(ious.argmax()) if ious.size else -1
            if max_iou >= iou_thresh and max_idx not in matched[img_id]:
                tp[idx] = 1
                matched[img_id].add(max_idx)
            else:
                fp[idx] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(gt_count, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

        recall_points = np.linspace(0, 1, 101)
        precisions = []
        for point in recall_points:
            best_precision = precision[recall >= point].max() if np.any(recall >= point) else 0.0
            precisions.append(best_precision)
        return float(np.mean(precisions))

    def _severity_errors(self, iou_thresh: float = 0.5) -> Tuple[float, float]:
        errors: List[float] = []
        for pred, target in zip(self.preds, self.targets):
            if pred.severity is None or target.severity is None:
                continue
            used = set()
            for pb, pl, ps in zip(pred.boxes, pred.labels, pred.severity):
                gt_mask = target.labels == pl
                gt_boxes = target.boxes[gt_mask]
                gt_severity = target.severity[gt_mask] if target.severity is not None else None
                if gt_severity is None or gt_boxes.size == 0:
                    continue
                ious = _box_iou(pb[None, :], gt_boxes)[0]
                max_iou = float(ious.max()) if ious.size else 0.0
                max_idx = int(ious.argmax()) if ious.size else -1
                if max_iou >= iou_thresh and max_idx not in used:
                    errors.append(float(ps - gt_severity[max_idx]))
                    used.add(max_idx)
        if not errors:
            return float("nan"), float("nan")
        err = np.asarray(errors, dtype=np.float32)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        return mae, rmse

    def compute(self, class_name_map: Dict[int, str] | None = None) -> Dict[str, object]:
        class_ids = self._class_ids(class_name_map)
        per_class = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        map50_values = []
        map50_95_values = []

        for class_id in class_ids:
            counts = self._match_counts_for_class(class_id, 0.5)
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            ap50 = self._average_precision_for_class(class_id, 0.5)
            ap50_95 = float(np.mean([self._average_precision_for_class(class_id, t) for t in self.iou_thresholds]))

            row = {
                "class_id": class_id,
                "class_name": self._class_label(class_id, class_name_map),
                "images": counts["images"],
                "instances": counts["instances"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": ap50,
                "map50_95": ap50_95,
            }
            per_class.append(row)

            if counts["instances"] > 0:
                total_tp += tp
                total_fp += fp
                total_fn += fn
                map50_values.append(ap50)
                map50_95_values.append(ap50_95)

        summary = {
            "images": len(self.targets),
            "instances": int(sum(len(target.boxes) for target in self.targets)),
            "precision": total_tp / max(total_tp + total_fp, 1),
            "recall": total_tp / max(total_tp + total_fn, 1),
            "f1": (2 * total_tp) / max(2 * total_tp + total_fp + total_fn, 1e-6),
            "map50": float(np.mean(map50_values)) if map50_values else 0.0,
            "map50_95": float(np.mean(map50_95_values)) if map50_95_values else 0.0,
        }
        if self.has_severity:
            mae, rmse = self._severity_errors(0.5)
            summary["severity_mae"] = mae
            summary["severity_rmse"] = rmse

        return {
            "summary": summary,
            "per_class": per_class,
        }
