from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from damage_assessment.severity_score import severity_from_xyxy
from dataset.prompt_generator import DEFAULT_TEMPLATES, generate_prompt
from dataset.transforms import get_transforms


def _cxcywh_norm_to_xyxy_pixels(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x_c = boxes[:, 0] * img_w
    y_c = boxes[:, 1] * img_h
    w = boxes[:, 2] * img_w
    h = boxes[:, 3] * img_h
    return np.stack([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2], axis=-1)


def _sanitize_xyxy_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return boxes.reshape(0, 4), np.zeros((0,), dtype=bool)

    clipped = boxes.reshape(-1, 4).astype(np.float32, copy=True)
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0.0, float(img_w))
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0.0, float(img_h))
    keep = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    return clipped[keep], keep


def _dedupe_prompts(prompts: Iterable[str]) -> List[str]:
    seen = set()
    unique = []
    for prompt in prompts:
        clean = str(prompt).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        unique.append(clean)
    return unique


def _select_prompts(
    prompts: List[str],
    max_prompts: int,
    prompt_strategy: str,
    priority_prompts: List[str] | None = None,
) -> List[str]:
    prompts = _dedupe_prompts(prompts)
    if not max_prompts or len(prompts) <= max_prompts:
        return prompts

    if prompt_strategy == "random":
        return random.sample(prompts, max_prompts)

    if prompt_strategy == "coverage":
        selected = []
        for prompt in _dedupe_prompts(priority_prompts or []):
            if prompt in prompts and prompt not in selected:
                selected.append(prompt)
            if len(selected) >= max_prompts:
                return selected
        for prompt in prompts:
            if prompt not in selected:
                selected.append(prompt)
            if len(selected) >= max_prompts:
                break
        return selected

    return prompts[:max_prompts]


def _compute_metrics(xc: float, yc: float, w: float, h: float) -> List[float]:
    area = w * h
    min_side = max(min(w, h), 1e-8)
    elongation = max(w, h) / min_side
    edge = 1.0 - min(xc, yc, 1 - xc, 1 - yc)
    raw = 0.6 * area + 0.25 * elongation + 0.15 * edge
    return [area, elongation, edge, raw]


def _aggregate_metrics(boxes_norm: np.ndarray) -> np.ndarray:
    if boxes_norm.size == 0:
        return np.zeros((4,), dtype=np.float32)
    metrics = []
    for xc, yc, w, h in boxes_norm:
        metrics.append(_compute_metrics(float(xc), float(yc), float(w), float(h)))
    return np.mean(np.asarray(metrics, dtype=np.float32), axis=0)


def _load_class_names(path: str | Path | None) -> Dict[int, str]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    names = {}
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        name = line.strip()
        if name:
            names[idx] = name
    return names


class LabelFolderDataset(Dataset):
    """Dataset for image folder + label folder (YOLO-style txt labels)."""

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        image_size: int = 640,
        train: bool = True,
        class_names_path: str | Path | None = None,
        prompt_templates: List[str] | None = None,
        max_prompts: int = 8,
        prompt_strategy: str = "random",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = get_transforms(image_size=image_size, train=train)
        self.prompt_templates = prompt_templates or DEFAULT_TEMPLATES
        self.max_prompts = max_prompts
        self.prompt_strategy = prompt_strategy
        self.class_names = _load_class_names(class_names_path)
        self.id_to_label = dict(self.class_names)

        self.items = []
        for label_path in sorted(self.labels_dir.glob("*.txt")):
            stem = label_path.stem
            image_path = self._find_image_for_stem(stem)
            if image_path is None:
                continue
            self.items.append({"image": image_path, "label": label_path})

    def _find_image_for_stem(self, stem: str) -> Path | None:
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        for ext in exts:
            candidate = self.images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        image_path = item["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        label_path = item["label"]
        boxes_norm = []
        labels = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            boxes_norm.append([xc, yc, w, h])
            labels.append(class_id)

        boxes_norm_np = np.asarray(boxes_norm, dtype=np.float32)
        boxes = _cxcywh_norm_to_xyxy_pixels(boxes_norm_np, img_w, img_h)
        boxes, keep_mask = _sanitize_xyxy_boxes(boxes, img_w, img_h)
        if len(labels) == len(keep_mask):
            labels = [label for label, keep in zip(labels, keep_mask.tolist()) if keep]
            boxes_norm_np = boxes_norm_np[keep_mask]

        prompts = []
        for class_id in labels:
            label_name = self.class_names.get(class_id, f"class_{class_id}")
            prompts.append(generate_prompt(label_name, self.prompt_templates))
        priority_prompts = list(prompts)

        prompts = _select_prompts(
            prompts,
            self.max_prompts,
            self.prompt_strategy,
            priority_prompts=priority_prompts,
        )

        metrics_vec = _aggregate_metrics(boxes_norm_np)

        transformed = self.transforms(image=image, bboxes=boxes.tolist(), labels=labels)
        image_tensor = transformed["image"]
        boxes_tensor = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        if boxes_tensor.numel() == 0:
            boxes_tensor = boxes_tensor.reshape(0, 4)
        labels_tensor = torch.as_tensor(transformed["labels"], dtype=torch.long)
        metrics_tensor = torch.as_tensor(metrics_vec, dtype=torch.float32)
        img_h_t, img_w_t = image_tensor.shape[1:]
        severity_tensor = severity_from_xyxy(boxes_tensor, img_w_t, img_h_t)

        return {
            "image": image_tensor,
            "prompt": prompts if prompts else ["object"],
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "severity": severity_tensor,
            "metrics": metrics_tensor,
            "image_id": image_path.stem,
        }
