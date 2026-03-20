from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from damage_assessment.severity_score import severity_from_xyxy
from dataset.json_parser import load_dataset
from dataset.prompt_generator import DEFAULT_TEMPLATES, build_prompts, generate_prompt
from dataset.transforms import get_transforms


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x, y, x + w, y + h], axis=-1)


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


def _dedupe_prompts(prompts: List[str]) -> List[str]:
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


def _aggregate_metrics(annotations: List[Dict]) -> np.ndarray:
    metrics_list = []
    for ann in annotations:
        metrics = ann.get("metrics", {}) or {}
        metrics_list.append(
            [
                float(metrics.get("area_ratio", 0.0) or 0.0),
                float(metrics.get("elongation", 0.0) or 0.0),
                float(metrics.get("edge_factor", 0.0) or 0.0),
                float(metrics.get("raw_severity_score", 0.0) or 0.0),
            ]
        )
    if not metrics_list:
        return np.zeros((4,), dtype=np.float32)
    return np.mean(np.asarray(metrics_list, dtype=np.float32), axis=0)


def _resolve_image_path(image_root: Path, image_name: str, split: str | None) -> Path:
    direct = image_root / image_name
    if direct.exists():
        return direct

    if split:
        candidate = image_root / "images" / split / image_name
        if candidate.exists():
            return candidate
        candidate = image_root / split / "images" / image_name
        if candidate.exists():
            return candidate
        candidate = image_root / split / image_name
        if candidate.exists():
            return candidate

    candidate = image_root / "images" / image_name
    if candidate.exists():
        return candidate

    return direct




class MILDDetectionDataset(Dataset):
    """Dataset that returns image tensor, prompt, boxes, and labels."""

    def __init__(
        self,
        json_path: str | Path,
        image_root: str | Path,
        image_size: int = 640,
        train: bool = True,
        prompt_templates: List[str] | None = None,
        split: str | None = None,
        max_prompts: int = 8,
        prompt_strategy: str = "random",
        include_description: bool = True,
        include_definition: bool = True,
    ) -> None:
        self.items = load_dataset(json_path, split=split)
        self.image_root = Path(image_root)
        self.transforms = get_transforms(image_size=image_size, train=train)
        self.prompt_templates = prompt_templates or DEFAULT_TEMPLATES
        self.max_prompts = max_prompts
        self.prompt_strategy = prompt_strategy
        self.include_description = include_description
        self.include_definition = include_definition
        self.label_to_id = self._build_label_map(self.items)
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    @staticmethod
    def _build_label_map(items: List[Dict]) -> Dict[str, int]:
        labels = set()
        for item in items:
            for label in item.get("labels", []):
                labels.add(label)
        return {label: idx for idx, label in enumerate(sorted(labels))}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        image_path = _resolve_image_path(self.image_root, item["image"], item.get("split"))
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        boxes: np.ndarray
        if "boxes_norm" in item:
            boxes_norm = np.array(item.get("boxes_norm", []), dtype=np.float32)
            boxes = _cxcywh_norm_to_xyxy_pixels(boxes_norm, img_w, img_h)
        else:
            boxes = np.array(item.get("boxes", []), dtype=np.float32)
            if item.get("boxes_format") == "xyxy":
                boxes = boxes.reshape(-1, 4)
            else:
                boxes = _xywh_to_xyxy(boxes)

        labels = item.get("labels", [])
        boxes, keep_mask = _sanitize_xyxy_boxes(boxes, img_w, img_h)
        if len(labels) == len(keep_mask):
            labels = [label for label, keep in zip(labels, keep_mask.tolist()) if keep]
        label_ids = [self.label_to_id[label] for label in labels]
        priority_prompts = [f"{label} on aircraft surface" for label in dict.fromkeys(labels)]

        prompts: List[str] = []
        if item.get("annotations"):
            for ann in item["annotations"]:
                prompts.extend(
                    build_prompts(
                        ann,
                        include_description=self.include_description,
                        include_definition=self.include_definition,
                    )
                )
        if not prompts and item.get("prompts"):
            raw_prompts = item.get("prompts")
            if isinstance(raw_prompts, str):
                prompts = [raw_prompts]
            else:
                prompts = list(raw_prompts)
        if not prompts:
            prompt_label = labels[0] if labels else "object"
            prompts = [generate_prompt(prompt_label, self.prompt_templates)]

        prompts = _select_prompts(
            priority_prompts + prompts,
            self.max_prompts,
            self.prompt_strategy,
            priority_prompts=priority_prompts,
        )

        metrics_vec = _aggregate_metrics(item.get("annotations", []))

        transformed = self.transforms(image=image, bboxes=boxes.tolist(), labels=label_ids)
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
            "prompt": prompts,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "severity": severity_tensor,
            "metrics": metrics_tensor,
            "image_id": item.get("image_id", item.get("id", idx)),
        }
