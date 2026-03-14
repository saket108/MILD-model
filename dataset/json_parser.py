from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _parse_rich(images: List[Dict[str, Any]], split: str | None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i, image_entry in enumerate(images):
        split_value = image_entry.get("split")
        if split and split_value and split_value != split:
            continue

        file_name = (
            image_entry.get("file_name")
            or image_entry.get("image")
            or image_entry.get("filename")
        )
        if file_name is None:
            raise ValueError(f"Missing file_name for image index {i}.")

        annotations = image_entry.get("annotations", []) or []
        parsed_annotations = []
        boxes_norm = []
        labels = []

        for ann in annotations:
            label = ann.get("category_name") or ann.get("label")
            if label is None and ann.get("category_id") is not None:
                label = str(ann.get("category_id"))
            if label is None:
                label = "object"

            bbox_norm = (
                ann.get("bounding_box_normalized")
                or ann.get("bbox_normalized")
                or ann.get("bbox")
            )
            if isinstance(bbox_norm, dict):
                x_center = bbox_norm.get("x_center")
                y_center = bbox_norm.get("y_center")
                width = bbox_norm.get("width")
                height = bbox_norm.get("height")
                if None in (x_center, y_center, width, height):
                    continue
                boxes_norm.append([x_center, y_center, width, height])
            elif isinstance(bbox_norm, (list, tuple)) and len(bbox_norm) == 4:
                boxes_norm.append([float(v) for v in bbox_norm])
            else:
                continue

            parsed_annotations.append(
                {
                    "label": label,
                    "zone": ann.get("zone_estimation"),
                    "severity": (ann.get("risk_assessment") or {}).get("severity_level"),
                    "class_definition": ann.get("class_definition"),
                    "description": ann.get("description"),
                    "metrics": ann.get("damage_metrics", {}),
                    "category_id": ann.get("category_id"),
                }
            )
            labels.append(label)

        items.append(
            {
                "image": file_name,
                "image_id": image_entry.get("image_id", i),
                "boxes_norm": boxes_norm,
                "labels": labels,
                "annotations": parsed_annotations,
                "split": split_value,
            }
        )

    return items


def _parse_simple(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for i, item in enumerate(items):
        image = item.get("image") or item.get("file_name") or item.get("filename")
        if image is None:
            raise ValueError(f"Missing image field at index {i}.")
        boxes = item.get("boxes", [])
        labels = item.get("labels", [])
        normalized.append(
            {
                "image": image,
                "boxes": boxes,
                "labels": labels,
                "prompts": item.get("prompts"),
                "boxes_format": item.get("boxes_format"),
                "id": item.get("id", i),
            }
        )
    return normalized


def load_dataset(json_path: str | Path, split: str | None = None) -> List[Dict[str, Any]]:
    """Load a dataset JSON file.

    Supported formats:
    - Rich: {"images": [...]} with per-image annotations and metadata.
    - Simple: list of items with keys `image`, `boxes`, `labels`.
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "images" in data:
        return _parse_rich(data["images"], split)
    if isinstance(data, dict) and "items" in data:
        return _parse_simple(data["items"])
    if isinstance(data, list):
        return _parse_simple(data)

    raise ValueError("Unsupported JSON structure.")
