from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset.json_parser import load_dataset
from dataset.prompt_generator import build_prompts


def _resolve_image_path(image_root: Path, image_name: str, split: str | None) -> Path:
    direct = image_root / image_name
    if direct.exists():
        return direct

    if split:
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


def _cxcywh_norm_to_xyxy_pixels(
    boxes: np.ndarray, img_w: int, img_h: int
) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x_c = boxes[:, 0] * img_w
    y_c = boxes[:, 1] * img_h
    w = boxes[:, 2] * img_w
    h = boxes[:, 3] * img_h
    return np.stack([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2], axis=-1)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x, y, x + w, y + h], axis=-1)


def _build_prompts(item: dict, include_description: bool, include_definition: bool) -> List[str]:
    prompts: List[str] = []
    for ann in item.get("annotations", []):
        prompts.extend(
            build_prompts(
                ann,
                include_description=include_description,
                include_definition=include_definition,
            )
        )
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to dataset JSON.")
    parser.add_argument("--image-root", required=True, help="Dataset root folder.")
    parser.add_argument("--index", type=int, default=0, help="Sample index.")
    parser.add_argument("--split", default=None, help="Optional split filter (train/valid/test).")
    parser.add_argument("--save", default=None, help="Optional output image path.")
    parser.add_argument("--no-desc", action="store_true", help="Exclude description prompts.")
    parser.add_argument("--no-def", action="store_true", help="Exclude definition prompts.")
    args = parser.parse_args()

    items = load_dataset(args.json, split=args.split)
    if not items:
        raise SystemExit("No items found. Check JSON path or split.")
    if args.index < 0 or args.index >= len(items):
        raise SystemExit(f"Index out of range. Dataset size={len(items)}")

    item = items[args.index]
    image_root = Path(args.image_root)
    image_path = _resolve_image_path(image_root, item["image"], item.get("split"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    if "boxes_norm" in item:
        boxes = np.asarray(item.get("boxes_norm", []), dtype=np.float32)
        boxes_xyxy = _cxcywh_norm_to_xyxy_pixels(boxes, w, h)
    else:
        boxes = np.asarray(item.get("boxes", []), dtype=np.float32)
        if item.get("boxes_format") == "xyxy":
            boxes_xyxy = boxes.reshape(-1, 4)
        else:
            boxes_xyxy = _xywh_to_xyxy(boxes)

    labels = item.get("labels", [])
    prompts = _build_prompts(
        item, include_description=not args.no_desc, include_definition=not args.no_def
    )

    print("\nSAMPLE PREVIEW\n")
    print(f"Image: {image_path}")
    print(f"Boxes: {len(boxes_xyxy)}")
    if labels:
        print("Labels:", ", ".join(labels))
    if prompts:
        print("\nPrompts:")
        for p in prompts[:10]:
            print(f"- {p}")
        if len(prompts) > 10:
            print(f"... ({len(prompts)-10} more)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.tolist()
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            color="lime",
            linewidth=2,
        )
        ax.add_patch(rect)
        if i < len(labels):
            ax.text(
                x1,
                y1 - 4,
                labels[i],
                color="lime",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )
    ax.axis("off")
    ax.set_title("Preview Sample")

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"\nSaved preview to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
