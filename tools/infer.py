from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch
import yaml

from models.mild_model import build_model
from utils.checkpoint import load_checkpoint
from utils.detection_postprocess import postprocess_detections
from utils.device import resolve_device
from utils.visualizer import draw_boxes


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--metrics", nargs=4, type=float)
    args = parser.parse_args()

    cfg_model = load_yaml(args.model_config)
    device = resolve_device(args.device)
    model = build_model(cfg_model).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    original_h, original_w = image.shape[:2]
    image_resized = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    metrics_tensor = None
    if args.metrics is not None:
        metrics_tensor = torch.tensor([args.metrics], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(image_tensor, [args.prompt], metrics_tensor)

    result = postprocess_detections(
        pred_logits=outputs["pred_logits"],
        pred_boxes=outputs["pred_boxes"],
        pred_severity=outputs.get("pred_severity"),
        score_threshold=args.score_threshold,
        nms_iou=args.nms_iou,
        top_k=args.top_k,
        max_detections=args.max_detections,
    )[0]

    boxes = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    labels = result["labels"].cpu().numpy()
    severities = result["severity"].cpu().numpy() if result["severity"] is not None else None

    if boxes.size > 0:
        scale = torch.tensor([original_w, original_h, original_w, original_h], dtype=torch.float32)
        boxes = (torch.from_numpy(boxes) * scale).numpy()
    label_text = []
    for i, lab in enumerate(labels):
        if severities is not None:
            label_text.append(f"{int(lab)}|sev:{severities[i]:.2f}")
        else:
            label_text.append(str(int(lab)))

    vis = draw_boxes(image, boxes, labels=label_text, scores=scores)
    out_path = Path("runs") / "inference.png"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
