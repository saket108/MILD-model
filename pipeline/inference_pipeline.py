from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch
import json
import yaml

from damage_assessment.adas_config import load_adas_config
from damage_assessment.adas_metrics import compute_metrics as adas_metrics
from damage_assessment.adas_metrics import compute_score as adas_score
from damage_assessment.severity_classifier import SeverityThresholds
from losses.box_losses import box_cxcywh_to_xyxy
from models.mild_model import build_model
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_names(path: str | Path | None) -> Dict[int, str]:
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


def run_inference(
    image_path: str | Path,
    prompt: str,
    model_config: str | Path,
    checkpoint: str | Path,
    thresholds_path: str | Path = "configs/severity_thresholds.yaml",
    classes_path: str | Path | None = None,
    adas_config_path: str | Path | None = "configs/adas.yaml",
    metrics: List[float] | None = None,
    image_size: int = 640,
    score_threshold: float = 0.3,
    device: str | None = None,
) -> List[dict]:
    device = resolve_device(device)
    cfg_model = load_yaml(model_config)
    model = build_model(cfg_model).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    original_h, original_w = image.shape[:2]
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)

    metrics_tensor = None
    if metrics is not None:
        metrics_tensor = torch.tensor([metrics], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(image_tensor, [prompt], metrics_tensor)

    scores, labels = outputs["pred_logits"].sigmoid().max(dim=-1)
    pred_boxes_cxcywh = outputs["pred_boxes"].detach().cpu().numpy()[0]
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"]).cpu().numpy()[0]
    scores = scores.cpu().numpy()[0]
    labels = labels.cpu().numpy()[0]
    severities = outputs.get("pred_severity")
    severities = severities.cpu().numpy()[0] if severities is not None else None

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    pred_boxes_cxcywh = pred_boxes_cxcywh[keep]
    if severities is not None:
        severities = severities[keep]

    if boxes.size > 0:
        scale = torch.tensor([original_w, original_h, original_w, original_h], dtype=torch.float32)
        boxes = (torch.from_numpy(boxes) * scale).numpy()

    thresholds = SeverityThresholds.from_yaml(thresholds_path)
    adas_cfg = load_adas_config(adas_config_path)
    class_names = load_class_names(classes_path)
    results = []
    for i in range(len(labels)):
        sev = float(severities[i]) if severities is not None else 0.0
        cx, cy, w, h = pred_boxes_cxcywh[i].tolist()
        area_ratio, elongation, edge_factor = adas_metrics(cx, cy, w, h)
        category = class_names.get(int(labels[i]), int(labels[i]))
        adas = adas_score(
            area_ratio,
            elongation,
            edge_factor,
            weights=adas_cfg["weights"],
            category=category,
            zone=None,
            class_ranks=adas_cfg["class_ranks"],
            zone_weights=adas_cfg["zone_weights"],
            default_rank=adas_cfg["defaults"]["rank"],
            default_zone_weight=adas_cfg["defaults"]["zone_weight"],
        )
        adas_level = thresholds.classify(adas)
        neural_level = thresholds.classify(sev)
        agreement = adas_level == neural_level

        results.append(
            {
                "id": i,
                "label_id": int(labels[i]),
                "label": class_names.get(int(labels[i])),
                "score": float(scores[i]),
                "box": {
                    "format": "xyxy",
                    "x1": float(boxes[i][0]),
                    "y1": float(boxes[i][1]),
                    "x2": float(boxes[i][2]),
                    "y2": float(boxes[i][3]),
                    "normalized": False,
                },
                "box_norm": {
                    "format": "cxcywh",
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                },
                "adas": {
                    "score": float(adas),
                    "level": adas_level,
                    "metrics": {
                        "area_ratio": float(area_ratio),
                        "elongation": float(elongation),
                        "edge_factor": float(edge_factor),
                    },
                },
                "neural": {
                    "score": float(sev),
                    "level": neural_level,
                },
                "hybrid": {
                    "agreement": agreement,
                    "final_level": adas_level if agreement else adas_level,
                    "review_required": not agreement,
                },
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", default="configs/severity_thresholds.yaml")
    parser.add_argument("--classes")
    parser.add_argument("--adas-config", default="configs/adas.yaml")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--metrics", nargs=4, type=float)
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    results = run_inference(
        image_path=args.image,
        prompt=args.prompt,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        thresholds_path=args.thresholds,
        classes_path=args.classes,
        adas_config_path=args.adas_config,
        metrics=args.metrics,
        image_size=args.image_size,
        score_threshold=args.score_threshold,
        device=args.device,
    )

    report = {
        "image": {
            "path": str(args.image),
        },
        "prompt": args.prompt,
        "model": {
            "config": str(args.model_config),
            "checkpoint": str(args.checkpoint),
            "thresholds": str(args.thresholds),
        },
        "summary": {
            "num_predictions": len(results),
            "review_required": sum(1 for r in results if r["hybrid"]["review_required"]),
        },
        "predictions": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
