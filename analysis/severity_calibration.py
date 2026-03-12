from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml

from damage_assessment.adas_config import load_adas_config
from damage_assessment.adas_metrics import compute_metrics, compute_score


def load_class_names(path: Path | None) -> Dict[int, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")
    names = {}
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        name = line.strip()
        if name:
            names[idx] = name
    return names


def find_label_dirs(root: Path) -> List[Path]:
    candidates = []
    for split in ("train", "valid", "test"):
        path = root / split / "labels"
        if path.exists():
            candidates.append(path)
    if not candidates:
        flat = root / "labels"
        if flat.exists():
            candidates.append(flat)
    return candidates


def iter_label_files(label_dirs: Iterable[Path]) -> Iterable[Path]:
    for label_dir in label_dirs:
        for path in sorted(label_dir.glob("*.txt")):
            yield path


def parse_label_line(line: str) -> Tuple[float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        xc, yc, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    return xc, yc, w, h


def scan_yolo_scores(
    label_dirs: List[Path],
    adas_cfg: Dict,
    class_names: Dict[int, str],
) -> List[float]:
    scores: List[float] = []
    for path in iter_label_files(label_dirs):
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            class_id = int(float(line.strip().split()[0]))
            category = class_names.get(class_id, class_id)
            xc, yc, w, h = parsed
            area, elongation, edge = compute_metrics(xc, yc, w, h)
            scores.append(
                compute_score(
                    area,
                    elongation,
                    edge,
                    weights=adas_cfg["weights"],
                    category=category,
                    zone=None,
                    class_ranks=adas_cfg["class_ranks"],
                    zone_weights=adas_cfg["zone_weights"],
                    default_rank=adas_cfg["defaults"]["rank"],
                    default_zone_weight=adas_cfg["defaults"]["zone_weight"],
                )
            )
    return scores


def scan_json_scores(json_path: Path, adas_cfg: Dict) -> List[float]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "images" not in data:
        return []
    scores: List[float] = []
    for image_entry in data.get("images", []):
        for ann in image_entry.get("annotations", []) or []:
            bbox = ann.get("bounding_box_normalized") or ann.get("bbox_normalized")
            if not isinstance(bbox, dict):
                continue
            xc = bbox.get("x_center")
            yc = bbox.get("y_center")
            w = bbox.get("width")
            h = bbox.get("height")
            if None in (xc, yc, w, h):
                continue
            category = ann.get("category_name") or ann.get("category_id")
            zone = ann.get("zone_estimation")
            area, elongation, edge = compute_metrics(float(xc), float(yc), float(w), float(h))
            scores.append(
                compute_score(
                    area,
                    elongation,
                    edge,
                    weights=adas_cfg["weights"],
                    category=category,
                    zone=zone,
                    class_ranks=adas_cfg["class_ranks"],
                    zone_weights=adas_cfg["zone_weights"],
                    default_rank=adas_cfg["defaults"]["rank"],
                    default_zone_weight=adas_cfg["defaults"]["zone_weight"],
                )
            )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", default="dataset", help="Root containing train/valid/test folders.")
    parser.add_argument("--json", nargs="*", help="Optional rich JSON file(s) to include.")
    parser.add_argument("--classes", help="Optional class names file (one per line).")
    parser.add_argument("--adas-config", default="configs/adas.yaml")
    parser.add_argument("--p50", type=float, default=50)
    parser.add_argument("--p85", type=float, default=85)
    parser.add_argument("--output", default="configs/severity_thresholds.yaml")
    args = parser.parse_args()

    adas_cfg = load_adas_config(args.adas_config)
    class_names = load_class_names(Path(args.classes)) if args.classes else {}

    label_root = Path(args.label_root)
    label_dirs = find_label_dirs(label_root)
    if not label_dirs:
        print("No label folders found. Expected dataset/train/labels, dataset/valid/labels, dataset/test/labels.")
        return

    scores = scan_yolo_scores(label_dirs, adas_cfg, class_names)
    if args.json:
        for json_path_str in args.json:
            json_path = Path(json_path_str)
            if not json_path.exists():
                print(f"JSON not found: {json_path}")
                continue
            scores.extend(scan_json_scores(json_path, adas_cfg))

    if not scores:
        print("No scores found. Check label folders or JSON files.")
        return

    scores_arr = np.asarray(scores, dtype=np.float32)
    p50 = float(np.percentile(scores_arr, args.p50))
    p85 = float(np.percentile(scores_arr, args.p85))

    print("\nADAS SEVERITY CALIBRATION\n")
    print(f"Total samples: {len(scores)}")
    print(f"p50 threshold: {p50:.6f}")
    print(f"p85 threshold: {p85:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"p50": p50, "p85": p85}, f, sort_keys=False)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
