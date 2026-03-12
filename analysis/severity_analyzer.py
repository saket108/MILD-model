from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from damage_assessment.adas_config import load_adas_config
from damage_assessment.adas_metrics import compute_metrics, compute_score


def compute_metrics_wrapper(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float]:
    return compute_metrics(xc, yc, w, h)


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


def parse_label_line(line: str) -> Tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    return class_id, xc, yc, w, h


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


def scan_yolo_labels(
    label_dirs: List[Path],
    adas_cfg: Dict,
    class_names: Dict[int, str],
    p50: float,
    p85: float,
) -> Dict[str, object]:
    class_counts = Counter()
    severity_counts = Counter()
    areas: List[float] = []
    elongations: List[float] = []
    edges: List[float] = []
    scores: List[float] = []
    total_annotations = 0
    label_files = list(iter_label_files(label_dirs))

    for path in label_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            class_id, xc, yc, w, h = parsed
            area, elongation, edge = compute_metrics_wrapper(xc, yc, w, h)
            category = class_names.get(class_id, class_id)
            score = compute_score(
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
            areas.append(area)
            elongations.append(elongation)
            edges.append(edge)
            scores.append(score)

            if score < p50:
                severity_counts["low"] += 1
            elif score < p85:
                severity_counts["moderate"] += 1
            else:
                severity_counts["high"] += 1

            class_counts[class_id] += 1
            total_annotations += 1

    return {
        "label_files": label_files,
        "class_counts": class_counts,
        "severity_counts": severity_counts,
        "areas": areas,
        "elongations": elongations,
        "edges": edges,
        "scores": scores,
        "total_annotations": total_annotations,
    }


def parse_rich_json(json_path: Path, adas_cfg: Dict) -> Dict[str, object]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "images" not in data:
        return {}

    severity_counts = Counter()
    class_counts = Counter()
    metrics_acc = defaultdict(list)
    total_annotations = 0
    scores: List[float] = []

    for image_entry in data.get("images", []):
        for ann in image_entry.get("annotations", []) or []:
            label = ann.get("category_name")
            if label is None and ann.get("category_id") is not None:
                label = str(ann.get("category_id"))
            if label is None:
                label = "object"
            class_counts[label] += 1

            severity = (ann.get("risk_assessment") or {}).get("severity_level")
            if severity:
                severity_counts[str(severity).lower()] += 1

            metrics = ann.get("damage_metrics") or {}
            for key in ("area_ratio", "elongation", "edge_factor", "raw_severity_score"):
                if key in metrics and metrics[key] is not None:
                    metrics_acc[key].append(float(metrics[key]))

            bbox = ann.get("bounding_box_normalized") or ann.get("bbox_normalized")
            if isinstance(bbox, dict):
                xc = bbox.get("x_center")
                yc = bbox.get("y_center")
                w = bbox.get("width")
                h = bbox.get("height")
                if None not in (xc, yc, w, h):
                    area, elongation, edge = compute_metrics_wrapper(float(xc), float(yc), float(w), float(h))
                    scores.append(
                        compute_score(
                            area,
                            elongation,
                            edge,
                            weights=adas_cfg["weights"],
                            category=label,
                            zone=ann.get("zone_estimation"),
                            class_ranks=adas_cfg["class_ranks"],
                            zone_weights=adas_cfg["zone_weights"],
                            default_rank=adas_cfg["defaults"]["rank"],
                            default_zone_weight=adas_cfg["defaults"]["zone_weight"],
                        )
                    )

            total_annotations += 1

    return {
        "severity_counts": severity_counts,
        "class_counts": class_counts,
        "metrics_acc": metrics_acc,
        "scores": scores,
        "total_annotations": total_annotations,
    }


def balance_score(severity_counts: Counter) -> str:
    total = sum(severity_counts.values())
    if total == 0:
        return "UNKNOWN"
    target = {"low": 0.40, "moderate": 0.35, "high": 0.25}
    deviations = []
    for key, tgt in target.items():
        deviations.append(abs(severity_counts.get(key, 0) / total - tgt))
    max_dev = max(deviations)
    if max_dev <= 0.15:
        return "GOOD"
    if max_dev <= 0.30:
        return "MODERATE"
    return "BIASED"


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def format_count_distribution(counter: Counter, name_map: Dict[int, str] | None = None) -> List[str]:
    if not counter:
        return ["(none)"]
    lines = []
    for key, count in sorted(counter.items(), key=lambda x: (-x[1], str(x[0]))):
        if name_map is not None and isinstance(key, int) and key in name_map:
            label = f"{name_map[key]} ({key})"
        else:
            label = str(key)
        lines.append(f"{label}: {count}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", default="dataset", help="Root containing train/valid/test folders.")
    parser.add_argument("--json", nargs="*", help="Optional rich JSON file(s) to include.")
    parser.add_argument("--classes", help="Optional class names file (one per line).")
    parser.add_argument("--p50", type=float, default=0.15)
    parser.add_argument("--p85", type=float, default=0.90)
    parser.add_argument("--plot-dir", default="analysis/plots")
    parser.add_argument("--thresholds", help="Optional YAML file with p50/p85 thresholds.")
    parser.add_argument("--adas-config", default="configs/adas.yaml")
    args = parser.parse_args()

    label_root = Path(args.label_root)
    label_dirs = find_label_dirs(label_root)
    if not label_dirs:
        print("No label folders found. Expected dataset/train/labels, dataset/valid/labels, dataset/test/labels.")
        return

    adas_cfg = load_adas_config(args.adas_config)

    thresholds_path = Path(args.thresholds) if args.thresholds else Path("configs/severity_thresholds.yaml")
    if thresholds_path.exists():
        with thresholds_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        args.p50 = float(data.get("p50", args.p50))
        args.p85 = float(data.get("p85", args.p85))

    class_names = load_class_names(Path(args.classes)) if args.classes else {}

    yolo_report = scan_yolo_labels(label_dirs, adas_cfg, class_names, args.p50, args.p85)
    total_images = len(yolo_report["label_files"])
    total_annotations = yolo_report["total_annotations"]

    print("\nDATASET SEVERITY REPORT\n")
    print(f"Label roots: {', '.join(str(p) for p in label_dirs)}")
    print(f"Total Images (label files): {total_images}")
    print(f"Total Annotations: {total_annotations}")
    print(f"Thresholds: p50={args.p50:.4f}, p85={args.p85:.4f}")

    sev_counts = yolo_report["severity_counts"]
    total_sev = sum(sev_counts.values())
    print("\nSeverity Distribution (ADAS score)")
    for key in ("low", "moderate", "high"):
        count = sev_counts.get(key, 0)
        pct = (count / total_sev * 100.0) if total_sev else 0.0
        print(f"{key.capitalize():<9}: {count} ({pct:.1f}%)")

    print("\nDamage Class Distribution (YOLO)")
    for line in format_count_distribution(yolo_report["class_counts"], class_names):
        print(line)

    print("\nGeometry Stats (YOLO)")
    print(f"Mean Area Ratio: {mean_or_nan(yolo_report['areas']):.6f}")
    print(f"Mean Elongation: {mean_or_nan(yolo_report['elongations']):.6f}")
    print(f"Mean Edge Factor: {mean_or_nan(yolo_report['edges']):.6f}")
    print(f"Mean ADAS Score: {mean_or_nan(yolo_report['scores']):.6f}")

    print(f"\nDataset Balance Score: {balance_score(sev_counts)}")

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Severity distribution plot
    plt.figure(figsize=(6, 4))
    sev_labels = ["Low", "Moderate", "High"]
    sev_values = [sev_counts.get("low", 0), sev_counts.get("moderate", 0), sev_counts.get("high", 0)]
    plt.bar(sev_labels, sev_values, color=["#5cb85c", "#f0ad4e", "#d9534f"])
    plt.title("Severity Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plot_dir / "severity_distribution.png", dpi=150)
    plt.close()

    # Area ratio histogram
    if yolo_report["areas"]:
        plt.figure(figsize=(6, 4))
        plt.hist(yolo_report["areas"], bins=30, color="#4c72b0", edgecolor="black", linewidth=0.5)
        plt.title("Area Ratio Distribution")
        plt.xlabel("Area Ratio")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(plot_dir / "area_ratio_histogram.png", dpi=150)
        plt.close()

    # Elongation histogram
    if yolo_report["elongations"]:
        plt.figure(figsize=(6, 4))
        plt.hist(yolo_report["elongations"], bins=30, color="#55a868", edgecolor="black", linewidth=0.5)
        plt.title("Elongation Distribution")
        plt.xlabel("Elongation")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(plot_dir / "elongation_histogram.png", dpi=150)
        plt.close()

    # Class distribution plot
    class_counts = yolo_report["class_counts"]
    if class_counts:
        labels = []
        counts = []
        for key, count in sorted(class_counts.items(), key=lambda x: (-x[1], str(x[0]))):
            if isinstance(key, int) and key in class_names:
                labels.append(class_names[key])
            else:
                labels.append(str(key))
            counts.append(count)

        plt.figure(figsize=(max(6, len(labels) * 0.6), 4))
        plt.bar(labels, counts, color="#8172b2")
        plt.title("Damage Class Distribution")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / "damage_class_distribution.png", dpi=150)
        plt.close()

    print(f"\nPlots saved in {plot_dir}")

    if args.json:
        for json_path_str in args.json:
            json_path = Path(json_path_str)
            if not json_path.exists():
                print(f"\nJSON not found: {json_path}")
                continue
            parsed = parse_rich_json(json_path, adas_cfg)
            if not parsed:
                print(f"\nJSON skipped (unsupported format): {json_path}")
                continue

            print(f"\nJSON Summary: {json_path.name}")
            print(f"Total Annotations: {parsed['total_annotations']}")

            print("\nSeverity Distribution (JSON risk_assessment)")
            for line in format_count_distribution(parsed["severity_counts"]):
                print(line)

            print("\nDamage Class Distribution (JSON category_name)")
            for line in format_count_distribution(parsed["class_counts"]):
                print(line)

            metrics_acc = parsed["metrics_acc"]
            if metrics_acc:
                print("\nGeometry Stats (JSON damage_metrics)")
                for key in ("area_ratio", "elongation", "edge_factor", "raw_severity_score"):
                    values = metrics_acc.get(key, [])
                    if values:
                        print(f"Mean {key}: {float(np.mean(values)):.6f}")

            if parsed.get("scores"):
                print(f"Mean ADAS Score (JSON): {float(np.mean(parsed['scores'])):.6f}")


if __name__ == "__main__":
    main()
