from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_class_names(path: Path | None) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            names.append(name)
    return names


def _iter_image_files(images_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for path in images_dir.iterdir():
        if path.suffix.lower() in exts:
            yield path


def _resolve_image_path(image_root: Path, file_name: str, split: str | None) -> Path:
    direct = image_root / file_name
    if direct.exists():
        return direct

    if split:
        candidate = image_root / split / "images" / file_name
        if candidate.exists():
            return candidate
        candidate = image_root / split / file_name
        if candidate.exists():
            return candidate

    candidate = image_root / "images" / file_name
    if candidate.exists():
        return candidate

    return direct


def _validate_label_file(
    label_path: Path,
    class_count: int | None,
    issues: List[str],
    max_issues: int,
) -> int:
    count = 0
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            _add_issue(issues, max_issues, f"{label_path}: line {idx} has <5 fields")
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            _add_issue(issues, max_issues, f"{label_path}: line {idx} has non-numeric values")
            continue
        if class_id < 0:
            _add_issue(issues, max_issues, f"{label_path}: line {idx} has negative class_id")
        if class_count is not None and class_id >= class_count:
            _add_issue(
                issues,
                max_issues,
                f"{label_path}: line {idx} class_id {class_id} out of range (0..{class_count-1})",
            )
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            _add_issue(issues, max_issues, f"{label_path}: line {idx} center out of [0,1]")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            _add_issue(issues, max_issues, f"{label_path}: line {idx} width/height out of (0,1]")
        count += 1
    return count


def _add_issue(issues: List[str], max_issues: int, message: str) -> None:
    if len(issues) < max_issues:
        issues.append(message)


def _validate_label_split(
    images_dir: Path,
    labels_dir: Path,
    class_names: List[str],
    issues: List[str],
    max_issues: int,
) -> Tuple[int, int]:
    if not images_dir.exists():
        _add_issue(issues, max_issues, f"Missing images dir: {images_dir}")
        return 0, 0
    if not labels_dir.exists():
        _add_issue(issues, max_issues, f"Missing labels dir: {labels_dir}")
        return 0, 0

    image_stems = {p.stem for p in _iter_image_files(images_dir)}
    label_files = list(labels_dir.glob("*.txt"))
    label_stems = {p.stem for p in label_files}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    for stem in missing_labels[:max_issues]:
        _add_issue(issues, max_issues, f"Missing label for image: {images_dir / (stem + '.*')}")
    for stem in missing_images[:max_issues]:
        _add_issue(issues, max_issues, f"Missing image for label: {labels_dir / (stem + '.txt')}")

    class_count = len(class_names) if class_names else None
    total_labels = 0
    for label_path in label_files:
        total_labels += _validate_label_file(label_path, class_count, issues, max_issues)

    return len(image_stems), total_labels


def _validate_json(
    json_path: Path,
    image_root: Path | None,
    class_names: List[str],
    issues: List[str],
    max_issues: int,
) -> Tuple[int, int]:
    if not json_path.exists():
        _add_issue(issues, max_issues, f"JSON not found: {json_path}")
        return 0, 0

    data = json.loads(json_path.read_text(encoding="utf-8"))
    images = []
    if isinstance(data, dict) and "images" in data:
        images = data.get("images", [])
    elif isinstance(data, list):
        images = data
    else:
        _add_issue(issues, max_issues, f"JSON format not recognized: {json_path}")
        return 0, 0

    class_set = set(class_names)
    total_images = 0
    total_annotations = 0

    for item in images:
        file_name = item.get("file_name") or item.get("image")
        if not file_name:
            _add_issue(issues, max_issues, f"{json_path}: image entry missing file_name/image")
            continue
        total_images += 1

        if image_root is not None:
            split_value = item.get("split")
            img_path = _resolve_image_path(image_root, file_name, split_value)
            if not img_path.exists():
                _add_issue(issues, max_issues, f"Missing image file: {img_path}")

        annotations = item.get("annotations", [])
        if not isinstance(annotations, list):
            _add_issue(issues, max_issues, f"{json_path}: annotations not a list for {file_name}")
            continue

        for ann in annotations:
            total_annotations += 1
            bbox = ann.get("bounding_box_normalized")
            if not bbox:
                _add_issue(issues, max_issues, f"{json_path}: missing bbox for {file_name}")
                continue
            try:
                xc = float(bbox.get("x_center"))
                yc = float(bbox.get("y_center"))
                w = float(bbox.get("width"))
                h = float(bbox.get("height"))
            except (TypeError, ValueError):
                _add_issue(issues, max_issues, f"{json_path}: bbox has non-numeric values for {file_name}")
                continue

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                _add_issue(issues, max_issues, f"{json_path}: bbox center out of range for {file_name}")
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                _add_issue(issues, max_issues, f"{json_path}: bbox size out of range for {file_name}")

            if class_names:
                label = ann.get("category_name")
                if label and label not in class_set:
                    _add_issue(
                        issues,
                        max_issues,
                        f"{json_path}: category_name '{label}' not in class list",
                    )
                category_id = ann.get("category_id")
                if category_id is not None:
                    try:
                        cid = int(category_id)
                        if cid < 0 or cid >= len(class_names):
                            _add_issue(
                                issues,
                                max_issues,
                                f"{json_path}: category_id {cid} out of range (0..{len(class_names)-1})",
                            )
                    except ValueError:
                        _add_issue(issues, max_issues, f"{json_path}: category_id not an int")

    return total_images, total_annotations


def validate_dataset(
    dataset_cfg: dict,
    overrides: dict | None = None,
    max_issues: int = 50,
    verbose: bool = True,
) -> int:
    overrides = overrides or {}
    issues: List[str] = []

    class_names_path = overrides.get("class_names") or dataset_cfg.get("class_names")
    class_names = _load_class_names(Path(class_names_path)) if class_names_path else []

    total_images = 0
    total_annotations = 0

    train_json = overrides.get("train_json") or dataset_cfg.get("train_json")
    val_json = overrides.get("val_json") or dataset_cfg.get("val_json")
    json_paths = overrides.get("json_paths") or dataset_cfg.get("json_paths") or []
    image_root = overrides.get("image_root") or dataset_cfg.get("image_root")
    image_root_path = Path(image_root) if image_root else None

    for path in [train_json, val_json, *json_paths]:
        if not path:
            continue
        img_count, ann_count = _validate_json(
            Path(path),
            image_root_path,
            class_names,
            issues,
            max_issues,
        )
        total_images += img_count
        total_annotations += ann_count

    for split in ("train", "val", "valid", "test"):
        images_key = f"{split}_images"
        labels_key = f"{split}_labels"
        images_dir = overrides.get(images_key) or dataset_cfg.get(images_key)
        labels_dir = overrides.get(labels_key) or dataset_cfg.get(labels_key)
        if images_dir and labels_dir:
            img_count, ann_count = _validate_label_split(
                Path(images_dir),
                Path(labels_dir),
                class_names,
                issues,
                max_issues,
            )
            total_images += img_count
            total_annotations += ann_count

    if verbose:
        print("\nDATASET VALIDATION\n")
        print(f"Images checked: {total_images}")
        print(f"Annotations checked: {total_annotations}")
        if issues:
            print(f"\nIssues found: {len(issues)} (showing up to {max_issues})")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("\nNo issues found.")

    return len(issues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--val-json", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--max-issues", type=int, default=50)
    args = parser.parse_args()

    dataset_cfg = _load_yaml(Path(args.dataset_config))
    overrides = {}
    if args.train_json:
        overrides["train_json"] = args.train_json
    if args.val_json:
        overrides["val_json"] = args.val_json
    if args.image_root:
        overrides["image_root"] = args.image_root
    if args.class_names:
        overrides["class_names"] = args.class_names

    issue_count = validate_dataset(dataset_cfg, overrides=overrides, max_issues=args.max_issues, verbose=True)
    raise SystemExit(1 if issue_count else 0)


if __name__ == "__main__":
    main()
