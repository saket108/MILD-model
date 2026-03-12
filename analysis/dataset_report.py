from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def find_label_dirs(root: Path) -> Dict[str, Path]:
    splits = {}
    for split in ("train", "valid", "test"):
        path = root / split / "labels"
        if path.exists():
            splits[split] = path
    if not splits:
        flat = root / "labels"
        if flat.exists():
            splits["all"] = flat
    return splits


def iter_label_files(label_dir: Path) -> Iterable[Path]:
    for path in sorted(label_dir.glob("*.txt")):
        yield path


def parse_label_line(line: str) -> int | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        return int(float(parts[0]))
    except ValueError:
        return None


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", default="dataset", help="Root containing train/valid/test folders.")
    parser.add_argument("--classes", help="Optional class names file (one per line).")
    args = parser.parse_args()

    label_root = Path(args.label_root)
    splits = find_label_dirs(label_root)
    if not splits:
        print("No label folders found. Expected dataset/train/labels, dataset/valid/labels, dataset/test/labels.")
        return

    class_names = load_class_names(Path(args.classes)) if args.classes else {}

    total_images = 0
    total_annotations = 0
    total_counts = Counter()

    print("\nDATASET REPORT\n")
    for split, label_dir in splits.items():
        split_images = 0
        split_annotations = 0
        split_counts = Counter()

        for path in iter_label_files(label_dir):
            split_images += 1
            for line in path.read_text(encoding="utf-8").splitlines():
                class_id = parse_label_line(line)
                if class_id is None:
                    continue
                split_counts[class_id] += 1
                split_annotations += 1

        total_images += split_images
        total_annotations += split_annotations
        total_counts.update(split_counts)

        print(f"{split.upper()} | images: {split_images} | annotations: {split_annotations}")

    print(f"\nTOTAL images: {total_images}")
    print(f"TOTAL annotations: {total_annotations}")

    print("\nClass Distribution")
    for class_id, count in sorted(total_counts.items(), key=lambda x: (-x[1], x[0])):
        name = class_names.get(class_id, str(class_id))
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
