import argparse
import os
import shutil
import urllib.request
import zipfile

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def infer_default_output_save_dir():
    drive_root = "/content/drive/MyDrive"
    if os.path.isdir(drive_root):
        return os.path.join(drive_root, "MILD_runs")
    return None


def split_dir_exists(root, split):
    return os.path.isdir(os.path.join(root, split))


def grouped_split_dir_exists(root, split):
    return os.path.isdir(os.path.join(root, "images", split)) and os.path.isdir(
        os.path.join(root, "labels", split)
    )


def infer_val_split(root, preferred):
    if preferred:
        return preferred
    if split_dir_exists(root, "valid") or grouped_split_dir_exists(root, "valid"):
        return "valid"
    if split_dir_exists(root, "val") or grouped_split_dir_exists(root, "val"):
        return "val"
    return "valid"


def has_split_dirs(root):
    train_ok = split_dir_exists(root, "train")
    val_ok = split_dir_exists(root, "val") or split_dir_exists(root, "valid")
    test_ok = split_dir_exists(root, "test")
    return train_ok and val_ok and test_ok


def has_grouped_image_label_dirs(root):
    train_ok = grouped_split_dir_exists(root, "train")
    val_ok = grouped_split_dir_exists(root, "val") or grouped_split_dir_exists(root, "valid")
    test_ok = grouped_split_dir_exists(root, "test")
    return train_ok and val_ok and test_ok


def _split_aliases(split):
    if split == "val":
        return ["val", "valid"]
    if split == "valid":
        return ["valid", "val"]
    return [split]


def find_split_json(root, split):
    if not os.path.isdir(root):
        return None

    candidates = []
    aliases = _split_aliases(split)
    for name in os.listdir(root):
        lower = name.lower()
        if not lower.endswith(".json"):
            continue

        score = 0
        for alias in aliases:
            if lower == f"{alias}.json":
                score = max(score, 4)
            elif lower.endswith(f"_{alias}.json") or lower.endswith(f"-{alias}.json"):
                score = max(score, 3)
            elif lower.startswith(f"{alias}_") or lower.startswith(f"{alias}-"):
                score = max(score, 2)
            elif alias in lower:
                score = max(score, 1)
        if score:
            candidates.append((score, len(name), os.path.join(root, name)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][2]


def has_split_jsons(root):
    return bool(find_split_json(root, "train") and (find_split_json(root, "val") or find_split_json(root, "valid")))


def dataset_root_score(root):
    score = 0
    if has_split_dirs(root):
        score += 2
    if has_grouped_image_label_dirs(root):
        score += 3
    if has_split_jsons(root):
        score += 4
    return score


def resolve_dataset_root(root, max_depth=3):
    if not os.path.isdir(root):
        return root

    best_root = root
    best_score = dataset_root_score(root)
    root = os.path.abspath(root)
    base_depth = root.count(os.sep)

    for current_root, dirnames, _ in os.walk(root):
        depth = current_root.count(os.sep) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue

        score = dataset_root_score(current_root)
        if score > best_score:
            best_root = current_root
            best_score = score

    return best_root if best_score > 0 else root


def download_zip(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as f:
        shutil.copyfileobj(response, f)


def extract_zip(zip_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def infer_label_paths(root, train_split, val_split, test_split):
    if has_grouped_image_label_dirs(root):
        return {
            "train_images": os.path.join(root, "images", train_split),
            "train_labels": os.path.join(root, "labels", train_split),
            "val_images": os.path.join(root, "images", val_split),
            "val_labels": os.path.join(root, "labels", val_split),
            "test_images": os.path.join(root, "images", test_split),
            "test_labels": os.path.join(root, "labels", test_split),
        }

    if has_split_dirs(root):
        return {
            "train_images": os.path.join(root, train_split, "images"),
            "train_labels": os.path.join(root, train_split, "labels"),
            "val_images": os.path.join(root, val_split, "images"),
            "val_labels": os.path.join(root, val_split, "labels"),
            "test_images": os.path.join(root, test_split, "images"),
            "test_labels": os.path.join(root, test_split, "labels"),
        }

    return None


def infer_json_config(root, train_split, val_split, test_split, train_json, val_json, test_json):
    train_json = train_json or find_split_json(root, train_split)
    val_json = val_json or find_split_json(root, val_split)
    test_json = test_json or find_split_json(root, test_split)

    if not train_json or not val_json:
        return None

    return {
        "image_root": root,
        "train_json": train_json,
        "val_json": val_json,
        "test_json": test_json,
    }


def update_dataset_config(
    path,
    root,
    train_split,
    val_split,
    test_split,
    class_names,
    json_paths,
    train_json,
    val_json,
    test_json,
):
    cfg = load_yaml(path)
    label_paths = infer_label_paths(root, train_split, val_split, test_split)
    json_cfg = infer_json_config(root, train_split, val_split, test_split, train_json, val_json, test_json)

    if label_paths:
        cfg.update(label_paths)

    if json_cfg:
        cfg.update(json_cfg)
    elif not label_paths:
        raise ValueError(
            "Could not infer dataset layout. Expected split folders, images/labels folders, or split JSON files."
        )

    if class_names:
        cfg["class_names"] = class_names

    cfg["json_paths"] = list(json_paths or [])
    save_yaml(path, cfg)
    return cfg


def update_train_config(path, safe, batch_size, image_size, num_workers, epochs, output_save_dir):
    cfg = load_yaml(path)
    training = cfg.get("training", {})
    output = cfg.get("output", {})

    if safe:
        training["batch_size"] = 1
        training["image_size"] = 320
        training["num_workers"] = 0

    if batch_size is not None:
        training["batch_size"] = batch_size
    if image_size is not None:
        training["image_size"] = image_size
    if num_workers is not None:
        training["num_workers"] = num_workers
    if epochs is not None:
        training["epochs"] = epochs
    if output_save_dir is not None:
        output["save_dir"] = output_save_dir
    elif not output.get("save_dir") or output.get("save_dir") == "runs":
        default_output_save_dir = infer_default_output_save_dir()
        if default_output_save_dir is not None:
            output["save_dir"] = default_output_save_dir

    cfg["training"] = training
    cfg["output"] = output
    save_yaml(path, cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Colab setup helper for MILD")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root or parent folder containing split folders, images/labels folders, or split JSON files",
    )
    parser.add_argument("--dataset-url", default=None, help="Optional URL to a dataset zip")
    parser.add_argument("--dataset-zip", default=None, help="Optional path for downloaded zip")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--train-json", default=None, help="Optional explicit train JSON path")
    parser.add_argument("--val-json", default=None, help="Optional explicit val JSON path")
    parser.add_argument("--test-json", default=None, help="Optional explicit test JSON path")
    parser.add_argument("--json", nargs="*", default=None)
    parser.add_argument("--safe", action="store_true", help="Apply safe Colab defaults")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--output-save-dir",
        default=None,
        help="Directory to store run checkpoints and metrics. Defaults to /content/drive/MyDrive/MILD_runs when Drive is mounted.",
    )
    parser.add_argument("--print", action="store_true", help="Print updated config paths")
    args = parser.parse_args()

    root = args.dataset_root
    if args.dataset_url:
        zip_path = args.dataset_zip or os.path.join(root, "dataset.zip")
        if not os.path.isfile(zip_path):
            print("Downloading dataset zip:", args.dataset_url)
            download_zip(args.dataset_url, zip_path)
        else:
            print("Dataset zip already exists:", zip_path)
        print("Extracting dataset zip to:", root)
        extract_zip(zip_path, root)

    root = resolve_dataset_root(root)
    val_split = infer_val_split(root, args.val_split)

    dataset_cfg = update_dataset_config(
        args.dataset_config,
        root,
        args.train_split,
        val_split,
        args.test_split,
        args.class_names,
        args.json,
        args.train_json,
        args.val_json,
        args.test_json,
    )

    train_cfg = update_train_config(
        args.train_config,
        args.safe,
        args.batch_size,
        args.image_size,
        args.num_workers,
        args.epochs,
        args.output_save_dir,
    )

    if args.print:
        print("Resolved dataset root:", root)
        print("  image_root:", dataset_cfg.get("image_root"))
        print("  train_json:", dataset_cfg.get("train_json"))
        print("  val_json:", dataset_cfg.get("val_json"))
        print("  test_json:", dataset_cfg.get("test_json"))
        print("Updated dataset config:", args.dataset_config)
        print("  train_images:", dataset_cfg.get("train_images"))
        print("  train_labels:", dataset_cfg.get("train_labels"))
        print("  val_images:", dataset_cfg.get("val_images"))
        print("  val_labels:", dataset_cfg.get("val_labels"))
        print("  test_images:", dataset_cfg.get("test_images"))
        print("  test_labels:", dataset_cfg.get("test_labels"))
        print("  json_paths:", dataset_cfg.get("json_paths"))
        print("Updated train config:", args.train_config)
        print("  batch_size:", train_cfg.get("training", {}).get("batch_size"))
        print("  image_size:", train_cfg.get("training", {}).get("image_size"))
        print("  num_workers:", train_cfg.get("training", {}).get("num_workers"))
        print("  epochs:", train_cfg.get("training", {}).get("epochs"))
        print("  output.save_dir:", train_cfg.get("output", {}).get("save_dir"))


if __name__ == "__main__":
    main()
