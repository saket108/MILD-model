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


def infer_val_split(root, preferred):
    if preferred:
        return preferred
    if os.path.isdir(os.path.join(root, "valid")):
        return "valid"
    if os.path.isdir(os.path.join(root, "val")):
        return "val"
    return "valid"

def has_split_dirs(root):
    train_ok = os.path.isdir(os.path.join(root, "train"))
    val_ok = os.path.isdir(os.path.join(root, "val")) or os.path.isdir(os.path.join(root, "valid"))
    test_ok = os.path.isdir(os.path.join(root, "test"))
    return train_ok and val_ok and test_ok


def resolve_dataset_root(root):
    if has_split_dirs(root):
        return root
    candidates = []
    if not os.path.isdir(root):
        return root
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and has_split_dirs(path):
            candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    return root


def download_zip(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as f:
        shutil.copyfileobj(response, f)


def extract_zip(zip_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def update_dataset_config(path, root, train_split, val_split, test_split, class_names, json_paths):
    cfg = load_yaml(path)
    cfg["train_images"] = os.path.join(root, train_split, "images")
    cfg["train_labels"] = os.path.join(root, train_split, "labels")
    cfg["val_images"] = os.path.join(root, val_split, "images")
    cfg["val_labels"] = os.path.join(root, val_split, "labels")
    cfg["test_images"] = os.path.join(root, test_split, "images")
    cfg["test_labels"] = os.path.join(root, test_split, "labels")
    if class_names:
        cfg["class_names"] = class_names
    cfg["json_paths"] = json_paths or cfg.get("json_paths", []) or []
    save_yaml(path, cfg)
    return cfg


def update_train_config(path, safe, batch_size, image_size, num_workers, epochs):
    cfg = load_yaml(path)
    training = cfg.get("training", {})

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

    cfg["training"] = training
    save_yaml(path, cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Colab setup helper for MILD")
    parser.add_argument("--dataset-root", required=True, help="Root containing train/val/test folders")
    parser.add_argument("--dataset-url", default=None, help="Optional URL to a dataset zip")
    parser.add_argument("--dataset-zip", default=None, help="Optional path for downloaded zip")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--json", nargs="*", default=None)
    parser.add_argument("--safe", action="store_true", help="Apply safe Colab defaults")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
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
    )

    train_cfg = update_train_config(
        args.train_config,
        args.safe,
        args.batch_size,
        args.image_size,
        args.num_workers,
        args.epochs,
    )

    if args.print:
        print("Updated dataset config:", args.dataset_config)
        print("  train_images:", dataset_cfg.get("train_images"))
        print("  train_labels:", dataset_cfg.get("train_labels"))
        print("  val_images:", dataset_cfg.get("val_images"))
        print("  val_labels:", dataset_cfg.get("val_labels"))
        print("  test_images:", dataset_cfg.get("test_images"))
        print("  test_labels:", dataset_cfg.get("test_labels"))
        print("Updated train config:", args.train_config)
        print("  batch_size:", train_cfg.get("training", {}).get("batch_size"))
        print("  image_size:", train_cfg.get("training", {}).get("image_size"))
        print("  num_workers:", train_cfg.get("training", {}).get("num_workers"))
        print("  epochs:", train_cfg.get("training", {}).get("epochs"))


if __name__ == "__main__":
    main()
