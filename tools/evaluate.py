from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.collate_fn import collate_fn
from dataset.label_folder_loader import LabelFolderDataset
from dataset.loader import MILDDetectionDataset
from evaluation.evaluator import evaluate
from models.mild_model import build_model
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device
from utils.logger import Logger


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dataset_cfg(cfg_train: dict) -> dict:
    dataset_cfg = cfg_train.get("dataset", {})
    dataset_config_path = cfg_train.get("dataset_config")
    if dataset_config_path:
        dataset_cfg.update(load_yaml(dataset_config_path) or {})
    return dataset_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Training config (preferred).")
    parser.add_argument("--model-config", help="Model config override.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-json", help="Override val JSON path.")
    parser.add_argument("--image-root", help="Override image root.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--nms-iou", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    args = parser.parse_args()

    cfg_train = load_yaml(args.config) if args.config else {}
    dataset_cfg = merge_dataset_cfg(cfg_train) if cfg_train else {}
    train_cfg = cfg_train.get("training", cfg_train) if cfg_train else {}
    eval_cfg = cfg_train.get("evaluation", {}) if cfg_train else {}

    model_config_path = args.model_config or cfg_train.get("model_config")
    if model_config_path is None:
        raise ValueError("Provide --model-config or set model_config in the training config.")
    cfg_model = load_yaml(model_config_path)

    val_json = args.val_json or dataset_cfg.get("val_json")
    image_root = args.image_root or dataset_cfg.get("image_root")
    val_images = dataset_cfg.get("val_images")
    val_labels = dataset_cfg.get("val_labels")
    class_names = dataset_cfg.get("class_names")

    if val_json:
        if not image_root:
            raise ValueError("image_root must be set when using val_json.")
        dataset = MILDDetectionDataset(
            json_path=val_json,
            image_root=image_root,
            image_size=train_cfg.get("image_size", 640),
            train=False,
            max_prompts=train_cfg.get("val_max_prompts", train_cfg.get("max_prompts", 8)),
            prompt_strategy=train_cfg.get("val_prompt_strategy", train_cfg.get("prompt_strategy", "random")),
            include_description=train_cfg.get(
                "val_include_description",
                train_cfg.get("include_description", True),
            ),
            include_definition=train_cfg.get(
                "val_include_definition",
                train_cfg.get("include_definition", True),
            ),
        )
    elif val_images and val_labels:
        dataset = LabelFolderDataset(
            images_dir=val_images,
            labels_dir=val_labels,
            image_size=train_cfg.get("image_size", 640),
            train=False,
            class_names_path=class_names,
            max_prompts=train_cfg.get("val_max_prompts", train_cfg.get("max_prompts", 8)),
            prompt_strategy=train_cfg.get("val_prompt_strategy", train_cfg.get("prompt_strategy", "random")),
        )
    else:
        raise ValueError("Provide val_json or val_images/val_labels in config or via CLI.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    device = resolve_device(args.device)
    model = build_model(cfg_model).to(device)
    load_checkpoint(args.checkpoint, model)

    report = evaluate(
        model,
        dataloader,
        torch.device(device),
        score_threshold=(
            args.score_threshold
            if args.score_threshold is not None
            else float(eval_cfg.get("score_threshold", 0.0) or 0.0)
        ),
        nms_iou=args.nms_iou if args.nms_iou is not None else eval_cfg.get("nms_iou"),
        top_k=args.top_k if args.top_k is not None else eval_cfg.get("top_k"),
        max_detections=(
            args.max_detections if args.max_detections is not None else eval_cfg.get("max_detections")
        ),
    )
    Logger().log_detection_report("eval", 1, 1, report)


if __name__ == "__main__":
    main()
