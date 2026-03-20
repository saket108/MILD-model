from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from dataset.collate_fn import collate_fn
from dataset.label_folder_loader import LabelFolderDataset
from dataset.loader import MILDDetectionDataset
from evaluation.evaluator import evaluate
from models.mild_model import build_model
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device

DEFAULT_MODES = [
    "full",
    "label_only",
    "label_zone",
    "description_only",
    "definition_only",
    "generic",
]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dataset_cfg(cfg_train: dict) -> dict:
    dataset_cfg = cfg_train.get("dataset", {})
    dataset_config_path = cfg_train.get("dataset_config")
    if dataset_config_path:
        dataset_cfg.update(load_yaml(dataset_config_path) or {})
    return dataset_cfg


def build_val_dataset(dataset_cfg: dict, train_cfg: dict, prompt_mode: str):
    val_json = dataset_cfg.get("val_json")
    image_root = dataset_cfg.get("image_root")
    val_images = dataset_cfg.get("val_images")
    val_labels = dataset_cfg.get("val_labels")
    class_names = dataset_cfg.get("class_names")

    if val_json:
        if not image_root:
            raise ValueError("image_root must be set when using val_json.")
        return MILDDetectionDataset(
            json_path=val_json,
            image_root=image_root,
            image_size=train_cfg.get("image_size", 640),
            train=False,
            max_prompts=train_cfg.get("val_max_prompts", train_cfg.get("max_prompts", 8)),
            prompt_strategy=train_cfg.get("val_prompt_strategy", train_cfg.get("prompt_strategy", "random")),
            prompt_mode=prompt_mode,
            include_description=train_cfg.get(
                "val_include_description",
                train_cfg.get("include_description", True),
            ),
            include_definition=train_cfg.get(
                "val_include_definition",
                train_cfg.get("include_definition", True),
            ),
        )

    if val_images and val_labels:
        return LabelFolderDataset(
            images_dir=val_images,
            labels_dir=val_labels,
            image_size=train_cfg.get("image_size", 640),
            train=False,
            class_names_path=class_names,
            max_prompts=train_cfg.get("val_max_prompts", train_cfg.get("max_prompts", 8)),
            prompt_strategy=train_cfg.get("val_prompt_strategy", train_cfg.get("prompt_strategy", "random")),
            prompt_mode=prompt_mode,
        )

    raise ValueError("Provide val_json or val_images/val_labels in config.")


def fmt(value) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value != value:
        return "-"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint under multiple prompt-content modes.")
    parser.add_argument("--config", required=True, help="Training config.")
    parser.add_argument("--model-config", default=None, help="Optional model config override.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--nms-iou", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--save-json", default=None, help="Optional JSON path for raw reports.")
    args = parser.parse_args()

    cfg_train = load_yaml(args.config)
    dataset_cfg = merge_dataset_cfg(cfg_train)
    train_cfg = cfg_train.get("training", cfg_train)
    eval_cfg = cfg_train.get("evaluation", {})

    model_config_path = args.model_config or cfg_train.get("model_config")
    if model_config_path is None:
        raise ValueError("Provide --model-config or set model_config in the training config.")
    cfg_model = load_yaml(model_config_path)

    device = resolve_device(args.device)
    model = build_model(cfg_model).to(device)
    load_checkpoint(args.checkpoint, model)

    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else float(eval_cfg.get("score_threshold", 0.0) or 0.0)
    )
    nms_iou = args.nms_iou if args.nms_iou is not None else eval_cfg.get("nms_iou")
    top_k = args.top_k if args.top_k is not None else eval_cfg.get("top_k")
    max_detections = args.max_detections if args.max_detections is not None else eval_cfg.get("max_detections")

    console = Console()
    if not cfg_model.get("use_text", True):
        console.print("[yellow]Model config has use_text=false. Prompt-content ablation is not meaningful for this checkpoint.[/yellow]")

    reports = {}
    table = Table(title="Prompt Ablation", show_header=True, header_style="bold")
    table.add_column("Mode")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("mAP50", justify="right")
    table.add_column("mAP50-95", justify="right")
    table.add_column("F1", justify="right")

    for mode in args.modes:
        dataset = build_val_dataset(dataset_cfg, train_cfg, prompt_mode=mode)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
        report = evaluate(
            model,
            dataloader,
            torch.device(device),
            score_threshold=score_threshold,
            nms_iou=nms_iou,
            top_k=top_k,
            max_detections=max_detections,
        )
        reports[mode] = report
        summary = report.get("summary", {})
        table.add_row(
            mode,
            fmt(summary.get("precision")),
            fmt(summary.get("recall")),
            fmt(summary.get("map50")),
            fmt(summary.get("map50_95")),
            fmt(summary.get("f1")),
        )

    console.print(table)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        console.print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
