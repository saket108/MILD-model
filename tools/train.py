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
from torch.utils.data import DataLoader

from dataset.collate_fn import collate_fn
from dataset.label_folder_loader import LabelFolderDataset
from dataset.loader import MILDDetectionDataset
from evaluation.evaluator import evaluate
from losses.total_loss import build_loss
from models.mild_model import build_model
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.trainer import Trainer
from analysis.validate_dataset import validate_dataset
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device
from utils.run_notes import write_run_notebook
from utils.seed import set_seed


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dataset_cfg(cfg_train: dict) -> dict:
    dataset_cfg = cfg_train.get("dataset", {})
    dataset_config_path = cfg_train.get("dataset_config")
    if dataset_config_path:
        dataset_cfg.update(load_yaml(dataset_config_path) or {})
    return dataset_cfg


def next_run_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in base.glob("exp_*"):
        parts = path.name.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            existing.append(int(parts[1]))
    next_id = max(existing, default=0) + 1
    return base / f"exp_{next_id:03d}"


def extract_summary(report: dict | None) -> dict:
    if not isinstance(report, dict):
        return {}
    summary = report.get("summary")
    if isinstance(summary, dict):
        return summary
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Training config (new format).")
    parser.add_argument("--model-config", help="Model config.")
    parser.add_argument("--train-config", help="Legacy training config (flat format).")
    parser.add_argument("--train-json", help="Override train JSON path.")
    parser.add_argument("--val-json", help="Override val JSON path.")
    parser.add_argument("--image-root", help="Override image root.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-validate", action="store_true", help="Skip dataset validation.")
    parser.add_argument("--notes", default=None, help="Optional run notes for runs/exp_xxx/notes.ipynb")
    parser.add_argument("--resume", default=None, help="Resume from runs/exp_xxx/last.pt or best.pt")
    args = parser.parse_args()

    if args.config:
        cfg_train = load_yaml(args.config)
    elif args.train_config:
        cfg_train = load_yaml(args.train_config)
    else:
        raise ValueError("Provide --config (preferred) or --train-config (legacy).")

    model_config_path = args.model_config or cfg_train.get("model_config")
    if model_config_path is None:
        raise ValueError("Provide --model-config or set model_config in the training config.")
    cfg_model = load_yaml(model_config_path)

    train_cfg = cfg_train.get("training", cfg_train)
    dataset_cfg = merge_dataset_cfg(cfg_train)
    output_cfg = cfg_train.get("output", cfg_train)

    set_seed(train_cfg.get("seed", cfg_train.get("seed", 42)))

    train_json = args.train_json or dataset_cfg.get("train_json")
    val_json = args.val_json or dataset_cfg.get("val_json")
    image_root = args.image_root or dataset_cfg.get("image_root")
    train_images = dataset_cfg.get("train_images")
    train_labels = dataset_cfg.get("train_labels")
    val_images = dataset_cfg.get("val_images")
    val_labels = dataset_cfg.get("val_labels")
    class_names = dataset_cfg.get("class_names")
    if not train_json and not (train_images and train_labels):
        raise ValueError("Provide train_json or train_images/train_labels in dataset config.")

    if not args.no_validate:
        overrides = {}
        if args.train_json:
            overrides["train_json"] = args.train_json
        if args.val_json:
            overrides["val_json"] = args.val_json
        if args.image_root:
            overrides["image_root"] = args.image_root
        issues = validate_dataset(dataset_cfg, overrides=overrides, verbose=True)
        if issues:
            raise SystemExit(
                "Dataset validation failed. Fix issues or re-run with --no-validate."
            )

    if train_json:
        if not image_root:
            raise ValueError("image_root must be set when using train_json.")
        dataset = MILDDetectionDataset(
            json_path=train_json,
            image_root=image_root,
            image_size=train_cfg.get("image_size", 640),
            train=True,
            max_prompts=train_cfg.get("max_prompts", 8),
            prompt_strategy=train_cfg.get("prompt_strategy", "random"),
            include_description=train_cfg.get("include_description", True),
            include_definition=train_cfg.get("include_definition", True),
        )
    else:
        dataset = LabelFolderDataset(
            images_dir=train_images,
            labels_dir=train_labels,
            image_size=train_cfg.get("image_size", 640),
            train=True,
            class_names_path=class_names,
            max_prompts=train_cfg.get("max_prompts", 8),
            prompt_strategy=train_cfg.get("prompt_strategy", "random"),
        )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_json or (val_images and val_labels):
        if val_json:
            if not image_root:
                raise ValueError("image_root must be set when using val_json.")
            val_dataset = MILDDetectionDataset(
                json_path=val_json,
                image_root=image_root,
                image_size=train_cfg.get("image_size", 640),
                train=False,
                max_prompts=train_cfg.get("max_prompts", 8),
                prompt_strategy=train_cfg.get("prompt_strategy", "random"),
                include_description=train_cfg.get("include_description", True),
                include_definition=train_cfg.get("include_definition", True),
            )
        else:
            val_dataset = LabelFolderDataset(
                images_dir=val_images,
                labels_dir=val_labels,
                image_size=train_cfg.get("image_size", 640),
                train=False,
                class_names_path=class_names,
                max_prompts=train_cfg.get("max_prompts", 8),
                prompt_strategy=train_cfg.get("prompt_strategy", "random"),
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 2),
            collate_fn=collate_fn,
        )

    device = resolve_device(args.device)
    model = build_model(cfg_model).to(device)
    optimizer = build_optimizer(model, cfg_train)
    total_epochs = train_cfg.get("epochs", 1)
    scheduler = build_scheduler(optimizer, cfg_train, total_epochs)
    loss_fn = build_loss(cfg_model)

    start_epoch = 0
    history = []
    best_epoch = None
    best_report = None
    trainer_best_metric = float("-inf")
    if args.resume:
        resume_path = Path(args.resume)
        run_dir = resume_path.resolve().parent
        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = int(checkpoint.get("epoch") or 0)
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            history = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"Resuming from: {resume_path}")
        print(f"Run dir: {run_dir}")
        print(f"Start epoch: {start_epoch + 1}/{train_cfg.get('epochs', 1)}")
    else:
        run_dir = next_run_dir(output_cfg.get("save_dir", "runs"))
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump({"train": cfg_train, "model": cfg_model}, f, sort_keys=False)
        write_run_notebook(run_dir, args.notes, cfg_train, cfg_model, dataset_cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=torch.device(device),
        save_dir=run_dir,
        evaluator=evaluate if val_loader is not None else None,
        use_amp=train_cfg.get("amp", False),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 0.0),
    )
    if history:
        for entry in history:
            report = entry.get("val")
            summary = extract_summary(report)
            score = summary.get(trainer.best_metric_key)
            if score is not None and score > trainer_best_metric:
                trainer_best_metric = score
                best_epoch = entry.get("epoch")
                best_report = report if isinstance(report, dict) else {"summary": summary, "per_class": []}
        trainer.best_metric = trainer_best_metric

    trainer.fit(
        dataloader,
        epochs=train_cfg.get("epochs", 1),
        val_loader=val_loader,
        start_epoch=start_epoch,
        history=history,
        best_epoch=best_epoch,
        best_report=best_report,
    )


if __name__ == "__main__":
    main()
