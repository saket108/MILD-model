from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.collate_fn import collate_fn
from dataset.loader import MILDDetectionDataset
from evaluation.evaluator import evaluate
from losses.total_loss import build_loss
from models.mild_model import build_model
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.trainer import Trainer
from utils.device import resolve_device


def create_smoke_dataset(root: Path) -> tuple[Path, Path, Path]:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / "image_0001.jpg"

    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    plt.imsave(image_path, img)

    sample = {
        "images": [
            {
                "image_id": "image_0001",
                "file_name": "image_0001.jpg",
                "split": "train",
                "annotations": [
                    {
                        "annotation_id": "image_0001_0",
                        "category_id": 0,
                        "category_name": "dent",
                        "class_definition": "A localized surface deformation without material fracture.",
                        "zone_estimation": "central",
                        "bounding_box_normalized": {
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width": 0.4,
                            "height": 0.3,
                        },
                        "damage_metrics": {
                            "area_ratio": 0.12,
                            "elongation": 1.3,
                            "edge_factor": 0.5,
                            "raw_severity_score": 0.2,
                        },
                        "risk_assessment": {
                            "severity_level": "low",
                            "calibration_percentiles": {"p50": 0.15, "p85": 0.9},
                            "requires_manual_validation": False,
                        },
                        "description": "Low severity dent detected in the central region.",
                    }
                ],
            }
        ]
    }

    train_json = root / "train.json"
    val_json = root / "val.json"
    train_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    val_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    return images_dir, train_json, val_json


def main() -> None:
    smoke_root = Path("analysis/smoke_data")
    images_dir, train_json, val_json = create_smoke_dataset(smoke_root)

    cfg_model = yaml.safe_load(Path("configs/model.yaml").read_text(encoding="utf-8"))
    cfg_train = yaml.safe_load(Path("configs/train.yaml").read_text(encoding="utf-8"))

    train_cfg = cfg_train.get("training", cfg_train)
    device = resolve_device("auto")

    dataset = MILDDetectionDataset(
        json_path=train_json,
        image_root=images_dir,
        image_size=256,
        train=True,
        max_prompts=train_cfg.get("max_prompts", 2),
        prompt_strategy=train_cfg.get("prompt_strategy", "random"),
        include_description=True,
        include_definition=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_dataset = MILDDetectionDataset(
        json_path=val_json,
        image_root=images_dir,
        image_size=256,
        train=False,
        max_prompts=train_cfg.get("max_prompts", 2),
        prompt_strategy=train_cfg.get("prompt_strategy", "random"),
        include_description=True,
        include_definition=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = build_model(cfg_model).to(device)
    optimizer = build_optimizer(model, cfg_train)
    total_steps = 1
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)
    loss_fn = build_loss(cfg_model)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=torch.device(device),
        save_dir=smoke_root / "runs",
        evaluator=evaluate,
    )

    trainer.fit(dataloader, epochs=1, val_loader=val_loader)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
