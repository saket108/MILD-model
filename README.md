# MILD Multimodal Detector

Lightweight multimodal prompt-guided object detector for image, text, and structured damage metrics.  
Supports ADAS-style severity scoring, hybrid severity decisions, and research-friendly experiment tracking.

**Features**
- Image + text + numeric metric fusion
- DETR-style decoder with auxiliary losses
- ADAS severity scoring and neural severity head
- Dataset validation and calibration utilities
- Colab-ready training flow

**Quickstart**
```bash
pip install -r requirements.txt
python tools/train.py --config configs/train.yaml
```

**Colab**
Open `notebooks/colab_quickstart.ipynb` or run:
```bash
python tools/colab_setup.py --dataset-root /content/drive/MyDrive/your_dataset_root --safe --print
python tools/train.py --config configs/train.yaml --device auto
```
Optional dataset download:
```bash
python tools/colab_setup.py --dataset-root /content/dataset --dataset-url https://example.com/dataset.zip --safe --print
```
One-click smoke test: `notebooks/colab_one_click.ipynb`

**Repository Layout**
- `configs/` training and model configs
- `dataset/` loaders and transforms
- `models/` encoders, fusion, decoder, heads
- `losses/` matcher and loss functions
- `training/` trainer, optimizer, scheduler
- `evaluation/` metrics and evaluation loop
- `analysis/` validation, calibration, and reports
- `tools/` CLI entrypoints

**Configs**
- Model config: `configs/model.yaml`
- Train config: `configs/train.yaml`
- Dataset config: `configs/dataset.yaml`

**Dataset Formats**
Two formats are supported.

Simple list:
- `image`: image filename
- `boxes`: list of `[x, y, w, h]` boxes in pixels
- `labels`: list of string labels
```json
{
  "image": "img001.jpg",
  "boxes": [[10, 20, 120, 80]],
  "labels": ["crack"]
}
```

Rich JSON with metadata:
```json
{
  "images": [
    {
      "image_id": "image_00641",
      "file_name": "image_00641.jpg",
      "split": "train",
      "annotations": [
        {
          "category_name": "dent",
          "zone_estimation": "central",
          "bounding_box_normalized": {
            "x_center": 0.41,
            "y_center": 0.49,
            "width": 0.26,
            "height": 0.50
          },
          "damage_metrics": {
            "area_ratio": 0.13,
            "elongation": 1.92,
            "edge_factor": 0.58,
            "raw_severity_score": 0.13
          },
          "risk_assessment": { "severity_level": "low" },
          "class_definition": "A localized surface deformation without material fracture.",
          "description": "Low severity Dent detected in the central structural region."
        }
      ]
    }
  ]
}
```

**Train**
```bash
python tools/train.py --config configs/train.yaml
```
Optional run notes notebook:
```bash
python tools/train.py --config configs/train.yaml --notes "experiment: multiscale+aux+amp"
```

**Train With Label Folders (no JSON)**
Set these in `configs/dataset.yaml`:
```yaml
train_images: dataset/train/images
train_labels: dataset/train/labels
val_images: dataset/valid/images
val_labels: dataset/valid/labels
class_names: configs/classes.txt
```

**Evaluate**
```bash
python tools/evaluate.py --config configs/train.yaml --checkpoint runs/exp_001/best.pt
```

**Inference**
```bash
python tools/infer.py \
  --model-config configs/model.yaml \
  --checkpoint runs/exp_001/best.pt \
  --image path/to/image.jpg \
  --prompt "aircraft crack"
```
Optional numeric metrics for inference:
```bash
python tools/infer.py \
  --model-config configs/model.yaml \
  --checkpoint runs/exp_001/best.pt \
  --image path/to/image.jpg \
  --prompt "dent on aircraft surface" \
  --metrics 0.13 1.92 0.58 0.13
```

**Analysis**
- `python analysis/validate_dataset.py`
- `python analysis/severity_analyzer.py`
- `python analysis/severity_calibration.py`
- `python analysis/dataset_report.py`
- `python analysis/flops_report.py`
- `python analysis/verify_mild_net.py`

**Pipeline**
```bash
python pipeline/inference_pipeline.py \
  --image path/to/image.jpg \
  --prompt "dent on aircraft surface" \
  --model-config configs/model.yaml \
  --checkpoint runs/exp_001/best.pt \
  --adas-config configs/adas.yaml
```

**Inference Output (JSON)**
```json
{
  "image": { "path": "path/to/image.jpg" },
  "prompt": "dent on aircraft surface",
  "model": {
    "config": "configs/model.yaml",
    "checkpoint": "runs/exp_001/best.pt",
    "thresholds": "configs/severity_thresholds.yaml"
  },
  "summary": {
    "num_predictions": 1,
    "review_required": 0
  },
  "predictions": [
    {
      "id": 0,
      "label_id": 1,
      "label": "dent",
      "score": 0.94,
      "box": { "format": "xyxy", "x1": 120.0, "y1": 80.0, "x2": 360.0, "y2": 300.0, "normalized": false },
      "box_norm": { "format": "cxcywh", "cx": 0.38, "cy": 0.46, "w": 0.24, "h": 0.34 },
      "adas": {
        "score": 0.73,
        "level": "moderate",
        "metrics": { "area_ratio": 0.08, "elongation": 1.42, "edge_factor": 0.54 }
      },
      "neural": { "score": 0.69, "level": "moderate" },
      "hybrid": { "agreement": true, "final_level": "moderate", "review_required": false }
    }
  ]
}
```
