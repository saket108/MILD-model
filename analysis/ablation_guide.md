# MILD Ablation Guide

## 1. Core comparison runs

Use these three training setups:

1. Full multimodal:
   - `configs/model.yaml`
   - modalities: image + text + numeric metrics

2. No-text ablation:
   - `configs/model_no_text.yaml`
   - modalities: image + numeric metrics

3. Image-only baseline:
   - `configs/model_image_only.yaml`
   - modalities: image only

Recommended command pattern:

```bash
python tools/train.py --config configs/train.yaml --model-config configs/model.yaml
python tools/train.py --config configs/train.yaml --model-config configs/model_no_text.yaml
python tools/train.py --config configs/train.yaml --model-config configs/model_image_only.yaml
```

This gives a defensible answer to:
- how much text helps beyond image + metrics
- how much the full multimodal model helps beyond a plain vision detector

## 2. Prompt-content sensitivity on the same checkpoint

Evaluate one trained multimodal checkpoint under different prompt modes:

```bash
python tools/prompt_ablation.py --config configs/train.yaml --checkpoint runs/exp_xxx/best.pt
```

Supported modes:
- `full`
- `label_only`
- `label_zone`
- `description_only`
- `definition_only`
- `generic`

Interpretation:
- `full -> label_only` drop: extra text beyond class name matters
- `label_only -> generic` drop: damage type words matter
- `full -> label_zone` gap: description/definition text matters beyond type + coarse location
- little change across all modes: text branch is present but not carrying much useful signal

## 3. What the current prompts actually contain

From `dataset/prompt_generator.py`, the rich JSON prompt set can include:
- damage type label
- severity word
- coarse zone/location word
- class definition
- free-form description

Examples:
- `crack on aircraft surface`
- `high severity crack`
- `crack located in central aircraft structure`
- a class-definition sentence
- a free-form description sentence

## 4. What the prompts do NOT contain

The text does not provide:
- exact bounding-box coordinates
- exact pixel-level localization
- box width/height values

So the prompt is not telling the detector the exact box.
It only gives:
- what damage to look for
- sometimes a coarse zone
- sometimes descriptive semantics

The final box still comes from visual evidence and box supervision.

## 5. Defensible novelty claims from this repo

These are supported by the implementation:

1. Text affects the model in more than one place.
   - It is not only fused at the end.
   - Text gates the edge-aware branch in the image encoder.
   - Text also conditions the decoder queries through UDCM.

2. Metrics affect the model in more than one place.
   - Metrics are encoded separately.
   - They gate backbone features.
   - They also enter UDCM as context tokens.

3. Query initialization is context-conditioned.
   - Standard DETR uses fixed learned queries.
   - This model shifts queries toward the prompt/metric context before visual cross-attention.

4. Unified context memory is used during decoding.
   - Queries attend to visual tokens plus context tokens jointly.
   - So text/metrics remain available during decoding, not only before it.

## 6. Claims to avoid

Avoid claiming that:
- the prompt gives exact bounding-box detail
- the model performs grounded language localization from sentence supervision alone
- text alone determines where the damage is

That is not what the current code does.

The honest claim is:
- text provides semantic guidance about damage type and coarse context
- image evidence still determines the exact localization
