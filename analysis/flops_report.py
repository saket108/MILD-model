"""
analysis/flops_report.py
------------------------
FLOPs and parameter comparison: MILD-Net backbone vs ConvNeXt-Tiny.

Usage:
  python analysis/flops_report.py
  python analysis/flops_report.py --image-size 640 --output runs/flops_report.json
  python analysis/flops_report.py --csv runs/flops_report.csv

Requires:
  pip install thop timm
  (thop is optional; script falls back to param-only stats)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import timm

try:
    from thop import clever_format, profile

    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("[warn] thop not installed. Install with: pip install thop")

from models.image_encoder import EdgeAwareBranch, ImageEncoder, MILDBackbone, MILDBlock, MetricGate


def count_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def measure_macs(model: nn.Module, *dummy_inputs) -> float:
    if not HAS_THOP:
        return 0.0
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=dummy_inputs, verbose=False)
    return float(macs)


def memory_mb(model: nn.Module) -> float:
    total, _ = count_params(model)
    return total * 4 / (1024**2)


def module_breakdown(model: nn.Module, prefix: str = "") -> List[Dict]:
    rows = []
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        total, train = count_params(child)
        rows.append(
            {
                "module": full_name,
                "class": child.__class__.__name__,
                "params": total,
                "params_fmt": fmt_params(total),
                "trainable": train,
            }
        )
    return rows


def build_convnext_tiny(hidden_dim: int = 256) -> nn.Module:
    class OriginalImageEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(
                "convnext_tiny",
                pretrained=False,
                features_only=True,
                out_indices=(-1,),
            )
            in_ch = self.backbone.feature_info.channels()[-1]
            self.proj = nn.Conv2d(in_ch, hidden_dim, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.backbone(x)
            return self.proj(feats[-1])

    return OriginalImageEncoder()


def modification_delta_table(
    image_size: int,
) -> List[Dict]:
    channels = (64, 128, 256, 512)
    depths = (2, 2, 5, 2)

    blocks_total = sum(depths)
    kernel_7x7 = sum(49 * c * d for c, d in zip(channels, depths))
    kernel_5x5 = sum(25 * c * d for c, d in zip(channels, depths))
    mod4_param_saving = kernel_7x7 - kernel_5x5

    full_mlp = sum(8 * c * c * d for c, d in zip(channels, depths))
    grp_mlp = sum(2 * c * c * d for c, d in zip(channels, depths))
    mod5_param_saving = full_mlp - grp_mlp

    feat_dim = channels[0]
    prompt_dim = 256
    mod2_params = (
        2 * feat_dim * 3 * 3
        + feat_dim * 3 * 3
        + prompt_dim * feat_dim
        + feat_dim
    )

    mod3_params = 2 * (256 * 256 + 256 + 256 * 256 + 256)

    return [
        {
            "mod": "Mod 1 - Depth/width rescale",
            "description": "[2,2,5,2] x [64,128,256,512] vs ConvNeXt-Tiny",
            "param_effect": "Primary driver of param reduction",
            "flop_effect": "Primary driver of FLOPs reduction",
        },
        {
            "mod": "Mod 2 - Edge-aware branch",
            "description": "Sobel + DW conv + prompt gate on Stage-1",
            "param_effect": f"+{fmt_params(mod2_params)} added",
            "flop_effect": "Small (Sobel + 3x3 conv at /4)",
        },
        {
            "mod": "Mod 3 - Metric gate (stages 3 and 4)",
            "description": "Scale+shift conditioning from MetricsEncoder",
            "param_effect": f"+{fmt_params(mod3_params)} added",
            "flop_effect": "Small (two Linear layers)",
        },
        {
            "mod": "Mod 4 - 5x5 DW conv (replaces 7x7)",
            "description": f"Across {blocks_total} blocks in all stages",
            "param_effect": f"-{fmt_params(mod4_param_saving)} vs 7x7 DW",
            "flop_effect": "Roughly 49% less DW kernel FLOPs",
        },
        {
            "mod": "Mod 5 - Grouped 1x1 MLP (g=4)",
            "description": "Grouped pointwise expand and project",
            "param_effect": f"-{fmt_params(mod5_param_saving)} vs full 1x1",
            "flop_effect": "Roughly 75% less pointwise FLOPs",
        },
    ]


def _sep(width: int = 78) -> None:
    print("-" * width)


def print_report(report: Dict) -> None:
    print()
    print("MILD-Net vs ConvNeXt-Tiny - FLOPs and Parameter Report")
    print(f"Input resolution: {report['image_size']}x{report['image_size']}")
    print(f"hidden_dim: {report['hidden_dim']}")
    print()

    _sep()
    print(f"{'Metric':<35} {'MILD-Net':>14}  {'ConvNeXt-Tiny':>14}  {'Delta':>14}")
    _sep()
    for row in report["comparison"]:
        print(f"{row['metric']:<35} {row['mild']:>14}  {row['convnext']:>14}  {row['delta']:>14}")
    _sep()

    if report.get("flops_available"):
        print()
        print("GFLOPs = MACs x 2 (standard convention)")

    print()
    print("Per-modification contribution")
    _sep()
    for mod in report["modifications"]:
        print(mod["mod"])
        print(f"  {mod['description']}")
        print(f"  Params: {mod['param_effect']}")
        print(f"  FLOPs : {mod['flop_effect']}")
        print()

    print("MILD-Net module breakdown")
    _sep()
    print(f"{'Module':<30} {'Class':<22} {'Params':>10}  {'Trainable':>10}")
    _sep()
    for row in report["mild_breakdown"]:
        print(
            f"{row['module']:<30} {row['class']:<22} {row['params_fmt']:>10}  {fmt_params(row['trainable']):>10}"
        )
    _sep()
    print()


def run(image_size: int = 640, hidden_dim: int = 256) -> Dict:
    device = torch.device("cpu")

    mild = ImageEncoder(
        model_name="mild_net",
        hidden_dim=hidden_dim,
        metrics_dim=hidden_dim,
        prompt_dim=hidden_dim,
    ).to(device)
    convnxt = build_convnext_tiny(hidden_dim=hidden_dim).to(device)
    mild.eval()
    convnxt.eval()

    mild_total, mild_train = count_params(mild)
    convnxt_total, convnxt_train = count_params(convnxt)
    delta_pct = (mild_total - convnxt_total) / convnxt_total * 100

    dummy_img = torch.randn(1, 3, image_size, image_size)
    dummy_metrics = torch.randn(1, hidden_dim)
    dummy_prompt = torch.randn(1, hidden_dim)

    mild_macs = measure_macs(mild, dummy_img, dummy_metrics, dummy_prompt)
    convnxt_macs = measure_macs(convnxt, dummy_img)

    mild_mem = memory_mb(mild)
    convnxt_mem = memory_mb(convnxt)

    comparison = [
        {
            "metric": "Total parameters",
            "mild": fmt_params(mild_total),
            "convnext": fmt_params(convnxt_total),
            "delta": f"{delta_pct:+.1f}%",
        },
        {
            "metric": "Trainable parameters",
            "mild": fmt_params(mild_train),
            "convnext": fmt_params(convnxt_train),
            "delta": f"{(mild_train - convnxt_train) / convnxt_train * 100:+.1f}%",
        },
        {
            "metric": "Param memory (fp32)",
            "mild": f"{mild_mem:.1f} MB",
            "convnext": f"{convnxt_mem:.1f} MB",
            "delta": f"{mild_mem - convnxt_mem:+.1f} MB",
        },
    ]

    if HAS_THOP and mild_macs > 0:
        mild_gf = mild_macs * 2 / 1e9
        convnxt_gf = convnxt_macs * 2 / 1e9
        comparison += [
            {
                "metric": "MACs (encoder)",
                "mild": clever_format([mild_macs], "%.3f")[0],
                "convnext": clever_format([convnxt_macs], "%.3f")[0],
                "delta": f"{(mild_gf - convnxt_gf):+.3f} GFLOPs",
            },
            {
                "metric": "GFLOPs (MACs x 2)",
                "mild": f"{mild_gf:.3f}",
                "convnext": f"{convnxt_gf:.3f}",
                "delta": f"{mild_gf - convnxt_gf:+.3f}",
            },
        ]

    report = {
        "image_size": image_size,
        "hidden_dim": hidden_dim,
        "flops_available": HAS_THOP,
        "comparison": comparison,
        "modifications": modification_delta_table(image_size),
        "mild_breakdown": module_breakdown(mild),
        "raw": {
            "mild_total_params": mild_total,
            "mild_train_params": mild_train,
            "convnext_total_params": convnxt_total,
            "convnext_train_params": convnxt_train,
            "mild_macs": mild_macs,
            "convnext_macs": convnxt_macs,
            "mild_mem_mb": mild_mem,
            "convnext_mem_mb": convnxt_mem,
        },
    }
    return report


def save_csv(report: Dict, path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mild", "convnext", "delta"])
        writer.writeheader()
        writer.writerows(report["comparison"])
    print(f"CSV saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FLOPs/param report: MILD-Net vs ConvNeXt-Tiny")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    report = run(image_size=args.image_size, hidden_dim=args.hidden_dim)
    print_report(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON saved: {out}")

    if args.csv:
        save_csv(report, Path(args.csv))

    if not HAS_THOP:
        print()
        print("Tip: install thop for GFLOPs output: pip install thop")


if __name__ == "__main__":
    main()
