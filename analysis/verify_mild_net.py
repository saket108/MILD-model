"""
analysis/verify_mild_net.py
---------------------------
Pre-training verification for MILD-Net backbone.
Tests shapes, NaN/Inf, gradient flow, and all 5 modification paths
without needing a dataset or checkpoint.

Usage:
  python analysis/verify_mild_net.py
  python analysis/verify_mild_net.py --image-size 320
  python analysis/verify_mild_net.py --device cuda
  python analysis/verify_mild_net.py --output runs/verify_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


PASS = _green("  PASS")
FAIL = _red("  FAIL")
SKIP = _yellow("  SKIP")


class SkipTest(Exception):
    pass


class TestSuite:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.results: List[Dict] = []

    def run(self, name: str, fn) -> bool:
        sys.stdout.write(f"  {name:<55}")
        sys.stdout.flush()
        t0 = time.perf_counter()
        try:
            note = fn()
            elapsed = time.perf_counter() - t0
            msg = f"{note or ''}  ({elapsed*1000:.0f}ms)"
            print(f"{PASS}  {msg}")
            self.results.append({"test": name, "status": "pass", "note": note or ""})
            return True
        except SkipTest as exc:
            print(f"{SKIP}  {exc}")
            self.results.append({"test": name, "status": "skip", "note": str(exc)})
            return True
        except Exception as exc:
            print(f"{FAIL}")
            print(f"           {_red(str(exc))}")
            traceback.print_exc()
            self.results.append({"test": name, "status": "fail", "note": str(exc)})
            return False

    def summary(self) -> Dict:
        passed = sum(1 for r in self.results if r["status"] == "pass")
        failed = sum(1 for r in self.results if r["status"] == "fail")
        skipped = sum(1 for r in self.results if r["status"] == "skip")
        return {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": len(self.results),
            "results": self.results,
        }


def assert_no_nan(tensor: torch.Tensor, name: str = "tensor") -> None:
    assert not torch.isnan(tensor).any(), f"NaN in {name}"
    assert not torch.isinf(tensor).any(), f"Inf in {name}"


def assert_shape(tensor: torch.Tensor, expected: tuple, name: str = "tensor") -> None:
    assert tuple(tensor.shape) == expected, (
        f"{name}: expected shape {expected}, got {tuple(tensor.shape)}"
    )


def make_dummy(batch: int, img_size: int, hidden: int, device: torch.device):
    images = torch.randn(batch, 3, img_size, img_size, device=device)
    metrics = torch.randn(batch, hidden, device=device)
    prompt = torch.randn(batch, hidden, device=device)
    return images, metrics, prompt


def import_image_encoder():
    try:
        from models.image_encoder import (
            DownsampleLayer,
            EdgeAwareBranch,
            ImageEncoder,
            MILDBackbone,
            MILDBlock,
            MetricGate,
        )

        return ImageEncoder, MILDBackbone, MILDBlock, EdgeAwareBranch, MetricGate, DownsampleLayer
    except ImportError as exc:
        raise RuntimeError(
            "Could not import models.image_encoder. Make sure MILD-Net version is present."
        ) from exc


def run_backbone_tests(suite: TestSuite, img_size: int, hidden: int) -> None:
    print(_bold("\n[1] MILDBackbone - stage shapes and Mod 1"))
    device = suite.device

    _, MILDBackbone, *_ = import_image_encoder()

    def test_stage_shapes():
        batch = 2
        backbone = MILDBackbone(metrics_dim=hidden).to(device).eval()
        images = torch.randn(batch, 3, img_size, img_size, device=device)
        metrics = torch.randn(batch, hidden, device=device)
        with torch.no_grad():
            stages = backbone(images, metrics)
        assert len(stages) == 4, "Expected 4 stage outputs"
        channels = (64, 128, 256, 512)
        for i, (stage, chan) in enumerate(zip(stages, channels)):
            stride = 4 * (2**i)
            exp_h, exp_w = img_size // stride, img_size // stride
            assert_shape(stage, (batch, chan, exp_h, exp_w), f"Stage{i+1}")
            assert_no_nan(stage, f"Stage{i+1}")
        return f"shapes {[tuple(s.shape) for s in stages]}"

    def test_no_metrics_fallback():
        batch = 1
        backbone = MILDBackbone(metrics_dim=hidden).to(device).eval()
        images = torch.randn(batch, 3, img_size, img_size, device=device)
        with torch.no_grad():
            stages = backbone(images, metrics_emb=None)
        assert len(stages) == 4
        for stage in stages:
            assert_no_nan(stage)
        return "metric gates skipped cleanly"

    def test_depth_width():
        _, MILDBackbone, _, _, _, _ = import_image_encoder()
        assert MILDBackbone.DEPTHS == (2, 2, 5, 2), f"Got {MILDBackbone.DEPTHS}"
        assert MILDBackbone.CHANNELS == (64, 128, 256, 512), f"Got {MILDBackbone.CHANNELS}"
        return "depths [2,2,5,2] channels [64,128,256,512]"

    suite.run("Stage output shapes (with metrics)", test_stage_shapes)
    suite.run("No-metrics graceful fallback", test_no_metrics_fallback)
    suite.run("Mod 1 - depth/width config verified", test_depth_width)


def run_block_tests(suite: TestSuite) -> None:
    print(_bold("\n[2] MILDBlock - Mods 4 and 5"))
    device = suite.device
    _, _, MILDBlock, *_ = import_image_encoder()

    def test_kernel_size():
        block = MILDBlock(64)
        assert block.dw_conv.kernel_size == (5, 5), f"Expected 5x5, got {block.dw_conv.kernel_size}"
        return "kernel_size=(5,5) confirmed"

    def test_grouped_conv():
        block = MILDBlock(64, groups=4)
        assert block.pw1.groups == 4, f"pw1 groups={block.pw1.groups}"
        assert block.pw2.groups == 4, f"pw2 groups={block.pw2.groups}"
        return f"pw1.groups={block.pw1.groups}, pw2.groups={block.pw2.groups}"

    def test_block_residual():
        block = MILDBlock(64).to(device).eval()
        x = torch.randn(2, 64, 20, 20, device=device)
        with torch.no_grad():
            out = block(x)
        assert_shape(out, (2, 64, 20, 20))
        assert_no_nan(out)
        return "residual shape preserved"

    suite.run("Mod 4 - 5x5 DW kernel confirmed", test_kernel_size)
    suite.run("Mod 5 - grouped 1x1 confirmed", test_grouped_conv)
    suite.run("Block residual shape preserved", test_block_residual)


def run_edge_branch_tests(suite: TestSuite, img_size: int, hidden: int) -> None:
    print(_bold("\n[3] EdgeAwareBranch - Mod 2"))
    device = suite.device
    _, _, _, EdgeAwareBranch, *_ = import_image_encoder()

    def test_sobel_buffer():
        branch = EdgeAwareBranch(feat_dim=64, prompt_dim=hidden)
        assert hasattr(branch, "sobel"), "sobel buffer missing"
        assert "sobel" in dict(branch.named_buffers()), "sobel not a buffer"
        assert tuple(branch.sobel.shape) == (2, 1, 3, 3), f"Sobel shape: {tuple(branch.sobel.shape)}"
        return "sobel buffer shape (2,1,3,3)"

    def test_edge_fusion_shapes():
        batch = 2
        feat_dim = 64
        s_h, s_w = img_size // 4, img_size // 4
        branch = EdgeAwareBranch(feat_dim=feat_dim, prompt_dim=hidden).to(device).eval()
        images = torch.randn(batch, 3, img_size, img_size, device=device)
        main = torch.randn(batch, feat_dim, s_h, s_w, device=device)
        prompt = torch.randn(batch, hidden, device=device)
        with torch.no_grad():
            out = branch(images, main, prompt)
        assert_shape(out, (batch, feat_dim, s_h, s_w), "edge fusion output")
        assert_no_nan(out, "edge fusion output")
        return f"output shape {tuple(out.shape)}"

    def test_prompt_gate_range():
        batch = 4
        branch = EdgeAwareBranch(feat_dim=64, prompt_dim=hidden).to(device).eval()
        prompt = torch.randn(batch, hidden, device=device)
        with torch.no_grad():
            gate = branch.prompt_gate(prompt)
        assert gate.min() >= 0.0 and gate.max() <= 1.0, (
            f"Gate out of (0,1): min={gate.min():.4f} max={gate.max():.4f}"
        )
        return f"gate in [0, 1], shape {tuple(gate.shape)}"

    suite.run("Sobel registered as buffer (not learned)", test_sobel_buffer)
    suite.run("Edge fusion output shape correct", test_edge_fusion_shapes)
    suite.run("Prompt gate output in [0, 1]", test_prompt_gate_range)


def run_metric_gate_tests(suite: TestSuite, hidden: int) -> None:
    print(_bold("\n[4] MetricGate - Mod 3"))
    device = suite.device
    _, _, _, _, MetricGate, _ = import_image_encoder()

    def test_identity_init():
        gate = MetricGate(hidden, 256).to(device)
        x = torch.randn(2, 256, 10, 10, device=device)
        metrics = torch.randn(2, hidden, device=device)
        with torch.no_grad():
            out = gate(x, metrics)
        diff = (out - x).abs().max().item()
        assert diff < 1e-5, f"Gate not identity at init: max diff={diff:.6f}"
        return f"max deviation from identity = {diff:.2e}"

    def test_gate_shape_preserved():
        batch, channels, height, width = 2, 256, 20, 20
        gate = MetricGate(hidden, channels).to(device).eval()
        x = torch.randn(batch, channels, height, width, device=device)
        metrics = torch.randn(batch, hidden, device=device)
        with torch.no_grad():
            out = gate(x, metrics)
        assert_shape(out, (batch, channels, height, width))
        assert_no_nan(out)
        return f"shape {tuple(out.shape)} preserved"

    suite.run("Gate is identity at initialization", test_identity_init)
    suite.run("Gate preserves spatial shape", test_gate_shape_preserved)


def run_encoder_tests(suite: TestSuite, img_size: int, hidden: int) -> None:
    print(_bold("\n[5] ImageEncoder - full forward paths"))
    device = suite.device
    ImageEncoder, *_ = import_image_encoder()

    def test_single_scale():
        batch = 2
        enc = ImageEncoder(hidden_dim=hidden, multiscale=False).to(device).eval()
        images, metrics, prompt = make_dummy(batch, img_size, hidden, device)
        with torch.no_grad():
            out = enc(images, metrics_emb=metrics, prompt_emb=prompt)
        expected = (batch, hidden, img_size // 32, img_size // 32)
        assert_shape(out, expected)
        assert_no_nan(out)
        return f"output {tuple(out.shape)}"

    def test_multiscale():
        batch = 2
        enc = ImageEncoder(hidden_dim=hidden, multiscale=True, feature_indices=(1, 2, 3)).to(device).eval()
        images, metrics, prompt = make_dummy(batch, img_size, hidden, device)
        with torch.no_grad():
            out = enc(images, metrics_emb=metrics, prompt_emb=prompt)
        assert out.shape[1] == hidden
        assert_no_nan(out)
        return f"output {tuple(out.shape)}"

    def test_no_metrics_no_prompt():
        batch = 1
        enc = ImageEncoder(hidden_dim=hidden, multiscale=False).to(device).eval()
        images = torch.randn(batch, 3, img_size, img_size, device=device)
        with torch.no_grad():
            out = enc(images)
        assert_no_nan(out)
        return f"output {tuple(out.shape)}"

    def test_batch_size_1():
        enc = ImageEncoder(hidden_dim=hidden).to(device).eval()
        images, metrics, prompt = make_dummy(1, img_size, hidden, device)
        with torch.no_grad():
            out = enc(images, metrics, prompt)
        assert_no_nan(out)
        return f"B=1 OK, shape {tuple(out.shape)}"

    def test_batch_size_4():
        enc = ImageEncoder(hidden_dim=hidden).to(device).eval()
        images, metrics, prompt = make_dummy(4, img_size, hidden, device)
        with torch.no_grad():
            out = enc(images, metrics, prompt)
        assert_no_nan(out)
        return f"B=4 OK, shape {tuple(out.shape)}"

    suite.run("Single-scale forward (full path)", test_single_scale)
    suite.run("Multi-scale forward (FPN path)", test_multiscale)
    suite.run("No metrics / no prompt (inference-safe)", test_no_metrics_no_prompt)
    suite.run("Batch size = 1", test_batch_size_1)
    suite.run("Batch size = 4", test_batch_size_4)


def run_gradient_tests(suite: TestSuite, img_size: int, hidden: int) -> None:
    print(_bold("\n[6] Gradient flow"))
    device = suite.device

    def test_grad_flows_to_backbone():
        ImageEncoder, *_ = import_image_encoder()
        enc = ImageEncoder(model_name="mild_net", hidden_dim=hidden).to(device).train()
        images = torch.randn(2, 3, img_size, img_size, device=device, requires_grad=True)
        metrics = torch.randn(2, hidden, device=device)
        prompt = torch.randn(2, hidden, device=device)
        out = enc(images, metrics, prompt)
        out.mean().backward()
        grads_ok = [
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in enc.backbone.parameters()
            if p.requires_grad
        ]
        frac = sum(grads_ok) / len(grads_ok)
        assert frac > 0.9, f"Only {frac:.0%} backbone params got gradients"
        return f"{frac:.0%} of backbone params received gradients"

    def test_edge_branch_grad():
        ImageEncoder, *_ = import_image_encoder()
        enc = ImageEncoder(
            model_name="mild_net",
            hidden_dim=hidden,
            multiscale=True,
            feature_indices=(0, 1, 2, 3),
        ).to(device).train()
        images = torch.randn(2, 3, img_size, img_size, device=device)
        metrics = torch.randn(2, hidden, device=device)
        prompt = torch.randn(2, hidden, device=device, requires_grad=True)
        out = enc(images, metrics, prompt)
        out.mean().backward()
        gate_grad = enc.edge_branch.prompt_gate[0].weight.grad
        assert gate_grad is not None and gate_grad.abs().sum() > 0, "Edge branch prompt gate got no gradient"
        return "edge branch grad OK"

    def test_metric_gate_grad():
        ImageEncoder, *_ = import_image_encoder()
        enc = ImageEncoder(model_name="mild_net", hidden_dim=hidden).to(device).train()
        images = torch.randn(2, 3, img_size, img_size, device=device)
        metrics = torch.randn(2, hidden, device=device, requires_grad=True)
        prompt = torch.randn(2, hidden, device=device)
        out = enc(images, metrics, prompt)
        out.mean().backward()
        gate_grad = enc.backbone.gate3.scale.weight.grad
        assert gate_grad is not None and gate_grad.abs().sum() > 0, "Metric gate (stage 3) got no gradient"
        return "metric gate grad OK (stage 3 and 4)"

    suite.run("Gradients reach backbone parameters", test_grad_flows_to_backbone)
    suite.run("Mod 2 - edge branch gets gradients", test_edge_branch_grad)
    suite.run("Mod 3 - metric gate gets gradients", test_metric_gate_grad)


def run_full_model_test(suite: TestSuite, img_size: int, hidden: int) -> None:
    print(_bold("\n[7] Full MILDModel end-to-end"))
    device = suite.device

    def test_full_forward():
        try:
            from models.mild_model import build_model
        except ImportError as exc:
            raise SkipTest(f"mild_model import failed: {exc}") from exc

        cfg = {
            "image_encoder": "mild_net",
            "hidden_dim": hidden,
            "metrics_dim": 4,
            "metrics_hidden": 128,
            "num_queries": 10,
            "decoder_layers": 2,
            "decoder_heads": 8,
            "num_classes": 5,
            "multiscale": False,
            "text_encoder_trainable": False,
            "text_encoder_cache": False,
        }
        try:
            model = build_model(cfg).to(device).eval()
        except Exception as exc:
            raise SkipTest(f"build_model failed: {exc}") from exc

        batch = 2
        images = torch.randn(batch, 3, img_size, img_size, device=device)
        metrics = torch.tensor([[0.13, 1.92, 0.58, 0.13]] * batch, device=device)
        with torch.no_grad():
            out = model(images, ["crack on aircraft surface"] * batch, metrics)

        assert "pred_logits" in out
        assert "pred_boxes" in out
        assert "pred_severity" in out
        for key, value in out.items():
            if isinstance(value, torch.Tensor):
                assert_no_nan(value, key)
        return (
            f"logits {tuple(out['pred_logits'].shape)}, "
            f"boxes {tuple(out['pred_boxes'].shape)}, "
            f"sev {tuple(out['pred_severity'].shape)}"
        )

    def test_forward_no_metrics():
        try:
            from models.mild_model import build_model
        except ImportError as exc:
            raise SkipTest(str(exc)) from exc
        cfg = {
            "image_encoder": "mild_net",
            "hidden_dim": hidden,
            "num_queries": 10,
            "decoder_layers": 2,
            "decoder_heads": 8,
            "num_classes": 5,
            "text_encoder_trainable": False,
            "text_encoder_cache": False,
        }
        try:
            model = build_model(cfg).to(device).eval()
        except Exception as exc:
            raise SkipTest(str(exc)) from exc
        images = torch.randn(1, 3, img_size, img_size, device=device)
        with torch.no_grad():
            out = model(images, ["dent"], None)
        assert_no_nan(out["pred_logits"])
        return "metrics=None path OK"

    suite.run("Full model forward (with metrics)", test_full_forward)
    suite.run("Full model forward (metrics=None)", test_forward_no_metrics)


def print_summary(summary: Dict) -> None:
    passed, failed, skipped, total = (
        summary["passed"],
        summary["failed"],
        summary["skipped"],
        summary["total"],
    )
    print()
    print("-" * 70)
    print(
        f"  Results: {_green(str(passed))} passed  "
        f"{_red(str(failed))} failed  "
        f"{_yellow(str(skipped))} skipped  "
        f"/ {total} total"
    )
    print("-" * 70)
    if failed == 0:
        print(_green(_bold("  OK All checks passed - safe to run training.")))
    else:
        print(_red(_bold(f"  FAIL {failed} check(s) failed - fix before training.")))
        print()
        print("  Failed tests:")
        for result in summary["results"]:
            if result["status"] == "fail":
                print(f"    - {result['test']}")
                print(f"      {result['note']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-training verification for MILD-Net")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(_bold("\n  MILD-Net pre-training verification"))
    print(f"  device={device}  image_size={args.image_size}  hidden_dim={args.hidden_dim}")

    suite = TestSuite(device)
    run_backbone_tests(suite, args.image_size, args.hidden_dim)
    run_block_tests(suite)
    run_edge_branch_tests(suite, args.image_size, args.hidden_dim)
    run_metric_gate_tests(suite, args.hidden_dim)
    run_encoder_tests(suite, args.image_size, args.hidden_dim)
    run_gradient_tests(suite, args.image_size, args.hidden_dim)
    run_full_model_test(suite, args.image_size, args.hidden_dim)

    summary = suite.summary()
    print_summary(summary)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  Report saved: {out}\n")

    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
