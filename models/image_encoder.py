from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import timm


def _sobel_kernel() -> Tensor:
    gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    gy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    return torch.stack([gx, gy]).unsqueeze(1)


class MILDBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 4, groups: int = 4) -> None:
        super().__init__()
        inner = dim * expand
        g = min(groups, dim)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv2d(dim, inner, kernel_size=1, groups=g)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(inner, dim, kernel_size=1, groups=g)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x + residual


class DownsampleLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.conv(x)


class MetricGate(nn.Module):
    def __init__(self, metrics_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(metrics_dim, feat_dim)
        self.shift = nn.Linear(metrics_dim, feat_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: Tensor, metrics_emb: Tensor) -> Tensor:
        scale = self.scale(metrics_emb).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(metrics_emb).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift


class EdgeAwareBranch(nn.Module):
    def __init__(self, feat_dim: int, prompt_dim: int) -> None:
        super().__init__()
        self.register_buffer("sobel", _sobel_kernel())
        self.edge_conv = nn.Sequential(
            nn.Conv2d(2, feat_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim),
        )
        self.norm = nn.GroupNorm(1, feat_dim)
        self.prompt_gate = nn.Sequential(
            nn.Linear(prompt_dim, feat_dim),
            nn.Sigmoid(),
        )

    def _edge_map(self, images: Tensor) -> Tensor:
        gray = images.mean(dim=1, keepdim=True)
        sobel = self.sobel.to(dtype=gray.dtype, device=gray.device)
        return F.conv2d(gray, sobel, padding=1)

    def forward(self, images: Tensor, main_feat: Tensor, prompt_emb: Tensor) -> Tensor:
        edges = self._edge_map(images)
        edges = F.interpolate(edges, size=main_feat.shape[-2:], mode="bilinear", align_corners=False)
        edge_feat = self.norm(self.edge_conv(edges))
        gate = self.prompt_gate(prompt_emb).unsqueeze(-1).unsqueeze(-1)
        return main_feat + edge_feat * gate


class MILDBackbone(nn.Module):
    DEPTHS = (2, 2, 5, 2)
    CHANNELS = (64, 128, 256, 512)

    def __init__(self, metrics_dim: int = 256) -> None:
        super().__init__()
        c = self.CHANNELS
        d = self.DEPTHS

        self.stem = nn.Sequential(
            nn.Conv2d(3, c[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, c[0]),
        )

        self.stage1 = self._make_stage(c[0], d[0])
        self.down1 = DownsampleLayer(c[0], c[1])
        self.stage2 = self._make_stage(c[1], d[1])
        self.down2 = DownsampleLayer(c[1], c[2])
        self.stage3 = self._make_stage(c[2], d[2])
        self.down3 = DownsampleLayer(c[2], c[3])
        self.stage4 = self._make_stage(c[3], d[3])

        self.gate3 = MetricGate(metrics_dim, c[2])
        self.gate4 = MetricGate(metrics_dim, c[3])

    @staticmethod
    def _make_stage(dim: int, depth: int) -> nn.Sequential:
        return nn.Sequential(*[MILDBlock(dim) for _ in range(depth)])

    def forward(self, x: Tensor, metrics_emb: Tensor | None = None) -> list[Tensor]:
        x = self.stem(x)

        x = self.stage1(x)
        s1 = x
        x = self.down1(x)

        x = self.stage2(x)
        s2 = x
        x = self.down2(x)

        x = self.stage3(x)
        if metrics_emb is not None:
            x = self.gate3(x, metrics_emb)
        s3 = x
        x = self.down3(x)

        x = self.stage4(x)
        if metrics_emb is not None:
            x = self.gate4(x, metrics_emb)
        s4 = x
        return [s1, s2, s3, s4]


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        hidden_dim: int = 256,
        multiscale: bool = False,
        feature_indices: tuple[int, ...] | None = None,
        metrics_dim: int | None = None,
        prompt_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.multiscale = multiscale
        self.metrics_dim = metrics_dim or hidden_dim
        self.prompt_dim = prompt_dim or hidden_dim

        if model_name.lower() in {"mild", "mild_net", "mild_backbone"}:
            self.mode = "mild"
            self.backbone = MILDBackbone(metrics_dim=self.metrics_dim)
            self.edge_branch = EdgeAwareBranch(feat_dim=MILDBackbone.CHANNELS[2], prompt_dim=self.prompt_dim)

            if self.multiscale:
                use = list(feature_indices or (1, 2, 3))
                chs = [MILDBackbone.CHANNELS[i] for i in use]
                self.lateral = nn.ModuleList([nn.Conv2d(c, hidden_dim, kernel_size=1) for c in chs])
                self.out_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
                self._use_idx = use
            else:
                self.proj = nn.Conv2d(MILDBackbone.CHANNELS[-1], hidden_dim, kernel_size=1)
            return

        self.mode = "timm"
        if self.multiscale:
            out_indices = feature_indices or (1, 2, 3)
        else:
            out_indices = (-1,)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = self.backbone.feature_info.channels()
        if self.multiscale:
            self.lateral = nn.ModuleList([nn.Conv2d(c, hidden_dim, kernel_size=1) for c in channels])
            self.out_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        else:
            self.proj = nn.Conv2d(channels[-1], hidden_dim, kernel_size=1)

    def forward(
        self,
        images: Tensor,
        metrics_emb: Tensor | None = None,
        prompt_emb: Tensor | None = None,
    ) -> Tensor:
        if self.mode == "timm":
            feats = self.backbone(images)
            if not self.multiscale:
                return self.proj(feats[-1])
            projected = [conv(f) for conv, f in zip(self.lateral, feats)]
            fused = projected[-1]
            for feat in reversed(projected[:-1]):
                fused = F.interpolate(fused, size=feat.shape[-2:], mode="nearest")
                fused = fused + feat
            return self.out_conv(fused)

        feats = self.backbone(images, metrics_emb)
        if prompt_emb is not None:
            feats[2] = self.edge_branch(images, feats[2], prompt_emb)

        if not self.multiscale:
            return self.proj(feats[-1])

        projected = [conv(feats[i]) for conv, i in zip(self.lateral, self._use_idx)]
        fused = projected[-1]
        for feat in reversed(projected[:-1]):
            fused = F.interpolate(fused, size=feat.shape[-2:], mode="nearest")
            fused = fused + feat
        return self.out_conv(fused)
