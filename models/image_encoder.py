from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn


class ImageEncoder(nn.Module):
    """ConvNeXt image encoder wrapper."""

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        hidden_dim: int = 256,
        multiscale: bool = False,
        feature_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.multiscale = multiscale
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
            in_channels = channels[-1]
            self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        if not self.multiscale:
            return self.proj(feats[-1])

        projected = [conv(f) for conv, f in zip(self.lateral, feats)]
        fused = projected[-1]
        for feat in reversed(projected[:-1]):
            fused = F.interpolate(fused, size=feat.shape[-2:], mode="nearest")
            fused = fused + feat
        return self.out_conv(fused)
