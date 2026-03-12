from __future__ import annotations

import timm
import torch
from torch import nn


class ImageEncoder(nn.Module):
    """ConvNeXt image encoder wrapper."""

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        in_channels = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)[-1]
        return self.proj(features)
