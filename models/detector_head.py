from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DetectorHead(nn.Module):
    """Predicts class logits and normalized boxes (cx, cy, w, h)."""

    def __init__(self, hidden_dim: int = 256, num_classes: int = 5) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bbox_mlp = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(x)
        boxes = self.bbox_mlp(x).sigmoid()
        return logits, boxes
