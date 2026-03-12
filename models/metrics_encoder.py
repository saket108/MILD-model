from __future__ import annotations

import torch
from torch import nn


class MetricsEncoder(nn.Module):
    """Encode structured numeric metrics into the hidden dimension."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, output_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, metrics: torch.Tensor) -> torch.Tensor:
        return self.net(metrics)
