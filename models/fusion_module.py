from __future__ import annotations

import torch
from torch import nn


class PromptGuidedFusion(nn.Module):
    """Cross-attention fusion between visual tokens and text embedding."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.metrics_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, visual: torch.Tensor, text: torch.Tensor, metrics: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = visual.shape
        tokens = visual.flatten(2).transpose(1, 2)  # [B, HW, C]
        text_token = text.unsqueeze(1)  # [B, 1, C]
        attn_out, _ = self.attn(query=tokens, key=text_token, value=text_token)
        fused = self.norm(tokens + attn_out)
        if metrics is not None:
            gate = self.metrics_gate(metrics).unsqueeze(1)  # [B, 1, C]
            fused = fused * (1.0 + gate)
        return fused.transpose(1, 2).reshape(b, c, h, w)
