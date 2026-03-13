from __future__ import annotations

import math
import torch
from torch import nn


def _build_2d_sincos_pos_embed(
    h: int,
    w: int,
    dim: int,
    temperature: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by 4 for 2D sin-cos positional encoding.")
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    omega = torch.arange(dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (temperature ** (omega / (dim // 4)))

    out_x = grid_x[:, :, None] * omega[None, None, :]
    out_y = grid_y[:, :, None] * omega[None, None, :]

    pos = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=-1)
    pos = pos.reshape(h * w, dim)
    return pos.unsqueeze(0)


class TransformerDecoderModule(nn.Module):
    """DETR-style transformer decoder with learnable object queries."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)
        self.use_positional_encoding = use_positional_encoding
        self.return_intermediate = return_intermediate

    def forward(self, memory: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        b, c, h, w = memory.shape
        memory_tokens = memory.flatten(2).transpose(1, 2)  # [B, HW, C]
        if self.use_positional_encoding:
            pos = _build_2d_sincos_pos_embed(h, w, c, device=memory.device, dtype=memory.dtype)
            memory_tokens = memory_tokens + pos

        query = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        query_pos = self.query_pos.weight.unsqueeze(0).repeat(b, 1, 1)
        query = query + query_pos

        intermediates = []
        output = query
        for layer in self.layers:
            output = layer(tgt=output, memory=memory_tokens)
            if self.return_intermediate:
                intermediates.append(output)

        if self.return_intermediate:
            return intermediates
        return output
