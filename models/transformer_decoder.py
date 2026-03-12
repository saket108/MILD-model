from __future__ import annotations

import torch
from torch import nn


class TransformerDecoderModule(nn.Module):
    """DETR-style transformer decoder with learnable object queries."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        b, c, h, w = memory.shape
        memory_tokens = memory.flatten(2).transpose(1, 2)  # [B, HW, C]
        query = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        return self.decoder(tgt=query, memory=memory_tokens)
