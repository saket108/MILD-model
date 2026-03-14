from __future__ import annotations

"""
models/udcm.py
--------------
Unified Damage Context Module (UDCM).

Replaces PromptGuidedFusion + TransformerDecoderModule with a single
task-aware module. The key novelty over standard DETR-style decoders:

  Standard DETR:  fixed random queries → cross-attend to visual memory
  UDCM:           prompt+metrics → condition queries → cross-attend to
                  (visual + context) memory jointly

This means the queries already encode what kind of damage to look for
before attending to visual features, rather than learning this purely
from data over many epochs.

Tensor contract:
  Input:
    visual      [B, C, H, W]   from ImageEncoder
    text        [B, C]         from TextEncoder
    metrics_emb [B, C] | None  from MetricsEncoder
  Output:
    tokens      [B, Q, C]      fed directly into DetectorHead + SeverityHead
"""

import torch
import torch.nn as nn
from torch import Tensor


# ── 2-D sin-cos positional encoding ──────────────────────────────────────────

def _sincos_pos_embed(h: int, w: int, dim: int,
                      device: torch.device, dtype: torch.dtype) -> Tensor:
    if dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by 4.")
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    omega = 1.0 / (10000.0 ** (
        torch.arange(dim // 4, device=device, dtype=dtype) / (dim // 4)
    ))
    ox = gx[:, :, None] * omega[None, None, :]
    oy = gy[:, :, None] * omega[None, None, :]
    pos = torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], dim=-1)
    return pos.reshape(1, h * w, dim)


# ── Component 1: ContextTokenizer ────────────────────────────────────────────

class ContextTokenizer(nn.Module):
    """
    Projects text + metrics into 2 context tokens [B, 2, C].

    Token 0 = semantic context  (damage type / description)
    Token 1 = metric context    (area, elongation, edge factor, severity)

    Degrades gracefully when metrics_emb is None — uses a learned fallback
    vector so inference without numeric metrics still works.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.metrics_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.metrics_fallback = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, text: Tensor, metrics_emb: Tensor | None) -> Tensor:
        tok_text = self.text_proj(text).unsqueeze(1)               # [B, 1, C]
        if metrics_emb is not None:
            tok_met = self.metrics_proj(metrics_emb).unsqueeze(1)  # [B, 1, C]
        else:
            tok_met = self.metrics_fallback.expand(text.size(0), -1, -1)
        return torch.cat([tok_text, tok_met], dim=1)               # [B, 2, C]


# ── Component 2: PromptConditionedQueryInit ───────────────────────────────────

class PromptConditionedQueryInit(nn.Module):
    """
    Generates object queries conditioned on context tokens.

    DETR:  queries = fixed Embedding(Q, C)
    UDCM:  queries = base_embed + cross_attn(base_embed, context_tokens)

    Each query is shifted toward the damage type described by the prompt
    and metrics before any visual attention occurs. The base_embed is
    still fully learned — conditioning adds one MHA layer only.

    Output: [B, Q, C]
    """

    def __init__(self, hidden_dim: int, num_queries: int,
                 num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.base_embed = nn.Embedding(num_queries, hidden_dim)
        self.cond_attn  = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_tokens: Tensor) -> Tensor:
        B = context_tokens.size(0)
        base = self.base_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        cond, _ = self.cond_attn(base, context_tokens, context_tokens)
        return self.norm(base + cond)                                  # [B, Q, C]


# ── Component 3: UnifiedAttentionLayer ───────────────────────────────────────

class UnifiedAttentionLayer(nn.Module):
    """
    Transformer decoder layer where queries attend to unified memory:
    [visual tokens | context tokens] concatenated.

    Standard DETR decoder: queries cross-attend to visual memory only.
    UDCM:                   queries cross-attend to visual + context jointly.

    This lets the model re-attend to the text/metric signal at every
    layer without a separate fusion step upstream.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, queries: Tensor, unified_memory: Tensor) -> Tensor:
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.drop(q2))
        q3, _ = self.cross_attn(queries, unified_memory, unified_memory)
        queries = self.norm2(queries + self.drop(q3))
        queries = self.norm3(queries + self.drop(self.ffn(queries)))
        return queries


# ── UDCM: top-level module ────────────────────────────────────────────────────

class UDCM(nn.Module):
    """
    Unified Damage Context Module.

    Drop-in replacement for PromptGuidedFusion + TransformerDecoderModule.
    mild_model.py changes: swap self.fusion + self.decoder → self.udcm,
    and update the forward() call (see mild_model.py comments).

    Param comparison (hidden_dim=256, num_queries=20, num_layers=2):
      PromptGuidedFusion:        ~329K
      TransformerDecoderModule:  ~4.8M
      Total replaced:            ~5.1M

      UDCM:                      ~1.9M   (-63% vs what it replaces)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 20,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
        use_positional_encoding: bool = True,
        return_intermediate: bool = False,  # API compat — always returns final only
    ) -> None:
        super().__init__()
        self.use_positional_encoding = use_positional_encoding

        self.context_tokenizer = ContextTokenizer(hidden_dim)
        self.query_init = PromptConditionedQueryInit(
            hidden_dim, num_queries, num_heads, dropout
        )
        self.layers = nn.ModuleList([
            UnifiedAttentionLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual: Tensor,
        text: Tensor,
        metrics_emb: Tensor | None = None,
    ) -> Tensor:
        """
        visual:      [B, C, H, W]
        text:        [B, C]
        metrics_emb: [B, C] or None
        Returns:     [B, Q, C]
        """
        B, C, H, W = visual.shape

        # 1. Visual tokens + 2-D positional encoding
        vis_tokens = visual.flatten(2).transpose(1, 2)       # [B, HW, C]
        if self.use_positional_encoding:
            pos = _sincos_pos_embed(H, W, C, visual.device, visual.dtype)
            vis_tokens = vis_tokens + pos

        # 2. Context tokens from text + metrics  [B, 2, C]
        ctx_tokens = self.context_tokenizer(text, metrics_emb)

        # 3. Unified memory: visual + context  [B, HW+2, C]
        memory = torch.cat([vis_tokens, ctx_tokens], dim=1)

        # 4. Prompt-conditioned query initialisation  [B, Q, C]
        queries = self.query_init(ctx_tokens)

        # 5. Stacked unified attention
        for layer in self.layers:
            queries = layer(queries, memory)

        return self.out_norm(queries)                         # [B, Q, C]
