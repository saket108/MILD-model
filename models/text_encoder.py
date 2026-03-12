from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """MiniLM text encoder with mean pooling."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hidden_dim: int = 256) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.model.config.hidden_size, hidden_dim)
        self.hidden_dim = hidden_dim

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = self.model(**tokens)
        last_hidden = outputs.last_hidden_state
        attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1.0)
        return self.proj(pooled)

    def forward(self, texts: List[str] | List[List[str]]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.hidden_dim), device=next(self.parameters()).device)

        if isinstance(texts[0], (list, tuple)):
            batch_prompts = texts  # type: ignore[assignment]
            lengths = [len(p) for p in batch_prompts]
            flat = [p for prompts in batch_prompts for p in prompts]
            if not flat:
                return torch.zeros((len(batch_prompts), self.hidden_dim), device=next(self.parameters()).device)

            flat_embeddings = self._encode(flat)
            embeddings = []
            idx = 0
            for length in lengths:
                if length == 0:
                    embeddings.append(torch.zeros((self.hidden_dim,), device=flat_embeddings.device))
                else:
                    embeddings.append(flat_embeddings[idx : idx + length].mean(dim=0))
                idx += length
            return torch.stack(embeddings, dim=0)

        return self._encode(texts)  # type: ignore[arg-type]
