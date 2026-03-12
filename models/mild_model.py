from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from models.image_encoder import ImageEncoder
from models.metrics_encoder import MetricsEncoder
from models.severity_head import SeverityHead
from models.text_encoder import TextEncoder
from models.fusion_module import PromptGuidedFusion
from models.transformer_decoder import TransformerDecoderModule
from models.detector_head import DetectorHead


class MILDModel(nn.Module):
    """Main model: image encoder + text encoder + fusion + decoder + head."""

    def __init__(
        self,
        image_encoder: str = "convnext_tiny",
        text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        metrics_dim: int = 4,
        metrics_hidden: int = 128,
        num_queries: int = 100,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(image_encoder, pretrained=True, hidden_dim=hidden_dim)
        self.text_encoder = TextEncoder(text_encoder, hidden_dim=hidden_dim)
        self.metrics_encoder = MetricsEncoder(
            input_dim=metrics_dim,
            hidden_dim=metrics_hidden,
            output_dim=hidden_dim,
        )
        self.fusion = PromptGuidedFusion(hidden_dim=hidden_dim, num_heads=decoder_heads)
        self.decoder = TransformerDecoderModule(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
        )
        self.head = DetectorHead(hidden_dim=hidden_dim, num_classes=num_classes)
        self.severity_head = SeverityHead(hidden_dim=hidden_dim)

    def forward(
        self,
        images: torch.Tensor,
        prompts: List[str] | List[List[str]],
        metrics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        visual = self.image_encoder(images)
        text = self.text_encoder(prompts)
        metrics_emb = self.metrics_encoder(metrics) if metrics is not None else None
        fused = self.fusion(visual, text, metrics_emb)
        tokens = self.decoder(fused)
        pred_logits, pred_boxes = self.head(tokens)
        pred_severity = self.severity_head(tokens).squeeze(-1)
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_severity": pred_severity,
        }


def build_model(cfg: Dict) -> MILDModel:
    return MILDModel(
        image_encoder=cfg.get("image_encoder", "convnext_tiny"),
        text_encoder=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        hidden_dim=cfg.get("hidden_dim", 256),
        metrics_dim=cfg.get("metrics_dim", 4),
        metrics_hidden=cfg.get("metrics_hidden", 128),
        num_queries=cfg.get("num_queries", 100),
        decoder_layers=cfg.get("decoder_layers", 3),
        decoder_heads=cfg.get("decoder_heads", 8),
        num_classes=cfg.get("num_classes", 5),
    )
