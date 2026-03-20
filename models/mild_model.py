from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from models.image_encoder import ImageEncoder
from models.metrics_encoder import MetricsEncoder
from models.severity_head import SeverityHead
from models.text_encoder import TextEncoder
from models.udcm import UDCM
from models.detector_head import DetectorHead


class MILDModel(nn.Module):
    """
    MILD multimodal detector — Option C architecture.

    Forward flow:
      1. text_encoder    → text embedding      [B, C]
      2. metrics_encoder → metrics embedding   [B, C]
      3. image_encoder   → visual features     [B, C, H, W]
         (Mods 2 & 3 inside backbone use text + metrics)
      4. UDCM            → query tokens        [B, Q, C]
         (replaces PromptGuidedFusion + TransformerDecoderModule)
      5. detector_head   → logits + boxes
      6. severity_head   → severity scores
    """

    def __init__(
        self,
        image_encoder: str = "mild_net",
        text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        metrics_dim: int = 4,
        metrics_hidden: int = 128,
        num_queries: int = 20,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        num_classes: int = 5,
        multiscale: bool = False,
        feature_indices: tuple[int, ...] | None = None,
        use_positional_encoding: bool = True,
        return_intermediate: bool = False,
        text_trainable: bool = False,
        text_cache: bool = True,
        text_cache_max_size: int = 4096,
        use_text: bool = True,
        use_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_text = use_text
        self.use_metrics = use_metrics

        self.text_encoder = None
        if self.use_text:
            self.text_encoder = TextEncoder(
                text_encoder,
                hidden_dim=hidden_dim,
                trainable=text_trainable,
                cache=text_cache,
                cache_max_size=text_cache_max_size,
            )

        self.metrics_encoder = None
        if self.use_metrics:
            self.metrics_encoder = MetricsEncoder(
                input_dim=metrics_dim,
                hidden_dim=metrics_hidden,
                output_dim=hidden_dim,
            )
        self.image_encoder = ImageEncoder(
            image_encoder,
            pretrained=True,
            hidden_dim=hidden_dim,
            multiscale=multiscale,
            feature_indices=feature_indices,
            metrics_dim=hidden_dim,
            prompt_dim=hidden_dim,
        )
        # UDCM replaces both PromptGuidedFusion and TransformerDecoderModule
        self.udcm = UDCM(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            use_positional_encoding=use_positional_encoding,
            return_intermediate=return_intermediate,
        )
        self.head          = DetectorHead(hidden_dim=hidden_dim, num_classes=num_classes)
        self.severity_head = SeverityHead(hidden_dim=hidden_dim)

    def forward(
        self,
        images: torch.Tensor,
        prompts: List[str] | List[List[str]],
        metrics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:

        # Modality encoders
        if self.use_text and self.text_encoder is not None:
            text = self.text_encoder(prompts)
            prompt_emb = text
        else:
            text = torch.zeros((images.shape[0], self.hidden_dim), device=images.device, dtype=images.dtype)
            prompt_emb = None

        if self.use_metrics and self.metrics_encoder is not None and metrics is not None:
            metrics_emb = self.metrics_encoder(metrics)
        else:
            metrics_emb = None

        # MILD-Net backbone (Mods 2 & 3 active when text/metrics provided)
        visual = self.image_encoder(images, metrics_emb=metrics_emb, prompt_emb=prompt_emb)

        # UDCM: single unified pass → query tokens [B, Q, C]
        tokens = self.udcm(visual, text, metrics_emb)

        # Prediction heads (unchanged)
        pred_logits, pred_boxes = self.head(tokens)
        pred_severity = self.severity_head(tokens).squeeze(-1)

        return {
            "pred_logits":   pred_logits,
            "pred_boxes":    pred_boxes,
            "pred_severity": pred_severity,
            "aux_outputs":   [],           # no aux outputs (return_intermediate=False)
        }


def build_model(cfg: Dict) -> MILDModel:
    return MILDModel(
        image_encoder=cfg.get("image_encoder", "mild_net"),
        text_encoder=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        hidden_dim=cfg.get("hidden_dim", 256),
        metrics_dim=cfg.get("metrics_dim", 4),
        metrics_hidden=cfg.get("metrics_hidden", 128),
        num_queries=cfg.get("num_queries", 20),
        decoder_layers=cfg.get("decoder_layers", 2),
        decoder_heads=cfg.get("decoder_heads", 8),
        num_classes=cfg.get("num_classes", 5),
        multiscale=cfg.get("multiscale", False),
        feature_indices=tuple(cfg.get("feature_indices", [])) or None,
        use_positional_encoding=cfg.get("use_positional_encoding", True),
        return_intermediate=cfg.get("return_intermediate", False),
        text_trainable=cfg.get("text_encoder_trainable", False),
        text_cache=cfg.get("text_encoder_cache", True),
        text_cache_max_size=cfg.get("text_encoder_cache_max_size", 4096),
        use_text=cfg.get("use_text", True),
        use_metrics=cfg.get("use_metrics", True),
    )
