from __future__ import annotations

from typing import Dict, List

import torch

from losses.matcher import HungarianMatcher
from losses.box_losses import box_cxcywh_to_xyxy, l1_loss, giou_loss
from losses.focal_loss import sigmoid_focal_loss
from damage_assessment.severity_score import severity_from_cxcywh
from losses.severity_loss import severity_loss


class TotalLoss:
    """Combines bbox, giou, and classification loss."""

    def __init__(self, matcher: HungarianMatcher, num_classes: int, weight_dict: Dict[str, float] | None = None) -> None:
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict or {
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_cls": 1.0,
            "loss_severity": 1.0,
            "loss_aux": 0.5,
        }

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_queries, _ = outputs["pred_logits"].shape
        target_classes = torch.zeros(batch_size, num_queries, self.num_classes, device=outputs["pred_logits"].device)

        losses = {
            "loss_bbox": torch.tensor(0.0, device=outputs["pred_logits"].device),
            "loss_giou": torch.tensor(0.0, device=outputs["pred_logits"].device),
            "loss_cls": torch.tensor(0.0, device=outputs["pred_logits"].device),
            "loss_severity": torch.tensor(0.0, device=outputs["pred_logits"].device),
        }

        for b, (idx_pred, idx_tgt) in enumerate(indices):
            if idx_pred.numel() == 0:
                continue
            tgt_labels = targets[b]["labels"][idx_tgt]
            target_classes[b, idx_pred, tgt_labels] = 1.0

            pred_boxes = outputs["pred_boxes"][b, idx_pred]
            tgt_boxes = targets[b]["boxes"][idx_tgt]
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
            tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)

            losses["loss_bbox"] = losses["loss_bbox"] + l1_loss(pred_boxes, tgt_boxes)
            losses["loss_giou"] = losses["loss_giou"] + giou_loss(pred_boxes_xyxy, tgt_boxes_xyxy)

            if "pred_severity" in outputs:
                pred_sev = outputs["pred_severity"][b, idx_pred]
                if pred_sev.dim() > 1:
                    pred_sev = pred_sev.squeeze(-1)

                target_sev = targets[b].get("severity")
                if target_sev is None or target_sev.numel() == 0:
                    target_sev = severity_from_cxcywh(tgt_boxes)
                else:
                    target_sev = target_sev[idx_tgt]
                losses["loss_severity"] = losses["loss_severity"] + severity_loss(pred_sev, target_sev)

        losses["loss_cls"] = sigmoid_focal_loss(outputs["pred_logits"], target_classes)
        return losses

    def __call__(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        indices = self.matcher(outputs, targets)

        losses = self._compute_losses(outputs, targets, indices)
        total = (
            self.weight_dict["loss_bbox"] * losses["loss_bbox"]
            + self.weight_dict["loss_giou"] * losses["loss_giou"]
            + self.weight_dict["loss_cls"] * losses["loss_cls"]
            + self.weight_dict["loss_severity"] * losses["loss_severity"]
        )

        aux_outputs = outputs.get("aux_outputs") or []
        if aux_outputs:
            aux_total = torch.tensor(0.0, device=outputs["pred_logits"].device)
            for i, aux in enumerate(aux_outputs):
                aux_losses = self._compute_losses(aux, targets, indices)
                aux_total = aux_total + (
                    self.weight_dict["loss_bbox"] * aux_losses["loss_bbox"]
                    + self.weight_dict["loss_giou"] * aux_losses["loss_giou"]
                    + self.weight_dict["loss_cls"] * aux_losses["loss_cls"]
                    + self.weight_dict["loss_severity"] * aux_losses["loss_severity"]
                )
                losses[f"loss_bbox_aux_{i}"] = aux_losses["loss_bbox"]
                losses[f"loss_giou_aux_{i}"] = aux_losses["loss_giou"]
                losses[f"loss_cls_aux_{i}"] = aux_losses["loss_cls"]
                losses[f"loss_severity_aux_{i}"] = aux_losses["loss_severity"]
            total = total + self.weight_dict.get("loss_aux", 0.0) * aux_total

        return total, losses


def build_loss(cfg_model: Dict) -> TotalLoss:
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    num_classes = cfg_model.get("num_classes", 5)
    weight_dict = cfg_model.get("loss_weights")
    return TotalLoss(matcher=matcher, num_classes=num_classes, weight_dict=weight_dict)
