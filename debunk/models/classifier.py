from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .backbone import NetMambaBackbone
from .head import ClassifierHead


class TrafficClassifier(nn.Module):
    """NetMambaBackbone + ClassifierHead returning (logits, features)."""

    def __init__(self, backbone: NetMambaBackbone, num_classes: int, head_dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim
        self.head = ClassifierHead(self.feature_dim, num_classes, dropout=head_dropout)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(input_ids=input_ids, lengths=lengths)
        logits, pooled = self.head(feats)
        return logits, pooled


