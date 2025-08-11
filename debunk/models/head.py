from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features: (B, T, C)
        x = features.transpose(1, 2)  # (B, C, T)
        pooled = self.pool(x).squeeze(-1)  # (B, C)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits, pooled


