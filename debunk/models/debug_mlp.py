from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class DebugMLPClassifier(nn.Module):
    """A tiny MLP that consumes per-sample float vectors.

    This is for debugging tabular ARFF features. It maps input_dim -> hidden_dims -> num_classes.
    forward returns (logits, features) to match the classifier interface.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = hidden_dims[-1] if hidden_dims else self.input_dim
        dims = [self.input_dim] + (hidden_dims or [128, 64])
        layers: List[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None):
        # input_ids shape: (B, D)
        x = input_ids.float()
        feats = self.mlp(x)
        logits = self.head(feats)
        return logits, feats
