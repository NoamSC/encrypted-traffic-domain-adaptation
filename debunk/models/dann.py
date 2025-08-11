from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def set_lambda(self, lambd: float) -> None:
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int] = [256, 128], dropout: float = 0.1) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DannModel(nn.Module):
    def __init__(self, classifier: nn.Module, discriminator: DomainDiscriminator, grl: GradientReversal) -> None:
        super().__init__()
        self.classifier = classifier
        self.discriminator = discriminator
        self.grl = grl

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, features = self.classifier(input_ids=input_ids, lengths=lengths)
        rev_features = self.grl(features)
        dom_logits = self.discriminator(rev_features)
        return logits, features, dom_logits


