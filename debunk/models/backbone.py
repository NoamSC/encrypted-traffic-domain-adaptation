from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class NetMambaBlock(nn.Module):
    """A lightweight linear-time sequence mixing block approximating Mamba.

    This block uses depthwise-separable conv + gated feedforward with residuals.
    It is self-contained and efficient while keeping an interface compatible
    with Mamba-style encoders.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.pwconv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.pwconv(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        return x


class NetMambaBackbone(nn.Module):
    """Backbone encoder that outputs sequence features.

    Attributes
    ----------
    feature_dim: int
        The size of the last dimension of the encoded features.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 6, dropout: float = 0.1, max_len: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.feature_dim = d_model
        # 0..255 bytes + meta tokens region (>=256). Provide a buffer size.
        self.embed = nn.Embedding(256 + 64, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.layers = nn.ModuleList([NetMambaBlock(d_model, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids: (B, T)
        B, T = input_ids.shape
        x = self.embed(input_ids.clamp(min=0, max=self.embed.num_embeddings - 1))
        pos = self.pos[:, :T, :]
        x = x + pos
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x  # (B, T, C)


