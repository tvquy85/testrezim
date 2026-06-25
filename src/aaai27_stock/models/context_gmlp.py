from __future__ import annotations

import torch
from torch import nn
from .common import MultiScaleTemporalEncoder, PredictionHead, init_residual_zero


class ContextGatedMLPBlock(nn.Module):
    """gMLP-like hidden-channel gate shifted by causal market context."""
    def __init__(self, hidden_dim: int, context_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ctx = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        init_residual_zero(self.out)
        self.act = nn.Hardswish()

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.norm(h))
        u, v = z.chunk(2, dim=-1)
        bias = self.ctx(context).unsqueeze(1)
        gated = self.act(u) * torch.sigmoid(v + bias)
        return h + self.out(gated)


class ContextGMLPStockForecaster(nn.Module):
    def __init__(self, num_assets: int, lookback: int, feature_dim: int, context_dim: int, hidden_dim: int = 64, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.encoder = MultiScaleTemporalEncoder(lookback, feature_dim, hidden_dim, dropout=dropout)
        self.block = ContextGatedMLPBlock(hidden_dim, context_dim, dropout=dropout)
        self.head = PredictionHead(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, sectors: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            raise ValueError("ContextGMLPStockForecaster requires context tensor.")
        h = self.encoder(x)
        h = self.block(h, context)
        return self.head(h)
