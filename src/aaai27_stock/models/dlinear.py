from __future__ import annotations

import torch
from torch import nn


class DLinearStock(nn.Module):
    """Strong simple baseline: per-stock linear forecast from the lookback window."""
    def __init__(self, num_assets: int, lookback: int, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.0, **kwargs) -> None:
        super().__init__()
        self.lookback = lookback
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(lookback * feature_dim),
            nn.Dropout(dropout),
            nn.Linear(lookback * feature_dim, 1),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, sectors: torch.Tensor | None = None) -> torch.Tensor:
        b, n, l, f = x.shape
        return self.net(x.reshape(b, n, l * f)).squeeze(-1)
