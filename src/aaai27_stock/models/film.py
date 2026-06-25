from __future__ import annotations

import torch
from torch import nn
from .common import MultiScaleTemporalEncoder, StaticStockMixer, PredictionHead


class FiLMRegimeMixer(nn.Module):
    """Context-conditioned FiLM modulation plus static stock mixing."""
    def __init__(self, num_assets: int, lookback: int, feature_dim: int, context_dim: int, hidden_dim: int = 64, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.encoder = MultiScaleTemporalEncoder(lookback, feature_dim, hidden_dim, dropout=dropout)
        self.ctx = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.static = StaticStockMixer(num_assets, hidden_dim, dropout=dropout)
        self.head = PredictionHead(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, sectors: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            raise ValueError("FiLMRegimeMixer requires context tensor.")
        h = self.encoder(x)
        gamma, beta = self.ctx(context).chunk(2, dim=-1)
        h = self.norm(h) * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1)) + 0.1 * beta.unsqueeze(1)
        h = self.static(h)
        return self.head(h)
