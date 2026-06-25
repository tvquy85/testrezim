from __future__ import annotations

import torch
from torch import nn
from .common import MultiScaleTemporalEncoder, StaticStockMixer, PredictionHead


class StockMixerBugFixed(nn.Module):
    """Bug-fixed, compact StockMixer-style baseline.

    This is not a byte-for-byte copy of StockMixer. It is a clean-room baseline
    with registered ModuleList layers, chronological training compatibility and
    stable tensor shapes.
    """
    def __init__(self, num_assets: int, lookback: int, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.encoder = MultiScaleTemporalEncoder(lookback, feature_dim, hidden_dim, dropout=dropout)
        self.stock_mixer = StaticStockMixer(num_assets, hidden_dim, dropout=dropout)
        self.post = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.head = PredictionHead(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, sectors: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(x)
        h = self.stock_mixer(h)
        h = h + self.post(h)
        return self.head(h)
