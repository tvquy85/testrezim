from __future__ import annotations

import math
import torch
from torch import nn
from .common import MultiScaleTemporalEncoder, StaticStockMixer, PredictionHead


class RegimeLoRAStockMixer(nn.Module):
    """Causal Regime-Conditioned Low-Rank stock mixer.

    For each date/context c, this applies a low-rank stock-axis adapter:
        A(c) = A_static + U diag(g(c)) V^T
    where rank << number of assets. This keeps parameter count controlled while
    allowing cross-sectional relations to change across market regimes.
    """
    def __init__(
        self,
        num_assets: int,
        lookback: int,
        feature_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        rank: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.rank = rank
        self.encoder = MultiScaleTemporalEncoder(lookback, feature_dim, hidden_dim, dropout=dropout)
        self.static = StaticStockMixer(num_assets, hidden_dim, dropout=dropout)
        self.u = nn.Parameter(torch.empty(num_assets, rank))
        self.v = nn.Parameter(torch.empty(num_assets, rank))
        nn.init.normal_(self.u, std=1.0 / math.sqrt(max(1, num_assets)))
        nn.init.normal_(self.v, std=1.0 / math.sqrt(max(1, num_assets)))
        self.gate = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rank),
            nn.Tanh(),
        )
        self.mix_norm = nn.LayerNorm(hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.post = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.head = PredictionHead(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, sectors: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            raise ValueError("RegimeLoRAStockMixer requires context tensor.")
        h = self.encoder(x)       # B,N,D
        h = self.static(h)        # static residual stock mixing
        z = self.mix_norm(h)
        gate = self.gate(context) # B,R in [-1,1]
        # V^T x for every hidden channel: B,N,D and N,R -> B,D,R
        proj = torch.einsum("bnd,nr->bdr", z, self.v)
        proj = proj * gate.unsqueeze(1)
        delta = torch.einsum("bdr,nr->bnd", proj, self.u)
        h = h + self.alpha * delta
        h = h + self.post(h)
        return self.head(h)
