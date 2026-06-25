from __future__ import annotations

import torch
from torch import nn
from .common import MultiScaleTemporalEncoder, StaticStockMixer, PredictionHead


class SectorMoERegimeMixer(nn.Module):
    """Mixture of stock-mixer experts gated by market context.

    Optional sector embeddings allow the same context to be interpreted differently
    for sector buckets. The model still returns a full cross-section forecast.
    """
    def __init__(
        self,
        num_assets: int,
        lookback: int,
        feature_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        num_experts: int = 4,
        num_sectors: int | None = None,
        use_sector: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_sector = use_sector
        self.encoder = MultiScaleTemporalEncoder(lookback, feature_dim, hidden_dim, dropout=dropout)
        self.sector_emb = nn.Embedding(max(1, num_sectors or 1), hidden_dim) if use_sector else None
        self.experts = nn.ModuleList([StaticStockMixer(num_assets, hidden_dim, dropout=dropout) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.head = PredictionHead(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, sectors: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            raise ValueError("SectorMoERegimeMixer requires context tensor.")
        h = self.encoder(x)
        if self.use_sector and self.sector_emb is not None and sectors is not None:
            # sectors: B,N or N. Dataloader returns B,N.
            sec = sectors if sectors.ndim == 2 else sectors.unsqueeze(0).expand(h.shape[0], -1)
            h = h + 0.1 * self.sector_emb(sec)
        weights = torch.softmax(self.gate(context), dim=-1)  # B,E
        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=-1)  # B,N,D,E
        h = torch.einsum("bnde,be->bnd", expert_outs, weights)
        return self.head(h)
