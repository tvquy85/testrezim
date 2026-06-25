from __future__ import annotations

import math
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        act: nn.Module
        if activation == "hardswish":
            act = nn.Hardswish()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            act = nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalFeatureEncoder(nn.Module):
    """A compact MLP encoder for each asset's lookback window.

    Input:  (B, N, L, F)
    Output: (B, N, D)
    """
    def __init__(self, lookback: int, feature_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lookback = lookback
        self.feature_dim = feature_dim
        in_dim = lookback * feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, l, f = x.shape
        if l != self.lookback or f != self.feature_dim:
            raise ValueError(f"Expected lookback/features {(self.lookback, self.feature_dim)}, got {(l, f)}")
        return self.net(x.reshape(b, n, l * f))


class MultiScaleTemporalEncoder(nn.Module):
    """MLP temporal encoder using full, half and quarter windows.

    This is intentionally simple, stable and shape-safe. It keeps the spirit of
    StockMixer's multi-scale temporal mixing while avoiding fragile ModuleList/
    ParameterList bugs from older prototypes.
    """
    def __init__(self, lookback: int, feature_dim: int, hidden_dim: int, dropout: float = 0.0, scales: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.lookback = lookback
        self.feature_dim = feature_dim
        self.scales = tuple(s for s in scales if lookback // s >= 2)
        branch_dim = max(8, hidden_dim // max(1, len(self.scales)))
        self.branches = nn.ModuleList()
        self.branch_slices = []
        for s in self.scales:
            l = lookback // s
            self.branch_slices.append(l)
            self.branches.append(
                nn.Sequential(
                    nn.LayerNorm(l * feature_dim),
                    nn.Linear(l * feature_dim, branch_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(branch_dim, branch_dim),
                )
            )
        self.out = nn.Sequential(nn.LayerNorm(branch_dim * len(self.branches)), nn.Linear(branch_dim * len(self.branches), hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, l, f = x.shape
        outs = []
        for use_l, branch in zip(self.branch_slices, self.branches):
            xs = x[:, :, -use_l:, :].reshape(b, n, use_l * f)
            outs.append(branch(xs))
        return self.out(torch.cat(outs, dim=-1))


class StaticStockMixer(nn.Module):
    """Static MLP mixer along stock axis.

    Input/output shape: (B, N, D). For each channel D, it mixes the N-stock vector.
    """
    def __init__(self, num_assets: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.net = nn.Sequential(
            nn.LayerNorm(num_assets),
            nn.Linear(num_assets, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_assets),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: B,N,D -> B,D,N -> mix -> B,N,D
        mixed = self.net(h.transpose(1, 2)).transpose(1, 2)
        return h + mixed


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_residual_zero(module: nn.Module) -> None:
    """Zero-init last Linear layer in residual adapters for stable warm start."""
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return
