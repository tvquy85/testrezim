from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np

EPS = 1e-8


@dataclass
class TrainStandardScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TrainStandardScaler":
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_ = np.where(self.std_ < EPS, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def _safe_log_return(close_window: np.ndarray) -> np.ndarray:
    close_window = np.asarray(close_window, dtype=np.float64)
    close_window = np.where(close_window <= 0, np.nan, close_window)
    logp = np.log(close_window)
    ret = np.diff(logp, axis=0)
    return np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)


def _pca_ratio_from_returns(r: np.ndarray) -> float:
    # r: time x assets. SVD avoids constructing huge N x N covariance.
    x = r - np.nanmean(r, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] < 2 or np.allclose(x, 0):
        return 0.0
    try:
        s = np.linalg.svd(x, compute_uv=False)
        denom = float(np.sum(s ** 2))
        if denom <= EPS:
            return 0.0
        return float((s[0] ** 2) / denom)
    except np.linalg.LinAlgError:
        return 0.0


def compute_market_context(
    close: np.ndarray,
    lookback: int,
    features: Iterable[str] = ("mean", "slope", "vol", "dispersion", "pca_ratio"),
) -> Tuple[np.ndarray, List[str]]:
    """Compute causal market context for every date t using only close[:t+1].

    Returns
    -------
    context : ndarray, shape (T, C)
        Rows before lookback-1 are zero.
    names : list[str]
        Ordered context feature names.
    """
    close = np.asarray(close, dtype=np.float64)
    if close.ndim != 2:
        raise ValueError(f"close must have shape (T, N), got {close.shape}")
    names = list(features)
    t_total, _ = close.shape
    ctx = np.zeros((t_total, len(names)), dtype=np.float32)

    for t in range(lookback - 1, t_total):
        w = close[t - lookback + 1 : t + 1]
        r = _safe_log_return(w)  # (lookback-1, N)
        market_r = np.nanmean(r, axis=1)
        cs_std = np.nanstd(r, axis=1)
        values = []
        for name in names:
            if name == "mean":
                values.append(float(np.nanmean(market_r)))
            elif name == "slope":
                # Causal market momentum over the window.
                m = np.nanmean(w, axis=1)
                if np.any(m <= 0):
                    values.append(0.0)
                else:
                    values.append(float(np.log(m[-1] / (m[0] + EPS)) / max(1, lookback - 1)))
            elif name == "vol":
                values.append(float(np.nanstd(market_r)))
            elif name == "dispersion":
                values.append(float(np.nanmean(cs_std)))
            elif name == "pca_ratio":
                values.append(_pca_ratio_from_returns(r))
            elif name == "breadth":
                values.append(float(np.nanmean(r > 0)))
            elif name == "downside_vol":
                down = np.minimum(market_r, 0.0)
                values.append(float(np.sqrt(np.nanmean(down ** 2))))
            elif name == "tail_abs":
                q95 = np.nanquantile(np.abs(market_r), 0.95) if market_r.size else 0.0
                values.append(float(q95))
            else:
                raise ValueError(f"Unknown context feature: {name}")
        ctx[t] = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0), names


def rolling_zscore_context(ctx: np.ndarray, window: int = 252) -> np.ndarray:
    out = np.zeros_like(ctx, dtype=np.float32)
    for t in range(ctx.shape[0]):
        s = max(0, t - window)
        hist = ctx[s : t + 1]
        mu = np.nanmean(hist, axis=0)
        sd = np.nanstd(hist, axis=0)
        sd = np.where(sd < EPS, 1.0, sd)
        out[t] = (ctx[t] - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def transform_context_train_only(
    ctx: np.ndarray,
    train_date_indices: np.ndarray,
    scaler_name: str = "train_standard",
    rolling_norm_window: int = 252,
) -> Tuple[np.ndarray, TrainStandardScaler | None]:
    if scaler_name == "none":
        return ctx.astype(np.float32), None
    if scaler_name == "rolling_zscore":
        return rolling_zscore_context(ctx, window=rolling_norm_window), None
    if scaler_name != "train_standard":
        raise ValueError(f"Unsupported context scaler: {scaler_name}")
    scaler = TrainStandardScaler().fit(ctx[train_date_indices])
    return scaler.transform(ctx).astype(np.float32), scaler


def apply_context_controls(ctx: np.ndarray, train_date_indices: np.ndarray, *, shuffle: bool = False, noise: bool = False, lag: int = 0, seed: int = 2027) -> np.ndarray:
    out = np.array(ctx, copy=True)
    rng = np.random.default_rng(seed)
    if lag > 0:
        shifted = np.zeros_like(out)
        shifted[lag:] = out[:-lag]
        out = shifted
    if shuffle:
        # Negative control: use a random permutation learned on all dates after causal transformation.
        perm = rng.permutation(out.shape[0])
        out = out[perm]
    if noise:
        mu = np.nanmean(out[train_date_indices], axis=0)
        sd = np.nanstd(out[train_date_indices], axis=0)
        sd = np.where(sd < EPS, 1.0, sd)
        out = rng.normal(mu, sd, size=out.shape).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
