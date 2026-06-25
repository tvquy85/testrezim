#!/usr/bin/env python
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def rolling_mean(x, window):
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        s = max(0, t - window + 1)
        out[t] = np.mean(x[s : t + 1], axis=0)
    return out


def rolling_std(x, window):
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        s = max(0, t - window + 1)
        out[t] = np.std(x[s : t + 1], axis=0)
    return out


def generate(path: Path, days: int, assets: int, sectors_n: int, seed: int, horizon: int) -> None:
    rng = np.random.default_rng(seed)
    sectors = rng.integers(0, sectors_n, size=assets, dtype=np.int64)
    beta = rng.normal(1.0, 0.25, size=assets)
    quality = rng.normal(0.0, 1.0, size=assets)
    sector_loading = rng.normal(0.0, 0.4, size=(sectors_n, assets))
    for i, s in enumerate(sectors):
        sector_loading[s, i] += 1.0

    # Markov regimes: 0 calm/up, 1 high-vol/down, 2 dispersion/stock-picking.
    P = np.array([[0.94, 0.04, 0.02], [0.08, 0.88, 0.04], [0.06, 0.04, 0.90]])
    regimes = np.zeros(days, dtype=np.int64)
    for t in range(1, days):
        regimes[t] = rng.choice(3, p=P[regimes[t - 1]])

    mkt = np.zeros(days)
    sector_ret = np.zeros((days, sectors_n))
    ret = np.zeros((days, assets))
    for t in range(1, days):
        reg = regimes[t]
        if reg == 0:
            mu, vol, disp = 0.0006, 0.006, 0.004
        elif reg == 1:
            mu, vol, disp = -0.0005, 0.018, 0.009
        else:
            mu, vol, disp = 0.0001, 0.010, 0.014
        mkt[t] = 0.08 * mkt[t - 1] + rng.normal(mu, vol)
        sector_ret[t] = 0.05 * sector_ret[t - 1] + rng.normal(0, vol * 0.45, size=sectors_n)
        # A regime-dependent alpha signal: quality works in calm/dispersion, reverses in stress.
        alpha = (0.00025 * quality if reg != 1 else -0.00020 * quality)
        mom = 0.10 * ret[t - 1] if reg == 0 else (-0.05 * ret[t - 1] if reg == 1 else 0.03 * ret[t - 1])
        sec_component = sector_ret[t, sectors]
        eps = rng.normal(0, disp, size=assets)
        ret[t] = beta * mkt[t] + sec_component + alpha + mom + eps

    close = 100.0 * np.exp(np.cumsum(ret, axis=0))
    log_close = np.log(close)
    labels = np.full((days, assets), np.nan, dtype=np.float32)
    labels[:-horizon] = (log_close[horizon:] - log_close[:-horizon]).astype(np.float32)

    r1 = np.vstack([np.zeros((1, assets)), np.diff(log_close, axis=0)])
    ma5 = rolling_mean(r1, 5)
    vol5 = rolling_std(r1, 5)
    ma20_price = rolling_mean(close, 20)
    rel_price = close / np.maximum(ma20_price, 1e-8) - 1.0
    market_ret = np.mean(r1, axis=1, keepdims=True).repeat(assets, axis=1)
    sector_id_scaled = sectors.astype(float) / max(1, sectors_n - 1)
    sector_feature = np.tile(sector_id_scaled.reshape(1, -1), (days, 1))
    features = np.stack([close, r1, ma5, vol5, rel_price, market_ret, sector_feature], axis=-1).astype(np.float32)
    mask = np.isfinite(labels)
    dates = pd.date_range("2010-01-01", periods=days, freq="B").astype(str).to_numpy()
    assets_names = np.asarray([f"SYN{i:04d}" for i in range(assets)])
    feature_names = np.asarray(["close", "ret1", "ret_ma5", "ret_vol5", "price_vs_ma20", "market_ret", "sector_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        close=close.astype(np.float32),
        mask=mask,
        dates=dates,
        assets=assets_names,
        sectors=sectors,
        regimes=regimes,
        feature_names=feature_names,
    )
    print(f"Saved {path} with features={features.shape}, labels={labels.shape}, sectors={sectors_n}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/synthetic_market.npz"))
    p.add_argument("--days", type=int, default=900)
    p.add_argument("--assets", type=int, default=120)
    p.add_argument("--sectors", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()
    generate(args.out, args.days, args.assets, args.sectors, args.seed, args.horizon)


if __name__ == "__main__":
    main()
