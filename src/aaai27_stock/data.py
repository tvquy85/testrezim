from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .context import TrainStandardScaler, compute_market_context, transform_context_train_only, apply_context_controls

EPS = 1e-8


@dataclass
class MarketArrays:
    features: np.ndarray  # (T, N, F)
    labels: np.ndarray    # (T, N), label at date t = future return after observing date t
    close: np.ndarray     # (T, N)
    mask: np.ndarray      # (T, N)
    dates: np.ndarray     # (T,)
    assets: np.ndarray    # (N,)
    sectors: Optional[np.ndarray] = None  # (N,)
    feature_names: Optional[List[str]] = None


def load_npz(path: str | Path, close_index: int = 0) -> MarketArrays:
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    if "features" not in z or "labels" not in z:
        raise ValueError("NPZ must contain at least 'features' and 'labels'.")
    features = z["features"].astype(np.float32)
    labels = z["labels"].astype(np.float32)
    if features.ndim != 3:
        raise ValueError(f"features must be (T,N,F), got {features.shape}")
    if labels.shape != features.shape[:2]:
        raise ValueError(f"labels must be (T,N), got {labels.shape} for features {features.shape}")
    close = z["close"].astype(np.float32) if "close" in z else features[:, :, close_index].astype(np.float32)
    mask = z["mask"].astype(bool) if "mask" in z else np.isfinite(labels) & np.all(np.isfinite(features), axis=-1)
    dates = z["dates"].astype(str) if "dates" in z else np.arange(features.shape[0]).astype(str)
    assets = z["assets"].astype(str) if "assets" in z else np.asarray([f"asset_{i:04d}" for i in range(features.shape[1])])
    sectors = z["sectors"].astype(np.int64) if "sectors" in z else None
    if "feature_names" in z:
        feature_names = [str(x) for x in z["feature_names"].tolist()]
    else:
        feature_names = [f"f{i}" for i in range(features.shape[-1])]
    return MarketArrays(
        features=np.nan_to_num(features),
        labels=np.nan_to_num(labels),
        close=np.nan_to_num(close),
        mask=mask,
        dates=dates,
        assets=assets,
        sectors=sectors,
        feature_names=feature_names,
    )


def load_csv_long(path: str | Path, csv_cfg: Dict, close_index: int = 0) -> MarketArrays:
    path = Path(path)
    df = pd.read_csv(path)
    date_col = csv_cfg.get("date_col", "date")
    asset_col = csv_cfg.get("asset_col", "asset")
    label_col = csv_cfg.get("label_col", "label")
    sector_col = csv_cfg.get("sector_col")
    feature_cols = csv_cfg.get("feature_cols")
    if feature_cols is None:
        exclude = {date_col, asset_col, label_col}
        if sector_col:
            exclude.add(sector_col)
        feature_cols = [c for c in df.columns if c not in exclude]
    dates = np.asarray(sorted(df[date_col].astype(str).unique()))
    assets = np.asarray(sorted(df[asset_col].astype(str).unique()))
    date_to_i = {d: i for i, d in enumerate(dates)}
    asset_to_i = {a: i for i, a in enumerate(assets)}
    t_total, n_assets, f_dim = len(dates), len(assets), len(feature_cols)
    features = np.full((t_total, n_assets, f_dim), np.nan, dtype=np.float32)
    labels = np.full((t_total, n_assets), np.nan, dtype=np.float32)
    mask = np.zeros((t_total, n_assets), dtype=bool)
    sectors = np.zeros((n_assets,), dtype=np.int64) if sector_col else None
    sector_map: Dict[str, int] = {}
    for _, row in df.iterrows():
        t = date_to_i[str(row[date_col])]
        a = asset_to_i[str(row[asset_col])]
        features[t, a] = row[feature_cols].to_numpy(dtype=np.float32)
        labels[t, a] = np.float32(row[label_col])
        mask[t, a] = bool(np.isfinite(labels[t, a]) and np.all(np.isfinite(features[t, a])))
        if sector_col and sectors is not None:
            sec = str(row[sector_col])
            if sec not in sector_map:
                sector_map[sec] = len(sector_map)
            sectors[a] = sector_map[sec]
    close = features[:, :, close_index]
    return MarketArrays(
        features=np.nan_to_num(features),
        labels=np.nan_to_num(labels),
        close=np.nan_to_num(close),
        mask=mask,
        dates=dates,
        assets=assets,
        sectors=sectors,
        feature_names=list(feature_cols),
    )


def load_market_arrays(cfg: Dict) -> MarketArrays:
    fmt = cfg.get("format", "npz")
    path = cfg["path"]
    close_index = int(cfg.get("close_index", 0))
    if fmt == "npz":
        return load_npz(path, close_index=close_index)
    if fmt == "csv_long":
        return load_csv_long(path, cfg.get("csv", {}), close_index=close_index)
    raise ValueError(f"Unsupported data format: {fmt}")


def make_sample_dates(t_total: int, lookback: int, horizon: int, mask: np.ndarray) -> np.ndarray:
    last_t = t_total - horizon - 1
    candidates = np.arange(lookback - 1, last_t + 1, dtype=np.int64)
    # Keep dates with at least 2 valid assets so IC/ranking metrics are meaningful.
    valid = [t for t in candidates if np.sum(mask[t]) >= 2]
    return np.asarray(valid, dtype=np.int64)


def chronological_split(sample_dates: np.ndarray, split_cfg: Dict) -> Dict[str, np.ndarray]:
    n = len(sample_dates)
    tr = float(split_cfg.get("train", 0.6))
    va = float(split_cfg.get("valid", 0.2))
    n_train = max(1, int(round(n * tr)))
    n_valid = max(1, int(round(n * va)))
    n_train = min(n_train, n - 2) if n >= 3 else max(1, n - 1)
    n_valid = min(n_valid, n - n_train - 1) if n - n_train >= 2 else max(0, n - n_train)
    return {
        "train": sample_dates[:n_train],
        "valid": sample_dates[n_train : n_train + n_valid],
        "test": sample_dates[n_train + n_valid :],
    }


def fit_transform_features_train_only(features: np.ndarray, train_dates: np.ndarray, lookback: int, scaler_name: str) -> Tuple[np.ndarray, Optional[TrainStandardScaler]]:
    if scaler_name == "none":
        return features.astype(np.float32), None
    if scaler_name == "cs_zscore":
        # Causal per-date cross-sectional normalization. This uses only same-date observables.
        mu = np.nanmean(features, axis=1, keepdims=True)
        sd = np.nanstd(features, axis=1, keepdims=True)
        sd = np.where(sd < EPS, 1.0, sd)
        return np.nan_to_num((features - mu) / sd).astype(np.float32), None
    if scaler_name != "train_standard":
        raise ValueError(f"Unsupported feature scaler: {scaler_name}")
    # Fit on raw train dates only, not validation/test.
    # Include all lookback dates that are observable inside train samples.
    train_obs_dates = set()
    for t in train_dates:
        for s in range(t - lookback + 1, t + 1):
            train_obs_dates.add(int(s))
    train_obs = np.asarray(sorted(train_obs_dates), dtype=np.int64)
    x_train = features[train_obs].reshape(-1, features.shape[-1])
    scaler = TrainStandardScaler().fit(x_train)
    transformed = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape).astype(np.float32)
    return np.nan_to_num(transformed), scaler


class StockWindowDataset(Dataset):
    def __init__(
        self,
        arrays: MarketArrays,
        sample_dates: np.ndarray,
        lookback: int,
        context: np.ndarray,
        sectors: Optional[np.ndarray] = None,
    ) -> None:
        self.arrays = arrays
        self.sample_dates = np.asarray(sample_dates, dtype=np.int64)
        self.lookback = int(lookback)
        self.context = context.astype(np.float32)
        self.sectors = sectors.astype(np.int64) if sectors is not None else np.zeros((arrays.features.shape[1],), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sample_dates)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = int(self.sample_dates[idx])
        x = self.arrays.features[t - self.lookback + 1 : t + 1]  # L,N,F
        x = np.transpose(x, (1, 0, 2))  # N,L,F
        y = self.arrays.labels[t]
        m = self.arrays.mask[t]
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "mask": torch.tensor(m, dtype=torch.bool),
            "context": torch.tensor(self.context[t], dtype=torch.float32),
            "date_index": torch.tensor(t, dtype=torch.long),
            "sectors": torch.tensor(self.sectors, dtype=torch.long),
        }


def make_dataloaders(cfg: Dict) -> Tuple[Dict[str, DataLoader], Dict]:
    data_cfg = cfg["data"]
    arrays = load_market_arrays(data_cfg)
    lookback = int(data_cfg["lookback"])
    horizon = int(data_cfg.get("horizon", 1))
    sample_dates = make_sample_dates(arrays.features.shape[0], lookback, horizon, arrays.mask)
    splits = chronological_split(sample_dates, data_cfg.get("split", {}))

    features_scaled, feature_scaler = fit_transform_features_train_only(
        arrays.features, splits["train"], lookback, data_cfg.get("feature_scaler", "train_standard")
    )
    arrays = MarketArrays(
        features=features_scaled,
        labels=arrays.labels,
        close=arrays.close,
        mask=arrays.mask,
        dates=arrays.dates,
        assets=arrays.assets,
        sectors=arrays.sectors,
        feature_names=arrays.feature_names,
    )

    ctx_cfg = cfg.get("context", {})
    if ctx_cfg.get("enabled", True):
        ctx_raw, ctx_names = compute_market_context(
            arrays.close,
            lookback=int(ctx_cfg.get("lookback", lookback)),
            features=ctx_cfg.get("features", ["mean", "slope", "vol", "dispersion", "pca_ratio"]),
        )
        ctx_scaled, ctx_scaler = transform_context_train_only(
            ctx_raw,
            splits["train"],
            scaler_name=ctx_cfg.get("scaler", "train_standard"),
            rolling_norm_window=int(ctx_cfg.get("rolling_norm_window", 252)),
        )
        ctx = apply_context_controls(
            ctx_scaled,
            splits["train"],
            shuffle=bool(ctx_cfg.get("shuffle", False)),
            noise=bool(ctx_cfg.get("noise", False)),
            lag=int(ctx_cfg.get("lag", 0)),
            seed=int(cfg.get("seed", 2027)),
        )
    else:
        ctx_names = []
        ctx_scaler = None
        ctx = np.zeros((arrays.features.shape[0], 1), dtype=np.float32)

    loaders: Dict[str, DataLoader] = {}
    for split, dates in splits.items():
        ds = StockWindowDataset(arrays, dates, lookback, ctx, arrays.sectors)
        loaders[split] = DataLoader(
            ds,
            batch_size=int(data_cfg.get("batch_size", 8)),
            shuffle=(split == "train"),
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", False)),
        )

    meta = {
        "num_assets": int(arrays.features.shape[1]),
        "lookback": int(lookback),
        "feature_dim": int(arrays.features.shape[-1]),
        "context_dim": int(ctx.shape[-1]),
        "context_names": ctx_names,
        "num_sectors": int(np.max(arrays.sectors) + 1) if arrays.sectors is not None else 1,
        "feature_names": arrays.feature_names,
        "dates": arrays.dates.tolist(),
        "assets": arrays.assets.tolist(),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "split_date_indices": {k: v.astype(int).tolist() for k, v in splits.items()},
        "feature_scaler": "fitted" if feature_scaler is not None else None,
        "context_scaler": "fitted" if ctx_scaler is not None else None,
    }
    return loaders, meta


def write_metadata(meta: Dict, output_dir: str | Path) -> None:
    path = Path(output_dir) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
