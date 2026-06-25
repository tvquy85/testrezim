from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import copy
import yaml


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 2027,
    "device": "auto",
    "output_dir": "outputs/run",
    "data": {
        "path": "data/synthetic_market.npz",
        "format": "npz",  # npz or csv_long
        "lookback": 16,
        "horizon": 1,
        "close_index": 0,
        "feature_scaler": "train_standard",  # train_standard, cs_zscore, none
        "split": {"train": 0.60, "valid": 0.20, "test": 0.20},
        "num_workers": 0,
        "batch_size": 8,
        "pin_memory": False,
        "csv": {
            "date_col": "date",
            "asset_col": "asset",
            "label_col": "label",
            "feature_cols": None,
            "sector_col": None,
        },
    },
    "context": {
        "enabled": True,
        "lookback": 16,
        "features": ["mean", "slope", "vol", "dispersion", "pca_ratio"],
        "scaler": "train_standard",  # train_standard, rolling_zscore, none
        "rolling_norm_window": 252,
        "shuffle": False,
        "noise": False,
        "lag": 0,
    },
    "model": {
        "name": "crc_lora",  # stockmixer, context_gmlp, crc_lora, film, sector_moe, dlinear
        "hidden_dim": 64,
        "dropout": 0.10,
        "rank": 8,
        "num_experts": 4,
        "use_sector": False,
        "num_sectors": None,
    },
    "loss": {
        "mse_weight": 1.0,
        "bpr_weight": 0.10,
        "ic_weight": 0.0,
        "portfolio_weight": 0.0,
        "turnover_weight": 0.0,
        "topk": 10,
    },
    "optim": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 20,
        "patience": 5,
        "grad_clip": 1.0,
    },
    "eval": {
        "topk": 10,
        "trading_days": 252,
        "transaction_cost_bps": [0, 5, 10],
    },
}


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, loaded)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
