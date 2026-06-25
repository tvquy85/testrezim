from __future__ import annotations

from typing import Dict, Iterable, Tuple
import numpy as np
from scipy.stats import rankdata

EPS = 1e-12


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return np.nan
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.std(a) * np.std(b)
    if denom < EPS:
        return np.nan
    return float(np.mean(a * b) / denom)


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return np.nan
    equity = np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, EPS) - 1.0
    return float(np.min(dd))


def _long_short_weights(scores: np.ndarray, valid: np.ndarray, topk: int) -> np.ndarray:
    idx = np.where(valid)[0]
    w = np.zeros_like(scores, dtype=np.float64)
    if idx.size < 2:
        return w
    k = min(topk, idx.size // 2 if idx.size >= 2 else 1)
    order = idx[np.argsort(scores[idx])]
    short_idx = order[:k]
    long_idx = order[-k:]
    w[long_idx] = 1.0 / k
    w[short_idx] = -1.0 / k
    return w


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    topk: int = 10,
    trading_days: int = 252,
    transaction_cost_bps: Iterable[float] = (0, 5, 10),
) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}, mask {mask.shape}")

    daily_ic, daily_ric, prec, overlap, long_ret, ls_ret = [], [], [], [], [], []
    weights = []
    for t in range(pred.shape[0]):
        valid = mask[t] & np.isfinite(pred[t]) & np.isfinite(target[t])
        idx = np.where(valid)[0]
        if idx.size < 2:
            continue
        p = pred[t, idx]
        y = target[t, idx]
        daily_ic.append(_corr(p, y))
        daily_ric.append(_corr(rankdata(p), rankdata(y)))
        k = min(topk, idx.size)
        top_pred = idx[np.argsort(pred[t, idx])[-k:]]
        bottom_pred = idx[np.argsort(pred[t, idx])[:k]]
        true_top = set(idx[np.argsort(target[t, idx])[-k:]].tolist())
        prec.append(float(np.mean(target[t, top_pred] > 0)))
        overlap.append(float(len(set(top_pred.tolist()) & true_top) / k))
        long_ret.append(float(np.mean(target[t, top_pred])))
        ls_ret.append(float(np.mean(target[t, top_pred]) - np.mean(target[t, bottom_pred])))
        weights.append(_long_short_weights(pred[t], valid, topk=k))

    def mean_std_ir(x):
        arr = np.asarray(x, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan, np.nan, np.nan
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=0))
        ir = float(mu / (sd + EPS))
        return mu, sd, ir

    ic, ic_std, icir = mean_std_ir(daily_ic)
    ric, ric_std, ricir = mean_std_ir(daily_ric)
    long_arr = np.asarray(long_ret, dtype=np.float64)
    ls_arr = np.asarray(ls_ret, dtype=np.float64)
    metrics: Dict[str, float] = {
        "ic": ic,
        "ic_std": ic_std,
        "icir": icir,
        "rank_ic": ric,
        "rank_ic_std": ric_std,
        "rank_icir": ricir,
        f"precision_at_{topk}": float(np.nanmean(prec)) if prec else np.nan,
        f"topk_overlap_at_{topk}": float(np.nanmean(overlap)) if overlap else np.nan,
        "long_return_daily": float(np.nanmean(long_arr)) if long_arr.size else np.nan,
        "long_short_return_daily": float(np.nanmean(ls_arr)) if ls_arr.size else np.nan,
        "long_sharpe": float(np.sqrt(trading_days) * np.nanmean(long_arr) / (np.nanstd(long_arr) + EPS)) if long_arr.size else np.nan,
        "long_short_sharpe": float(np.sqrt(trading_days) * np.nanmean(ls_arr) / (np.nanstd(ls_arr) + EPS)) if ls_arr.size else np.nan,
        "long_max_drawdown": _max_drawdown(long_arr) if long_arr.size else np.nan,
        "long_short_max_drawdown": _max_drawdown(ls_arr) if ls_arr.size else np.nan,
        "num_days": float(len(long_arr)),
    }

    if weights:
        w = np.stack(weights, axis=0)
        turnover = np.mean(np.sum(np.abs(w[1:] - w[:-1]), axis=1)) if w.shape[0] > 1 else 0.0
        metrics["turnover_daily"] = float(turnover)
        for bps in transaction_cost_bps:
            cost = float(bps) / 10000.0 * turnover
            cost_adj = ls_arr.copy()
            if cost_adj.size > 1:
                cost_adj[1:] = cost_adj[1:] - cost
            key = f"cost_{int(bps)}bps"
            metrics[f"{key}_ls_return_daily"] = float(np.nanmean(cost_adj)) if cost_adj.size else np.nan
            metrics[f"{key}_ls_sharpe"] = float(np.sqrt(trading_days) * np.nanmean(cost_adj) / (np.nanstd(cost_adj) + EPS)) if cost_adj.size else np.nan
    else:
        metrics["turnover_daily"] = np.nan
    return metrics


def bootstrap_ci_daily(values: np.ndarray, n_boot: int = 1000, seed: int = 2027) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
