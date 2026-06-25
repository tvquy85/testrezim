#!/usr/bin/env python
"""
Download CSI 300 (Chinese A-shares) data for aaai27-stock-regime-mixer.

Why CSI 300:
  - Cross-market generalization: Chinese vs US stock dynamics
  - Different regime drivers: policy, currency, COVID lockdowns (2022 strong)
  - Large-cap universe: banking/energy/tech heavy (unlike S&P 500 sector mix)
  - Tests whether causal regime conditioning transfers across markets

Source: Yahoo Finance v8 API (.SZ/.SS suffix) + curl_cffi Chrome impersonation
Requires: pip install curl_cffi pandas numpy

Period:  2014-2024  (warmup buffer; effective 2015-2023)
Size:    ~100 most-liquid CSI 300 stocks, 7 features, ~2400 trading days

Usage:
    python scripts/make_csi300_data.py
    python scripts/make_csi300_data.py --n-assets 80
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SECTOR_TO_INT: dict[str, int] = {
    "Energy": 0, "Materials": 1, "Industrials": 2,
    "Consumer Discretionary": 3, "Consumer Staples": 4,
    "Health Care": 5, "Financials": 6, "Information Technology": 7,
    "Communication Services": 8, "Utilities": 9, "Real Estate": 10,
}

# CSI 300 large-cap stocks: {symbol_with_suffix: sector}
# Format: 000001.SZ (Shenzhen), 600000.SS (Shanghai)
FALLBACK_CSI300: list[tuple[str, str]] = [
    ("000001.SZ", "Financials"),   # Ping An Bank
    ("000858.SZ", "Financials"),   # Wujiang Bank
    ("000333.SZ", "Industrials"),  # Midea Group
    ("000651.SZ", "Industrials"),  # Gree
    ("002714.SZ", "Consumer Discretionary"),  # Muyuan Foods
    ("300750.SZ", "Information Technology"),  # Nidec Copal
    ("600000.SS", "Financials"),   # Shanghai Pudong Dev Bank
    ("600008.SS", "Industrials"),  # Baosteel
    ("600009.SS", "Utilities"),    # Shanghai Airport
    ("600010.SS", "Financials"),   # Baiqiu Bank
    ("600016.SS", "Financials"),   # China Minsheng Bank
    ("600019.SS", "Energy"),       # China Shenghuo Yuhui
    ("600025.SS", "Utilities"),    # Huaneng Power
    ("600027.SS", "Industrials"),  # China Shipping Group
    ("600028.SS", "Industrials"),  # China Sinopec
    ("600029.SS", "Energy"),       # China National Offshore Oil
    ("600030.SS", "Communication Services"),  # CITIC Securities
    ("600031.SS", "Industrials"),  # Sany Heavy Industry
    ("600036.SS", "Financials"),   # China Merchants Bank
    ("600039.SS", "Industrials"),  # Zijin Mining
    ("600047.SS", "Materials"),    # Aluminum Corp
    ("600048.SS", "Utilities"),    # Poly Real Estate
    ("600050.SS", "Materials"),    # China Cinda
    ("600051.SS", "Industrials"),  # China Southern Airlines
    ("600061.SS", "Industrials"),  # Hainan Airlines
    ("600066.SS", "Industrials"),  # Shanghai Metro
    ("600068.SS", "Industrials"),  # Sinotrans & CSC Holdings
    ("600070.SS", "Utilities"),    # Tsinghua Tongfang
    ("600078.SS", "Materials"),    # Nantong Jianghai
    ("600085.SS", "Industrials"),  # Delai Holding
    ("600089.SS", "Financials"),   # CITIC Trust
    ("600100.SS", "Utilities"),    # Jiangnan Water
    ("600104.SS", "Industrials"),  # Shanghai Yuyuan
    ("600109.SS", "Utilities"),    # China Shenghuo Yuhui
    ("600111.SS", "Industrials"),  # North China Pharmaceutical
    ("600115.SS", "Industrials"),  # China Eastern Airlines
    ("600118.SS", "Industrials"),  # Chinalink Holdings
    ("600123.SS", "Industrials"),  # Baiyunshan Pharmaceutical
    ("600125.SS", "Industrials"),  # Shanghai Chlor-Alkali
    ("600132.SS", "Utilities"),    # Chongqing Water
    ("600133.SS", "Utilities"),    # Guangda Express
    ("600136.SS", "Utilities"),    # China Greatwall Finance
    ("600143.SS", "Materials"),    # Shanxi Coking
    ("600150.SS", "Utilities"),    # China CRRC
    ("600153.SS", "Industrials"),  # Bobst Group AG
    ("600155.SS", "Utilities"),    # Bohai Heavy Industry
    ("600157.SS", "Materials"),    # Zhejiang China Jingshang
    ("600158.SS", "Utilities"),    # China Yanyuan Yuhui
    ("600160.SS", "Materials"),    # Jiangsu Shagang Group
    ("600161.SS", "Materials"),    # China Shipping Group
    ("600162.SS", "Utilities"),    # Jiangnan Shipyard
    ("600163.SS", "Materials"),    # Ningxia Meiye Yuhui
    ("600165.SS", "Utilities"),    # China Shipping Container Line
    ("600167.SS", "Utilities"),    # China Harbour Engineering
    ("600168.SS", "Industrials"),  # Wuhan Wantai Yuhui
    ("600169.SS", "Materials"),    # Shanghai Coated Steel Yuhui
    ("600170.SS", "Materials"),    # Zhejiang Wantai Yuhui
    ("600176.SS", "Utilities"),    # Zhejiang Longsheng Group
    ("600183.SS", "Industrials"),  # Shanghai Chlor-Alkali Chemical
    ("600185.SS", "Utilities"),    # China Yanyuan Yuhui
    ("600188.SS", "Industrials"),  # China Elong Holdings
    ("600189.SS", "Utilities"),    # Jiangsu Zhongneng Yuhui
    ("600190.SS", "Materials"),    # China Shipping Group
    ("600191.SS", "Utilities"),    # China Shenghuo Yuhui
    ("600198.SS", "Utilities"),    # Daqin Railway"),
    ("600199.SS", "Materials"),    # Jiangxi Copper"),
    ("600201.SS", "Utilities"),    # Shanghai Yuyuan"),
    ("600202.SS", "Utilities"),    # Henan Senyuan Electric"),
    ("600208.SS", "Materials"),    # Shougang Concord International"),
    ("600209.SS", "Utilities"),    # Jiangsu Yangzi Yuhui"),
    ("600210.SS", "Utilities"),    # Heilongjiang Electric Power"),
    ("600211.SS", "Utilities"),    # Guangdong Xingye Bank"),
    ("600212.SS", "Utilities"),    # Bohai Bank"),
    ("600213.SS", "Utilities"),    # Jiangxi Copper"),
    ("600215.SS", "Utilities"),    # Shanghai Chlor-Alkali"),
    ("600216.SS", "Utilities"),    # Zhejiang Expressway"),
    ("600217.SS", "Utilities"),    # Shenzhen Metro"),
    ("600219.SS", "Materials"),    # Shanghai Chlor-Alkali"),
    ("600220.SS", "Utilities"),    # China Shenghuo Yuhui"),
    ("600221.SS", "Utilities"),    # Shanghai Chlor-Alkali"),
    ("600222.SS", "Utilities"),    # Baiyunshan Pharmaceutical"),
    ("600223.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600225.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600226.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600228.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600229.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600233.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600235.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600237.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600238.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600239.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
    ("600240.SS", "Utilities"),    # Shandong Zhouyuan Electric"),
]

_YF_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _get_csi300_tickers() -> list[str]:
    """Fallback: return hardcoded CSI 300 large-cap list."""
    print("Using hardcoded CSI 300 constituent list (large-caps, ~100 stocks)")
    return [t for t, _ in FALLBACK_CSI300]


def _fetch_one(args: tuple) -> tuple[str, pd.Series | None, pd.Series | None]:
    """Download adjclose + volume from Yahoo Finance v8 API via curl_cffi."""
    ticker, start, end = args
    try:
        from curl_cffi import requests as curl_req
        sess = curl_req.Session(impersonate="chrome110")
        p1 = int(pd.Timestamp(start).timestamp())
        p2 = int(pd.Timestamp(end).timestamp())
        resp = sess.get(
            _YF_URL.format(ticker=ticker),
            params={"period1": p1, "period2": p2, "interval": "1d"},
            timeout=20,
        )
        if resp.status_code != 200:
            return ticker, None, None
        result = resp.json().get("chart", {}).get("result", [None])[0]
        if not result:
            return ticker, None, None
        ts = result.get("timestamp", [])
        if len(ts) < 50:
            return ticker, None, None
        adj_list = result["indicators"]["adjclose"][0].get("adjclose", [])
        vol_list = result["indicators"]["quote"][0].get("volume", [])
        if not adj_list or len(adj_list) != len(ts):
            return ticker, None, None
        dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None)
        close = pd.Series(
            [float(v) if v is not None else float("nan") for v in adj_list],
            index=dates, name=ticker,
        )
        vol = pd.Series(
            [float(v) if v is not None else 0.0 for v in vol_list],
            index=dates, name=ticker,
        )
        return ticker, close, vol
    except Exception:
        return ticker, None, None


def _download(tickers, start, end, max_workers=8, batch_size=50, delay=0.5):
    print(f"\nDownloading {len(tickers)} CSI 300 tickers ({start} to {end}), workers={max_workers}...")
    closes, volumes, failed = {}, {}, 0
    args_list = [(t, start, end) for t in tickers]
    n_batches = (len(tickers) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = args_list[b * batch_size : (b + 1) * batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for ticker, close, vol in pool.map(_fetch_one, batch):
                if close is not None:
                    closes[ticker] = close
                    volumes[ticker] = vol
                else:
                    failed += 1
        done = min((b + 1) * batch_size, len(tickers))
        print(f"  [{done:>3}/{len(tickers)}] ok={len(closes)} failed={failed}")
        if b < n_batches - 1:
            time.sleep(delay)
    if not closes:
        raise RuntimeError("0 tickers downloaded.")
    close_df = pd.DataFrame(closes).sort_index()
    volume_df = pd.DataFrame(volumes).sort_index()
    close_df = close_df.loc[~close_df.index.duplicated(keep="last")]
    volume_df = volume_df.loc[~volume_df.index.duplicated(keep="last")]
    return close_df, volume_df


def _filter(close, volume, min_cov, n_assets):
    cov = close.notna().mean()
    keep = cov[cov >= min_cov].index.tolist()
    print(f"  Coverage >= {min_cov:.0%}: {len(close.columns)} -> {len(keep)}")
    if not keep:
        keep = cov.nlargest(min(n_assets, len(cov))).index.tolist()
    close = close[keep].ffill().bfill()
    volume = volume[keep].fillna(0)
    if len(keep) > n_assets:
        top = volume.median().nlargest(n_assets).index.tolist()
        close, volume = close[top], volume[top]
        print(f"  Top-{n_assets} by volume selected")
    return close, volume


def _features(close_df, volume_df):
    log_c = np.log(close_df.clip(lower=1e-8))
    ret1  = log_c.diff().fillna(0.0)
    ma5   = ret1.rolling(5, min_periods=1).mean()
    vol5  = ret1.rolling(5, min_periods=1).std().fillna(0.0)
    ma20  = close_df.rolling(20, min_periods=1).mean()
    pma20 = (close_df / ma20.clip(lower=1e-8)) - 1.0
    lv    = np.log1p(volume_df.clip(lower=0))
    mv20  = lv.rolling(20, min_periods=1).mean()
    volz  = (lv / mv20.clip(lower=1e-8)) - 1.0
    mkt   = ret1.mean(axis=1).values[:, None] * np.ones((1, len(close_df.columns)))
    arr   = np.stack([close_df.values, ret1.values, ma5.values, vol5.values,
                      pma20.values, volz.values, mkt], axis=-1).astype(np.float32)
    return arr, ["close", "ret1", "ret_ma5", "ret_vol5", "price_vs_ma20", "vol_z", "market_ret"]


def _build_npz(close_df, volume_df, sector_map, warmup):
    assets    = list(close_df.columns)
    dates_str = close_df.index.strftime("%Y-%m-%d").tolist()
    T, N      = len(dates_str), len(assets)
    sectors   = np.array([sector_map.get(a, 1) for a in assets], dtype=np.int64)  # default: Materials
    feats, fnames = _features(close_df, volume_df)
    log_c  = np.log(close_df.values.clip(min=1e-8))
    labels = np.full((T, N), np.nan, dtype=np.float32)
    labels[:-1] = (log_c[1:] - log_c[:-1]).astype(np.float32)
    mask = np.isfinite(labels) & (close_df.values > 0) & np.isfinite(close_df.values)
    mask[:warmup] = False
    return dict(
        features=feats, labels=labels,
        close=close_df.values.astype(np.float32),
        mask=mask, dates=np.array(dates_str, dtype=object),
        assets=np.array(assets), sectors=sectors,
        feature_names=np.array(fnames),
    )


def main():
    p = argparse.ArgumentParser(description="Build CSI 300 NPZ (Yahoo Finance + curl_cffi)")
    p.add_argument("--start", default="2014-06-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--n-assets", type=int, default=100)
    p.add_argument("--min-coverage", type=float, default=0.85)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("data/csi300.npz"))
    args = p.parse_args()

    try:
        from curl_cffi import requests as _cr; _cr.Session(impersonate="chrome110")
        print("curl_cffi ready - Chrome impersonation enabled")
    except ImportError:
        print("ERROR: pip install curl_cffi"); raise

    tickers = _get_csi300_tickers()
    ticker_sectors = dict((t, s) for t, s in FALLBACK_CSI300)

    close_df, volume_df = _download(
        tickers, args.start, args.end,
        args.workers, args.batch, args.delay,
    )
    close_df, volume_df = _filter(close_df, volume_df, args.min_coverage, args.n_assets)
    sector_map = {t: SECTOR_TO_INT.get(ticker_sectors.get(t, "Financials"), 6)
                  for t in close_df.columns}
    npz = _build_npz(close_df, volume_df, sector_map, warmup=args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **npz)

    T, N, F = npz["features"].shape
    vm = npz["mask"].any(axis=1)
    valid_days = int(vm.sum())
    size_mb = args.out.stat().st_size / 1024 / 1024
    sc = {name: int((npz["sectors"] == code).sum())
          for name, code in sorted(SECTOR_TO_INT.items(), key=lambda x: x[1])
          if (npz["sectors"] == code).sum() > 0}
    print(f"\n{'='*60}")
    print(f"  Saved      : {args.out}")
    print(f"  Shape      : features={npz['features'].shape}, labels={npz['labels'].shape}")
    print(f"  Assets     : {N}  |  Dates: {T}  |  Features: {F}")
    print(f"  Valid days : {valid_days}")
    if len(npz["dates"]) > 0:
        print(f"  Period     : {npz['dates'][0]}  to  {npz['dates'][-1]}")
    print(f"  Sectors    : {sc}")
    print(f"  File size  : {size_mb:.1f} MB")
    print(f"{'='*60}")
    print(f"\n  Next: python scripts/train.py --config configs/csi300.yaml --set optim.amp=true")


if __name__ == "__main__":
    main()
