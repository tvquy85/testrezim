#!/usr/bin/env python
"""
Download NASDAQ-100 data for aaai27-stock-regime-mixer.

Why NASDAQ-100:
  - Tech/growth-heavy (IT ~60%, CommSvcs ~20%) vs S&P 500 broad market
  - Distinct regime signatures: 2020 COVID boom, 2021 ZIRP peak, 2022 rate-shock bear
  - Narrow sector concentration amplifies cross-stock regime co-movement
    -> stronger test of causal regime conditioning

Source: Yahoo Finance v8 API + curl_cffi Chrome impersonation
Requires: pip install curl_cffi

Period:  2014-2024  (warmup buffer; effective 2015-2023 after masking)
Size:    ~95 most-liquid NASDAQ-100 stocks, 7 features, ~2400 trading days

Usage:
    python scripts/make_nasdaq100_data.py
    python scripts/make_nasdaq100_data.py --n-assets 80 --out data/nasdaq100_small.npz
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import io
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

GICS_TO_INT: dict[str, int] = {
    "Energy": 0, "Materials": 1, "Industrials": 2,
    "Consumer Discretionary": 3, "Consumer Staples": 4,
    "Health Care": 5, "Financials": 6, "Information Technology": 7,
    "Communication Services": 8, "Utilities": 9, "Real Estate": 10,
}

# NASDAQ-100 hardcoded list (stable large-cap members, as of 2024).
# Wikipedia scrape is tried first; this is the fallback.
FALLBACK_NDX: list[tuple[str, str]] = [
    ("AAPL","Information Technology"),("ABNB","Consumer Discretionary"),("ADBE","Information Technology"),
    ("ADI","Information Technology"),("ADP","Information Technology"),("ADSK","Information Technology"),
    ("AEP","Utilities"),("AMAT","Information Technology"),("AMD","Information Technology"),
    ("AMGN","Health Care"),("AMZN","Consumer Discretionary"),("ANSS","Information Technology"),
    ("ASML","Information Technology"),("AVGO","Information Technology"),("BIDU","Communication Services"),
    ("BIIB","Health Care"),("BKNG","Consumer Discretionary"),("BKR","Energy"),
    ("CCEP","Consumer Staples"),("CDNS","Information Technology"),("CEG","Utilities"),
    ("CHTR","Communication Services"),("CMCSA","Communication Services"),("COST","Consumer Staples"),
    ("CPRT","Industrials"),("CRWD","Information Technology"),("CSCO","Information Technology"),
    ("CSX","Industrials"),("CTAS","Industrials"),("CTSH","Information Technology"),
    ("DDOG","Information Technology"),("DLTR","Consumer Staples"),("DXCM","Health Care"),
    ("EA","Communication Services"),("ENPH","Information Technology"),("EXC","Utilities"),
    ("FANG","Energy"),("FAST","Industrials"),("FTNT","Information Technology"),
    ("GEHC","Health Care"),("GILD","Health Care"),("GOOG","Communication Services"),
    ("GOOGL","Communication Services"),("HON","Industrials"),("IDXX","Health Care"),
    ("ILMN","Health Care"),("INTC","Information Technology"),("INTU","Information Technology"),
    ("ISRG","Health Care"),("JD","Consumer Discretionary"),("KDP","Consumer Staples"),
    ("KLAC","Information Technology"),("LRCX","Information Technology"),("MAR","Consumer Discretionary"),
    ("MELI","Consumer Discretionary"),("META","Communication Services"),("MNST","Consumer Staples"),
    ("MRNA","Health Care"),("MRVL","Information Technology"),("MSFT","Information Technology"),
    ("MU","Information Technology"),("NFLX","Communication Services"),("NTES","Communication Services"),
    ("NVDA","Information Technology"),("NXPI","Information Technology"),("ODFL","Industrials"),
    ("ON","Information Technology"),("ORLY","Consumer Discretionary"),("PANW","Information Technology"),
    ("PAYX","Information Technology"),("PCAR","Industrials"),("PDD","Consumer Discretionary"),
    ("PEP","Consumer Staples"),("PYPL","Information Technology"),("QCOM","Information Technology"),
    ("REGN","Health Care"),("ROST","Consumer Discretionary"),("SBUX","Consumer Discretionary"),
    ("SIRI","Communication Services"),("SNPS","Information Technology"),("TEAM","Information Technology"),
    ("TMUS","Communication Services"),("TSLA","Consumer Discretionary"),("TTWO","Communication Services"),
    ("TXN","Information Technology"),("VRSK","Industrials"),("VRTX","Health Care"),
    ("WDAY","Information Technology"),("XEL","Utilities"),("ZS","Information Technology"),
]

_WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_YF_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _get_ndx_info() -> pd.DataFrame:
    """Scrape NASDAQ-100 components from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        print("Fetching NASDAQ-100 constituent list from Wikipedia...")
        req = urllib.request.Request(url, headers=_WIKI_HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        # Find table with Ticker/Symbol column
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                sym_col = next(c for c in tbl.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower())
                sec_col = next((c for c in tbl.columns if "sector" in str(c).lower() or "gics" in str(c).lower()), None)
                df = tbl[[sym_col]].copy()
                df.columns = ["ticker"]
                df["sector"] = tbl[sec_col] if sec_col else "Information Technology"
                df = df.dropna(subset=["ticker"]).drop_duplicates("ticker")
                df["ticker"] = df["ticker"].str.strip()
                if len(df) >= 80:
                    print(f"  Found {len(df)} NASDAQ-100 tickers across {df['sector'].nunique()} sectors")
                    return df
    except Exception as e:
        print(f"  Wikipedia scrape failed ({type(e).__name__})")
    print(f"  Using built-in fallback ({len(FALLBACK_NDX)} tickers)")
    return pd.DataFrame(FALLBACK_NDX, columns=["ticker", "sector"])


def _fetch_one(args: tuple) -> tuple[str, pd.Series | None, pd.Series | None]:
    """Download adjclose + volume directly from Yahoo Finance v8 API via curl_cffi."""
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
        timestamps = result.get("timestamp", [])
        if len(timestamps) < 50:
            return ticker, None, None
        ind = result.get("indicators", {})
        adj_list = ind.get("adjclose", [{}])[0].get("adjclose", [])
        vol_list = ind.get("quote", [{}])[0].get("volume", [])
        if not adj_list or len(adj_list) != len(timestamps):
            return ticker, None, None
        dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
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
    print(f"\nDownloading {len(tickers)} NASDAQ-100 tickers ({start} to {end}), workers={max_workers}...")
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
    sectors   = np.array([sector_map.get(a, 7) for a in assets], dtype=np.int64)  # default: IT
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
    p = argparse.ArgumentParser(description="Build NASDAQ-100 NPZ (Yahoo Finance + curl_cffi)")
    p.add_argument("--start", default="2014-06-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--n-assets", type=int, default=95)
    p.add_argument("--min-coverage", type=float, default=0.90)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("data/nasdaq100.npz"))
    args = p.parse_args()

    try:
        from curl_cffi import requests as _cr; _cr.Session(impersonate="chrome110")
        print("curl_cffi ready - Chrome impersonation enabled")
    except ImportError:
        print("ERROR: pip install curl_cffi"); raise

    info = _get_ndx_info()
    ticker_sector = dict(zip(info["ticker"], info["sector"]))

    close_df, volume_df = _download(
        info["ticker"].tolist(), args.start, args.end,
        args.workers, args.batch, args.delay,
    )
    close_df, volume_df = _filter(close_df, volume_df, args.min_coverage, args.n_assets)
    sector_map = {t: GICS_TO_INT.get(ticker_sector.get(t, "Information Technology"), 7)
                  for t in close_df.columns}
    npz = _build_npz(close_df, volume_df, sector_map, warmup=args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **npz)

    T, N, F = npz["features"].shape
    vm = npz["mask"].any(axis=1)
    valid_days = int(vm.sum())
    size_mb = args.out.stat().st_size / 1024 / 1024
    sc = {name: int((npz["sectors"] == code).sum())
          for name, code in sorted(GICS_TO_INT.items(), key=lambda x: x[1])
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
    print(f"\n  Next: python scripts/train.py --config configs/nasdaq100.yaml --set optim.amp=true")


if __name__ == "__main__":
    main()
