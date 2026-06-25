#!/usr/bin/env python
"""
Download real S&P 500 data for aaai27-stock-regime-mixer.

Uses curl_cffi (Chrome impersonation) + yfinance to bypass Yahoo Finance bot detection.
Requires: pip install curl_cffi yfinance

Period:  2014-2024  (buffer for warm-up; paper uses 2015-2023)
Size:    ~150 most-liquid stocks, 7 features, ~2250 trading days.

Usage:
    python scripts/make_real_data.py                    # default 150 stocks
    python scripts/make_real_data.py --n-assets 80      # quick test
    python scripts/make_real_data.py --start 2018-01-01 --end 2023-12-31
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

# Hardcoded fallback: 160 stable S&P 500 large-caps.
# Used only if Wikipedia scrape fails.
FALLBACK_TICKERS: list[tuple[str, str]] = [
    ("AAPL","Information Technology"),("ABBV","Health Care"),("ABT","Health Care"),
    ("ACN","Information Technology"),("ADBE","Information Technology"),("ADI","Information Technology"),
    ("ADP","Information Technology"),("AEP","Utilities"),("AIG","Financials"),
    ("AMAT","Information Technology"),("AMD","Information Technology"),("AMGN","Health Care"),
    ("AMZN","Consumer Discretionary"),("AON","Financials"),("APD","Materials"),
    ("APH","Information Technology"),("AXP","Financials"),("AZO","Consumer Discretionary"),
    ("BA","Industrials"),("BAC","Financials"),("BAX","Health Care"),("BDX","Health Care"),
    ("BKR","Energy"),("BLK","Financials"),("BMY","Health Care"),("BSX","Health Care"),
    ("BX","Financials"),("C","Financials"),("CAT","Industrials"),("CB","Financials"),
    ("CDNS","Information Technology"),("CI","Health Care"),("CL","Consumer Staples"),
    ("CME","Financials"),("CMCSA","Communication Services"),("CMG","Consumer Discretionary"),
    ("CMI","Industrials"),("CNC","Health Care"),("COF","Financials"),("COP","Energy"),
    ("COST","Consumer Staples"),("CRM","Information Technology"),("CSCO","Information Technology"),
    ("CSX","Industrials"),("CTAS","Industrials"),("CVS","Health Care"),("CVX","Energy"),
    ("D","Utilities"),("DAL","Industrials"),("DE","Industrials"),("DHR","Health Care"),
    ("DIS","Communication Services"),("DOV","Industrials"),("DUK","Utilities"),("DVN","Energy"),
    ("ECL","Materials"),("ED","Utilities"),("EL","Consumer Staples"),("EMR","Industrials"),
    ("EOG","Energy"),("ETN","Industrials"),("ETR","Utilities"),("EW","Health Care"),
    ("EXC","Utilities"),("F","Consumer Discretionary"),("FCX","Materials"),("FDX","Industrials"),
    ("FIS","Information Technology"),("FISV","Information Technology"),("FITB","Financials"),
    ("GD","Industrials"),("GE","Industrials"),("GILD","Health Care"),("GM","Consumer Discretionary"),
    ("GOOGL","Communication Services"),("GPC","Consumer Discretionary"),("GS","Financials"),
    ("HAL","Energy"),("HD","Consumer Discretionary"),("HON","Industrials"),("HPQ","Information Technology"),
    ("HUM","Health Care"),("IBM","Information Technology"),("ICE","Financials"),("IDXX","Health Care"),
    ("IFF","Materials"),("INTC","Information Technology"),("INTU","Information Technology"),
    ("ISRG","Health Care"),("ITW","Industrials"),("JNJ","Health Care"),("JPM","Financials"),
    ("KEY","Financials"),("KMB","Consumer Staples"),("KO","Consumer Staples"),("KR","Consumer Staples"),
    ("LH","Health Care"),("LIN","Materials"),("LLY","Health Care"),("LMT","Industrials"),
    ("LOW","Consumer Discretionary"),("LYB","Materials"),("MA","Information Technology"),
    ("MAR","Consumer Discretionary"),("MCD","Consumer Discretionary"),("MCO","Financials"),
    ("MDT","Health Care"),("META","Communication Services"),("MMC","Financials"),("MMM","Industrials"),
    ("MO","Consumer Staples"),("MRK","Health Care"),("MS","Financials"),("MSFT","Information Technology"),
    ("MU","Information Technology"),("NEE","Utilities"),("NFLX","Communication Services"),
    ("NKE","Consumer Discretionary"),("NOC","Industrials"),("NSC","Industrials"),("NTRS","Financials"),
    ("NUE","Materials"),("NVDA","Information Technology"),("OKE","Energy"),("ORCL","Information Technology"),
    ("OXY","Energy"),("PANW","Information Technology"),("PAYX","Information Technology"),
    ("PEP","Consumer Staples"),("PFE","Health Care"),("PG","Consumer Staples"),
    ("PGR","Financials"),("PH","Industrials"),("PM","Consumer Staples"),("PNC","Financials"),
    ("PRU","Financials"),("PSA","Real Estate"),("PSX","Energy"),("PYPL","Information Technology"),
    ("QCOM","Information Technology"),("REGN","Health Care"),("RF","Financials"),("RJF","Financials"),
    ("ROST","Consumer Discretionary"),("RTX","Industrials"),("SBUX","Consumer Discretionary"),
    ("SCHW","Financials"),("SHW","Materials"),("SLB","Energy"),("SO","Utilities"),
    ("SPGI","Financials"),("SPG","Real Estate"),("STT","Financials"),("STZ","Consumer Staples"),
    ("SYK","Health Care"),("T","Communication Services"),("TGT","Consumer Discretionary"),
    ("TMO","Health Care"),("TROW","Financials"),("TRV","Financials"),("TSLA","Consumer Discretionary"),
    ("TXN","Information Technology"),("UNH","Health Care"),("UNP","Industrials"),("UPS","Industrials"),
    ("URI","Industrials"),("USB","Financials"),("V","Information Technology"),("VLO","Energy"),
    ("VRTX","Health Care"),("VZ","Communication Services"),("WFC","Financials"),("WMT","Consumer Staples"),
    ("XOM","Energy"),("XYL","Industrials"),("YUM","Consumer Discretionary"),("ZTS","Health Care"),
]

_WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def _get_sp500_info() -> pd.DataFrame:
    """Scrape S&P 500 tickers + GICS sectors from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        print("Fetching S&P 500 constituent list from Wikipedia...")
        req = urllib.request.Request(url, headers=_WIKI_HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        df = tables[0][["Symbol", "GICS Sector"]].copy()
        df.columns = ["ticker", "sector"]
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df = df.dropna(subset=["ticker", "sector"]).drop_duplicates("ticker")
        print(f"  Found {len(df)} tickers across {df['sector'].nunique()} GICS sectors")
        return df
    except Exception as e:
        print(f"  Wikipedia scrape failed ({type(e).__name__}), using built-in fallback list ({len(FALLBACK_TICKERS)} tickers)")
        return pd.DataFrame(FALLBACK_TICKERS, columns=["ticker", "sector"])


# ── Yahoo Finance v8 API via curl_cffi Chrome session ─────────────────────────

_YF_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _make_curl_session():
    """Create a Chrome-impersonating requests session (bypasses Yahoo Finance bot detection)."""
    from curl_cffi import requests as curl_req
    return curl_req.Session(impersonate="chrome110")


def _fetch_one_ticker(args: tuple) -> tuple[str, pd.Series | None, pd.Series | None]:
    """
    Download adjusted close + volume directly from Yahoo Finance v8/chart API.
    Uses curl_cffi Chrome impersonation — no yfinance wrapper needed.
    """
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
        payload = resp.json()
        result = payload.get("chart", {}).get("result", None)
        if not result:
            return ticker, None, None
        result = result[0]
        timestamps = result.get("timestamp", [])
        if len(timestamps) < 50:
            return ticker, None, None
        ind = result.get("indicators", {})
        adj_list = ind.get("adjclose", [{}])[0].get("adjclose", [])
        vol_list = ind.get("quote", [{}])[0].get("volume", [])
        if not adj_list or len(adj_list) != len(timestamps):
            return ticker, None, None
        dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
        # Replace None with NaN
        adj_vals = [float(v) if v is not None else float("nan") for v in adj_list]
        vol_vals = [float(v) if v is not None else 0.0 for v in vol_list]
        close = pd.Series(adj_vals, index=dates, name=ticker)
        vol = pd.Series(vol_vals, index=dates, name=ticker)
        return ticker, close, vol
    except Exception:
        return ticker, None, None


def _download_prices(
    tickers: list[str],
    start: str,
    end: str,
    max_workers: int = 8,
    batch_size: int = 50,
    delay: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close + volume for all tickers.
    Uses curl_cffi (Chrome impersonation) to bypass Yahoo Finance bot detection.
    """
    print(f"\nDownloading {len(tickers)} tickers ({start} to {end}), workers={max_workers}...")
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    failed = 0
    args_list = [(t, start, end) for t in tickers]
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        batch_args = args_list[b_idx * batch_size : (b_idx + 1) * batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one_ticker, a): a[0] for a in batch_args}
            for fut in as_completed(futures):
                ticker, close, vol = fut.result()
                if close is not None:
                    closes[ticker] = close
                    volumes[ticker] = vol
                else:
                    failed += 1
        done = min((b_idx + 1) * batch_size, len(tickers))
        print(f"  [{done:>3}/{len(tickers)}] downloaded={len(closes)} failed={failed}")
        if b_idx < n_batches - 1:
            time.sleep(delay)

    if not closes:
        raise RuntimeError(
            "0 tickers downloaded. Ensure curl_cffi is installed: pip install curl_cffi\n"
            "Test connectivity: python -c \"from curl_cffi import requests as r; s=r.Session(impersonate='chrome110'); print(s.get('https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=5d').status_code)\""
        )

    print(f"  Aligning {len(closes)} tickers on trading calendar...")
    close_df = pd.DataFrame(closes).sort_index()
    volume_df = pd.DataFrame(volumes).sort_index()
    close_df = close_df.loc[~close_df.index.duplicated(keep="last")]
    volume_df = volume_df.loc[~volume_df.index.duplicated(keep="last")]
    return close_df, volume_df


def _filter_assets(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    min_coverage: float,
    n_assets: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop low-coverage stocks; keep top-N by median daily volume."""
    coverage = close.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    print(f"  Coverage >= {min_coverage:.0%}: {len(close.columns)} stocks -> {len(keep)} pass")
    if not keep:
        keep = coverage.nlargest(min(n_assets, len(coverage))).index.tolist()
        print(f"  Relaxed: keeping top-{len(keep)} by coverage")

    close = close[keep].ffill().bfill()
    volume = volume[keep].fillna(0)

    if len(keep) > n_assets:
        top_n = volume.median().nlargest(n_assets).index.tolist()
        close = close[top_n]
        volume = volume[top_n]
        print(f"  Top-{n_assets} by median volume selected")

    return close, volume


def _compute_features(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute (T, N, 7) causal feature tensor.
    Matches feature structure of make_synthetic.py:
        0 close          - adjusted close
        1 ret1           - 1-day log return
        2 ret_ma5        - 5d rolling mean(ret1)
        3 ret_vol5       - 5d rolling std(ret1)
        4 price_vs_ma20  - close/MA20 - 1
        5 vol_z          - log_vol/MA20_vol - 1
        6 market_ret     - cross-sectional mean(ret1)  [causal market factor]
    """
    log_c = np.log(close_df.clip(lower=1e-8))
    ret1 = log_c.diff().fillna(0.0)
    ma5 = ret1.rolling(5, min_periods=1).mean()
    vol5 = ret1.rolling(5, min_periods=1).std().fillna(0.0)
    ma20_px = close_df.rolling(20, min_periods=1).mean()
    price_vs_ma20 = (close_df / ma20_px.clip(lower=1e-8)) - 1.0
    log_vol = np.log1p(volume_df.clip(lower=0))
    ma20_vol = log_vol.rolling(20, min_periods=1).mean()
    vol_z = (log_vol / ma20_vol.clip(lower=1e-8)) - 1.0
    mkt = ret1.mean(axis=1).values[:, None] * np.ones((1, len(close_df.columns)))
    stacked = np.stack([
        close_df.values, ret1.values, ma5.values, vol5.values,
        price_vs_ma20.values, vol_z.values, mkt,
    ], axis=-1).astype(np.float32)
    names = ["close", "ret1", "ret_ma5", "ret_vol5", "price_vs_ma20", "vol_z", "market_ret"]
    return stacked, names


def _build_npz(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    sector_map: dict[str, int],
    warmup_days: int,
) -> dict:
    assets = list(close_df.columns)
    dates_str = close_df.index.strftime("%Y-%m-%d").tolist()
    T, N = len(dates_str), len(assets)
    sectors = np.array([sector_map.get(a, 0) for a in assets], dtype=np.int64)
    features, feat_names = _compute_features(close_df, volume_df)
    log_c = np.log(close_df.values.clip(min=1e-8))
    labels = np.full((T, N), np.nan, dtype=np.float32)
    labels[:-1] = (log_c[1:] - log_c[:-1]).astype(np.float32)
    mask = np.isfinite(labels) & (close_df.values > 0) & np.isfinite(close_df.values)
    mask[:warmup_days] = False
    return dict(
        features=features, labels=labels,
        close=close_df.values.astype(np.float32),
        mask=mask,
        dates=np.array(dates_str, dtype=object),
        assets=np.array(assets),
        sectors=sectors,
        feature_names=np.array(feat_names),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real S&P 500 NPZ for aaai27-stock-regime-mixer (source: Yahoo Finance v8 + curl_cffi)."
    )
    parser.add_argument("--start", default="2014-06-01",
                        help="Download start (buffer for rolling warm-up)")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--n-assets", type=int, default=150,
                        help="Keep top-N stocks by median daily volume")
    parser.add_argument("--min-coverage", type=float, default=0.90,
                        help="Minimum fraction of non-NaN days to keep a stock")
    parser.add_argument("--warmup", type=int, default=60,
                        help="Mask out first N days (rolling feature warm-up)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download workers (Chrome curl sessions)")
    parser.add_argument("--batch", type=int, default=50,
                        help="Tickers per batch")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to pause between batches")
    parser.add_argument("--out", type=Path, default=Path("data/sp500_real.npz"))
    args = parser.parse_args()

    # 1. Universe
    info = _get_sp500_info()
    ticker_sector = dict(zip(info["ticker"], info["sector"]))

    # Verify curl_cffi available
    try:
        from curl_cffi import requests as _cr
        _cr.Session(impersonate="chrome110")
        print("curl_cffi ready - Chrome impersonation enabled")
    except ImportError:
        print("ERROR: curl_cffi not found. Run: pip install curl_cffi")
        raise

    # 2. Download via Yahoo Finance + curl_cffi
    close_df, volume_df = _download_prices(
        info["ticker"].tolist(),
        start=args.start, end=args.end,
        max_workers=args.workers,
        batch_size=args.batch,
        delay=args.delay,
    )

    # 3. Filter
    close_df, volume_df = _filter_assets(close_df, volume_df, args.min_coverage, args.n_assets)
    sector_map = {t: GICS_TO_INT.get(ticker_sector.get(t, ""), 0) for t in close_df.columns}

    # 4. Build NPZ
    npz = _build_npz(close_df, volume_df, sector_map, warmup_days=args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **npz)

    # 5. Summary
    T, N, F = npz["features"].shape
    vm = npz["mask"].any(axis=1)
    valid_days = int(vm.sum())
    avg_valid = float(npz["mask"].sum(axis=1)[vm].mean()) if valid_days > 0 else 0
    n_sectors = int(np.unique(npz["sectors"]).size)
    sector_counts = {
        name: int((npz["sectors"] == code).sum())
        for name, code in sorted(GICS_TO_INT.items(), key=lambda x: x[1])
        if (npz["sectors"] == code).sum() > 0
    }
    size_mb = args.out.stat().st_size / 1024 / 1024

    print(f"\n{'='*64}")
    print(f"  Saved     : {args.out}")
    print(f"  Shape     : features={npz['features'].shape}, labels={npz['labels'].shape}")
    print(f"  Assets    : {N}  |  Dates: {T}  |  Features: {F}")
    print(f"  Valid days: {valid_days}  (avg {avg_valid:.0f} assets/day)")
    if len(npz["dates"]) > 0:
        print(f"  Period    : {npz['dates'][0]}  to  {npz['dates'][-1]}")
    print(f"  Sectors   : {n_sectors}  ->  {sector_counts}")
    print(f"  File size : {size_mb:.1f} MB")
    print(f"{'='*64}")
    print("\nNext:")
    print(f"  python scripts/inspect_dataset.py --data {args.out}")
    print(f"  python scripts/train.py --config configs/real_sp500.yaml --set optim.amp=true")


if __name__ == "__main__":
    main()
