#!/usr/bin/env python
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aaai27_stock.config import load_config, save_config
from aaai27_stock.seed import seed_everything, resolve_device
from aaai27_stock.data import make_dataloaders, write_metadata
from aaai27_stock.models import build_model
from aaai27_stock.trainer import fit


def parse_override(items):
    # Minimal dotlist override: a.b.c=value, JSON parsed when possible.
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate regime-conditioned stock forecasting models.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--set", nargs="*", default=[], help="Dotlist overrides, e.g. model.name=stockmixer optim.epochs=5")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_override(args.set))
    seed_everything(int(cfg.get("seed", 2027)), benchmark=bool(cfg.get("cudnn_benchmark", False)))
    device = resolve_device(cfg.get("device", "auto"))
    outdir = Path(cfg.get("output_dir", "outputs/run"))
    outdir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, outdir / "config.resolved.yaml")

    loaders, meta = make_dataloaders(cfg)
    write_metadata(meta, outdir)
    model = build_model(cfg, meta).to(device)
    print(f"device={device} model={cfg['model']['name']} params={getattr(model, 'num_parameters', 'n/a')} meta={meta['split_sizes']}", flush=True)
    result = fit(model, loaders, cfg, device, meta)
    test = result.get("metrics", {}).get("test", {})
    print("DONE", {"best_epoch": result.get("best_epoch"), "test_rank_ic": test.get("rank_ic"), "test_ls_sharpe": test.get("long_short_sharpe"), "results": str(outdir / "results.json")}, flush=True)


if __name__ == "__main__":
    main()
