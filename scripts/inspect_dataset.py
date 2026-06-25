#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aaai27_stock.config import load_config
from aaai27_stock.data import make_dataloaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    _, meta = make_dataloaders(cfg)
    for k, v in meta.items():
        if k in {"dates", "assets"}:
            print(k, len(v), v[:3], "...", v[-3:])
        elif k == "split_date_indices":
            print(k, {s: (len(x), x[0] if x else None, x[-1] if x else None) for s, x in v.items()})
        else:
            print(k, v)

if __name__ == "__main__":
    main()
