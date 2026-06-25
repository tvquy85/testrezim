#!/usr/bin/env python
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def run_job(cmd: list, outdir: Path, label: str) -> dict:
    _log(f"[START] {label}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    # Print last N lines of stdout so parallel output is readable
    stdout_tail = "\n".join(proc.stdout.strip().splitlines()[-6:]) if proc.stdout else ""
    if proc.returncode != 0:
        with _print_lock:
            print(f"[FAIL] {label}\n--- stderr ---\n{proc.stderr[-2000:]}\n--- stdout tail ---\n{stdout_tail}", flush=True)
        raise RuntimeError(f"Job failed: {label}\n{proc.stderr[-1000:]}")
    with _print_lock:
        if stdout_tail:
            print(f"[OUT] {label}\n{stdout_tail}", flush=True)
    with open(outdir / "results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a small model sweep from one base config.")
    p.add_argument("--base", type=str, default="configs/synthetic_crc_lora.yaml")
    p.add_argument("--models", nargs="+", default=["dlinear", "stockmixer", "context_gmlp", "film", "crc_lora", "sector_moe"])
    p.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel training jobs. E.g. --workers 4 runs 4 jobs simultaneously on the same GPU.")
    p.add_argument("--amp", action="store_true",
                   help="Enable AMP (fp16 autocast) for faster GPU training (passed as optim.amp=true)")
    p.add_argument("--benchmark", action="store_true",
                   help="Enable cudnn.benchmark for speed (passed as cudnn_benchmark=true)")
    p.add_argument("--prefix", type=str, default="sweep",
                   help="Output directory prefix, e.g. 'real_sweep' to avoid overwriting synthetic runs")
    args = p.parse_args()

    # Build task list
    tasks: list[tuple[str, int, Path, list]] = []
    for model_name in args.models:
        for seed in args.seeds:
            outdir = ROOT / "outputs" / f"{args.prefix}_{model_name}_seed{seed}"
            cmd = [
                sys.executable, str(ROOT / "scripts" / "train.py"),
                "--config", str(ROOT / args.base),
                "--set",
                f"model.name={json.dumps(model_name)}",
                f"seed={seed}",
                f"output_dir={json.dumps(str(outdir))}",
            ]
            if args.epochs is not None:
                cmd.append(f"optim.epochs={args.epochs}")
            if args.amp:
                cmd.append("optim.amp=true")
            if args.benchmark:
                cmd.append("cudnn_benchmark=true")
            tasks.append((model_name, seed, outdir, cmd))

    total = len(tasks)
    workers = min(args.workers, total)
    print(f"Sweep: {total} jobs | workers={workers} | amp={args.amp} | benchmark={args.benchmark}", flush=True)

    summary = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_info = {
            executor.submit(run_job, cmd, outdir, f"{model}[seed={seed}]"): (model, seed, outdir)
            for model, seed, outdir, cmd in tasks
        }
        for future in as_completed(future_to_info):
            model_name, seed, outdir = future_to_info[future]
            done += 1
            label = f"{model_name}[seed={seed}]"
            try:
                res = future.result()
                row = {"model": model_name, "seed": seed, "output_dir": str(outdir)}
                for split in ["valid", "test"]:
                    for k, v in res["metrics"][split].items():
                        row[f"{split}_{k}"] = v
                summary.append(row)
                test = res["metrics"].get("test", {})
                _log(
                    f"[DONE {done}/{total}] {label} "
                    f"rank_ic={test.get('rank_ic', float('nan')):.4f} "
                    f"ls_sharpe={test.get('long_short_sharpe', float('nan')):.3f}"
                )
            except Exception as exc:
                _log(f"[ERROR {done}/{total}] {label}: {exc}")

    out = ROOT / "outputs" / f"{args.prefix}_summary.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved sweep summary ({len(summary)}/{total} jobs succeeded): {out}", flush=True)


if __name__ == "__main__":
    main()
