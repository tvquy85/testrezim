from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any
import csv
import json
import time
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .losses import composite_loss
from .metrics import compute_metrics


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def collect_predictions(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, masks, dates = [], [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch["x"], batch.get("context"), batch.get("sectors"))
        preds.append(out.detach().cpu().numpy())
        ys.append(batch["y"].detach().cpu().numpy())
        masks.append(batch["mask"].detach().cpu().numpy())
        dates.append(batch["date_index"].detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0), np.concatenate(masks, axis=0), np.concatenate(dates, axis=0)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, eval_cfg: Dict[str, Any]) -> Dict[str, float]:
    pred, y, mask, _ = collect_predictions(model, loader, device)
    return compute_metrics(
        pred,
        y,
        mask,
        topk=int(eval_cfg.get("topk", 10)),
        trading_days=int(eval_cfg.get("trading_days", 252)),
        transaction_cost_bps=eval_cfg.get("transaction_cost_bps", [0, 5, 10]),
    )


def train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device, loss_cfg: Dict[str, Any], grad_clip: float = 1.0, scaler=None) -> Dict[str, float]:
    model.train()
    use_amp = scaler is not None
    totals: Dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="train", leave=False, disable=not sys.stderr.isatty()):
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(batch["x"], batch.get("context"), batch.get("sectors"))
            loss, logs = composite_loss(pred, batch["y"], batch["mask"], loss_cfg)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        count += 1
    return {k: v / max(1, count) for k, v in totals.items()}


def fit(model: nn.Module, loaders: Dict, cfg: Dict[str, Any], device: torch.device, meta: Dict[str, Any]) -> Dict[str, Any]:
    outdir = Path(cfg.get("output_dir", "outputs/run"))
    outdir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"].get("lr", 1e-3)),
        weight_decay=float(cfg["optim"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["optim"].get("epochs", 20))
    patience = int(cfg["optim"].get("patience", 5))
    grad_clip = float(cfg["optim"].get("grad_clip", 1.0))
    use_amp = bool(cfg.get("optim", {}).get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print(f"AMP enabled (fp16 autocast + GradScaler)", flush=True)
    best_score = -1e18
    best_epoch = -1
    history = []
    t0 = time.time()
    ckpt_path = outdir / "best_model.pt"
    log_csv = outdir / "history.csv"

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = None
        for epoch in range(1, epochs + 1):
            train_logs = train_one_epoch(model, loaders["train"], optimizer, device, cfg.get("loss", {}), grad_clip=grad_clip, scaler=scaler)
            valid_metrics = evaluate(model, loaders["valid"], device, cfg.get("eval", {}))
            score = valid_metrics.get("rank_ic", np.nan)
            if not np.isfinite(score):
                score = -train_logs.get("loss_total", 0.0)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_logs.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
            history.append(row)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(f"epoch={epoch:03d} train_loss={train_logs.get('loss_total', float('nan')):.6f} valid_rank_ic={valid_metrics.get('rank_ic', float('nan')):.6f} valid_ls_sharpe={valid_metrics.get('long_short_sharpe', float('nan')):.3f}")
            if score > best_score:
                best_score = float(score)
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "cfg": cfg, "meta": meta, "epoch": epoch, "score": best_score}, ckpt_path)
            elif epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
                break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    final_metrics = {split: evaluate(model, loader, device, cfg.get("eval", {})) for split, loader in loaders.items()}
    result = {
        "best_epoch": best_epoch,
        "best_valid_rank_ic": best_score,
        "num_parameters": int(getattr(model, "num_parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))),
        "elapsed_sec": time.time() - t0,
        "metrics": final_metrics,
    }
    with open(outdir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result
