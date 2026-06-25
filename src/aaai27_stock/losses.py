from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn.functional as F

EPS = 1e-8


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff.masked_fill(~mask, 0.0)
    denom = mask.float().sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_corr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = []
    for p, y, m in zip(pred, target, mask):
        p = p[m]
        y = y[m]
        if p.numel() < 2:
            continue
        p = p - p.mean()
        y = y - y.mean()
        denom = p.std(unbiased=False).clamp_min(EPS) * y.std(unbiased=False).clamp_min(EPS)
        vals.append((p * y).mean() / denom)
    if not vals:
        return pred.new_tensor(0.0)
    return torch.stack(vals).mean()


def bpr_rank_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, max_pairs: int = 4096) -> torch.Tensor:
    losses = []
    rng = torch.Generator(device=pred.device)
    rng.manual_seed(2027)
    for p, y, m in zip(pred, target, mask):
        idx = torch.where(m)[0]
        if idx.numel() < 2:
            continue
        p = p[idx]
        y = y[idx]
        n = p.numel()
        if n * n <= max_pairs:
            dy = y.unsqueeze(0) - y.unsqueeze(1)
            dp = p.unsqueeze(0) - p.unsqueeze(1)
            pair_mask = dy > 0
            if pair_mask.any():
                losses.append(F.softplus(-dp[pair_mask]).mean())
        else:
            i = torch.randint(0, n, (max_pairs,), device=pred.device, generator=rng)
            j = torch.randint(0, n, (max_pairs,), device=pred.device, generator=rng)
            good = y[i] > y[j]
            if good.any():
                losses.append(F.softplus(-(p[i[good]] - p[j[good]])).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def soft_portfolio_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, topk: int = 10, temperature: float = 0.05) -> torch.Tensor:
    # Differentiable long-only utility proxy. Invalid assets get -inf weight.
    scores = pred.masked_fill(~mask, -1e9)
    weights = torch.softmax(scores / temperature, dim=-1)
    ret = (weights * target.masked_fill(~mask, 0.0)).sum(dim=-1)
    return -ret.mean()


def composite_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cfg: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
    mse_w = float(cfg.get("mse_weight", 1.0))
    bpr_w = float(cfg.get("bpr_weight", 0.0))
    ic_w = float(cfg.get("ic_weight", 0.0))
    port_w = float(cfg.get("portfolio_weight", 0.0))
    topk = int(cfg.get("topk", 10))
    loss = pred.new_tensor(0.0)
    logs: Dict[str, float] = {}
    if mse_w:
        v = masked_mse(pred, target, mask)
        loss = loss + mse_w * v
        logs["loss_mse"] = float(v.detach().cpu())
    if bpr_w:
        v = bpr_rank_loss(pred, target, mask, max_pairs=int(cfg.get("max_pairs", 4096)))
        loss = loss + bpr_w * v
        logs["loss_bpr"] = float(v.detach().cpu())
    if ic_w:
        corr = masked_corr(pred, target, mask)
        v = -corr
        loss = loss + ic_w * v
        logs["loss_neg_ic"] = float(v.detach().cpu())
        logs["batch_ic"] = float(corr.detach().cpu())
    if port_w:
        v = soft_portfolio_loss(pred, target, mask, topk=topk)
        loss = loss + port_w * v
        logs["loss_portfolio"] = float(v.detach().cpu())
    logs["loss_total"] = float(loss.detach().cpu())
    return loss, logs
