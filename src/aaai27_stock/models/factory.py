from __future__ import annotations

from typing import Dict, Any
from torch import nn

from .stock_mixer import StockMixerBugFixed
from .context_gmlp import ContextGMLPStockForecaster
from .regime_lora import RegimeLoRAStockMixer
from .film import FiLMRegimeMixer
from .sector_moe import SectorMoERegimeMixer
from .dlinear import DLinearStock
from .common import count_parameters


MODEL_REGISTRY = {
    "stockmixer": StockMixerBugFixed,
    "context_gmlp": ContextGMLPStockForecaster,
    "crc_lora": RegimeLoRAStockMixer,
    "film": FiLMRegimeMixer,
    "sector_moe": SectorMoERegimeMixer,
    "dlinear": DLinearStock,
}


def build_model(cfg: Dict[str, Any], meta: Dict[str, Any]) -> nn.Module:
    model_cfg = dict(cfg.get("model", {}))
    name = model_cfg.pop("name", "crc_lora")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    kwargs = {
        "num_assets": meta["num_assets"],
        "lookback": meta["lookback"],
        "feature_dim": meta["feature_dim"],
        "context_dim": meta["context_dim"],
        "num_sectors": model_cfg.get("num_sectors") or meta.get("num_sectors", 1),
    }
    kwargs.update(model_cfg)
    model = cls(**kwargs)
    model.num_parameters = count_parameters(model)  # type: ignore[attr-defined]
    return model
