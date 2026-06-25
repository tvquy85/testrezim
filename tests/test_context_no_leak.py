from __future__ import annotations
import numpy as np
from aaai27_stock.context import compute_market_context, transform_context_train_only


def test_context_uses_only_past_window():
    close = np.ones((30, 5), dtype=np.float32) * 100
    close[20:, :] = 200  # future jump
    ctx1, _ = compute_market_context(close, lookback=5)
    close2 = close.copy()
    close2[25:, :] = 10000  # change far future
    ctx2, _ = compute_market_context(close2, lookback=5)
    # Context at t=19 cannot depend on t>=25.
    assert np.allclose(ctx1[19], ctx2[19])


def test_train_only_scaler_differs_from_global_leakage():
    ctx = np.zeros((20, 2), dtype=np.float32)
    ctx[:10] = 1
    ctx[10:] = 100
    scaled, scaler = transform_context_train_only(ctx, np.arange(10), scaler_name="train_standard")
    assert scaler is not None
    assert abs(float(np.mean(scaled[:10]))) < 1e-5
    assert float(np.mean(scaled[10:])) > 10
