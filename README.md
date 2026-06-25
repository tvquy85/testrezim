# AAAI27 Stock Regime Mixer

Leakage-safe research framework for **causal regime-conditioned MLP stock forecasting**. The main model is `crc_lora`, a low-rank stock-axis adapter conditioned on causal market context.

Start here:

```bash
pip install -e .
python scripts/train.py --config configs/synthetic_crc_lora.yaml --set optim.epochs=1 output_dir=outputs/smoke
```

Detailed instructions are in [`guide.md`](guide.md).
