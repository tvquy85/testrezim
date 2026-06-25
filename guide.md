# Guide chạy project `aaai27-stock-regime-mixer`

Project này là framework thực nghiệm độc lập để phát triển hướng paper AAAI 2027 từ ý tưởng **Causal Regime-Conditioned Low-Rank Stock Mixer**. Code không copy trực tiếp repo StockMixer/gMLP mà tái thiết kế lại theo hướng dễ tái lập: split theo thời gian, scaler fit trên train-only, market context causal, negative controls, nhiều baseline và metric tài chính.

## 1. Cấu trúc thư mục

```text
prj/
├── configs/
│   ├── synthetic_crc_lora.yaml              # model chính: causal regime-conditioned low-rank mixer
│   ├── synthetic_stockmixer.yaml            # baseline StockMixer-style bug-fixed
│   ├── synthetic_context_gmlp.yaml          # context-gated MLP variant
│   ├── synthetic_negative_shuffle.yaml      # negative control: shuffle context
│   └── csv_long_template.yaml               # template dùng dữ liệu thật dạng CSV long
├── data/
│   └── synthetic_market.npz                 # synthetic dataset mẫu, chạy được ngay
├── scripts/
│   ├── make_synthetic.py                    # sinh dữ liệu synthetic có regime
│   ├── train.py                             # train/evaluate một config
│   ├── run_sweep.py                         # chạy nhiều model + nhiều seed
│   └── inspect_dataset.py                   # kiểm tra shape/split/context
├── src/aaai27_stock/
│   ├── data.py                              # loader, chronological split, train-only scaler
│   ├── context.py                           # causal market context encoder
│   ├── losses.py                            # MSE, BPR rank, IC loss, portfolio proxy
│   ├── metrics.py                           # IC, RankIC, Prec@K, Sharpe, turnover, cost-aware metrics
│   ├── trainer.py                           # training loop, early stopping, checkpoint, logs
│   └── models/
│       ├── stock_mixer.py                   # baseline StockMixer-style đã sửa lỗi module registration
│       ├── regime_lora.py                   # model chính CRC-LoRA
│       ├── context_gmlp.py                  # context-gated gMLP-like block
│       ├── film.py                          # FiLM regime modulation
│       ├── sector_moe.py                    # mixture-of-experts theo context/sector
│       └── dlinear.py                       # baseline tuyến tính mạnh
└── tests/
    └── test_context_no_leak.py              # unit test chống leakage context
```

## 2. Cài đặt

Các script đã tự set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `TORCH_NUM_THREADS=1` nếu môi trường chưa set. Điều này giúp chạy CPU ổn định hơn khi thử nghiệm nhanh.

## 3. Chạy nhanh để kiểm tra pipeline

Dataset synthetic mẫu đã có sẵn tại `data/synthetic_market.npz`. Chạy 1 epoch:

```bash
python scripts/train.py \
  --config configs/synthetic_crc_lora.yaml \
  --set optim.epochs=1 model.hidden_dim=32 model.rank=4 output_dir=outputs/smoke_crc_lora
```

Kết quả nằm trong:

```text
outputs/smoke_crc_lora/
├── best_model.pt
├── config.resolved.yaml
├── history.csv
├── metadata.json
└── results.json
```

`results.json` có metric cho train/valid/test: `ic`, `rank_ic`, `precision_at_10`, `topk_overlap_at_10`, `long_short_sharpe`, `turnover_daily`, và Sharpe sau transaction cost ở các mức bps.

## 4. Sinh lại synthetic dataset

Synthetic generator có 3 regime: calm/up, high-vol/down, và dispersion/stock-picking. Regime ảnh hưởng tương quan, volatility, sector factor và quality alpha, nên context-gated model có điều kiện để chứng minh có ích.

```bash
python scripts/make_synthetic.py \
  --out data/synthetic_market.npz \
  --days 900 \
  --assets 120 \
  --sectors 8 \
  --seed 2027
```

Sau đó chạy lại model chính:

```bash
python scripts/train.py --config configs/synthetic_crc_lora.yaml
```

## 5. Chạy baseline và biến thể

Baseline StockMixer-style bug-fixed:

```bash
python scripts/train.py --config configs/synthetic_stockmixer.yaml
```

Context-gMLP variant:

```bash
python scripts/train.py --config configs/synthetic_context_gmlp.yaml
```

Negative control shuffle context:

```bash
python scripts/train.py --config configs/synthetic_negative_shuffle.yaml
```

Chạy sweep nhiều model/nhiều seed:

```bash
python scripts/run_sweep.py \
  --base configs/synthetic_crc_lora.yaml \
  --models dlinear stockmixer context_gmlp film crc_lora sector_moe \
  --seeds 2027 2028 2029 \
  --epochs 5
```

File tổng hợp sẽ nằm ở:

```text
outputs/sweep_summary.json
```

## 6. Các mô hình đã cài sẵn

### `dlinear`
Baseline tuyến tính per-stock, không stock mixing. Dùng để chống lỗi “deep model tốt hơn chỉ vì phức tạp hơn”.

### `stockmixer`
Baseline StockMixer-style viết lại sạch. Có multi-scale temporal encoder và static stock-axis MLP mixer. Không dùng `ParameterList` sai mục đích; các layer được register bằng `ModuleList`/`Sequential` chuẩn PyTorch.

### `context_gmlp`
Biến thể giống ý tưởng gMLP cải tiến: hidden representation được gated bởi causal market context. Đây là bản gần với prototype ban đầu nhất.

### `film`
Dùng FiLM modulation:

```text
h' = LayerNorm(h) * (1 + gamma(c)) + beta(c)
```

Phù hợp để kiểm tra xem context chỉ cần scale/shift representation hay phải dynamic stock-mixing thật sự.

### `crc_lora`
Model chính. Với context `c_t`, stock mixer có adapter low-rank:

```text
A(c_t) = A_static + U diag(g(c_t)) V^T
```

Ý nghĩa: quan hệ cross-sectional giữa cổ phiếu không cố định, mà thay đổi theo market regime. Rank nhỏ giúp giảm tham số so với ma trận `N x N` đầy đủ.

### `sector_moe`
Mixture-of-experts được gate bởi market context. Có thể bật sector embedding bằng config:

```yaml
model:
  name: sector_moe
  hidden_dim: 64
  num_experts: 4
  use_sector: true
```

## 7. Dùng dữ liệu thật

### Cách 1: NPZ, khuyến nghị cho tốc độ

Tạo file `.npz` có các key:

```python
features: shape (T, N, F), float32
labels:   shape (T, N),    float32
close:    shape (T, N),    float32
mask:     shape (T, N),    bool, optional
sectors:  shape (N,),      int64, optional
dates:    shape (T,),      string, optional
assets:   shape (N,),      string, optional
```

Quy ước quan trọng: `labels[t, i]` phải là future return của asset `i` sau khi quan sát feature tại ngày `t`. Ví dụ label 1 ngày:

```python
labels[t, i] = log(close[t+1, i]) - log(close[t, i])
```

Không đưa giá/return tương lai vào `features[t]`.

### Cách 2: CSV long

Dùng template `configs/csv_long_template.yaml`. CSV dạng:

```csv
date,asset,sector,label,close,ret1,ret_ma5,ret_vol5,price_vs_ma20,volume_z
2019-01-02,AAPL,Tech,0.0031,157.9,0.010,...
2019-01-02,MSFT,Tech,-0.0012,101.1,...
```

Trong config:

```yaml
data:
  path: data/your_market_long.csv
  format: csv_long
  csv:
    date_col: date
    asset_col: asset
    label_col: label
    sector_col: sector
    feature_cols: [close, ret1, ret_ma5, ret_vol5, price_vs_ma20, volume_z]
```

## 8. Thiết kế chống leakage

Project này đã cài các nguyên tắc sau:

1. Split train/valid/test theo thứ tự thời gian, không shuffle ngày giữa các split.
2. Feature scaler `train_standard` chỉ fit trên các ngày quan sát xuất hiện trong train windows.
3. Market context tại ngày `t` chỉ dùng `close[t-lookback+1:t+1]`.
4. Context scaler `train_standard` chỉ fit trên train sample dates.
5. Có negative controls: `context.shuffle=true`, `context.noise=true`, `context.lag=k`.
6. Metrics được tính theo từng ngày cross-section, sau đó aggregate.

Chạy unit test context:

```bash
PYTHONPATH=src pytest -q tests
```

## 9. Protocol thực nghiệm đề xuất cho paper AAAI 2027

Tối thiểu nên chạy các bảng sau:

### Bảng 1: Main comparison

```text
dlinear
stockmixer
context_gmlp
film
crc_lora
sector_moe
```

Mỗi model chạy ít nhất 10 seed nếu compute cho phép. Với bản submission mạnh, nên chạy 20 seed cho bảng chính.

### Bảng 2: Context ablation

```text
no_context
5_context_causal
context_lag_1
context_lag_5
context_shuffle
context_noise
single_context_vol
single_context_dispersion
single_context_pca_ratio
```

Nếu `context_shuffle` hoặc `context_noise` vẫn tốt gần bằng model chính, phải kiểm tra lại vì có thể model không thật sự dùng context.

### Bảng 3: Objective ablation

Thay đổi trong config `loss`:

```yaml
loss:
  mse_weight: 1.0
  bpr_weight: 0.0 / 0.1 / 0.5
  ic_weight: 0.0 / 0.05 / 0.2
  portfolio_weight: 0.0 / 0.01
```

Mục tiêu: chứng minh improvement ở IC/RankIC có chuyển được sang portfolio metric sau transaction cost hay không.

### Bảng 4: Regime-sliced analysis

Hiện code tính context nhưng chưa tự động cắt metric theo high-vol/low-vol. Nên bổ sung script phân nhóm ngày test theo percentile của `vol`, `dispersion`, `pca_ratio`, rồi báo metric riêng. Đây là bảng rất quan trọng cho thesis “regime-conditioned”.

### Bảng 5: Reproducibility audit

So sánh:

```text
stockmixer_static
stockmixer_static + causal_scaler
crc_lora + causal_context
crc_lora + shuffled_context
crc_lora + future-leaky context, chỉ để audit, không dùng claim chính
```

## 10. Checklist trước khi submit

- Chạy nhiều seed, lưu `config.resolved.yaml`, `metadata.json`, `history.csv`, `results.json`.
- Ghi rõ data source, universe construction, missing data mask, delisting/survivorship handling.
- Báo cả signal metrics và portfolio metrics sau cost.
- Báo parameter count/runtime.
- Không claim “strong accept chắc chắn”; claim đúng phạm vi thực nghiệm.
- Nếu dùng StockMixer gốc làm baseline, phải ghi rõ bạn dùng bản clean-room/bug-fixed hay code gốc, và không dựa vào reported number không tái lập.

## 11. Lệnh mẫu để bắt đầu nghiêm túc

```bash
# 1) Tạo synthetic lớn hơn
python scripts/make_synthetic.py --out data/synthetic_market.npz --days 900 --assets 120 --sectors 8 --seed 2027

# 2) Chạy baseline
python scripts/train.py --config configs/synthetic_stockmixer.yaml --set output_dir=outputs/base_stockmixer_seed2027

# 3) Chạy model chính
python scripts/train.py --config configs/synthetic_crc_lora.yaml --set output_dir=outputs/crc_lora_seed2027

# 4) Chạy negative control
python scripts/train.py --config configs/synthetic_negative_shuffle.yaml --set output_dir=outputs/crc_lora_shuffle_seed2027

# 5) Chạy sweep nhanh
python scripts/run_sweep.py --base configs/synthetic_crc_lora.yaml --epochs 5
```

## 12. Ghi chú quan trọng

Project này là nền code thực nghiệm có thể chạy được, không phải bằng chứng đảm bảo paper sẽ được AAAI chấp nhận. Để nhắm mức strong accept, bạn cần chạy benchmark lớn trên dữ liệu thật, thêm nhiều baseline mạnh, thống kê nhiều seed, kiểm định significance, và viết phần theory/negative-control thật chặt.
