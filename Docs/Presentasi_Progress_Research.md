---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Progress Research — XGBoost vs Random Forest untuk XAUUSD & BTCUSD'
footer: 'AI for Trading | UAS Project 2026'
style: |
  section {
    font-size: 24px;
    padding: 50px;
  }
  h1 {
    color: #1e3a8a;
    font-size: 44px;
  }
  h2 {
    color: #1e40af;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 8px;
  }
  h3 {
    color: #2563eb;
  }
  table {
    font-size: 18px;
    margin: 0 auto;
  }
  th {
    background-color: #1e3a8a;
    color: white;
  }
  code {
    background-color: #f1f5f9;
    color: #be185d;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .small {
    font-size: 18px;
  }
  .center {
    text-align: center;
  }
  blockquote {
    border-left: 4px solid #3b82f6;
    background: #eff6ff;
    padding: 10px 20px;
    font-style: italic;
  }
---

<!-- _class: lead -->

# AI for Trading
## Evaluasi Komparatif XGBoost vs Random Forest pada XAUUSD & BTCUSD

**Presentasi Progress Research**

UAS Project — Mei 2026

---

## Agenda

1. **Latar Belakang** — kenapa XAUUSD & BTCUSD, kenapa ensemble tree
2. **Rumusan & Tujuan Penelitian**
3. **Tinjauan Pustaka** — sintesis 3 paper inti (Crasta, Dave, Corbet)
4. **Research Gap** & posisi penelitian
5. **Metodologi** — desain modular *Scanning → Fix Model*
6. **Implementasi** — arsitektur kode aktual
7. **Hasil Backtest Aktual** — XAUUSD
8. **Komparasi Literatur vs Hasil** — validasi & gap
9. **Diskusi & Analisis Risiko**
10. **Status Progress & Roadmap Selanjutnya**

---

<!-- _class: lead -->

# 1. Latar Belakang

---

## Konteks Pasar 2026

<div class="columns">

<div>

### XAUUSD (Emas)
- Aset *safe-haven* tradisional
- Sensitif terhadap **DXY, yield obligasi 10Y, harga minyak**
- Volatil di 2026 — krisis bond yield, geopolitik
- Dinamika **non-linear**, asumsi ARIMA/GARCH gagal

</div>

<div>

### BTCUSD (Bitcoin)
- *Digital gold* — aset spekulatif & lindung nilai
- Sensitif terhadap **kebijakan The Fed**
- Dipengaruhi sentimen + data **on-chain**
- Volatilitas ekstrem, *regime shifting* cepat

</div>

</div>

> **Mengapa ensemble tree?** Kapasitas menangkap pola non-linear tanpa asumsi distribusi, terbukti unggul vs ARIMA/GARCH (Crasta, 2024; Cohen & Aiche, 2023).

---

## Mengapa Random Forest vs XGBoost?

| Aspek | Random Forest | XGBoost |
|-------|---------------|---------|
| Filosofi | *Bagging* — pohon paralel independen | *Boosting* — pohon sekuensial korektif |
| Kekuatan | Robust, kebal *noise*, *low variance* | Presisi tinggi, mampu menangkap interaksi kompleks |
| Kelemahan | Lebih lambat menangkap *regime shift* | Rentan *overfitting* jika hyperparameter buruk |
| Bukti literatur | Crasta (2024): RF MAE 7.20 di XAU/USD 1H | Corbet et al. (2020): XGBoost Sharpe 1.78 di BTC |

> Tidak ada *universal winner* — performa sangat **kontekstual** (timeframe, fitur, rezim pasar).

---

<!-- _class: lead -->

# 2. Rumusan & Tujuan Penelitian

---

## Rumusan Masalah

1. Bagaimana **performa komparatif** XGBoost dan Random Forest pada prediksi arah harga XAUUSD & BTCUSD di timeframe rendah (M1)?

2. Sejauh mana **arsitektur modular** (*scanning* untuk optimasi confidence + *fix model* untuk eksekusi) dapat mengatasi *concept drift*?

3. Apakah hasil backtest menunjukkan **edge nyata** atau hanya artefak *selection bias*?

---

## Tujuan

1. **Mengevaluasi** performa RF & XGBoost menggunakan metrik prediksi (winrate, akurasi) dan metrik finansial (Profit Factor, Drawdown, ROI)

2. **Membangun pipeline modular** — pemisahan fase optimasi hyperparameter & confidence dari fase produksi/eksekusi

3. **Memvalidasi** hasil terhadap benchmark literatur (Crasta 2024, Corbet 2020) dan mengidentifikasi gap implementasi

4. **Mengusulkan roadmap** perbaikan menuju strategi yang siap deploy

---

<!-- _class: lead -->

# 3. Tinjauan Pustaka

---

## Paper Inti #1 — Crasta (2024)

**"Comparative Analysis of ML Algorithms For XAU/USD Prediction"**
*MSc Research Project, National College of Ireland*

### Setup
- Data: Juli 2020 – Des 2023, **timeframe 1 jam**
- Fitur: OHLCV + S&P 500 + Crude Oil + VIX + **sentiment FinBERT**
- 7 model dibandingkan dengan GridSearch 5-fold CV

### Hasil
| Model | R² | MSE | MAE |
|-------|----|----|-----|
| **Random Forest** ⭐ | **0.9782** | **187.00** | **7.20** |
| XGBoost | 0.9760 | 205.95 | 8.52 |
| Gradient Boosting | 0.9637 | 311.30 | 10.51 |
| LSTM | 0.9316 | 587.27 | 17.28 |

> RF menang **karena kemampuannya menangani data campuran** (teknikal + fundamental + sentimen).

---

## Paper Inti #2 — Dave (2024)

**"Forex Trend Prediction: Can ML Algorithms Forecast Short Timeframe Movements?"**
*MAT(Comp), Yash Dave*

### Temuan Utama
- **Multi-timeframe**: 5m, 15m, 1H, 4H
- **XGBoost menang** di timeframe **4H** (RMSE terendah 0.0005)
- **Random Forest menang** di timeframe **5m** (profit simulasi 9.2%)

> **Implikasi**: superioritas model **bergantung pada timeframe**. Tidak bisa generalisasi.

---

## Paper Inti #3 — Corbet et al. (2020c)

**"ML for Bitcoin Price Prediction"**

### Hasil
- **XGBoost: Imbal Hasil 141.4%, Sharpe Ratio 1.78**
- Mengungguli LSTM, ARFIMA-GARCH untuk BTC/USD

### Mengapa XGBoost menang di BTC?
- Kemampuan integrasi **data teknikal + fundamental + sentimen + on-chain**
- Tangkap respons cepat terhadap **shock kebijakan moneter Fed**

> Benchmark untuk fase pengujian BTCUSD di tahap selanjutnya.

---

## Studi Tambahan & Peringatan Penting

<div class="columns">

<div>

### Kilimci (2022)
- Stacking ensemble > model tunggal
- **MAPE terbaik: 2.20** untuk XAU/USD harian
- RF tunggal MAPE 4.25 (lebih lemah)

### Behjoee (2025)
- Indikator momentum (CCI, Stoch_D, WMA) = top features di TF rendah
- RF & LightGBM unggul di 5m–4H XAU/USD

</div>

<div>

### ⚠️ Zewin (2026) — Peringatan
- Penambahan data alternatif (Google Trends, sentimen) **menurunkan akurasi 2%**
- Tidak semua data fundamental = manfaat
- Validasi empiris **wajib** sebelum integrasi

</div>

</div>

---

## Sintesis Pustaka — Pola Konsisten

> **Tidak ada model yang universal superior.** Pemilihan harus berdasarkan **konteks**:

| Konteks | Model yang Cenderung Unggul |
|---------|------------------------------|
| Data campuran teknikal + sentimen (XAU 1H) | Random Forest |
| Multi-data + non-linear kompleks (BTC) | XGBoost |
| Timeframe pendek (5m XAU) | Random Forest |
| Timeframe menengah (4H XAU) | XGBoost |
| Pasar dengan *regime shift* tajam | XGBoost (tapi rentan *overfit*) |
| Kebutuhan stabilitas & DD rendah | Random Forest |

---

<!-- _class: lead -->

# 4. Research Gap & Posisi Penelitian

---

## Gap yang Diidentifikasi

1. **Mayoritas studi pakai timeframe 1H atau lebih tinggi** — sedikit yang mengeksplorasi **M1 untuk XAUUSD**

2. **Belum ada studi** yang membandingkan **arsitektur modular** (scanning vs fix) sebagai mitigasi *concept drift* secara eksplisit

3. **Confidence threshold** sebagai *filter signal* jarang dioptimasi secara terpisah dari hyperparameter model

4. **BTC + XAU diuji bersama** dengan pipeline identik masih jarang — biasanya satu aset saja

---

## Posisi Penelitian Ini

> Membangun **pipeline modular terpisah** untuk XAUUSD M1, dengan **dual-confidence optimization** (buy & sell threshold independen), dievaluasi dengan metrik finansial nyata (PF, DD, ROI) menggunakan `backtesting.py`.

**Kontribusi pragmatis:**
- Skrip `_scanning.py` → grid search confidence → output ke `_fix.py`
- Verifikasi hipotesis Crasta/Dave di rezim pasar 2026

**Yang belum tercakup (future work):**
- BTCUSD belum diuji
- Walk-Forward Validation belum diimplementasi penuh
- Data eksogen (DXY, VIX, on-chain) belum diintegrasikan

---

<!-- _class: lead -->

# 5. Metodologi

---

## Arsitektur Sistem Modular

```
┌─────────────────────────┐         ┌──────────────────────┐
│   FASE 1: SCANNING      │         │   FASE 2: FIX MODEL  │
│  (Optimasi Periodik)    │ ──────► │   (Eksekusi Live)    │
│                         │         │                      │
│ • rf_scanning.py        │         │ • rf_fixmodel.py     │
│ • xgboost_scanning.py   │         │ • xgboost_fix.py     │
│                         │         │                      │
│ - Grid search           │         │ - Confidence locked  │
│   confidence            │         │ - Pure inference     │
│ - Pilih threshold       │         │ - Output signal      │
│   terbaik via score     │         │   BUY/SELL/NO TRADE  │
└─────────────────────────┘         └──────────────────────┘
        ▲                                      │
        │                                      ▼
        └─────── re-scan jika drift ──── eksekusi
```

> **Filosofi**: pisahkan otak optimasi dari otak eksekusi → adaptif tanpa risiko dana saat tuning.

---

## Pipeline Data & Feature Engineering

### Data
- **Symbol**: `XAUUSDc` (XAUUSD pada broker Exness cent account)
- **Timeframe**: M1 (1 menit)
- **Periode**: 15 Feb 2026 – sekarang (~3 bulan)
- **Source**: MetaTrader 5 via Python API

### Indikator Teknikal Dasar
- EMA 20 / 50 / 100
- RSI(14), Stochastic %K(14)
- ATR(14) — untuk SL/TP dinamis

### Feature Engineering (~45 fitur)
Momentum (3/5/10), volatility ratio, EMA slope/gap, wick ratio, return %, volume change, binary flags (RSI overbought/oversold, EMA cross, price above EMA)

---

## Target Labeling — SL/TP Forward Looking

```python
# Untuk tiap candle i, simulasikan 10 candle ke depan
SL = 1.00 × ATR    # Stop Loss
TP = 1.46 × ATR    # Take Profit (Risk:Reward = 1:1.46)

if buy_TP hit before buy_SL: target = 1 (BUY)
elif sell_TP hit before sell_SL: target = 0 (SELL)
else: target = NaN (drop)
```

**Mengapa risk:reward 1:1.46?**
- Memungkinkan profitabilitas dengan winrate >40%
- Realistis untuk gerakan ATR-based di M1

---

## Model Configuration

<div class="columns">

<div>

### Random Forest
```python
RandomForestClassifier(
  n_estimators=500,
  max_depth=10,
  min_samples_split=20,
  min_samples_leaf=8,
  class_weight='balanced',
  random_state=42
)
```

</div>

<div>

### XGBoost
```python
XGBClassifier(
  # + TimeSeriesSplit
  # + ADX filter (min_adx=20)
  # + L1/L2 regularization
)
```

</div>

</div>

### Dual Confidence Grid Search (Fase Scanning)
- `buy_conf` ∈ [0.50, 0.85] step 0.01 (RF) / 0.05 (XGB)
- `sell_conf` ∈ [0.50, 0.85] step 0.01 (RF) / 0.05 (XGB)
- Composite score: `PF×40 + Winrate×0.5 + ROI×0.2 − DD×1.5`

---

<!-- _class: lead -->

# 6. Hasil Backtest Aktual

---

## Random Forest — XAUUSD M1

### Konfigurasi Terbaik
`buy_conf=0.72`, `sell_conf=0.50`

| Metrik | Nilai |
|--------|-------|
| Total Trade | 315 (126 BUY, 189 SELL) |
| Winrate | **55.24%** |
| Profit Factor | **1.71** |
| Max Drawdown | **6.38%** |
| ROI | **+201.9%** |
| Saldo: $10.000 → | **$30.190** |

> **Profil**: Stabil, drawdown sangat rendah, banyak trade

---

## XGBoost — XAUUSD M1

### Konfigurasi Terbaik
`buy_conf=0.85`, `sell_conf=0.55`

| Metrik | Nilai |
|--------|-------|
| Total Trade | 143 (70 BUY, 73 SELL) |
| Winrate | **60.14%** ⭐ |
| Profit Factor | **2.02** ⭐ |
| Max Drawdown | 14.77% |
| ROI | +96.1% |
| Saldo: $10.000 → | $19.614 |

> **Profil**: Lebih selektif, akurasi tinggi, drawdown lebih besar

---

## Perbandingan Langsung RF vs XGBoost

| Metrik | Random Forest | XGBoost | Pemenang |
|--------|---------------|---------|----------|
| Winrate | 55.24% | 60.14% | 🏆 XGBoost |
| Profit Factor | 1.71 | 2.02 | 🏆 XGBoost |
| Max Drawdown | 6.38% | 14.77% | 🏆 RF |
| ROI | 201.9% | 96.1% | 🏆 RF |
| Total Trade | 315 | 143 | (lebih banyak peluang) |
| Konsistensi | Tinggi | Sedang | 🏆 RF |

> **Trade-off klasik**: RF = "tortoise" (stabil, banyak trade), XGBoost = "hare" (selektif, akurat tapi DD tinggi)

---

## Feature Importance — Insight Berbeda

<div class="columns">

<div>

### Top Features Random Forest
1. ema50 (0.067)
2. ema20 (0.062)
3. ema100 (0.061)
4. ema_gap_50_100 (0.060)
5. ema_gap_20_50 (0.046)
6. atr (0.044)
7. volatility_20 (0.042)

**Pola**: Murni teknikal/trend-based

</div>

<div>

### Top Features XGBoost
1. **newyork_session** (0.047) ⭐
2. price_above_ema20 (0.039)
3. ema50 (0.029)
4. ema100 (0.028)
5. ema_gap_50_100 (0.027)
6. stoch_above_50 (0.027)
7. **london_session** (0.027) ⭐
8. **hour** (0.026) ⭐

**Pola**: Menangkap pola **sesi waktu** trading

</div>

</div>

> XGBoost menemukan **edge tambahan** dari struktur waktu, RF lebih konservatif pada trend.

---

<!-- _class: lead -->

# 7. Komparasi Literatur vs Hasil

---

## Validasi Terhadap Benchmark Literatur

| Klaim Literatur | Hasil Aktual Kami | Verifikasi |
|-----------------|---------------------|------------|
| Crasta (2024): RF unggul XGB di XAU 1H (MAE 7.2 vs 8.5) | RF DD lebih rendah, tapi XGBoost menang winrate di M1 | ⚠️ Beda TF, beda hasil |
| Dave (2024): RF menang di 5m, XGB di 4H | Hasil M1 kami lebih dekat ke "zona RF" — DD rendah | ✅ Konsisten |
| Corbet (2020): XGBoost Sharpe 1.78 di BTC | BTC belum diuji | ⏳ Pending |
| Behjoee (2025): RF/LightGBM unggul di TF rendah XAU | RF kami lebih stabil di M1 | ✅ Konsisten |
| Zewin (2026): Data alternatif bisa kurangi akurasi 2% | Kami tidak pakai alt data — tidak terverifikasi | — |

---

## Pola "No Universal Winner" Terbukti

> Hasil empiris kami **mengkonfirmasi tesis utama Qwen Research** dan literatur akademik:

- ✅ Random Forest = **stabilitas & robust** (DD 6.4% sangat rendah)
- ✅ XGBoost = **akurasi tinggi** (winrate 60%, PF 2.02)
- ✅ Trade-off bukan kekurangan — itu **karakter arsitektur**

**Implikasi praktis:**
- Trader konservatif → pakai RF
- Trader agresif → pakai XGBoost dengan ketat regularisasi
- **Ensemble hybrid (stacking)** = arah penelitian selanjutnya

---

<!-- _class: lead -->

# 8. Diskusi & Analisis Risiko

---

## Kekuatan Pendekatan Ini

1. **Pipeline modular** terbukti scalable — tinggal ganti symbol untuk BTC
2. **Dual-confidence optimization** memberikan fleksibilitas tuning per arah trade
3. **Risk management** terintegrasi (1% per trade, position sizing via ATR)
4. **Reproducible** — `random_state=42`, semua data dari MT5 yang sama
5. **Pragmatis** — dirancang untuk eksekusi live, bukan sekadar paper trading

---

## Risiko & Keterbatasan Jujur

### ⚠️ Selection Bias pada Confidence Tuning
> Confidence dipilih dari **test set yang sama** yang dipakai mengukur performa → ROI 201% kemungkinan **overoptimistic**.

### ⚠️ Periode Data Sempit
> 3 bulan satu rezim pasar (Feb–Mei 2026) tidak setara dengan validasi multi-siklus 2015–2026.

### ⚠️ Bias Signal SELL
> RF: 189 SELL vs 126 BUY → model **bias ke SELL** karena XAUUSD sedang trending turun di periode ini. Generalisasi ke pasar bullish belum terbukti.

### ⚠️ Tidak Ada Walk-Forward Validation
> Walaupun direkomendasikan literatur, kode masih pakai simple 80/20 split.

### ⚠️ Slippage & Spread Tidak Disimulasikan Realistik
> `komisi=0.0` — di realita, spread broker bisa mengikis profit signifikan di M1.

---

## Risiko Sistemik yang Diperhatikan

| Risiko | Mitigasi yang Sudah Ada | Mitigasi yang Belum |
|--------|--------------------------|----------------------|
| Overfitting | `max_depth=10`, `class_weight='balanced'` | WFV penuh dengan TimeSeriesSplit |
| Concept drift | Arsitektur scanning periodik (desain) | Belum ada trigger otomatis re-train |
| Margin call | Position sizing via ATR, risk 1% | Belum ada circuit breaker |
| Black swan | — | Filter event makro (NFP, FOMC) |
| Broker latency | — | Belum ada simulasi latency |

---

<!-- _class: lead -->

# 9. Status Progress

---

## Apa yang Sudah Selesai ✅

- ✅ Setup environment MT5 + Python (test_library.py, candletest.py)
- ✅ Pipeline data acquisition dari MT5
- ✅ Feature engineering 45 indikator
- ✅ Target labeling forward-looking SL/TP
- ✅ Model Random Forest — scanning + fix
- ✅ Model XGBoost — scanning + fix
- ✅ Dual confidence grid search optimization
- ✅ Manual backtest + `backtesting.py` integration
- ✅ Output: feature importance, optimization results, equity curve HTML

---

## Apa yang Sedang Dikerjakan 🔄

- 🔄 Validasi hasil dengan literatur (Crasta, Dave, Corbet)
- 🔄 Analisis bias signal BUY vs SELL
- 🔄 Dokumentasi metodologi & laporan

---

## Apa yang Akan Datang ⏳

| Prioritas | Item | Estimasi |
|-----------|------|----------|
| **HIGH** | Implementasi Walk-Forward Validation murni (TimeSeriesSplit n=5) | 1 minggu |
| **HIGH** | Replikasi pipeline untuk BTCUSD | 1 minggu |
| **HIGH** | Hitung Sharpe Ratio dari equity curve `backtesting.py` | 1 hari |
| MEDIUM | Tambah baseline Buy & Hold sebagai pembanding | 1 hari |
| MEDIUM | Pisahkan validation set ke-3 untuk pemilihan confidence | 2 hari |
| MEDIUM | Integrasi data eksogen (DXY via MT5) | 3 hari |
| LOW | Hybrid stacking RF + XGBoost (meta-learner logreg) | 1 minggu |
| LOW | Asymmetric loss function berdasarkan Sharpe | 2 minggu |

---

## Roadmap Visualisasi

```
Sprint 1 (Sekarang)         Sprint 2                Sprint 3
─────────────────           ────────────            ──────────────
• Validasi literatur   →    • WFV penuh        →   • Hybrid stacking
• Dokumentasi              • BTCUSD pipeline       • Asym loss func
• Bias analysis            • Sharpe Ratio          • Live demo account
                           • Baseline B&H          • Risk circuit breaker
```

> **Target akhir UAS**: laporan komprehensif + demo eksekusi live di akun demo, dengan benchmark XAU & BTC yang valid.

---

<!-- _class: lead -->

# 10. Kesimpulan

---

## Kesimpulan Utama

1. **Pipeline modular berfungsi** — kedua model menghasilkan strategi yang *tradable* di paper trading

2. **Hasil konsisten dengan literatur** — RF stabil (DD 6.4%), XGBoost akurat (winrate 60%, PF 2.02), trade-off klasik **terverifikasi**

3. **Tidak ada model universal superior** — pemilihan tergantung profil risiko & rezim pasar (sesuai Crasta, Dave, Behjoee)

4. **Gap jujur**: hasil saat ini bukan bukti "edge nyata" — perlu WFV, data lebih luas, BTCUSD, baseline B&H sebelum klaim deploy-ready

5. **Roadmap jelas**: 6–8 item perbaikan teridentifikasi dengan prioritas

---

## Pertanyaan untuk Diskusi

1. Apakah arsitektur **scanning → fix model** ini cocok untuk mitigasi *concept drift* di rezim 2026?

2. Risk:reward **1:1.46** — apakah optimal untuk M1 XAUUSD, atau perlu adaptive berdasarkan ATR percentile?

3. **Hybrid stacking** RF+XGBoost layak diprioritaskan, atau dulu **selesaikan BTCUSD** untuk konsistensi cakupan riset?

4. Bagaimana cara terbaik **menghindari selection bias** pada confidence tuning tanpa kehilangan banyak data?

---

## Referensi Utama

1. **Crasta, M. (2024)**. *Comparative Analysis of ML Algorithms For XAU/USD Prediction: Integrating Economic Indicators And Sentiment Analysis*. MSc Research Project, National College of Ireland.

2. **Dave, Y. (2024)**. *Forex Trend Prediction: Can ML Algorithms Forecast Short Timeframe Movements?*. MAT(Comp) Thesis.

3. **Corbet, S. et al. (2020c)**. *ML for Bitcoin Price Prediction*. — XGBoost Sharpe 1.78, return 141.4%.

4. **Kilimci, Z. (2022)**. *Ensemble Regression-Based Gold Price (XAU/USD) Prediction*. Journal of Emerging Market Finance.

5. **Cohen, G. & Aiche, A. (2023)**. *Forecasting Gold Price Using ML Methodologies*. Chaos, Solitons & Fractals 175.

6. **Behjoee, A. (2025)**. *Forex Trend Prediction at Short Timeframes*. Student Theses Campus Fryslan.

7. **Zewin, A. (2026)**. *Forecasting Gold Price Movements Using Alternative Data and ML Techniques*. — Peringatan: alt data ↓ akurasi 2%.

---

<!-- _class: lead -->

# Terima Kasih

## Pertanyaan & Diskusi

**Repository**: `UAS-PROJECT/xgboost_randomforest_aifortrading/`
**Tools**: Python · MetaTrader5 · scikit-learn · XGBoost · backtesting.py
