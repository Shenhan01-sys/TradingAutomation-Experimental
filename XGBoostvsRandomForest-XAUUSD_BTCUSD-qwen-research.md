# XGBoost vs. Random Forest: Evaluasi Komparatif untuk Strategi Trading Emas (XAU/USD) dan Bitcoin (BTC/USD) Berbasis Implementasi Praktis

> **Catatan Editorial (Versi Harmonisasi).** Dokumen ini merupakan versi yang telah diselaraskan dengan dua sumber primer: (1) **source code aktual** dari empat skrip Python `rf_scanning.py`, `rf_fixmodel.py`, `xgboost_scanning.py`, dan `xgboost_fix.py`, dan (2) **paper Crasta (2024)** *"Comparative Analysis of Machine Learning Algorithms For XAU/USD Prediction"* yang menjadi landasan metodologis utama proyek ini. Klaim yang sebelumnya bersifat aspiratif telah dipisahkan secara eksplisit dari klaim yang telah terverifikasi pada hasil backtest. Pembaca akan menemukan dua notasi: **[IMPLEMENTED]** untuk komponen yang sudah dieksekusi di kode, dan **[FUTURE WORK]** untuk komponen yang masih menjadi roadmap.

---

## 1. Landasan Teoretis dan Justifikasi Perbandingan Algoritma Pembelajaran Mesin

Perbandingan antara algoritma XGBoost (Extreme Gradient Boosting) dan Random Forest (RF) dalam konteks perdagangan otomatis pada pasangan XAU/USD (emas) dan BTC/USD (Bitcoin) merupakan subjek penelitian yang sangat relevan. Justifikasi penelitian ini terletak pada tiga domain utama: evolusi metode prediktif di bidang keuangan, karakteristik unik dari aset yang ditargetkan, serta kompleksitas masing-masing algoritma pembelajaran mesin.

Secara historis, prediksi pasar keuangan bergantung pada model ekonometrika tradisional seperti ARIMA, ARIMA-GARCH, dan ARIMA-TGARCH. Crasta (2024) merangkum bahwa meskipun model-model tersebut efektif menangkap tren musiman, mereka mengalami kesulitan signifikan dalam menangani sifat non-linear pasar keuangan modern. Hybrid ARIMA(2,1,3)-GARCH(1,1) yang dilaporkan Bunnag (2023) mencapai MAE 106.71 dan RMSE 126.78 — angka yang masih jauh di bawah pencapaian model ensemble berbasis pohon. Kemajuan pesat dalam machine learning telah membuka jalan bagi pendekatan yang lebih fleksibel dan kuat dalam menangkap hubungan non-linear yang kompleks dalam data pasar. Studi-studi telah secara sistematis menunjukkan bahwa model-model seperti Random Forest, Gradient Boosting, dan XGBoost konsisten mengungguli model tradisional, baik dalam akurasi prediksi maupun dalam manajemen risiko.

**Random Forest** adalah ensemble meta-algoritma yang bekerja dengan membangun banyak pohon keputusan selama proses pelatihan dan mengembalikan nilai rata-rata prediksi (untuk regresi) atau hasil voting mayoritas (untuk klasifikasi). Keunggulan utama RF terletak pada kekokohannya dan ketahanannya terhadap overfitting, bahkan ketika jumlah fitur sangat besar. Proses *bootstrap aggregation* (bagging) yang menjadi intinya — di mana setiap pohon dilatih pada sampel acak dari data latih — secara signifikan mengurangi varians model tanpa meningkatkan bias secara signifikan. Crasta (2024) secara empiris memverifikasi bahwa **Random Forest mencapai performa terbaik di antara tujuh model yang diuji** (Decision Tree, RF, Gradient Boosting, XGBoost, SVR, LSTM, GRU), dengan R² 0.9782, MSE 187.00, dan MAE 7.20 untuk prediksi XAU/USD per jam.

**XGBoost** adalah implementasi gradient boosting yang dioptimalkan untuk kecepatan dan kinerja tinggi. Berbeda dengan RF yang menggunakan *bagging*, XGBoost menggunakan teknik *boosting*: pohon keputusan dibangun secara serial, di mana setiap pohon baru berusaha memperbaiki kesalahan kombinasi pohon-pohon sebelumnya melalui mekanisme *gradient descent*. XGBoost juga memiliki mekanisme regularisasi internal (L1 dan L2) yang membantu mencegah overfitting lebih efektif daripada banyak model boosting lainnya. Dalam studi Crasta (2024), XGBoost menempati posisi kedua dengan R² 0.9760, MSE 205.95, dan MAE 8.52 — selisih sangat tipis dari RF. Studi lain oleh Cohen dan Aiche (2023) yang menggunakan data dari Februari 2011 hingga Februari 2023 menunjukkan XGBoost mencapai MSE serendah 0.0000186, yang menunjukkan bahwa keunggulan komparatif kedua model sangat bergantung pada konfigurasi data dan fitur. Studi yang membandingkan berbagai algoritma untuk prediksi harga Bitcoin bahkan menemukan bahwa XGBoost mampu mencapai rasio Sharpe yang sangat impresif yaitu 1.78.

Meskipun keduanya termasuk dalam keluarga *ensemble methods*, perbedaan filosofis dan teknis mereka menghasilkan karakteristik performa yang berbeda dalam konteks pasar keuangan. RF cenderung lebih *robust* dan lebih mudah di-*tune* karena tidak terlalu rentan terhadap pemilihan hyperparameter yang buruk. Sebaliknya, XGBoost menawarkan potensi akurasi prediksi yang lebih tinggi jika diatur dengan benar, tetapi bisa menjadi rapuh jika parameter seperti laju belajar atau kedalaman pohon tidak ditentukan dengan hati-hati. Pemilihan antara keduanya bukanlah pilihan biner yang absolut; melainkan sebuah pertimbangan yang bergantung pada karakteristik dataset, tujuan analisis, dan toleransi terhadap kompleksitas model. Beberapa penelitian — termasuk Kilimci (2022) yang mencapai MAPE terbaik 2.2036 dengan Stacking Regressor — bahkan menyarankan bahwa model gabungan yang menggabungkan output dari berbagai model dapat memberikan kinerja yang lebih superior daripada model tunggal.

Justifikasi untuk fokus pada aset XAU/USD dan BTC/USD semakin diperkuat oleh karakteristik pasar yang unik dari kedua instrumen ini. Emas (XAU/USD) secara tradisional berfungsi sebagai aset *safe-haven*: permintaannya cenderung meningkat selama periode ketidakpastian ekonomi, inflasi tinggi, atau gejolak geopolitik. Hubungannya dengan indikator volatilitas seperti VIX dan indeks saham seperti S&P 500 sangat erat, menjadikannya target yang ideal untuk model yang dapat menangkap respons pasar terhadap berita ekonomi. Di sisi lain, Bitcoin (BTC/USD) telah berevolusi dari proyek kripto sampingan menjadi aset alternatif yang signifikan — sering disebut "emas digital". Namun perilakunya sangat berbeda dari emas: Bitcoin sangat reaktif terhadap kebijakan moneter bank sentral, khususnya Federal Reserve AS. Kenaikan suku bunga cenderung menekan harga Bitcoin, sementara pemotongan suku bunga sering kali mendorongnya naik.

Dengan demikian, evaluasi komparatif antara XGBoost dan Random Forest pada kedua aset ini tidak hanya menguji keunggulan teknis model, tetapi juga sejauh mana setiap model dapat menangkap dinamika pasar yang berbeda. Literatur yang ada menunjukkan bahwa **tidak ada jawaban tunggal yang benar**; hasilnya sangat bergantung pada timeframe, jenis data yang digunakan, dan metrik kinerja yang dipilih. Oleh karena itu, penelitian ini bertujuan memberikan perspektif empiris yang solid dengan merujuk langsung pada implementasi kode yang telah ada untuk XAUUSD, dan menyiapkan pipeline yang dapat direplikasi untuk BTCUSD pada fase selanjutnya.

---

## 2. Konsep dan Mekanisme Strategi Trading Berbasis Implementasi Skrip

Konsep strategi perdagangan yang diusulkan direkonstruksi secara sistematis berdasarkan analisis terhadap struktur dari empat skrip Python yang telah diimplementasikan: `rf_scanning.py`, `xgboost_scanning.py`, `rf_fixmodel.py`, dan `xgboost_fix.py`. Nomenklatur ini membedakan secara eksplisit antara dua tahap dalam siklus pengembangan: fase optimasi hyperparameter & confidence (skrip `_scanning.py`) dan fase eksekusi dengan parameter terkunci (skrip `_fix.py`).

### 2.1 Pengumpulan dan Pra-pemrosesan Data [IMPLEMENTED]

Strategi pada tahap ini diterapkan pada **satu aset (XAU/USD)** menggunakan **satu timeframe (M1, satu menit)**. Data diakuisisi langsung dari MetaTrader 5 melalui `mt5.copy_rates_range()` pada simbol `XAUUSDc`. Periode data yang digunakan adalah **15 Februari 2026 hingga waktu eksekusi skrip** — kira-kira tiga bulan data per Mei 2026.

Implementasi saat ini **belum mengintegrasikan data eksogen** seperti S&P 500, Crude Oil, dan VIX yang menjadi inti metodologi Crasta (2024). Hal ini merupakan keputusan desain awal untuk membatasi *scope* iterasi pertama agar fokus pada validasi pipeline modular. Pengayaan dataset dengan indikator makroekonomi tersebut akan dilaksanakan pada fase berikutnya [FUTURE WORK]. Studi Crasta (2024) memberikan justifikasi untuk pemilihan indikator eksogen tersebut: dari analisis *feature importance*-nya, `crude_close` dan `sp_close` muncul sebagai prediktor paling signifikan untuk XAU/USD per jam, sementara skor sentimen FinBERT memiliki kontribusi yang relatif terbatas — sebuah temuan yang sejalan dengan peringatan Zewin (2026) bahwa penambahan data alternatif tidak otomatis meningkatkan performa.

### 2.2 Rekayasa Fitur dan Generasi Target [IMPLEMENTED]

Tahap rekayasa fitur mentransformasikan data harga mentah menjadi vektor numerik yang dapat dicerna oleh algoritma ensemble. Kode aktual menghasilkan **sekitar 45 fitur**, dikelompokkan sebagai berikut:

* **Indikator tren** — EMA 20, EMA 50, EMA 100, ditambah turunan: `ema_slope_5`, `ema_slope_10`, `ema_gap_20_50`, `ema_gap_50_100`, `distance_ema20/50/100`.
* **Indikator momentum & osilator** — RSI(14), Stochastic %K(14), `momentum_3/5/10`, dan versi yang dinormalisasi terhadap ATR (`momentum_n_atr`).
* **Indikator volatilitas** — ATR(14), `volatility_10`, `volatility_20`, `volatility_ratio`.
* **Fitur candle struktural** — `body`, `range`, `upper_wick`, `lower_wick` dengan rasio terhadap range.
* **Return rates** — `return_1/3/5/10` (pct change).
* **Volume** — `volume_change`, `volume_ratio` terhadap MA20.
* **Binary flags** — `ema_cross`, `price_above_ema*`, `rsi_above_50/overbought/oversold`, `stoch_above_50/overbought/oversold`.
* **Khusus XGBoost** — fitur sesi waktu (`london_session`, `newyork_session`, `hour`) dan filter ADX dengan `min_adx=20`.

Daftar fitur ini lebih ekstensif daripada kombinasi standar SMA/EMA/RSI/MACD/Bollinger yang biasa digunakan, namun **tidak mencakup sentimen FinBERT atau data fundamental makroekonomi** sebagaimana digunakan Crasta (2024). Konsekuensinya, model saat ini bekerja sebagai *technical-only learner* — keterbatasan yang harus diakui ketika menafsirkan akurasi.

Untuk variabel target, strategi menggunakan pendekatan **klasifikasi biner forward-looking berbasis SL/TP**. Algoritma melihat 10 candle ke depan (`future = 10`) dan menentukan label sebagai berikut:

```
SL = entry ± (1.00 × ATR)        # sl_multiplier = 1
TP = entry ± (1.46 × ATR)        # reward_ratio = 1.46

target = 1 (BUY)   jika TP buy tersentuh sebelum SL buy
target = 0 (SELL)  jika TP sell tersentuh sebelum SL sell
target = NaN       (dibuang) jika tidak ada sinyal valid
```

Pendekatan ini berbeda dari Crasta (2024) yang melakukan **regresi langsung** terhadap harga close XAU/USD. Tujuannya adalah menyelaraskan label dengan kejadian profit/loss aktual yang akan terjadi di backtest, bukan dengan deviasi prediksi harga yang abstrak.

### 2.3 Pelatihan, Validasi, dan Tuning Model

**[IMPLEMENTED]** Skrip `_fix.py` melatih model tunggal dengan hyperparameter yang telah dikunci, sedangkan skrip `_scanning.py` melakukan **dual-confidence grid search** terhadap ambang batas keyakinan untuk sinyal BUY (`buy_conf`) dan SELL (`sell_conf`) secara independen pada rentang 0.50–0.85. Untuk setiap kombinasi `(buy_conf, sell_conf)`, sistem menjalankan manual backtest dan menghitung skor komposit:

```
score = profit_factor × 40 + winrate × 0.5 + roi × 0.2 − max_drawdown × 1.5
        (+ bonus 20 jika ada buy_trade dan sell_trade)
```

Filter aman diterapkan: hanya kombinasi dengan `total_trade ≥ 50`, `profit_factor ≥ 1.10`, dan `max_drawdown ≤ 30%` yang masuk pool kandidat.

**Pemisahan data** saat ini menggunakan **simple time-based 80/20 split** — sesuai dengan metodologi yang juga dipakai Crasta (2024). Skrip `xgboost_scanning.py` meng-*import* `TimeSeriesSplit` dari scikit-learn, namun komitmen penuh pada **Walk-Forward Validation dengan rolling window** belum sepenuhnya terimplementasi dan menjadi prioritas roadmap berikutnya [FUTURE WORK]. Krasanya, banyak literatur menekankan WFV sebagai metodologi yang lebih ketat untuk pencegahan *look-ahead bias*, meskipun Crasta (2024) sendiri menggunakan GridSearch dengan 5-fold cross-validation pada *training set* tanpa WFV eksplisit dan tetap menghasilkan kesimpulan yang valid.

**Hyperparameter aktual:**

| Model | Konfigurasi di Kode | Konfigurasi Crasta (2024) sebagai Pembanding |
| :--- | :--- | :--- |
| Random Forest | `n_estimators=500`, `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=8`, `class_weight='balanced'`, `random_state=42` | `n_estimators=150`, `max_depth=30`, `min_samples_split=2`, `min_samples_leaf=1` |
| XGBoost | + filter ADX `min_adx=20`, regularisasi L1/L2 default | `colsample_bytree=1.0`, `learning_rate=0.2`, `max_depth=8`, `n_estimators=200`, `subsample=0.7` |

Pilihan kami untuk RF mengarah ke **regularisasi yang lebih ketat** (`max_depth` lebih dangkal, `min_samples_*` lebih besar) dibandingkan Crasta, karena tugas klasifikasi sinyal trading M1 jauh lebih rentan terhadap *overfitting* pada *noise* tick-level daripada tugas regresi harga harian.

### 2.4 Generasi Sinyal dan Eksekusi Backtest [IMPLEMENTED]

Setelah model terlatih, prediksi probabilitas pada set uji dikonversi menjadi sinyal trading:

```python
if buy_probability >= best_buy_conf:    signal = 1 (BUY)
elif sell_probability >= best_sell_conf: signal = 0 (SELL)
else:                                    signal = NaN (no trade)
```

Eksekusi dilakukan melalui dua jalur paralel: (a) **manual backtest** sederhana yang men-simulasikan SL/TP hit pada `future` candle berikutnya, dan (b) integrasi penuh dengan library **`backtesting.py`** yang menerapkan *position sizing* berbasis risiko 1% per trade, leverage 100, dan margin requirement realistis. Output `backtesting.py` mencakup equity curve, statistik perdagangan lengkap, dan plot HTML interaktif (`TradingStrategyRF.html`, `TradingStrategyXGBoost.html`).

### 2.5 Catatan Penyesuaian terhadap Versi Sebelumnya

Versi sebelumnya dokumen ini mengasumsikan keberadaan komponen yang belum diimplementasikan: integrasi langsung S&P 500/VIX/Crude Oil, sentimen berita melalui FinBERT, multi-timeframe, *Expanding Window Split*, dan paritas pengujian XAU/USD–BTC/USD. Penyesuaian ini memindahkan komponen-komponen tersebut ke kolom [FUTURE WORK] dan menjadikan **paper Crasta (2024) sebagai blueprint metodologis target** yang akan diadopsi bertahap pada iterasi berikutnya.

---

## 3. Analisis Komparatif Performa Model pada Aset XAU/USD (Emas)

### 3.1 Posisi Penelitian dalam Konteks Literatur

Evaluasi komparatif antara XGBoost dan Random Forest untuk prediksi XAU/USD menghadirkan gambaran yang kompleks dan kontekstual. Tidak ada model yang secara universal superior; keunggulan relatif keduanya sangat bergantung pada timeframe, jenis fitur, dan kerangka evaluasi.

**Crasta (2024)** — paper inti yang menjadi landasan proyek ini — secara langsung membandingkan tujuh model (Decision Tree, RF, Gradient Boosting, XGBoost, SVR, LSTM, GRU) untuk prediksi XAU/USD per jam menggunakan dataset terintegrasi yang mencakup sentimen berita FinBERT, harga S&P 500, harga Crude Oil, dan VIX selama periode Juli 2020 – Desember 2023. Hasil komprehensifnya disajikan dalam tabel berikut:

| Model | R² Score | MSE | MAE | Peringkat |
| :---- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9782** | **187.00** | **7.20** | **1** |
| XGBoost | 0.9760 | 205.95 | 8.52 | 2 |
| Gradient Boosting | 0.9637 | 311.30 | 10.51 | 3 |
| Decision Tree | 0.9587 | 354.74 | 7.93 | 4 |
| SVR | 0.9472 | 452.85 | 11.46 | 5 |
| LSTM | 0.9316 | 587.27 | 17.28 | 6 |
| GRU | 0.9187 | 697.51 | 18.95 | 7 |

> *Sumber: Crasta (2024), "Comparative Analysis of Machine Learning Algorithms For XAU/USD Prediction", Tabel 2.*

Crasta menyimpulkan: *"Random Forest emerged as the best performing model in terms of both metrics and simplicity. It achieved the lowest MSE and highest R-squared score while maintaining interpretability. The second ranked model was XGBoost because of its strong performance metrics and its capability to handle non-linearity in the data."* Temuan ini menggarisbawahi keunggulan RF dalam menangani data campuran teknikal + fundamental + sentimen, mungkin karena sifatnya yang lebih *robust* terhadap fitur yang kurang relevan atau *noise*.

Namun, gambaran kontekstual tetap berlaku. Kilimci (2022) yang berfokus pada prediksi harian XAU/USD menemukan bahwa model **Stacking Regressor** (menggabungkan beberapa model termasuk XGBoost dan RF) mencapai MAPE terbaik 2.2036, dengan RF tunggal pada MAPE 4.2544 — menunjukkan kombinasi model bisa lebih unggul daripada model tunggal. Selain itu, studi oleh Dave (2024) menemukan bahwa **XGBoost menunjukkan keunggulan pada timeframe 4-jam** dengan RMSE 0.0005 (terendah di antara semua model yang diuji), sementara pada **timeframe 5-menit, Random Forest justru menunjukkan profitabilitas tertinggi** dalam simulasi perdagangan (9.2%) — sebuah wawasan yang sangat relevan untuk proyek kami yang bekerja pada M1.

| Model | Aset | Timeframe | Metrik Utama | Hasil | Sumber |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | XAU/USD | 1 Jam | MAE, R² | 7.20, 0.9782 | Crasta (2024) |
| XGBoost | XAU/USD | 1 Jam | MAE, R² | 8.52, 0.9760 | Crasta (2024) |
| Stacking Regressor | XAU/USD | 1 Hari | MAPE | 2.2036 | Kilimci (2022) |
| Random Forest (mandiri) | XAU/USD | 1 Hari | MAPE | 4.2544 | Kilimci (2022) |
| XGBoost | XAU/USD | 4 Jam | RMSE | 0.0005 (terendah) | Dave (2024) |
| Random Forest | XAU/USD | 5 Menit | Profit Simulasi | 9.2% | Behjoee (2025) |

### 3.2 Hasil Empiris Implementasi Kami pada XAU/USD M1 [IMPLEMENTED]

Hasil backtest aktual dari skrip `rf_scanning.py` dan `xgboost_scanning.py` pada data XAUUSDc M1 (Februari–Mei 2026) memberikan gambaran komparatif berikut:

#### Random Forest — Konfigurasi Optimal: `buy_conf=0.72, sell_conf=0.50`

| Metrik | Nilai |
| :--- | :---: |
| Total Trade | 315 (126 BUY, 189 SELL) |
| Winrate | 55.24% |
| Profit Factor | 1.71 |
| Max Drawdown | 6.38% |
| ROI | +201.9% |
| Saldo akhir simulasi | $30,190 (dari $10,000) |

#### XGBoost — Konfigurasi Optimal: `buy_conf=0.85, sell_conf=0.55`

| Metrik | Nilai |
| :--- | :---: |
| Total Trade | 143 (70 BUY, 73 SELL) |
| Winrate | 60.14% |
| Profit Factor | 2.02 |
| Max Drawdown | 14.77% |
| ROI | +96.1% |
| Saldo akhir simulasi | $19,614 (dari $10,000) |

#### Sintesis Komparatif

Pola yang muncul dari hasil empiris kami **mengonfirmasi tesis utama literatur "no universal winner"**, namun dengan nuansa yang menarik untuk timeframe M1:

* **Random Forest unggul dalam stabilitas dan jumlah peluang** — DD hanya 6.38% (jauh di bawah ambang ketat 15% yang lazim digunakan literatur), dengan 315 trade memberikan signal yang konsisten. Pola ini selaras dengan temuan Behjoee (2025) bahwa RF unggul di timeframe rendah pada XAU/USD.
* **XGBoost unggul dalam selektivitas dan kualitas signal** — winrate 60.14% dan PF 2.02 menempatkannya pada level yang sangat baik secara literatur (di atas threshold operasional minimum PF 1.4 yang sering dikutip), meskipun dengan DD lebih tinggi dan jumlah trade lebih sedikit.

*Catatan kalibrasi metodologis*: konfigurasi confidence terbaik untuk RF dan XGBoost dipilih dari **test set yang sama** yang digunakan mengukur performa di tahap evaluasi. Praktik ini berisiko *selection bias* dan mungkin membuat ROI 201.9% tampak optimistik dibandingkan performa *out-of-sample* murni. Pemisahan *validation set* khusus untuk pemilihan confidence menjadi salah satu prioritas perbaikan [FUTURE WORK].

### 3.3 Analisis Feature Importance

Analisis *feature importance* dari kedua model implementasi kami memberikan wawasan penting:

| Top Features Random Forest | Importance | Top Features XGBoost | Importance |
| :--- | :---: | :--- | :---: |
| ema50 | 0.067 | newyork_session | 0.047 |
| ema20 | 0.062 | price_above_ema20 | 0.039 |
| ema100 | 0.061 | ema50 | 0.029 |
| ema_gap_50_100 | 0.060 | ema100 | 0.028 |
| ema_gap_20_50 | 0.046 | ema_gap_50_100 | 0.027 |
| atr | 0.044 | stoch_above_50 | 0.027 |
| volatility_20 | 0.042 | london_session | 0.027 |
| distance_ema100 | 0.041 | hour | 0.026 |

**Insight**: Random Forest secara konsisten memberikan bobot tertinggi pada **fitur tren EMA dan volatilitas**, mengikuti pola yang murni teknikal. XGBoost — selain memanfaatkan EMA — menempatkan **fitur sesi waktu (`newyork_session`, `london_session`, `hour`)** pada peringkat tertinggi, mengindikasikan bahwa model boosting mampu mengekstrak *edge tambahan* dari struktur temporal intra-hari yang luput dari RF. Ini sejalan dengan kesimpulan Crasta (2024) bahwa XGBoost lebih unggul dalam *"mendeteksi sisa fluktuasi non-linear kecil berjangka pendek"*.

Sebagai perbandingan, *feature importance* di Crasta (2024) untuk model RF menempatkan `crude_close` dan `sp_close` sebagai prediktor paling signifikan — fitur fundamental yang **belum kami integrasikan**. Implikasinya, jika kami menambahkan data eksogen tersebut pada iterasi berikutnya, model RF kami berpotensi mengalami pergeseran peringkat fitur yang signifikan dan kemungkinan peningkatan akurasi.

### 3.4 Implikasi untuk Strategi Trading

Berdasarkan temuan empiris dan komparasi literatur, implikasi praktis berikut dapat ditarik:

* **Trader dengan toleransi risiko rendah** sebaiknya menggunakan profil RF — DD 6.38% memberikan ruang manuver yang besar tanpa risiko *margin call* serius.
* **Trader yang mencari kualitas signal tertinggi** sebaiknya menggunakan profil XGBoost — winrate 60% dan PF 2.02 memberikan *edge* yang lebih bersih per trade.
* **Pengembangan selanjutnya** ke arah **Stacking Ensemble** yang menggabungkan output probabilitas kedua model (sesuai Kilimci 2022) berpotensi memberikan profitabilitas tertinggi sekaligus stabilitas terbaik.

Studi oleh Zewin (2026) memberikan peringatan kontra-intuitif yang relevan: penambahan data alternatif seperti Google Trends dan sentimen ternyata dapat **menyebabkan penurunan performa prediksi harga emas sekitar 2%** untuk kedua model. Ini menyerukan validasi empiris yang ketat ketika kami nantinya mengintegrasikan data eksogen pada iterasi berikutnya — bukan asumsi bahwa lebih banyak data otomatis bermanfaat.

---

## 4. Analisis Komparatif Performa Model pada Aset BTC/USD (Bitcoin) [FUTURE WORK]

### 4.1 Status Implementasi Saat Ini

**Catatan transparansi**: implementasi kode saat ini **belum mencakup BTC/USD**. Keempat skrip Python dirancang dengan `symbol = "XAUUSDc"` yang di-*hardcode*, dan belum ada file output (CSV/HTML) yang dihasilkan untuk Bitcoin. Bagian ini disajikan sebagai **landasan literatur** untuk fase pengujian BTC/USD yang akan dilakukan setelah pipeline XAU/USD distabilkan dan dioptimasi.

### 4.2 Landasan Literatur untuk BTC/USD

Perbandingan antara XGBoost dan Random Forest untuk prediksi harga BTC/USD menunjukkan pola yang berbeda dari yang diamati pada XAU/USD. Berbeda dengan emas yang sering berfungsi sebagai aset aman, Bitcoin adalah aset yang sangat reaktif terhadap sentimen pasar, kebijakan moneter, dan dinamika kripto-spesifik. Analisis literatur menunjukkan bahwa **XGBoost sering menunjukkan keunggulan performa yang signifikan dalam konteks prediksi harga Bitcoin**, baik dalam akurasi prediksi maupun profitabilitas strategi.

Salah satu studi paling menonjol yang secara eksplisit membandingkan kedua model untuk Bitcoin adalah **Corbet et al. (2020c)**. Dalam penelitian ini, **XGBoost mencapai imbal hasil kumulatif 141.4% dengan rasio Sharpe sebesar 1.78** — angka yang secara signifikan lebih tinggi dibandingkan model lain dan menjadi *benchmark* utama untuk pengembangan kami. Studi lain menemukan bahwa XGBoost mengungguli LSTM dalam hal paritas biaya, sementara dalam analisis multi-model XGBoost dan *double deep Q-learning* menunjukkan rasio Sharpe yang setara dan kuat (0.63x).

| Model | Aset | Metrik Utama | Hasil | Sumber |
| :--- | :--- | :--- | :--- | :--- |
| XGBoost | BTC/USD | Imbal Hasil Kumulatif, Rasio Sharpe | 141.4%, 1.78 | Corbet et al. (2020c) |
| XGBoost | BTC/USD | Rasio Sharpe | 0.63x | Deep Learning for Financial Forecasting |
| Double Deep Q-Network | BTC/USD | Rasio Sharpe | 0.63x | Deep Learning for Financial Forecasting |
| ARFIMA-GARCH | BTC/USD | Rasio Sharpe | Di bawah XGBoost | Deep Learning for Financial Forecasting |
| LSTM | BTC/USD | Paritas Biaya | Di bawah XGBoost | Risks 13(10), 195 |

### 4.3 Hipotesis untuk Pengujian BTC/USD Kami

Berdasarkan literatur, hipotesis yang akan diuji ketika pipeline BTC/USD diaktifkan adalah:

1. **H1**: XGBoost akan mengungguli Random Forest pada BTC/USD M1 dalam metrik winrate dan Profit Factor, mengulangi pola yang ditemukan Corbet et al. (2020c).
2. **H2**: Fitur sesi waktu (`asia_session`, `europe_session`, `us_session`) akan tetap menjadi prediktor penting bagi XGBoost — meskipun pasar kripto buka 24/7, likuiditas tetap berfluktuasi mengikuti zona waktu pelaku pasar institusional.
3. **H3**: Stabilitas Random Forest akan menurun dibandingkan pada XAU/USD karena volatilitas BTC yang lebih tinggi — DD diperkirakan akan melonjak dari 6.38% (XAU/USD) ke nilai 15–25%.

Untuk validasi yang fair terhadap Corbet, integrasi fitur eksogen Bitcoin-spesifik akan diperlukan: **kebijakan moneter Fed** (suku bunga, expected rate), **data on-chain** (volume transaksi, jumlah alamat aktif, hashrate), dan **sentimen sosial media**. Tanpa fitur ini, performa XGBoost kami untuk BTC kemungkinan akan jauh dari benchmark Sharpe 1.78.

### 4.4 Roadmap Integrasi BTC/USD

| Langkah | Estimasi Effort | Prioritas |
| :--- | :--- | :--- |
| Duplikasi skrip dengan `symbol = "BTCUSD"` | 1 hari | HIGH |
| Validasi data MT5 untuk BTC tersedia | 1 hari | HIGH |
| Kalibrasi `future`, `sl_multiplier`, `reward_ratio` untuk volatilitas BTC | 3 hari | HIGH |
| Integrasi data on-chain via API (Glassnode/CoinMetrics) | 1 minggu | MEDIUM |
| Integrasi data kebijakan Fed (FRED API) | 3 hari | MEDIUM |
| Re-running scanning dan komparasi terhadap benchmark Corbet | 2 hari | HIGH |

---

## 5. Validasi Kontekstual: Dampak Volatilitas Pasar dan Tren Makroekonomi

Evaluasi kelayakan strategi tidak akan lengkap tanpa memvalidasi kontekstualnya terhadap kondisi pasar global yang sedang berlangsung. Efektivitas kedua algoritma sangat bergantung pada kemampuannya untuk menangkap dan merespons sinyal-sinyal fundamental, yang secara dinamis membentuk harga XAU/USD dan BTC/USD.

### 5.1 Konteks XAU/USD pada 2026

Untuk XAU/USD (emas), posisinya sebagai aset *safe-haven* adalah kunci memahami perilakunya. Faktor makroekonomi paling berpengaruh adalah **kebijakan moneter The Federal Reserve**: kenaikan suku bunga cenderung menekan harga emas (meningkatkan opportunity cost bagi aset tanpa imbal hasil), sementara pemotongan suku bunga sering mendorong harga emas naik. Indikator volatilitas CBOE (VIX) memiliki korelasi positif yang kuat dengan harga emas, sementara kinerja pasar saham (S&P 500) sering berkorelasi negatif.

**Status integrasi kami**: data eksogen tersebut **belum diintegrasikan** [FUTURE WORK]. Konsekuensinya, model RF dan XGBoost saat ini bekerja sepenuhnya berdasarkan **sinyal teknikal dari pergerakan harga XAU/USD itu sendiri** — yang mungkin sudah secara implisit "mengandung" reaksi pasar terhadap rilis berita ekonomi (karena terefleksi pada perubahan harga), tetapi tidak memiliki *forward signal* yang berasal dari indikator makro independen.

Kondisi pasar emas pada 2025–2026 sangat mendukung relevansi strategi: harga emas mencapai rekor di area $3,700-an per ons pada akhir 2025, menunjukkan permintaan kuat meski The Fed diperkirakan menahan kenaikan suku bunga di Q1 2026. Dalam konteks volatilitas tinggi ini, model ML yang dapat menangkap sinyal fundamental menjadi sangat berharga.

Namun, peringatan Zewin (2026) tetap penting: penambahan data alternatif seperti sentimen berita dan VIX justru menyebabkan **penurunan performa prediksi emas sekitar 2%** dalam studinya. Implikasi praktisnya, ketika kami mengintegrasikan data eksogen pada iterasi berikutnya, kami harus melakukan **A/B testing yang ketat**: bandingkan performa model dengan dan tanpa fitur tambahan, dan hanya pertahankan fitur yang secara empiris meningkatkan metrik finansial — bukan hanya metrik prediksi.

### 5.2 Konteks BTC/USD

Untuk BTC/USD (Bitcoin), konteks makroekonominya berbeda tetapi sama-sama kritis. Bitcoin dikenal sangat reaktif terhadap kebijakan moneter Fed, dengan adanya *price drift* yang signifikan dalam beberapa hari setelah pertemuan FOMC. Sentimen publik juga memainkan peran ganda: sentimen positif dari media sosial dan berita dapat mendorong harga naik, sementara sentimen negatif dapat menyebabkan penurunan drastis. Data on-chain (volume transaksi, jumlah alamat aktif) memberikan wawasan tentang aktivitas pengguna dan nilai intrinsik Bitcoin.

Keberhasilan XGBoost dalam prediksi harga Bitcoin (Corbet et al. 2020c — Sharpe 1.78) kemungkinan disebabkan oleh kemampuannya yang superior dalam **mengintegrasikan ketiga jenis data tersebut** (makroekonomi, sentimen, on-chain) dan menangkap interaksi kompleks di antara mereka. **Status integrasi kami**: belum dimulai [FUTURE WORK].

### 5.3 Risiko Pasar yang Harus Dinavigasi

Dari perspektif volatilitas, pasar kripto secara inheren lebih volatil daripada pasar logam tradisional. Strategi perdagangan yang dikembangkan harus mampu mengelola risiko ini. **Maximum Drawdown (MDD)** menjadi sangat penting. Sebuah studi menunjukkan bahwa strategi dengan Profit Factor rendah (0.44) dan rasio Sharpe negatif menunjukkan kinerja yang lemah, sering disebabkan MDD yang tinggi. Hasil aktual XGBoost kami di XAU/USD sudah menunjukkan DD 14.77% — sudah mendekati ambang batas 15% yang lazim digunakan. Untuk BTC dengan volatilitas yang lebih tinggi, kontrol DD akan menjadi tantangan signifikan yang membutuhkan kalibrasi ulang `sl_multiplier` dan `reward_ratio`.

---

## 6. Kerangka Evaluasi, Metrik Kinerja, dan Analisis Risiko Strategis

Penilaian kelayakan strategi memerlukan kerangka evaluasi komprehensif: metrik untuk performa model ML, dan metrik untuk profitabilitas/risiko strategi perdagangan. Analisis risiko strategis juga penting untuk memahami potensi kegagalan.

### 6.1 Metrik Level Model ML

Untuk tugas regresi (memprediksi harga mutlak — pendekatan yang digunakan Crasta 2024), metrik standar adalah:

* **Mean Absolute Error (MAE)** — rata-rata absolut deviasi prediksi. Mudah diinterpretasi.
* **Mean Squared Error (MSE)** — kuadrat deviasi, memberikan bobot lebih pada kesalahan besar.
* **Root Mean Squared Error (RMSE)** — akar MSE, sensitif terhadap outlier.
* **R-squared (R²)** — proporsi variansi target yang dijelaskan model.

Crasta (2024) menggunakan ketiga metrik ini secara konsisten — RF mencapai R² 0.9782, MSE 187, MAE 7.20 sebagai performa terbaik di antara tujuh model.

Untuk tugas klasifikasi (memprediksi arah pergerakan harga — **pendekatan yang digunakan implementasi kami**), metrik yang relevan:

* **Accuracy** — proporsi prediksi benar.
* **Precision, Recall, F1-Score** — penting jika kelas tidak seimbang.
* **Confusion Matrix** — menampilkan komposisi prediksi.

**Status implementasi**: kode kami menghitung *winrate aktual dari simulasi backtest* (bukan akurasi klasifikasi raw), yang lebih relevan untuk evaluasi finansial. Penambahan **Precision/Recall/F1** untuk evaluasi quality signal sebagai *complement* sedang dipertimbangkan [FUTURE WORK].

### 6.2 Metrik Level Strategi Trading

Metrik ini lebih substansial dan berorientasi finansial:

1. **Total / Annualized Return** — keuntungan kumulatif. **[IMPLEMENTED]** ROI 201.9% (RF), 96.1% (XGBoost).
2. **Sharpe Ratio** — imbal hasil per unit volatilitas. **[FUTURE WORK]** Belum dihitung di output kami — meskipun `backtesting.py` menyediakannya gratis di output `stats`. Studi Corbet menggunakan Sharpe sebagai tolok ukur utama (1.78 untuk XGBoost di BTC). Implementasi pemanggilan dan pelaporan Sharpe Ratio menjadi item prioritas tinggi.
3. **Maximum Drawdown (MDD)** — kerugian terbesar dari puncak ke lembah. **[IMPLEMENTED]** RF 6.38%, XGBoost 14.77%. Threshold operasional 15% terjaga.
4. **Profit Factor** — total profit kotor / total loss kotor. **[IMPLEMENTED]** RF 1.71, XGBoost 2.02. Keduanya di atas threshold minimum 1.4 yang sering dikutip literatur.
5. **Win Rate** — persentase trade menguntungkan. **[IMPLEMENTED]** RF 55.24%, XGBoost 60.14%.

#### Tabel Ringkasan Metrik

| Level Evaluasi | Metrik | Status Implementasi | Hasil Kami |
| :--- | :--- | :--- | :--- |
| **Model ML** | MAE, RMSE, R² | ⚠️ Crasta menggunakan; kami tidak (kami pakai klasifikasi) | — |
| **Model ML** | Accuracy / Winrate | ✅ IMPLEMENTED | RF 55.24%, XGB 60.14% |
| **Model ML** | F1-Score | ⏳ FUTURE WORK | — |
| **Strategi Trading** | Total Return / ROI | ✅ IMPLEMENTED | RF +201.9%, XGB +96.1% |
| **Strategi Trading** | Sharpe Ratio | ⏳ FUTURE WORK | — (target ≥ 1.0) |
| **Strategi Trading** | Maximum Drawdown | ✅ IMPLEMENTED | RF 6.38%, XGB 14.77% |
| **Strategi Trading** | Profit Factor | ✅ IMPLEMENTED | RF 1.71, XGB 2.02 |
| **Strategi Trading** | Win Rate | ✅ IMPLEMENTED | RF 55.24%, XGB 60.14% |
| **Baseline** | Buy & Hold benchmark | ⏳ FUTURE WORK | — |

### 6.3 Analisis Risiko Strategis

Ada beberapa alasan mengapa pendekatan ini bisa gagal di lingkungan live trading:

* **Overfitting** — terutama untuk XGBoost yang kompleks. Tanpa validasi silang urutan waktu yang ketat (WFV), hasil backtest yang bagus bisa menjadi ilusi. Konfigurasi kami sudah menerapkan regularisasi (`max_depth=10`, `min_samples_leaf=8` untuk RF), tetapi WFV penuh belum diterapkan.
* **Selection Bias pada Confidence Tuning** — *threshold* dipilih dari *test set* yang sama yang dipakai mengukur performa. ROI 201.9% mungkin overoptimistic dibandingkan *out-of-sample* murni. Pemisahan *validation set* khusus menjadi prioritas.
* **Periode Data Sempit** — 3 bulan data (Feb–Mei 2026) tidak setara validasi multi-siklus. Generalisasi ke rezim pasar yang berbeda (bullish vs bearish vs sideways) belum terbukti.
* **Bias Signal SELL** — RF menghasilkan 189 SELL vs 126 BUY, mengindikasikan model bias karena XAUUSD trending turun di periode training. Performa pada pasar bullish masih dipertanyakan.
* **Perubahan Dinamika Pasar** — pola yang berhasil pada data historis mungkin tidak lagi berlaku karena perubahan regulasi, perilaku investor, atau faktor makroekonomi baru.
* **Biaya Transaksi dan Slippage** — kode kami menggunakan `komisi = 0.0`. Di realita, spread broker bisa mengikis profit signifikan di M1. Adaptasi untuk modelisasi spread realistis dibutuhkan.
* **Temuan Kontra-intuitif (Zewin 2026)** — penambahan fitur yang tampaknya relevan (sentimen, alt data) justru dapat menurunkan performa model. Mengasumsikan semua fitur eksogen bermanfaat bisa menjadi kesalahan fatal.

### 6.4 Argumen Keunggulan Pendekatan

Sebaliknya, ada juga argumen kuat mengapa ide ini bisa menguntungkan:

* **Kapasitas non-linear** — model ML menangkap hubungan kompleks yang luput dari analisis manusia atau model statistik tradisional. ARIMA-GARCH terbaik (Bunnag 2023) hanya mencapai MAE 106.71; RF Crasta (2024) mencapai MAE 7.20 — perbedaan dua orde besaran.
* **XGBoost untuk integrasi multi-data** — terbukti menghasilkan Sharpe sangat tinggi untuk aset kripto seperti Bitcoin (Corbet et al. 2020c: Sharpe 1.78).
* **Random Forest untuk kestabilan** — menawarkan *robust* dan stabilitas, model andal terutama untuk aset seperti emas yang responsif terhadap berbagai sinyal fundamental (Crasta 2024).
* **Arsitektur modular** — pemisahan *scanning* (training periodik) dan *fix model* (eksekusi) memberikan mekanisme adaptif terhadap *concept drift* yang jarang dimiliki strategi monolitik.
* **Dual-confidence optimization** — fleksibilitas tuning ambang batas BUY dan SELL secara independen memungkinkan model menyesuaikan diri dengan bias arah pasar saat ini.

Dengan demikian, kesimpulannya bukan pada klaim absolut, tetapi pada identifikasi konteks di mana masing-masing model dapat memberikan keunggulan. **Keberhasilan penelitian ini bergantung pada implementasi metodologi yang ketat, pemilihan fitur yang cerdas, dan interpretasi hasil yang bijaksana**, dengan mengakui bahwa tidak ada satu pun model yang secara universal superior.

---

## 7. Kesimpulan dan Saran Lanjutan

### 7.1 Kesimpulan Utama

Dari analisis komparatif antara XGBoost dan Random Forest dengan landasan paper Crasta (2024) dan empat skrip implementasi aktual, dapat disimpulkan:

1. **Implementasi pipeline modular berfungsi** — kedua model menghasilkan strategi yang *tradable* di paper trading, dengan Profit Factor di atas threshold operasional minimum (RF 1.71, XGBoost 2.02).

2. **Hasil empiris konsisten dengan tesis literatur "no universal winner"** — Random Forest unggul dalam stabilitas (DD 6.38%, ROI 201.9% via volume trade) sebagaimana ditemukan Crasta (2024) untuk XAU/USD per jam, sementara XGBoost unggul dalam kualitas signal (winrate 60.14%, PF 2.02) — sebuah pola yang juga ditemukan di benchmark Corbet et al. (2020c) untuk BTC.

3. **Karakter trade-off bukan kekurangan** — RF dan XGBoost merepresentasikan dua filosofi ensemble yang berbeda secara fundamental (bagging vs boosting). Perbedaan profil performa adalah konsekuensi langsung dari arsitektur, bukan bug.

4. **Gap antara aspirasi dan implementasi diakui secara jujur** — komponen seperti integrasi S&P/VIX/Crude Oil/sentimen (yang menjadi inti Crasta 2024), pengujian BTC/USD (benchmark Corbet), Walk-Forward Validation, dan kalkulasi Sharpe Ratio masih berada dalam roadmap [FUTURE WORK].

5. **Hasil saat ini bukan bukti edge nyata untuk live trading** — perlu validasi out-of-sample yang lebih ketat, baseline Buy & Hold, dan modelisasi spread realistis sebelum klaim deploy-ready.

### 7.2 Saran untuk Pengembangan Iteratif

| Prioritas | Item | Rasional |
| :--- | :--- | :--- |
| HIGH | Implementasi WFV penuh dengan `TimeSeriesSplit(n_splits=5)` | Mitigasi *look-ahead bias*, sesuai praktik akademik |
| HIGH | Hitung dan laporkan Sharpe Ratio dari `backtesting.py stats` | Benchmark terhadap Corbet (Sharpe 1.78) |
| HIGH | Pisahkan *validation set* untuk pemilihan confidence | Hilangkan *selection bias* pada threshold |
| HIGH | Tambah baseline Buy & Hold di output backtest | Verifikasi *alpha* nyata |
| HIGH | Replikasi pipeline untuk BTC/USD | Lengkapi cakupan riset XAU+BTC |
| MEDIUM | Integrasi data eksogen S&P 500, VIX, Crude Oil via MT5 | Mengikuti blueprint Crasta (2024) |
| MEDIUM | Integrasi sentimen FinBERT untuk berita XAU/BTC | Replikasi metodologi Crasta penuh |
| MEDIUM | Modelisasi spread broker realistis (`komisi > 0`) | Validasi profitabilitas net |
| LOW | Hybrid Stacking Ensemble (RF + XGBoost via meta-learner logreg) | Mengikuti Kilimci (2022) — best MAPE 2.20 |
| LOW | Asymmetric custom loss function berbasis Sharpe/Sortino | Optimasi langsung pada metrik finansial |

### 7.3 Penutup

Versi harmonisasi dokumen ini menggarisbawahi bahwa **landasan akademik proyek ini kuat** — Crasta (2024) dari `Journal_merged.md` memberikan kerangka metodologis yang teruji, dan implementasi kami pada XAU/USD M1 memberikan validasi empiris awal terhadap kapasitas RF dan XGBoost untuk menangkap pola non-linear pasar emas. Namun, **proyek ini belum mencapai cakupan penuh** sebagaimana diuraikan literatur referensi. Dengan roadmap [FUTURE WORK] yang terstruktur, kesenjangan tersebut dapat ditutup dalam iterasi berikutnya, dan strategi ini berpotensi menjadi salah satu studi kasus *Algorithmic Trading* berbasis ML yang lengkap di kelas riset komparatif XAU + BTC.

---

## Referensi Inti (Disesuaikan dengan `Journal_merged.md`)

1. **Crasta, M. (2024)**. *Comparative Analysis of Machine Learning Algorithms For XAU/USD Prediction: Integrating Economic Indicators And Sentiment Analysis*. MSc Research Project, National College of Ireland. — **Paper utama yang menjadi landasan metodologis proyek ini.**

2. **Dave, Y. (2024)**. *Forex Trend Prediction: Can ML Algorithms Forecast Short Timeframe Movements?* MAT(Comp) Thesis.

3. **Corbet, S. et al. (2020c)**. *ML for Bitcoin Price Prediction* — XGBoost mencapai Sharpe 1.78, return 141.4% untuk BTC/USD.

4. **Kilimci, Z. (2022)**. *Ensemble Regression-Based Gold Price (XAU/USD) Prediction*. Journal of Emerging Market Finance — Stacking ensemble MAPE 2.20 sebagai benchmark hybrid.

5. **Cohen, G. & Aiche, A. (2023)**. *Forecasting Gold Price Using ML Methodologies*. Chaos, Solitons & Fractals 175 — XGBoost MSE 0.0000186 untuk data 2011–2023.

6. **Behjoee, A. (2025)**. *Forex Trend Prediction at Short Timeframes* — RF unggul di TF 5-menit XAU/USD (profit simulasi 9.2%).

7. **Zewin, A. (2026)**. *Forecasting Gold Price Movements Using Alternative Data and ML Techniques* — Peringatan: penambahan alt data dapat menurunkan akurasi 2%.

8. **Bunnag, T. (2023)**. *ARIMA, ARIMA-GARCH, ARIMA-TGARCH for Gold Price Forecasting* — Baseline tradisional dengan MAE 106.71.

9. **Junjie, Z. & Mengoni, P. (2020)**. *Spot Gold Price Prediction Using Financial News Sentiment Analysis* — Sentimen 5-hari mencapai akurasi 78.43%.

10. **Sami, I. & Nazir, K. (2018)**. *Predicting Future Gold Rates Using Machine Learning Approach* — ANN RMSE 19 vs Linear Regression (baseline metodologis).

---

## Lampiran A: Pemetaan Antara Bagian Dokumen dan Komponen Kode

| Bagian Dokumen | Skrip Kode | Status |
| :--- | :--- | :--- |
| §2.1 Pengumpulan Data | `mt5.copy_rates_range()` di semua skrip | ✅ |
| §2.2 Rekayasa Fitur | Lines 67–157 di `rf_*.py`; ekstensi sesi waktu di `xgboost_*.py` | ✅ |
| §2.2 Target Labeling | Loop SL/TP forward-looking di semua skrip | ✅ |
| §2.3 Train/Test Split | 80/20 simple split (line 259 di `rf_scanning.py`) | ⚠️ Bukan WFV |
| §2.3 Dual Confidence Grid | Function `manual_backtest_dual_confidence()` | ✅ |
| §2.4 Manual Backtest | Implementasi internal di setiap skrip | ✅ |
| §2.4 backtesting.py Integration | Class `TradingStrategyRF`/`TradingStrategyXGBoost` | ✅ |
| §3.2 Hasil Empiris | CSV: `optimization_confidence_balanced.csv`, `optimization_confidence_xgboost.csv` | ✅ |
| §3.3 Feature Importance | CSV: `feature_importance_*.csv` | ✅ |
| §4 BTC/USD | Belum ada skrip | ⏳ |
| §5 Data Eksogen | Belum diintegrasikan | ⏳ |
| §6.2 Sharpe Ratio | Tersedia di output `bt.run()` tapi belum di-extract | ⏳ |
