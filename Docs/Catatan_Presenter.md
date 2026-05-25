# Catatan Presenter — Progress Research Presentation

**Referensi deck:** `Presentasi_Progress_Research.md`
**Total slide:** ~43 slide
**Durasi target:** 18–22 menit + 5–10 menit Q&A
**Audiens:** Dosen pembimbing & penguji UAS — asumsikan familiar dengan ML dasar tapi bukan ahli trading

---

## Cara Memakai Catatan Ini

Setiap slide diberi tiga bagian:
- **Apa yang dijelaskan** — inti pesan slide
- **Talking points** — kalimat kunci yang harus diucapkan
- **Catatan delivery** — tips, antisipasi pertanyaan, atau peringatan

**Aturan emas saat presentasi:**
1. **Jangan baca slide** — slide adalah "anchor", narasi dari mulut Anda
2. **Selalu link kembali ke source code** — sebutkan nama file Python ketika relevan, ini menunjukkan Anda menguasai implementasi
3. **Jujur tentang gap** — jika ditanya soal yang belum diimplementasi, jangan ngeles. Akui dan tunjukkan posisinya di roadmap
4. **Angka adalah teman** — RF: PF 1.71, DD 6.38%, ROI 201.9% | XGBoost: winrate 60.14%, PF 2.02, DD 14.77%. Hafalkan ini

---

## SLIDE 1 — Title Slide
**Apa yang dijelaskan:** Judul, identitas presenter, konteks UAS.

**Talking points:**
- "Selamat pagi/siang. Presentasi ini akan menyampaikan progress research saya tentang perbandingan algoritma XGBoost dan Random Forest untuk strategi trading XAUUSD dan BTCUSD."
- "Saya akan membahas selama sekitar 20 menit, dan akan saya akhiri dengan beberapa pertanyaan terbuka untuk diskusi."

**Catatan delivery:**
- Durasi: 30 detik
- Jangan langsung lanjut — pastikan audiens siap

---

## SLIDE 2 — Agenda
**Apa yang dijelaskan:** Roadmap 10 bagian presentasi.

**Talking points:**
- "Saya akan mulai dari latar belakang dan tujuan, lalu masuk ke tinjauan pustaka dari tiga paper inti, kemudian metodologi dan arsitektur sistem yang saya bangun."
- "Bagian terpenting di akhir: hasil backtest aktual, komparasi dengan literatur, dan analisis jujur tentang gap antara klaim akademik dan apa yang sudah saya implementasi."

**Catatan delivery:**
- Durasi: 30 detik
- Tunjuk dengan jari ke poin 6 (Hasil Backtest) — "ini bagian yang paling banyak datanya, jadi mohon perhatian khusus di sana"

---

## SLIDE 3 — Lead: Latar Belakang
**Apa yang dijelaskan:** Pembuka bagian 1.

**Talking points:**
- "Mari kita mulai dengan kenapa proyek ini relevan."

**Catatan delivery:**
- Durasi: 5 detik (slide pembatas)

---

## SLIDE 4 — Konteks Pasar 2026 (XAU vs BTC)
**Apa yang dijelaskan:** Mengapa dua aset ini dipilih — karakteristik berbeda yang menarik dibandingkan dalam satu kerangka ML.

**Talking points:**
- "XAUUSD adalah safe-haven klasik — sensitif terhadap kebijakan moneter Fed, yield obligasi 10-tahun, dan geopolitik. Di 2026 ini kita lihat volatilitas tinggi karena krisis bond yield."
- "BTCUSD berbeda — sering disebut digital gold, tapi perilakunya jauh lebih reaktif terhadap kebijakan Fed dan dipengaruhi data on-chain."
- "Yang menarik: kedua aset memiliki dinamika non-linear yang **gagal ditangkap oleh ARIMA/GARCH tradisional** — ini justifikasi pemakaian ensemble tree."

**Catatan delivery:**
- Durasi: 60 detik
- Antisipasi pertanyaan: "Kenapa tidak pakai LSTM?" → Jawab: "Crasta 2024 sudah membuktikan ensemble tree mengungguli LSTM untuk XAU/USD per jam, dengan R² 0.978 vs 0.93. Slide selanjutnya akan menunjukkan datanya."

---

## SLIDE 5 — Mengapa Random Forest vs XGBoost
**Apa yang dijelaskan:** Justifikasi pemilihan kedua algoritma — beda filosofi (bagging vs boosting) dengan kekuatan-kelemahan masing-masing.

**Talking points:**
- "Random Forest pakai *bagging* — banyak pohon paralel independen, robust terhadap noise. XGBoost pakai *boosting* — pohon sekuensial yang saling memperbaiki, presisi tinggi tapi rentan overfit."
- "Bukti literatur sudah ada: Crasta menemukan RF MAE 7.20 di XAU/USD 1H, Corbet menemukan XGBoost Sharpe 1.78 di BTC. **Tidak ada universal winner — sangat kontekstual.**"

**Catatan delivery:**
- Durasi: 60 detik
- Frase penting yang harus diucapkan: **"no universal winner"** — ini tesis utama yang akan diulang di kesimpulan

---

## SLIDE 6 — Lead: Rumusan & Tujuan
**Apa yang dijelaskan:** Pembuka bagian 2.

**Catatan delivery:** 5 detik

---

## SLIDE 7 — Rumusan Masalah
**Apa yang dijelaskan:** Tiga pertanyaan riset utama.

**Talking points:**
- "Tiga pertanyaan inti: bagaimana performa komparatif kedua model di M1, sejauh mana arsitektur modular bisa mitigasi concept drift, dan apakah hasilnya menunjukkan edge nyata atau cuma artefak."
- "RQ ketiga ini yang paling penting — saya akan jujur di slide hasil bahwa selection bias masih ada."

**Catatan delivery:**
- Durasi: 45 detik
- Kalau ada waktu sempit, bisa skip baca poin satu-satu — cukup sebutkan inti masing-masing

---

## SLIDE 8 — Tujuan
**Apa yang dijelaskan:** Empat tujuan operasional.

**Talking points:**
- "Tujuan saya: evaluasi performa, bangun pipeline modular, validasi terhadap literatur, dan susun roadmap perbaikan."
- "Perlu diperhatikan tujuan ke-3: **validasi terhadap literatur**. Saya tidak hanya lapor hasil — saya bandingkan ke Crasta dan Corbet."

**Catatan delivery:** 45 detik

---

## SLIDE 9 — Lead: Tinjauan Pustaka
**Apa yang dijelaskan:** Pembuka bagian 3 — yang paling padat literatur.

**Talking points:**
- "Tiga paper jadi inti landasan saya. Saya akan bahas satu per satu, mulai dari yang paling dekat dengan metodologi saya."

**Catatan delivery:** 10 detik

---

## SLIDE 10 — Paper Inti #1: Crasta (2024) ⭐ TERPENTING
**Apa yang dijelaskan:** Paper utama dari `Journal_merged.md` — landasan metodologis langsung proyek ini.

**Talking points:**
- "Crasta dari National College of Ireland membandingkan 7 model untuk XAU/USD per jam, dengan integrasi sentimen FinBERT + S&P 500 + Crude Oil + VIX."
- "Hasilnya: **Random Forest menang** dengan R² 0.9782, MAE 7.20. XGBoost di tempat kedua, MAE 8.52. Selisih tipis tapi konsisten."
- "Paper ini adalah blueprint metodologi saya. Yang berbeda: saya pakai M1 bukan 1H, dan saya klasifikasi sinyal bukan regresi harga."

**Catatan delivery:**
- Durasi: 90 detik
- **Slide paling penting di section ini** — luangkan waktu
- Tunjuk tabel dengan jari atau pointer
- Antisipasi pertanyaan: "Kenapa Anda tidak persis replikasi Crasta?" → Jawab: "Karena tujuan akhir saya adalah eksekusi live trading di M1, jadi pendekatan klasifikasi sinyal lebih relevan daripada regresi harga. Saya gunakan Crasta sebagai benchmark target untuk fase berikutnya."

---

## SLIDE 11 — Paper Inti #2: Dave (2024)
**Apa yang dijelaskan:** Multi-timeframe analysis — bukti bahwa pemenang tergantung TF.

**Talking points:**
- "Dave menunjukkan dengan jelas: XGBoost menang di 4 jam, Random Forest menang di 5 menit."
- "Implikasi langsung untuk saya yang pakai M1: hasil yang akan saya tunjukkan nanti **kemungkinan akan ada di zona Random Forest** — dan memang itulah yang terjadi."

**Catatan delivery:**
- Durasi: 60 detik
- Frase kunci: **"superioritas model bergantung pada timeframe"**

---

## SLIDE 12 — Paper Inti #3: Corbet et al. (2020c)
**Apa yang dijelaskan:** Benchmark untuk BTCUSD — Sharpe 1.78, return 141%.

**Talking points:**
- "Untuk BTC, Corbet adalah benchmark yang harus saya kalahkan atau setidaknya dekati."
- "XGBoost menang karena kemampuan integrasi data multi-sumber: teknikal, fundamental, sentimen, on-chain."
- "**Penting**: pengujian BTC belum saya lakukan — paper ini jadi target di roadmap saya."

**Catatan delivery:**
- Durasi: 60 detik
- Akui jujur bahwa BTC belum diimplementasi — jangan tunggu ditanya

---

## SLIDE 13 — Studi Tambahan & Peringatan Penting
**Apa yang dijelaskan:** 3 studi pendukung + peringatan kontra-intuitif Zewin.

**Talking points:**
- "Kilimci menemukan stacking ensemble mengungguli model tunggal — ini arah hybrid yang ada di roadmap saya."
- "Behjoee mengonfirmasi: RF dan LightGBM unggul di timeframe rendah XAU."
- "Yang paling penting dari Zewin: **menambah data alternatif justru bisa menurunkan akurasi 2%**. Ini peringatan untuk saya saat nanti mengintegrasi data eksogen — harus A/B test ketat."

**Catatan delivery:**
- Durasi: 75 detik
- **Frase kunci**: "Tidak semua data otomatis bermanfaat — validasi empiris wajib"
- Slide ini adalah tameng untuk pertanyaan "kenapa Anda belum integrasi S&P/VIX?" → Jawab: "Karena saya akan A/B test sesuai peringatan Zewin, bukan langsung asumsi bermanfaat."

---

## SLIDE 14 — Sintesis Pustaka — Pola Konsisten
**Apa yang dijelaskan:** Tabel ringkasan: model mana unggul di konteks mana.

**Talking points:**
- "Pola yang konsisten di seluruh literatur: tidak ada universal winner. Pemilihan model harus berdasarkan konteks: timeframe, jenis data, profil risiko, dan rezim pasar."
- "Tabel ini akan saya pakai sebagai prisma untuk membaca hasil saya nanti."

**Catatan delivery:**
- Durasi: 45 detik
- Slide ini transisi penting — setelah sini, audiens harus paham bahwa hasil saya nanti tidak dievaluasi dengan "siapa menang", tapi "konsisten dengan literatur atau tidak"

---

## SLIDE 15 — Lead: Research Gap
**Catatan delivery:** 5 detik

---

## SLIDE 16 — Gap yang Diidentifikasi
**Apa yang dijelaskan:** 4 celah literatur yang menjadi peluang penelitian.

**Talking points:**
- "Empat gap utama: mayoritas studi pakai 1H ke atas, jarang yang eksplor M1 XAUUSD; belum ada studi yang eksplisit compare arsitektur modular scanning vs fix; dual-confidence jarang dioptimasi terpisah; dan kombinasi BTC+XAU dengan pipeline identik masih jarang."

**Catatan delivery:** 60 detik

---

## SLIDE 17 — Posisi Penelitian Ini
**Apa yang dijelaskan:** Apa yang dilakukan vs apa yang belum.

**Talking points:**
- "Kontribusi saya: pipeline modular XAUUSD M1 dengan dual-confidence optimization, validasi metrik finansial nyata via backtesting.py."
- "Saya akui di slide ini bahwa BTCUSD belum diuji, WFV belum penuh, dan data eksogen belum diintegrasikan. Ketiganya di roadmap."

**Catatan delivery:**
- Durasi: 60 detik
- **Praktik kejujuran ini melindungi Anda dari pertanyaan menyerang** — dosen yang lihat Anda sendiri yang akui akan lebih lunak

---

## SLIDE 18 — Lead: Metodologi
**Catatan delivery:** 5 detik

---

## SLIDE 19 — Arsitektur Sistem Modular
**Apa yang dijelaskan:** Diagram alur Scanning ↔ Fix Model.

**Talking points:**
- "Sistem dipisah dua fase. Fase 1: scanning — dijalankan periodik untuk grid search confidence optimal. Fase 2: fix model — eksekusi sinyal dengan parameter yang sudah dikunci."
- "Filosofinya: pisahkan otak optimasi dari otak eksekusi. Saat tuning, tidak ada risiko dana. Saat eksekusi, parameter sudah teruji."
- "Empat file Python: `rf_scanning.py`, `rf_fixmodel.py`, `xgboost_scanning.py`, `xgboost_fix.py`."

**Catatan delivery:**
- Durasi: 90 detik
- **Tunjuk diagram dengan jari** mengikuti alur — "input data MT5 → scanning → confidence terbaik → fix model → output sinyal"
- Antisipasi: "Bagaimana memutuskan kapan re-scan?" → Jawab: "Saat ini manual/periodik. Trigger otomatis berbasis deteksi drift ada di roadmap."

---

## SLIDE 20 — Pipeline Data & Feature Engineering
**Apa yang dijelaskan:** Apa data inputnya, apa fitur yang dibuat.

**Talking points:**
- "Data: XAUUSDc dari MetaTrader 5, timeframe M1, periode Februari sampai Mei 2026 — sekitar 3 bulan."
- "Indikator dasar: EMA 20/50/100, RSI(14), Stochastic %K(14), ATR(14) untuk SL/TP dinamis."
- "Sekitar 45 fitur total — momentum, volatility ratio, EMA slope dan gap, wick ratio, return persentase, volume change, binary flags."

**Catatan delivery:**
- Durasi: 75 detik
- Antisipasi: "3 bulan terlalu pendek?" → Akui: "Ya, ini keterbatasan. Di roadmap saya perpanjang ke minimal 1 tahun untuk validasi multi-rezim."

---

## SLIDE 21 — Target Labeling SL/TP Forward Looking
**Apa yang dijelaskan:** Cara menghasilkan label klasifikasi berdasarkan kejadian SL/TP aktual di masa depan.

**Talking points:**
- "Berbeda dengan Crasta yang regresi harga, saya pakai klasifikasi biner berdasarkan kejadian SL/TP aktual."
- "Untuk tiap candle, simulasikan 10 candle ke depan. Kalau TP buy tersentuh dulu, label = 1. Kalau TP sell tersentuh dulu, label = 0. Kalau tidak ada hasil bersih, drop."
- "Risk:reward 1:1.46 dipilih agar memungkinkan profit dengan winrate di atas 40%."

**Catatan delivery:**
- Durasi: 75 detik
- Antisipasi: "Kenapa 1:1.46, kenapa bukan 1:2?" → Jawab: "Karena untuk M1 XAUUSD, RR 1:2 winrate-nya turun drastis. 1.46 adalah hasil eksperimen empiris."

---

## SLIDE 22 — Model Configuration
**Apa yang dijelaskan:** Hyperparameter konkret untuk RF dan XGBoost.

**Talking points:**
- "RF: 500 pohon, kedalaman maksimum 10, regularisasi ketat dengan min_samples_split 20. Lebih ketat dari Crasta karena data M1 lebih noisy."
- "XGBoost: ditambah filter ADX min 20 untuk hindari trading di pasar sideways."
- "Dual-confidence grid: buy dan sell threshold di-scan independen dari 0.50 sampai 0.85. Skor komposit menyeimbangkan PF, winrate, ROI, dan drawdown."

**Catatan delivery:**
- Durasi: 90 detik
- Antisipasi: "Kenapa skor komposit, bukan PF saja?" → Jawab: "Karena PF tinggi dengan DD tinggi tetap berisiko. Saya hukum DD dengan koefisien 1.5, dan reward PF dengan 40 — pendekatan pareto untuk balance return-risk."

---

## SLIDE 23 — Lead: Hasil Backtest Aktual
**Talking points:**
- "Sekarang bagian inti — apa yang sebenarnya terjadi saat kode dijalankan."

**Catatan delivery:** 10 detik

---

## SLIDE 24 — Random Forest XAUUSD M1 ⭐ DATA INTI
**Apa yang dijelaskan:** Hasil konkret dari `optimization_confidence_balanced.csv` baris pertama.

**Talking points:**
- "Konfigurasi terbaik RF: buy confidence 0.72, sell confidence 0.50."
- "Hasil: 315 trade, winrate 55.24%, **Profit Factor 1.71**, **Max Drawdown hanya 6.38%**, ROI 201.9%. Modal $10 ribu jadi $30 ribu di periode test."
- "Profil RF: **stabil, drawdown sangat rendah, banyak trade**. Sesuai prediksi literatur."

**Catatan delivery:**
- Durasi: 75 detik
- **HAFALKAN ANGKA INI** — PF 1.71, DD 6.38%, winrate 55.24%
- Antisipasi: "189 SELL vs 126 BUY — bias?" → Jawab: "Ya, ini saya bahas di slide risiko nanti. Periode test dominan bearish XAUUSD."

---

## SLIDE 25 — XGBoost XAUUSD M1 ⭐ DATA INTI
**Apa yang dijelaskan:** Hasil dari `optimization_confidence_xgboost.csv` baris pertama.

**Talking points:**
- "Konfigurasi terbaik XGBoost: buy confidence 0.85, sell confidence 0.55. Threshold buy sangat tinggi — model selektif."
- "Hasil: 143 trade, **winrate 60.14%**, **Profit Factor 2.02**, DD 14.77%, ROI 96.1%."
- "Profil XGBoost: **selektif, kualitas signal lebih tinggi, tapi DD lebih besar**. Trade lebih sedikit jadi compound returns juga lebih lambat."

**Catatan delivery:**
- Durasi: 75 detik
- **HAFALKAN ANGKA INI** — winrate 60.14%, PF 2.02, DD 14.77%

---

## SLIDE 26 — Perbandingan Langsung RF vs XGBoost ⭐ SLIDE KUNCI
**Apa yang dijelaskan:** Tabel head-to-head dengan emoji pemenang per metrik.

**Talking points:**
- "Trade-off sangat jelas: XGBoost menang di winrate dan PF (kualitas signal), RF menang di drawdown dan ROI total (volume + stabilitas)."
- "**Ini bukan kelemahan — ini karakter arsitektur**. RF adalah tortoise yang stabil, XGBoost adalah hare yang akurat tapi DD lebih besar."

**Catatan delivery:**
- Durasi: 90 detik
- **Slide paling kuat secara naratif** — gunakan analogi tortoise vs hare untuk memorability
- Pause sebentar setelah tabel agar audiens bisa lihat 6 baris

---

## SLIDE 27 — Feature Importance — Insight Berbeda
**Apa yang dijelaskan:** Top fitur RF vs XGBoost — perbedaan strategi.

**Talking points:**
- "RF: top fitur murni teknikal — EMA, ATR, volatility. Pola klasik trend-following."
- "XGBoost: top fitur termasuk **sesi waktu** — New York session, London session, hour. Model boosting menemukan edge tambahan dari struktur temporal."
- "Ini menunjukkan XGBoost lebih ekspresif dalam menangkap pola kompleks, sesuai temuan Crasta."

**Catatan delivery:**
- Durasi: 60 detik
- Frase kunci: **"XGBoost menemukan edge tambahan dari struktur sesi"**

---

## SLIDE 28 — Lead: Komparasi Literatur
**Catatan delivery:** 5 detik

---

## SLIDE 29 — Validasi Terhadap Benchmark Literatur
**Apa yang dijelaskan:** Tabel cross-check hasil saya dengan klaim 5 paper.

**Talking points:**
- "Hasil saya konsisten dengan Behjoee — RF unggul di TF rendah XAU."
- "Konsisten dengan Dave — pattern timeframe-dependent terjadi."
- "Crasta beda hasil karena TF dan task berbeda (1H regresi vs M1 klasifikasi) — bukan kontradiksi, tapi kontekstualisasi."
- "Corbet BTC belum bisa saya validasi karena belum implementasi — pending."

**Catatan delivery:**
- Durasi: 90 detik
- **Slide ini adalah bukti kredibilitas akademik** — Anda tidak hanya report hasil, Anda **uji terhadap literatur**

---

## SLIDE 30 — Pola "No Universal Winner" Terbukti
**Apa yang dijelaskan:** Kesimpulan section komparasi — tesis utama terverifikasi.

**Talking points:**
- "Hasil empiris saya mengonfirmasi tesis utama literatur: tidak ada universal winner."
- "RF stabil, XGBoost akurat. Trade-off bukan bug — itu fitur."
- "Implikasi praktis: trader konservatif → RF, trader agresif → XGBoost dengan regularisasi ketat. Stacking hybrid → arah next."

**Catatan delivery:**
- Durasi: 60 detik
- Frase kunci yang harus diucapkan: **"trade-off bukan kekurangan, itu karakter arsitektur"**

---

## SLIDE 31 — Lead: Diskusi & Risiko
**Catatan delivery:** 5 detik

---

## SLIDE 32 — Kekuatan Pendekatan
**Apa yang dijelaskan:** 5 kekuatan yang dibanggakan.

**Talking points:**
- "Pipeline modular scalable — tinggal ganti symbol untuk BTC."
- "Dual-confidence memberikan fleksibilitas tuning per arah trade."
- "Risk management terintegrasi via ATR sizing — bukan fixed lot."
- "Reproducible dengan random_state=42."
- "Dirancang untuk live, bukan paper trading."

**Catatan delivery:** 60 detik — cepat saja, tidak perlu lambat

---

## SLIDE 33 — Risiko & Keterbatasan Jujur ⭐ SLIDE KEJUJURAN
**Apa yang dijelaskan:** 5 keterbatasan diakui terbuka.

**Talking points:**
- "Selection bias: confidence dipilih dari test set yang sama yang diukur performanya. ROI 201% mungkin optimistic."
- "Periode data sempit: 3 bulan satu rezim bearish — generalisasi ke bullish belum terbukti."
- "Bias signal SELL: 189 vs 126 BUY, model terpapar pada rezim turun."
- "Belum ada Walk-Forward Validation penuh — masih simple 80/20."
- "Slippage tidak dimodelkan — `komisi=0.0` di kode, tidak realistis."

**Catatan delivery:**
- Durasi: 120 detik (slide paling panjang waktu — jangan terburu)
- **Slide paling penting untuk kredibilitas** — kejujuran di sini akan banyak membantu di sidang
- Anggap ini "vaccination" terhadap pertanyaan negatif
- Antisipasi: "Lalu kenapa Anda klaim PF 1.71?" → Jawab: "Itu hasil pada test set spesifik di periode ini. Bukan klaim universal — itulah kenapa saya tampilkan keterbatasan terbuka."

---

## SLIDE 34 — Risiko Sistemik yang Diperhatikan
**Apa yang dijelaskan:** Tabel risiko vs mitigasi yang sudah ada vs yang belum.

**Talking points:**
- "Tiga risiko sudah ada mitigasi sebagian: overfitting (regularisasi), concept drift (arsitektur scanning), margin call (position sizing ATR)."
- "Dua risiko belum dimitigasi: black swan event makro, broker latency."

**Catatan delivery:** 60 detik

---

## SLIDE 35 — Lead: Status Progress
**Catatan delivery:** 5 detik

---

## SLIDE 36 — Apa yang Sudah Selesai
**Apa yang dijelaskan:** Checklist 9 item ✅.

**Talking points:**
- "Setup environment, data pipeline, feature engineering 45 indikator, target labeling, dua model lengkap dengan scanning + fix, dual confidence optimization, manual backtest, integrasi backtesting.py, dan output CSV/HTML."
- "Singkatnya: end-to-end pipeline untuk XAUUSD M1 sudah berjalan."

**Catatan delivery:**
- Durasi: 60 detik
- Baca cepat — ini slide kredensial

---

## SLIDE 37 — Sedang Dikerjakan + Akan Datang
**Apa yang dijelaskan:** Roadmap berprioritas HIGH/MEDIUM/LOW dengan estimasi.

**Talking points:**
- "Sedang dikerjakan: validasi literatur (yang Anda lihat di slide 29), analisis bias, dokumentasi."
- "8 item future work dengan prioritas jelas. Top 3 HIGH: WFV penuh, replikasi BTC, hitung Sharpe Ratio."

**Catatan delivery:**
- Durasi: 90 detik
- Antisipasi: "Realistis selesai sebelum sidang akhir?" → Jawab: "3 item HIGH dalam 2–3 minggu sangat realistis. Item LOW seperti stacking hybrid mungkin perlu semester lanjutan."

---

## SLIDE 38 — Roadmap Visualisasi
**Apa yang dijelaskan:** Diagram sprint planning visual.

**Talking points:**
- "Sprint 1 sekarang: validasi & dokumentasi. Sprint 2: WFV + BTC + Sharpe + baseline. Sprint 3: hybrid stacking + asym loss + live demo."
- "Target akhir UAS: laporan komprehensif + demo eksekusi di akun demo."

**Catatan delivery:** 45 detik

---

## SLIDE 39 — Lead: Kesimpulan
**Catatan delivery:** 5 detik

---

## SLIDE 40 — Kesimpulan Utama
**Apa yang dijelaskan:** 5 takeaway utama presentasi.

**Talking points:**
- "Satu: pipeline modular berfungsi, hasil tradable di paper trading."
- "Dua: hasil konsisten dengan literatur — RF stabil DD 6.4%, XGBoost akurat winrate 60%."
- "Tiga: trade-off antar model adalah karakter, bukan kekurangan."
- "Empat: gap dengan klaim akademik diakui terbuka — bukan deploy-ready."
- "Lima: roadmap konkret 6–8 item teridentifikasi."

**Catatan delivery:**
- Durasi: 90 detik
- **Slide closing utama** — bicara lambat, jelas, percaya diri
- Akhiri dengan: "Itu kesimpulan saya. Sekarang saya undang pertanyaan terbuka."

---

## SLIDE 41 — Pertanyaan untuk Diskusi
**Apa yang dijelaskan:** 4 pertanyaan terbuka untuk memandu diskusi.

**Talking points:**
- "Saya sengaja menutup dengan pertanyaan terbuka, bukan klaim final. Ini bukti research masih living."
- "Empat pertanyaan yang saya ingin diskusi: arsitektur scanning vs fix cukup untuk concept drift? Risk:reward 1:1.46 optimal atau perlu adaptive? Prioritas next hybrid stacking atau selesaikan BTC dulu? Cara terbaik hindari selection bias?"

**Catatan delivery:**
- Durasi: 60 detik
- Tunggu reaksi audiens setelah pertanyaan #4 — mereka mungkin langsung respond

---

## SLIDE 42 — Referensi Utama
**Apa yang dijelaskan:** 7 paper inti.

**Talking points:**
- "Referensi terdokumentasi lengkap di laporan tertulis. Yang paling penting: Crasta 2024 sebagai blueprint, Corbet 2020 sebagai benchmark BTC, dan Zewin 2026 sebagai peringatan untuk integrasi data alternatif."

**Catatan delivery:** 30 detik (slide referensi, lewat saja)

---

## SLIDE 43 — Terima Kasih
**Talking points:**
- "Terima kasih. Saya siap menerima pertanyaan dan kritik."

**Catatan delivery:** 10 detik — jangan tutup terlalu cepat, beri momen audiens menyiapkan pertanyaan

---

## 🎯 Antisipasi Pertanyaan Sulit

### Q1: "Kenapa 3 bulan saja datanya? Ini terlalu sedikit untuk klaim ML."
**Jawab:** "Setuju 100% — itulah kenapa di slide risiko saya akui sebagai keterbatasan. Tiga bulan adalah scope iterasi pertama untuk validasi pipeline. Roadmap fase berikutnya memperpanjang ke minimal 1 tahun, dan idealnya 2015–2026 seperti yang dilakukan Cohen & Aiche 2023."

### Q2: "ROI 201% dari $10k jadi $30k — kelihatan terlalu bagus?"
**Jawab:** "Benar, dan itulah kenapa saya bilang ini mungkin overoptimistic karena selection bias confidence. Yang lebih bisa dipercaya adalah Profit Factor 1.71 — metrik per-trade yang lebih robust. Untuk klaim ROI yang valid, saya perlu pisahkan validation set khusus untuk pemilihan threshold, ada di roadmap HIGH."

### Q3: "Walk-Forward Validation kan standar emas time series. Kenapa belum?"
**Jawab:** "Saya akui ini gap besar. Saat ini saya pakai 80/20 split — sama seperti Crasta 2024 sebetulnya. WFV penuh dengan TimeSeriesSplit n=5 ada di roadmap HIGH dengan estimasi 1 minggu. Saya bisa tunjukkan implementasinya di sidang akhir."

### Q4: "Kenapa XGBoost pakai filter ADX, RF tidak?"
**Jawab:** "Eksperimen awal — ADX filter di RF justru memotong terlalu banyak trade dan ROI drop. XGBoost sudah selektif by design via confidence tinggi, jadi filter ADX adalah safety net tambahan tanpa mengurangi banyak trade. Bisa dilihat di line 48 xgboost_scanning.py."

### Q5: "Bagaimana cara handle kalau strategi mulai loss saat live?"
**Jawab:** "Arsitektur scanning ↔ fix model didesain untuk itu. Kalau performa drop, scanning di-trigger ulang untuk cari confidence baru. Saat ini trigger manual periodik, tapi roadmap MEDIUM ada auto-trigger berbasis deteksi drift via rolling Sharpe."

### Q6: "Kenapa tidak pakai LSTM saja?"
**Jawab:** "Crasta 2024 sudah membuktikan secara empiris: LSTM R² 0.93, RF 0.978 untuk XAU/USD per jam dengan dataset yang sama. Ensemble tree menang. Selain itu LSTM butuh compute jauh lebih mahal — tidak cocok untuk re-training periodik di arsitektur scanning saya."

### Q7: "Bisa demo live execution?"
**Jawab:** "Saat ini paper trading via backtesting.py sudah berjalan. Live demo account akan saya tunjukkan di sidang akhir — itu Sprint 3 di roadmap saya. Untuk hari ini, saya bisa tunjukkan output HTML equity curve dan trade log CSV."

---

## 🎬 Skenario Timing (20 menit total)

| Section | Slide | Durasi |
| :--- | :--- | :--- |
| Opening | 1–2 | 1 min |
| Latar Belakang | 3–5 | 2 min |
| Rumusan & Tujuan | 6–8 | 1.5 min |
| Tinjauan Pustaka | 9–14 | 5 min ⭐ |
| Research Gap | 15–17 | 2 min |
| Metodologi | 18–22 | 3.5 min |
| Hasil Backtest | 23–27 | 4 min ⭐ |
| Komparasi Literatur | 28–30 | 2 min |
| Diskusi & Risiko | 31–34 | 3.5 min ⭐ |
| Status Progress | 35–38 | 2 min |
| Kesimpulan | 39–43 | 2 min |
| **Total** | | **~28 min** |

> Kalau dosen membatasi 15 menit: skip slide divider, percepat tinjauan pustaka jadi 3 menit (skip slide 13), hilangkan slide 38 roadmap visualisasi.

---

## ✅ Checklist Pre-Presentation

- [ ] Hafalkan 4 angka utama: RF (PF 1.71, DD 6.38%, ROI 201.9%) & XGBoost (winrate 60.14%, PF 2.02)
- [ ] Buka folder source code di laptop — siap demo file jika ditanya
- [ ] Buka CSV `optimization_confidence_balanced.csv` di sheet siap show
- [ ] Buka `TradingStrategyRF.html` dan `TradingStrategyXGBoost.html` di browser tab
- [ ] Test proyektor 5 menit sebelumnya
- [ ] Bawa cadangan deck di USB & cloud drive
- [ ] Air minum di podium
- [ ] Senyum di slide 1 — set positive tone

---

**Selamat presentasi! Anda sudah punya substansi yang kuat dan strategi delivery yang jujur — formula menang untuk sidang UAS.**
