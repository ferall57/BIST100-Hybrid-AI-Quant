# 📈 BIST 100 Hybrid AI Quant & Multi-Agent Trading System

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-102.3M_Params-ee4c2c.svg)
![Statsmodels](https://img.shields.io/badge/Econometrics-ADF_%26_Monte_Carlo-orange.svg)
![Backtesting](https://img.shields.io/badge/Backtest-Walk--Forward_Alpha-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-00a498.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

Bu proje, Borsa İstanbul (BIST) hisse senetleri için geliştirilmiş; **Derin Öğrenme (Deep Learning) tabanlı kantitatif fiyat tahmini**, **Klasik Ekonometrik Doğrulama (ADF & Parkinson Volatilitesi)**, **1.000 Yollu Stokastik Monte Carlo Simülasyonu**, **Walk-Forward Backtesting Motoru**, **Bilanço Rasyoları**, **Canlı KAP/Haber NLP Duyarlılık Füzyonu**, **VİOP Çift Yönlü (Long/Short) Türev Motoru** ve **Çoklu-Ajan (Multi-Agent) Komite Tartışması (TradingAgents)** kurgusunu birleştiren kurumsal düzeyde hibrit bir yatırım fonu, tarama ve doğrulama platformudur.

---

## 🚀 Proje Vizyonu
Piyasalardaki klasik indikatör botlarının veya kara kutu (black box) yapay zekaların aksine, bu sistem kararlarını tek bir modele bağlamaz. Karar alma süreci, klasik ekonometri (ADF durağanlık, mevsimsellik, volatilite rejimi), 1.000 yollu Monte Carlo stokastik simülasyonu, Walk-Forward geçmiş performans doğrulaması, canlı KAP haber duyarlılığı ve gerçek bir Wall Street araştırma masasındaki gibi farklı disiplinlerden gelen yapay zeka ajanlarının masada kıyasıya tartışmasıyla (**Boğa vs Ayı Debate**) ve Baş Portföy Yöneticisinin nihai **Açıklanabilir Yapay Zeka (XAI)** kararını vermesiyle sonuçlanır.

---

## 🧠 Sistem Mimarisi

Sistem birbirine entegre çalışan **7 ana Çekirdek (Core)** üzerinden çalışır:

```mermaid
graph TD
    A[Canlı Piyasa: Yahoo Finance] --> B(Çekirdek 1: Kronos Quant AI)
    A --> E_CON(Çekirdek 2: Klasik Ekonometri & Monte Carlo)
    A --> C[Canlı KAP & Google News RSS]
    A --> F[Bilanço Rasyoları: F/K, PD/DD, ROE]
    A --> G[Canlı Makro: XU100, USD/TRY]
    
    C --> NLP(Çekirdek 6: NLP Haber & KAP Duyarlılık Füzyonu)
    NLP --> D(Çekirdek 3: TradingAgents Komitesi)
    B -->|1H & 15-30G Mum Projeksiyonu| D
    E_CON -->|ADF Testi, Parkinson Volatilite, VaR, Monte Carlo| D
    F -->|Temel Finansal Çarpanlar| D
    G -->|Piyasa & Döviz Yönü| D
    
    subgraph Committee [Yapay Zeka Komitesi - Gemini 3'lü Rotator]
    D1[Temel Analist]
    D2[Teknik & Ekonometri Analisti]
    D3[Boğa Araştırmacısı]
    D4[Ayı Araştırmacısı]
    D1 --> D5{Baş Portföy Müdürü}
    D2 --> D5
    D3 --> D5
    D4 --> D5
    end
    
    D5 -->|AL / SAT / TUT, Dinamik Stop, XAI Ağırlıkları| E[Nihai Yatırım Raporu]
    
    H(Çekirdek 4: BIST Screener Tarama Motoru) -->|Tüm Evreni Tara: BIST 30 / 100| B
    H -->|En Yüksek Potansiyelli Top N Hisse| D
    H -->|Konsolide Bülten| I[BIST Keşif & Tarama Bülteni]

    J(Çekirdek 5: Walk-Forward Backtesting) -->|Lookahead-Free Rolling Window| K[Equity Curve & Sharpe/MDD Raporu]
    
    L(Çekirdek 7: VİOP Çift Yönlü Türev Motoru) -->|Kaldıraçlı Long & Short + Nemalandırma| M[Piyasa Nötr Türev Getirisi]
```

---

### 🔹 Çekirdek 1: Kronos-Base Quant Model (PyTorch)
* **102.3 Milyon parametreli** Transformer tabanlı finansal zaman serisi tahmin modelidir.
* Geçmiş 256 günlük mum grafiğini alarak **1 Haftalık (Kısa Vade)** ve **15-30 Günlük (Orta Vade)** çift vadeli matematiksel projeksiyonunu (Destek, Direnç, Beklenen Getiri) hesaplar ve görsel dark-mode projeksiyon grafiği çizer.
* `holidays` entegrasyonu sayesinde Türkiye'nin resmi ve dini tatil günlerini otomatik algılayıp projeksiyondan atlar.

### 🔹 Çekirdek 2: Klasik Ekonometri & 1.000 Yollu Monte Carlo Motoru
* **Augmented Dickey-Fuller (ADF) Durağanlık Testi:** Serinin birim kök ve trend karakterini *p*-değeri ile matematiksel olarak ispatlar.
* **Mevsimsellik (Seasonality):** Günlük/haftalık getiri varyansını ve anomali günlerini ölçer.
* **Parkinson High-Low Volatilitesi:** Gün içi oynaklık dalga boyunu ölçerek volatilite rejimini (Düşük / Normal / Yüksek Risk) belirler.
* **1.000 Yollu Geometrik Brown Hareketi (GBM):** 1.000 bağımsız stokastik simülasyon ile hem 1 haftalık hem orta vadeli medyan getiri, **%95 Güven Aralığı Bandı**, **Yükseliş Olasılığı (Win Rate %)** ve **Parametrik VaR (%95)** hesaplar.

### 🔹 Çekirdek 3: Multi-Agent Tartışma Komitesi & XAI (Gemini API Rotator)
* **Bilanço & Değerleme Rasyoları:** F/K, İleri F/K, PD/DD, FD/FAVÖK, ROE, Temettü Verimi ve 52 Haftalık Zirve/Dip marjları otomatik çekilip analiz edilir.
* **Canlı KAP & Haber RSS Akışı:** Google News TR ve KAP altyapısına doğrudan XML/RSS ile bağlanarak en taze 25 haberi çeker.
* **Açıklanabilir Yapay Zeka (Explainable AI - XAI):** Karara etki eden faktörlerin ağırlık dağılımı (% Bilanço İskontosu, % Risk/Ödül, % Quant/Monte Carlo Getirisi, % KAP Katalizörü) şeffaf şekilde raporlanır.
* **3'lü Gemini Akıllı Rotasyon Motoru:** Kota sınırlarını (Rate Limit / 429) ve 503 sunucu yoğunluklarını önlemek için akıllı bekleme (backoff) ile kesintisiz geçiş yapar.

### 🔹 Çekirdek 4: BIST 30 / 100 Otomatik Tarama Motoru (Screener)
* **2 Aşamalı Hibrit Tarama (2-Stage Funnel):**
  1. **1. Aşama (Hızlı Ön Eleme):** Tüm evrendeki hisseler saniyeler içinde taranır; 1H ve 15G Quant getiri potansiyeli, 52 haftalık zirveye iskonto ve hacim artışına göre puanlanarak sıralanır.
  2. **2. Aşama (Derin Komite Analizi):** En yüksek potansiyelli ilk **Top N** hisse seçilerek tam yapay zeka komite tartışmasından geçirilir.
* Tarama bitiminde konsolide bir **BIST Keşif Bülteni** (`outputs/reports/BIST_SCANNER_...md`) üretilir.

### 🔹 Çekirdek 5: Walk-Forward Backtesting & Finansal Doğrulama Motoru
* **Zaman Sızıntısız (Lookahead-Free) Rolling Window:** Model her adımda sadece o günün gerisindeki mumları görerek geçmiş periyotta işlem açar.
* **Dinamik Risk Yönetimi & İz Süren Stop:** Volatiliteye duyarlı ATR stop-loss ve trend sürme (Trailing Stop) ile kârı sonuna kadar koşturur.
* **Wall Street Performans Metrikleri:** Sharpe Oranı, Sortino Oranı, **Kazanma Oranı (Win Rate %)**, **Kâr Faktörü (Profit Factor)**, **Maksimum Çekilme (MDD %)** ve **Alpha (α)**.
* **Görsel Sermaye Eğrisi (Equity Curve):** 100.000 TL başlangıç sermayesinin Al-Tut (Buy & Hold) karşısındaki büyüme eğrisini çizer.

### 🔹 Çekirdek 6: Çok Modlu (Multi-Modal) NLP Haber & KAP Duyarlılık Füzyon Motoru
* **Finansal NLP Duyarlılık Skorlaması:** Canlı KAP bildirimleri ve Google News TR akışını analiz edip `[-1.0, +1.0]` arasında sayısal duyarlılık skoru ve `[%0, %100]` etki şiddeti üretir.
* **Pozitif / Negatif Katalizör Tespiti:** Ciro artırıcı dev ihaleler, bedelsiz sermaye, pay geri alımları veya ceza/fabrika durdurma krizlerini anında etiketler.
* **Matematiksel Hibrit Füzyon Matrisi:**
  ```text
  R_fused = (1 - w_news) * R_tech + w_news * (S_news * I_impact * σ_volatility)
  ```
* **Dinamik Eşik & Kârı Koşturma Stop Mesafesi:** Güçlü pozitif katalizörlü hisselerde teknik giriş barajını düşürür (`min_thresh` ↓) ve erken silkelenmeyi önlemek için İz Süren Stop mesafesini genişletir (`%4.0 -> %7.5`).

### 🔹 Çekirdek 7: VİOP Çift Yönlü (Long/Short) Türev & Nemalandırma Motoru
* **Çift Yönlü Kazanç (Bi-directional Alpha):** Yükseliş trendinde Kaldıraçlı Long, düşüş trendinde **Kaldıraçlı Short (Açığa Satış)** açarak ayı piyasalarından devasa kârlar üretir.
* **Ters İz Süren Stop (Inverted Trailing Stop):** Short pozisyonda fiyat düştükçe kâr seviyesini kilitler, dipten ani tepki geldiğinde kârı cebe atar.
* **Takasbank Nemalandırma Faizi:** Pozisyondayken veya nakitteyken portföye her gün gecelik Takasbank faizi (yıllık %45) tahakkuk ettirir.

---

## 🏆 Gerçekleşen Backtest Doğrulama Sonuçları (Case Studies)

Sistemin geçmiş veriler üzerinde hiçbir **zaman sızıntısı olmadan (Lookahead Bias %0)** gerçekleştirdiği bağımsız test sonuçları:

| Hisse Senedi | Test Periyodu | Hissenin Kendisi (Al ve Tut) | KRONOS Yapay Zekası | Üretilen Alpha (α) | Risk & Performans Metrikleri |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FROTO.IS** (VİOP Long/Short) | **Son 12 Ay** | **-%25.28** 🔴 | **%+101.64** 🟢 | **🚀 +%126.91 ALPHA** | **Düşüşte 2x Kâr** \| PF: 2.07x \| Sortino: 4.44 |
| **ISCTR.IS** (Fine-Tuned AI) | **Son 6 Ay** | **-%23.19** 🔴 | **%+0.83** 🟢 | **🚀 +%24.02 ALPHA** | **Kriz Koruması** \| MDD: -%6.91 \| Kârda Kapanış |
| **FROTO.IS** (Spot Defansif) | **Son 12 Ay** | **-%25.28** 🔴 | **%+0.00** 🟢 | **🚀 +%25.28 ALPHA** | **Tam Koruma** \| MDD: -%0.00 (%100 Nakit) |
| **ASELS.IS** (Trailing Stop)| **Son 12 Ay** | **%+118.88** 🟢 | **%+50.65** 🟢 | **🛡️ MDD: -%5.99** | **Win Rate: %53.8** \| Sharpe: 2.03 \| PF: 3.76x |

> 💡 **Öne Çıkan Başarılar (Kriz Kalkanı & Çift Yönlü Kazanç):**
> 1. **FROTO VİOP Short Zaferi:** Hisse 1 yıl boyunca **-%25.28 çökerken**, VİOP Çift Yönlü motorumuz düşüş trendinde **Kısa Pozisyon (Short / Açığa Satış)** açarak ve Takasbank nemalandırmasıyla **100.000 TL'lik kasayı 201.635 TL'ye (+%101.64 Net Kâr)** çıkarmış ve hisseye **+%126.91 ALPHA** farkı atmıştır!
> 2. **ISCTR Kriz Kalkanı:** ISCTR son 6 ayda bankacılık baskısıyla **%23.19 erirken**, BIST üzerinde fine-tune edilmiş modelimiz doğru dip seviyelerini yakalayarak hisse çökerken **kârda kalmayı başarmış (+%0.83)** ve hisseye **+%24.02 Alpha farkı** atmıştır.
> 3. **ASELSAN Trend Koşusu:** ASELSAN'ın parabolik yükseliş rallisinde **İz Süren Stop (Trailing Stop)** motorumuz trendi erken bırakmayıp **%+50.65 net getiri**, **3.76x Kâr Faktörü** ve **2.03 Sharpe Oranı** yakalamıştır.

---

## 💻 Kurulum ve Kullanım

### 1. Kurulum
```bash
git clone https://github.com/ferall57/BIST100-Hybrid-AI-Quant.git
cd BIST100-Hybrid-AI-Quant
pip install -r requirements.txt
```

### 2. Ortam Değişkenleri (.env)
Kök dizinde `.env` dosyası oluşturup Gemini API anahtarlarınızı girin:
```env
GOOGLE_API_KEY_1=AIzaSy...
GOOGLE_API_KEY_2=AIzaSy...
GOOGLE_API_KEY_3=AIzaSy...
```

---

## ⚡ Kullanım Komutları

### 1. Tekil Hisse Derin Analizi (Çift Vade: 1H & 15G)
```bash
python main.py --analyze ISCTR.IS --model gemini-2.5-flash
```

### 2. Canlı KAP ve Haber NLP Duyarlılık & Füzyon Karnesi
```bash
# Tek komutla hissenin anlık KAP haber duyarlılığını ve katalizörlerini puanla
python main.py --sentiment ASELS.IS
python main.py --sentiment FROTO.IS
```

### 3. VİOP Çift Yönlü (Long/Short) Sinyal Taraması & Backtest
```bash
# BIST 30 için günün Canlı Long ve Short kontrat fırsatlarını listele
python main.py --viop-signals --top 10

# 12 Aylık Çift Yönlü VİOP Backtesti (Kaldıraç: 1.5x, Nemalandırma %45)
python main.py --backtest FROTO.IS --months 12 --use-kronos-backtest --viop
```

### 4. Walk-Forward Spot Backtest & Performans Doğrulama
```bash
# 6 Aylık Spot Backtest (Stop-Loss %3.5, Take-Profit %8.0)
python main.py --backtest ISCTR.IS --months 6
```

### 5. BIST 30 / 100 Otomatik Tarama (Screener)
```bash
# BIST 30 En İyi 5 Fırsat
python main.py --scan bist30 --top 5 --model gemini-2.5-flash

# BIST 100 Geniş Evren Taraması
python main.py --scan bist100 --top 10 --model gemini-2.5-flash
```

### 6. BIST Veri Setlerini İndirme & Kronos Eğitimi
```bash
# BIST 100 verilerini indir
python main.py --download-all --download-mode bist100

# Kronos modelini BIST 100 üzerinde ince ayar (fine-tune) yap
python main.py --train-kronos
```

---

## 📁 Çıktılar
* **Komite Raporları:** `outputs/reports/<SEMBOL>_committee_report.md`
* **Backtest Raporları:** `outputs/reports/<SEMBOL>_backtest_report.md`
* **Sermaye Eğrileri (Equity Curve):** `outputs/charts/<SEMBOL>_backtest_equity_curve.png`
* **Tarama Bültenleri:** `outputs/reports/BIST_SCANNER_<EVREN>_<TARIH>.md`
* **Fiyat Projeksiyon Grafikleri:** `outputs/charts/<SEMBOL>_kronos_forecast.png`

---

## ⚠️ Yasal Uyarı (Disclaimer)
Bu proje tamamen eğitim, araştırma ve algoritmik modelleme amacıyla geliştirilmiştir. Üretilen çıktılar, fiyat tahminleri ve komite kararları **kesinlikle doğrudan yatırım tavsiyesi (YTD) niteliği taşımaz**. Gerçek piyasalarda işlem yapmadan önce kendi araştırmanızı yapınız.
