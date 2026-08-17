import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.bist_100_tickers import get_tickers
from bist_quant.bist_downloader import download_ticker_data, RAW_DATA_DIR
from hybrid_agents.bist_committee import BistHybridCommittee

REPORTS_DIR = os.path.join(ROOT_DIR, "outputs", "reports")

class BistScanner:
    """
    BIST 100 / BIST 30 hisse evrenini otomatik tarayan, 
    2 Aşamalı Hızlı Quant & Derin Komite Filtreleme Motoru (Screener).
    """
    def __init__(self, gemini_model: str = "gemini-2.5-pro", temperature: float = 0.2):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self.gemini_model = gemini_model
        self.temperature = temperature
        self.committee = BistHybridCommittee(gemini_model=gemini_model, temperature=temperature)
        self.quant_engine = self.committee.quant_engine

    def _quick_screen_ticker(self, ticker: str, forecast_days: int = 15):
        """Tek bir hisse için hızlı teknik, rasyo ve quant getiri ön değerlendirmesi yapar."""
        try:
            # 1. Veri güncelle
            download_ticker_data(ticker, period="2y", interval="1d", save_dir=RAW_DATA_DIR)
            csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}_1d.csv")
            if not os.path.exists(csv_path):
                return None
                
            df = pd.read_csv(csv_path)
            if len(df) < 50:
                return None
                
            current_close = df["close"].iloc[-1]
            prev_close = df["close"].iloc[-2] if len(df) >= 2 else current_close
            daily_change = ((current_close - prev_close) / prev_close) * 100.0
            
            # 20 günlük ortalama hacim
            avg_vol_20 = df["volume"].tail(20).mean()
            last_vol = df["volume"].iloc[-1]
            vol_ratio = (last_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0
            
            # 52 haftalık zirve / dip marjı
            high_52 = df["high"].tail(252).max()
            low_52 = df["low"].tail(252).min()
            discount_to_high = ((high_52 - current_close) / high_52) * 100.0
            
            expected_return = 0.0
            trend_label = "NÖTR"
            
            # Quant tahmin motoru devredeyse hızlı tahmin puanı al
            if self.quant_engine and hasattr(self.quant_engine, 'predictor') and self.quant_engine.predictor is not None:
                try:
                    lookback = min(128, len(df))
                    hist_df = df.iloc[-lookback:].copy().reset_index(drop=True)
                    x_df = hist_df[["open", "high", "low", "close", "volume", "amount"]]
                    x_ts = pd.to_datetime(hist_df["timestamps"])
                    
                    # Gelecek günleri hesapla
                    last_date = x_ts.iloc[-1]
                    y_timestamps = []
                    cur_d = last_date
                    while len(y_timestamps) < forecast_days:
                        cur_d += pd.Timedelta(days=1)
                        if cur_d.weekday() < 5:
                            y_timestamps.append(cur_d)
                            
                    pred_df = self.quant_engine.predictor.predict(
                        df=x_df,
                        x_timestamp=x_ts,
                        y_timestamp=pd.Series(y_timestamps),
                        pred_len=forecast_days,
                        T=0.8,
                        top_p=0.9,
                        sample_count=1
                    )
                    pred_close = pred_df["close"].iloc[-1]
                    expected_return = ((pred_close - current_close) / current_close) * 100.0
                except Exception:
                    # Hata olursa momentum bazlı yedek getiri skoru
                    sma_20 = df["close"].tail(20).mean()
                    expected_return = ((current_close - sma_20) / sma_20) * 50.0
            else:
                # Quant motoru yoksa teknik momentum skoru
                sma_20 = df["close"].tail(20).mean()
                sma_50 = df["close"].tail(50).mean()
                expected_return = (((current_close - sma_20) / sma_20) + ((sma_20 - sma_50) / sma_50)) * 50.0

            if expected_return > 3.0:
                trend_label = "🔥 YÜKSELİŞ"
            elif expected_return < -2.0:
                trend_label = "❄️ DÜŞÜŞ"

            # Skorlama: Quant getiri potansiyeli + hacim desteği + iskonto payı
            score = (expected_return * 1.5) + (discount_to_high * 0.3) + (min(vol_ratio, 3.0) * 2.0)

            return {
                "ticker": ticker,
                "close": current_close,
                "daily_change": daily_change,
                "expected_return": expected_return,
                "trend": trend_label,
                "vol_ratio": vol_ratio,
                "discount_to_high": discount_to_high,
                "score": score
            }
        except Exception:
            return None

    def scan_and_report(self, mode: str = "bist30", top_n: int = 5, forecast_days: int = 15):
        tickers = get_tickers(mode=mode)
        print(f"\n" + "="*80)
        print(f"🔍 BIST OTOMATIK TARAMA MOTORU (SCREENER) BASLATILDI")
        print(f"📊 Hedef Evren : {mode.upper()} ({len(tickers)} Hisse)")
        print(f"🎯 Hedef Filtre: En Yüksek Potansiyelli İlk {top_n} Hisse")
        print(f"🔮 Quant Vade  : {forecast_days} İşlem Günü")
        print("="*80)

        # 1. AŞAMA: Hızlı Ön Eleme (Funnel 1)
        print(f"\n[AŞAMA 1/2] {len(tickers)} Hisse İçin Teknik & Quant Hızlı Ön Tarama Yapılıyor...")
        candidates = []
        with tqdm(total=len(tickers), desc="Hisse Taraması") as pbar:
            for t in tickers:
                res = self._quick_screen_ticker(t, forecast_days=forecast_days)
                if res:
                    candidates.append(res)
                pbar.update(1)

        if not candidates:
            print("[HATA] Taranan hisselerden geçerli veri alınamadı.")
            return

        # Skora göre en yüksekten düşüğe sırala
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:top_n]

        print(f"\n🏆 [ÖN ELEME SONUCU] EN YÜKSEK POTANSİYELLİ İLK {top_n} HİSSE BELİRLENDİ:")
        print(f"{'Sıra':<5} {'Sembol':<10} {'Fiyat (TRY)':<14} {'Günlük %':<10} {'Quant Getiri %':<16} {'Trend':<14} {'Zirve İskonto %':<15}")
        print("-" * 85)
        for idx, c in enumerate(top_candidates, 1):
            print(f"{idx:<5} {c['ticker']:<10} {c['close']:<14.2f} {c['daily_change']:<+10.2f} {c['expected_return']:<+16.2f} {c['trend']:<14} %{c['discount_to_high']:<14.1f}")
        print("-" * 85)

        # 2. AŞAMA: Seçilen İlk N Hisse İçin Derin Komite Analizi
        print(f"\n[AŞAMA 2/2] İlk {top_n} Hisse İçin Çoklu Yapay Zeka Komitesi (Debate & Rasyolar) Toplanıyor...\n")
        
        committee_results = []
        for idx, c in enumerate(top_candidates, 1):
            t = c["ticker"]
            print(f"\n" + "-"*80)
            print(f"📌 [{idx}/{top_n}] {t} HİSSESİ İÇİN KOMİTE DEĞERLENDİRMESİ BAŞLIYOR...")
            print("-"*80)
            try:
                verdict, rep_path, chart_path = self.committee.analyze_ticker(t, forecast_days=forecast_days)
                
                # Karar raporundan önemli satırları ayıkla
                rating = "N/A"
                conf = "N/A"
                target_band = "N/A"
                alloc = "N/A"
                stop_loss = "N/A"
                
                import re
                for line in verdict.split("\n"):
                    clean_line = line.replace("*", "").strip()
                    if re.search(r"yat[ıi]r[ıi]m\s+karar[ıi]", clean_line, re.IGNORECASE) and ":" in clean_line:
                        rating = clean_line.split(":", 1)[1].strip()
                    elif re.search(r"g[uü]ven\s+katsay[ıi]s[ıi]", clean_line, re.IGNORECASE) and ":" in clean_line:
                        conf = clean_line.split(":", 1)[1].strip()
                    elif re.search(r"hedef\s+fiyat", clean_line, re.IGNORECASE) and ":" in clean_line:
                        target_band = clean_line.split(":", 1)[1].strip()
                    elif re.search(r"(portf[oö]y|tahsisat|pay[ıi])", clean_line, re.IGNORECASE) and ":" in clean_line:
                        alloc = clean_line.split(":", 1)[1].strip()
                    elif re.search(r"stop[\s\-_]*loss|zarar\s+kes", clean_line, re.IGNORECASE) and ":" in clean_line:
                        stop_loss = clean_line.split(":", 1)[1].strip()

                committee_results.append({
                    "rank": idx,
                    "ticker": t,
                    "close": c["close"],
                    "expected_return": c["expected_return"],
                    "rating": rating,
                    "confidence": conf,
                    "target_band": target_band,
                    "allocation": alloc,
                    "stop_loss": stop_loss,
                    "verdict": verdict,
                    "report_file": rep_path,
                    "chart_file": chart_path
                })
            except Exception as e:
                print(f"[UYARI] {t} komite analizi sırasında hata: {e}")

        # 3. KONSOLİDE TARAMA BÜLTENİ OLUŞTURMA
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        digest_file = os.path.join(REPORTS_DIR, f"BIST_SCANNER_{mode.upper()}_{file_ts}.md")

        table_rows = ""
        for r in committee_results:
            table_rows += f"| **#{r['rank']} {r['ticker']}** | {r['close']:.2f} TRY | **%{r['expected_return']:+.2f}** | **{r['rating']}** | {r['confidence']} | {r['target_band']} | {r['allocation']} | {r['stop_loss']} |\n"

        individual_sections = ""
        for r in committee_results:
            individual_sections += f"""
---
### 📌 #{r['rank']} - {r['ticker']} Komite Karar Özeti
* **Fiyat:** {r['close']:.2f} TRY | **Quant Tahmin Getirisi:** %{r['expected_return']:+.2f}
* **Yatırım Kararı:** **{r['rating']}** ({r['confidence']})
* **Hedef Fiyat:** {r['target_band']} | **Stop-Loss:** {r['stop_loss']}
* **Önerilen Portföy Payı:** {r['allocation']}
* 📄 [Tam Komite Tartışma Raporu](file:///{r['report_file'].replace('\\', '/')})
* 📊 [Projeksiyon Grafiği](file:///{r['chart_file'].replace('\\', '/') if r['chart_file'] else 'N/A'})

```markdown
{r['verdict']}
```
"""

        digest_md = f"""# 🏆 BIST 100 HİBRİT YAPAY ZEKA TARAMA & KEŞİF BÜLTENİ (SCREENER)
**Oluşturulma Tarihi:** {now_str} | **Tarama Evreni:** {mode.upper()} ({len(tickers)} Hisse)
**Seçilen Lider Hisseler:** İlk {len(committee_results)} Hisse | **Aktif Model:** Kronos-Base Quant + Gemini Rotational Multi-Agent Debate

---

## 📊 1. LİDER HİSSELER KARŞILAŞTIRMA TABLOSU

| Hisse | Anlık Fiyat | Quant Getiri (%) | Komite Kararı | Güven | Hedef Fiyat Bandı | Portföy Payı | Stop-Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_rows}

---

## 🔬 2. HİSSE BAZLI KOMİTE KARARLARI VE GEREKÇELERİ
{individual_sections}

---
*(Bu bülten Antigravity KRONOS Hibrit AI Komite Tarama Motoru tarafından otomatik üretilmiştir. Yatırım tavsiyesi içermez.)*
"""
        with open(digest_file, "w", encoding="utf-8") as f:
            f.write(digest_md)

        # 4. KONSOLA GÖRSEL ŞIK BİR ÖZET BAS
        print("\n" + "="*85)
        print(f"🏆 BIST {mode.upper()} TARAMA TAMAMLANDI - KONSOLİDE SONUÇ ÖZETİ")
        print("="*85)
        print(f"{'Sıra':<5} {'Hisse':<10} {'Fiyat':<10} {'Quant Getiri':<14} {'Karar':<18} {'Güven':<10} {'Hedef Fiyat':<18}")
        print("-" * 85)
        for r in committee_results:
            print(f"#{r['rank']:<4} {r['ticker']:<10} {r['close']:<10.2f} %{r['expected_return']:<12.2f} {r['rating']:<18} {r['confidence']:<10} {r['target_band']:<18}")
        print("="*85)
        print(f"📑 Konsolide Tarama Bülteni Kaydedildi -> {digest_file}")
        print("="*85 + "\n")
        return digest_file, committee_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BIST Otomatik Tarama Motoru")
    parser.add_argument("--mode", type=str, default="bist30", choices=["bist30", "bist100"], help="Tarama evreni")
    parser.add_argument("--top", type=int, default=3, help="Derin analize girecek hisse sayısı")
    parser.add_argument("--days", type=int, default=15, help="Quant tahmin gün sayısı")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Gemini modeli")
    
    args = parser.parse_args()
    scanner = BistScanner(gemini_model=args.model)
    scanner.scan_and_report(mode=args.mode, top_n=args.top, forecast_days=args.days)
