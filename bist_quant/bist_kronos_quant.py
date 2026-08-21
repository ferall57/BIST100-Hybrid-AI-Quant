import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Kronos kütüphane yolunu ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KRONOS_DIR = os.path.join(ROOT_DIR, "repos", "Kronos")
if KRONOS_DIR not in sys.path:
    sys.path.insert(0, KRONOS_DIR)

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    KRONOS_AVAILABLE = True
except Exception as e:
    KRONOS_AVAILABLE = False
    print(f"[UYARI] Kronos kütüphanesi yüklenirken hata oluştu: {e}")

RAW_DIR = os.path.join(ROOT_DIR, "bist_data", "raw")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "bist_kronos")

class BistKronosQuant:
    """
    Kronos-base AI modelini kullanarak BIST (Borsa İstanbul) hisseleri için
    geleceğe dönük fiyat, hacim tahmini ve Kantitatif Rapor üreten motor.
    """
    def __init__(self, use_base_model=True, device=None):
        os.makedirs(CHARTS_DIR, exist_ok=True)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🔮 BistKronosQuant başlatılıyor... (Cihaz: {self.device})")
        
        if not KRONOS_AVAILABLE:
            raise RuntimeError("Kronos modeli bulunamadı. Lütfen kurulum adımlarını kontrol edin.")
            
        # Tokenizer'ı yükle
        tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        finetuned_tok = os.path.join(MODELS_DIR, "tokenizer", "best_model")
        if os.path.exists(finetuned_tok):
            print(f"🌟 BIST özel eğitilmiş Tokenizer bulunuyor -> {finetuned_tok}")
            tokenizer_path = finetuned_tok
            
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        
        # Model (Predictor) Yükle (Kullanıcı talebiyle en üst model Kronos-base seçilir)
        model_path = "NeoQuasar/Kronos-base" if use_base_model else "NeoQuasar/Kronos-small"
        finetuned_model = os.path.join(MODELS_DIR, "basemodel", "best_model")
        if os.path.exists(finetuned_model):
            print(f"🏆 BIST 100 üzerinde ustalasarak fine-tune edilmiş Kronos modeli bulunuyor -> {finetuned_model}")
            model_path = finetuned_model
            
        print(f"🧠 Kronos Model Yüklendi: {model_path}")
        self.model = Kronos.from_pretrained(model_path)
        
        # Predictor nesnesini bağla
        self.predictor = KronosPredictor(self.model, self.tokenizer, max_context=512, device=self.device)

    def generate_quant_report(self, ticker: str, pred_len: int = 15, lookback: int = 256, sample_count: int = 3):
        """
        Belirtilen BIST hissesinin veri setinden gelecek tahmini yapar, plot çizer ve
        TradingAgents komitesine girecek Sözel & Sayısal Özet Rapor hazırlar.
        """
        if not ticker.endswith(".IS"):
            ticker += ".IS"
            
        csv_path = os.path.join(RAW_DIR, f"{ticker}_1d.csv")
        # Her analizde veriyi her zaman CANLI olarak guncelle (Yahoo Finance)
        from bist_quant.bist_downloader import download_ticker_data
        download_ticker_data(ticker, period="5y", interval="1d", save_dir=RAW_DIR)
            
        if not os.path.exists(csv_path):
            return f"❌ [HATA] {ticker} için geçmiş veri seti oluşturulamadı.", None
            
        df = pd.read_csv(csv_path)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.sort_values("timestamps").reset_index(drop=True)
        
        if len(df) < 50:
            return f"❌ {ticker} verisi tahmine uygun olacak kadar uzun değil ({len(df)} mum).", None
            
        # Bakış penceresi sınırla
        lookback = min(lookback, len(df))
        hist_df = df.iloc[-lookback:].copy().reset_index(drop=True)
        
        x_df = hist_df[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = hist_df["timestamps"]
        
        # Gelecekteki tarihleri hesapla (Hafta sonları ve Resmi Tatiller hariç BIST işlem günleri)
        import holidays
        last_date = x_timestamp.iloc[-1]
        y_timestamps = []
        cur_date = last_date
        
        tr_holidays = holidays.Turkey(years=[cur_date.year, cur_date.year + 1])
        
        while len(y_timestamps) < pred_len:
            cur_date += timedelta(days=1)
            # Cumartesi(5) ve Pazar(6) günlerini atla, resmi tatilleri atla
            if cur_date.weekday() < 5 and cur_date not in tr_holidays:
                y_timestamps.append(cur_date)
        y_timestamp_series = pd.Series(y_timestamps)
        
        print(f"⚡ {ticker} için {lookback} günlük geçmiş kullanılarak {pred_len} günlük Kronos tahmini hesaplanıyor...")
        
        # Tahmin üret
        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp_series,
            pred_len=pred_len,
            T=0.8,
            top_p=0.9,
            sample_count=sample_count
        )
        
        # Grafiği Çiz
        chart_path = self._plot_forecast(ticker, hist_df.tail(60), pred_df, y_timestamp_series)
        
        # Sayısal İstatistikleri Hesapla
        current_close = float(hist_df["close"].iloc[-1])
        
        # 1. 1-Haftalık (5 İşlem Günü / Kısa Vade) Projeksiyon İstatistikleri
        step_1w = min(4, len(pred_df) - 1)
        pred_1w_target = float(pred_df["close"].iloc[step_1w])
        pred_1w_max = float(pred_df["high"].iloc[:step_1w + 1].max())
        pred_1w_min = float(pred_df["low"].iloc[:step_1w + 1].min())
        expected_1w_return = ((pred_1w_target - current_close) / current_close) * 100.0
        max_1w_upside = ((pred_1w_max - current_close) / current_close) * 100.0
        max_1w_downside = ((pred_1w_min - current_close) / current_close) * 100.0
        
        # 2. Orta Vadeli (N Günlük, örn: 15-30 Gün) Projeksiyon İstatistikleri
        pred_target = float(pred_df["close"].iloc[-1])
        pred_max = float(pred_df["high"].max())
        pred_min = float(pred_df["low"].min())
        expected_return = ((pred_target - current_close) / current_close) * 100.0
        max_upside = ((pred_max - current_close) / current_close) * 100.0
        max_downside = ((pred_min - current_close) / current_close) * 100.0
        
        trend_1w = "🔥 YÜKSELİŞ" if expected_1w_return > 1.5 else ("❄️ DÜŞÜŞ" if expected_1w_return < -1.5 else "NÖTR / YATAY")
        trend_label = "🔥 YÜKSELİŞ (BULLISH)" if expected_return > 3.0 else ("❄️ DÜŞÜŞ (BEARISH)" if expected_return < -2.0 else "NÖTR / YATAY (NEUTRAL)")
        
        # Sözel Rapor Oluştur (TradingAgents Komitesine Gitmek Üzere)
        report = f"""# 📈 KRONOS-BASE QUANT AI - BIST TAHMİN RAPORU ({ticker})

**Anlık Güncel Kapanış:** {current_close:.2f} TRY
**Yapay Zeka Sinyali (Kısa Vade 1H):** **{trend_1w}** | **(Orta Vade {pred_len}G):** **{trend_label}**

## 📊 1. Çift Vadeli Sayısal Projeksiyon Tablosu

| Vade / Ufuk | Hedef Kapanış | Beklenen Getiri (%) | Tepe / Direnç (Max) | Dip / Destek (Min) |
| :--- | :--- | :--- | :--- | :--- |
| **1 Haftalık (5 İşlem Günü - Kısa Vade)** | **{pred_1w_target:.2f} TRY** | **%{expected_1w_return:+.2f}** | {pred_1w_max:.2f} TRY (+%{max_1w_upside:.2f}) | {pred_1w_min:.2f} TRY (%{max_1w_downside:+.2f}) |
| **Orta Vadeli ({pred_len} İşlem Günü)** | **{pred_target:.2f} TRY** | **%{expected_return:+.2f}** | {pred_max:.2f} TRY (+%{max_upside:.2f}) | {pred_min:.2f} TRY (%{max_downside:+.2f}) |

## 🔬 Kronos-Base Yapay Zeka Değerlendirmesi
Kronos-Base modeli, BIST piyasasının geçmiş mum örüntülerini analiz ederek 1 haftalık kısa vadede {pred_1w_target:.2f} TRY (%{expected_1w_return:+.2f}), {pred_len} işlem günlük orta vadede ise {pred_target:.2f} TRY (%{expected_return:+.2f}) yönlü bir fiyat rotası öngörmüştür. Model, 1 haftalık süreçte {pred_1w_min:.2f} TRY seviyesini kısa vadeli destek, {pred_max:.2f} TRY seviyesini ise ana hedef direnç koridoru olarak işaret etmektedir.
"""
        return report, chart_path

    def _plot_forecast(self, ticker, hist_df, pred_df, y_timestamps):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(hist_df["timestamps"], hist_df["close"], label="Geçmiş Gerçek Kapanış (TRY)", color="#00FFCC", linewidth=2)
        ax.plot(y_timestamps, pred_df["close"], label="Kronos-base AI Tahmin Rota", color="#FF3366", linestyle="--", linewidth=2, marker="o", markersize=4)
        
        # 1. Hafta (5. Gün) İşaretleyicisi
        if len(y_timestamps) >= 5:
            ts_1w = y_timestamps.iloc[4]
            price_1w = pred_df["close"].iloc[4]
            ax.axvline(x=ts_1w, color="#FFCC00", linestyle=":", alpha=0.7, label="1. Hafta (5G)")
            ax.scatter([ts_1w], [price_1w], color="#FFCC00", s=60, zorder=5)
            ax.annotate(f"1H: {price_1w:.2f}", (ts_1w, price_1w), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color="#FFCC00", fontweight='bold')
        
        # Güven / Volatile Bant (Low - High)
        ax.fill_between(y_timestamps, pred_df["low"], pred_df["high"], color="#FF3366", alpha=0.2, label="Tahmin Oynaklık Bandı (Low-High)")
        
        ax.set_title(f"Kronos-base AI BIST Fiyat Tahmini (1 Hafta & Orta Vade) -> {ticker}", fontsize=14, fontweight="bold", color="white")
        ax.set_xlabel("İşlem Tarihi", fontsize=11)
        ax.set_ylabel("Fiyat (TRY)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper left", framealpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_file = os.path.join(CHARTS_DIR, f"{ticker.replace('.', '_')}_kronos_forecast.png")
        plt.savefig(chart_file, dpi=150)
        plt.close()
        print(f"📊 Tahmin Grafiği Kaydedildi: {chart_file}")
        return chart_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kronos-base BIST Quant Motoru")
    parser.add_argument("--ticker", type=str, default="THYAO.IS", help="BIST Sembolü")
    parser.add_argument("--days", type=int, default=15, help="Kaç günlük tahmin")
    args = parser.parse_args()
    
    quant = BistKronosQuant(use_base_model=True)
    report, img = quant.generate_quant_report(args.ticker, pred_len=args.days)
    print("\n" + report)
