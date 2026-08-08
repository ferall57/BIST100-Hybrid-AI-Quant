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
        
        # Gelecekteki tarihleri hesapla (Hafta sonları hariç BIST işlem günleri)
        last_date = x_timestamp.iloc[-1]
        y_timestamps = []
        cur_date = last_date
        while len(y_timestamps) < pred_len:
            cur_date += timedelta(days=1)
            if cur_date.weekday() < 5: # Pazartesi(0)-Cuma(4)
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
        current_close = hist_df["close"].iloc[-1]
        pred_target = pred_df["close"].iloc[-1]
        pred_max = pred_df["high"].max()
        pred_min = pred_df["low"].min()
        expected_return = ((pred_target - current_close) / current_close) * 100.0
        max_upside = ((pred_max - current_close) / current_close) * 100.0
        max_downside = ((pred_min - current_close) / current_close) * 100.0
        
        trend_label = "🔥 YÜKSELİŞ (BULLISH)" if expected_return > 3.0 else ("❄️ DÜŞÜŞ (BEARISH)" if expected_return < -2.0 else "NÖTR / YATAY (NEUTRAL)")
        
        # Sözel Rapor Oluştur (TradingAgents Komitesine Gitmek Üzere)
        report = f"""# 📈 KRONOS-BASE QUANT AI - BIST TAHMİN RAPORU ({ticker})

**Tarih Penceresi:** Son 60 Günlük Kapanış: {current_close:.2f} TRY -> {pred_len} Günlük Projeksiyon Hedefi: {pred_target:.2f} TRY
**Yapay Zeka Sinyali:** **{trend_label}**

## 📊 Özet Matematiksel Öngörüler
* **Anlık Güncel Kapanış Fiyatı:** {current_close:.2f} TRY
* **Hedef Kapanış (N={pred_len} gün):** {pred_target:.2f} TRY (**Beklenen Getiri: %{expected_return:.2f}**)
* **Tahmin Edilen En Yüksek Nokta (Zirve):** {pred_max:.2f} TRY (Maksimum Potansiyel: +%{max_upside:.2f})
* **Tahmin Edilen En Düşük Nokta (Dip/Destek):** {pred_min:.2f} TRY (Maksimum Risk: %{max_downside:.2f})

## 🔬 Kronos-Base Yapay Zeka Değerlendirmesi
Kronos-Base temel finans modeli, BIST piyasasının geçmiş mum örüntülerini ve hacim akışlarını analiz ederek önümüzdeki {pred_len} işlem gününde %{expected_return:.2f} seviyesinde bir yönlü trend öngörmüştür. Model, fiyat kümülasyonlarında olası oynaklıklara karşı {pred_min:.2f} TRY seviyesini stratejik bir teknik destek, {pred_max:.2f} TRY seviyesini ise direnç bandı olarak işaret etmektedir.
"""
        return report, chart_path

    def _plot_forecast(self, ticker, hist_df, pred_df, y_timestamps):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(hist_df["timestamps"], hist_df["close"], label="Geçmiş Gerçek Kapanış (TRY)", color="#00FFCC", linewidth=2)
        ax.plot(y_timestamps, pred_df["close"], label="Kronos-base AI Tahmin Rota", color="#FF3366", linestyle="--", linewidth=2, marker="o", markersize=4)
        
        # Güven / Volatile Bant (Low - High)
        ax.fill_between(y_timestamps, pred_df["low"], pred_df["high"], color="#FF3366", alpha=0.2, label="Tahmin Oynaklık Bandı (Low-High)")
        
        ax.set_title(f"Kronos-base AI BIST Fiyat Tahmini -> {ticker}", fontsize=14, fontweight="bold", color="white")
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
