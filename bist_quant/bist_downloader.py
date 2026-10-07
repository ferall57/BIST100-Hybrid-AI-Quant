import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta

# Windows konsollarında Emoji ve UTF-8 karakter hatalarını önlemek için:
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

from bist_quant.bist_100_tickers import get_tickers

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "bist_data", "raw")

INTRADAY_SUFFIXES = ("m", "h")
PERIOD_PATTERN = re.compile(r"^(\d+)(d|wk|mo|y)$")
PERIOD_OFFSET_UNITS = {"d": "days", "wk": "weeks", "mo": "months", "y": "years"}
RESCALE_TOLERANCE = 0.005  # ortak mumlarda %0.5'ten büyük kapanış farkı = geçmiş yeniden ölçeklenmiş

def _candle_keys(timestamps: pd.Series, daily: bool) -> pd.Series:
    """Günlük mumlarda takvim gününü, gün içi mumlarda tam zaman damgasını anahtar yapar."""
    as_text = timestamps.astype(str)
    return as_text.str[:10] if daily else as_text

def merge_candles(existing: pd.DataFrame, fresh: pd.DataFrame, daily: bool = True) -> pd.DataFrame:
    """
    Diskteki geçmişi yeni indirilen mumlarla birleştirir. Aynı muma ait satırda yeni veri kazanır;
    kısa periyotlu bir indirme daha uzun geçmişi silmez.
    """
    if existing.empty:
        return fresh.assign(timestamps=fresh["timestamps"].astype(str)).reset_index(drop=True)
    combined = pd.concat(
        [existing.assign(timestamps=existing["timestamps"].astype(str)),
         fresh.assign(timestamps=fresh["timestamps"].astype(str))],
        ignore_index=True,
    )
    keys = _candle_keys(combined["timestamps"], daily)
    combined = combined.loc[~keys.duplicated(keep="last")]
    order = _candle_keys(combined["timestamps"], daily).sort_values(kind="stable").index
    return combined.loc[order].reset_index(drop=True)

def histories_are_consistent(existing: pd.DataFrame, fresh: pd.DataFrame, daily: bool = True) -> bool:
    """
    Ortak mumlarda kapanışlar uyuşuyor mu? Bölünme/bedelsiz sonrası Yahoo tüm geçmişi yeniden
    ölçekler; bu durumda eski dosya ile yeni veriyi birleştirmek sahte fiyat sıçraması üretir.
    Diskteki son mum seans içi yarım kalmış olabileceği için karşılaştırmaya girmez.
    """
    old = existing.assign(_key=_candle_keys(existing["timestamps"], daily)).iloc[:-1]
    new = fresh.assign(_key=_candle_keys(fresh["timestamps"], daily))
    overlap = old.merge(new, on="_key", suffixes=("_old", "_new"))
    if overlap.empty:
        return True
    drift = (overlap["close_new"] - overlap["close_old"]).abs() / overlap["close_old"].abs()
    return bool((drift <= RESCALE_TOLERANCE).all())

def trim_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Geçmişi son mumdan geriye doğru istenen periyoda ('6mo', '2y' vb.) kırpar; 'max' dokunmaz."""
    match = PERIOD_PATTERN.match(str(period))
    if not match or df.empty:
        return df
    days = pd.to_datetime(df["timestamps"].astype(str).str[:10])
    cutoff = days.max() - pd.DateOffset(**{PERIOD_OFFSET_UNITS[match.group(2)]: int(match.group(1))})
    return df.loc[days >= cutoff].reset_index(drop=True)

def download_ticker_data(ticker: str, period: str = "max", interval: str = "1d", save_dir: str = RAW_DATA_DIR):
    """
    Belirli bir BIST hissesi (.IS) için OHLCV veri setini Yahoo Finance üzerinden indirir,
    Kronos formasyonuna uydurmak için 'amount' (toplam işlem Türk Lirası tutarı yaklaşımı)
    hesaplar ve CSV olarak kaydeder.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{ticker}_{interval}.csv")
    
    try:
        # 1. Öncelikli olarak en güncel canlı veriyi getiren Ticker.history kullan
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period=period, interval=interval, auto_adjust=False)
        
        # Eğer boş geldiyse yf.download ile yedek dene
        if df is None or df.empty or len(df) < 1:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df is None or df.empty or len(df) < 1:
            return ticker, False, "Yetersiz veya boş veri seti"
            
        # Önce index'i sıfırla (Date / Datetime sütuna dönüşsün)
        df.reset_index(inplace=True)
        
        # Sütun isimlerini düzleştir ve küçült (MultiIndex tuples önlemi)
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                new_cols.append(str(col[0]).lower().strip())
            else:
                new_cols.append(str(col).lower().strip())
        df.columns = new_cols
        
        # Tarih sütununun adını Kronos için standardızasyon yapalım: 'timestamps'
        for dt_col in ["date", "datetime", "timestamp", "index", "price"]:
            if dt_col in df.columns and dt_col != "close":
                df.rename(columns={dt_col: "timestamps"}, inplace=True)
                break
                
        # Gerekli sütun kontrolü
        required_cols = ["open", "high", "low", "close", "volume"]
        for c in required_cols:
            if c not in df.columns:
                return ticker, False, f"Eksik sütun: {c}"
                
        # Boş / NaN satırları temizle ve sıfır hacimleri filtrele
        df = df.dropna(subset=required_cols)
        df = df[df["close"] > 0]
        
        # 'amount' (İşlem Hacmi TRY Tutarı) sütunu hesapla: Volume * Weighted Price
        df["amount"] = df["volume"] * ((df["high"] + df["low"] + 2 * df["close"]) / 4.0)
        
        # Sütunları sıraya koy
        final_cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
        df = df[final_cols].sort_values("timestamps").reset_index(drop=True)

        # Diskte daha uzun bir geçmiş varsa koru: farklı periyotlu çağrılar aynı dosyayı paylaşır
        if os.path.exists(file_path):
            try:
                existing = pd.read_csv(file_path)[final_cols]
                is_daily = not interval.endswith(INTRADAY_SUFFIXES)
                if histories_are_consistent(existing, df, daily=is_daily):
                    df = merge_candles(existing, df, daily=is_daily)
                elif len(existing) > len(df):
                    print(f"[UYARI] {ticker}: geçmiş fiyatlar yeniden ölçeklenmiş (bölünme/bedelsiz). Eski {len(existing)} mum atıldı, "
                          f"{len(df)} yeni mum yazıldı. Tam geçmiş için period='max' ile yeniden indirin.")
            except Exception as merge_err:
                print(f"[UYARI] {ticker}: mevcut veri okunamadı, yalnızca yeni indirilen veri yazılıyor: {merge_err}")

        # Atomik CSV kaydet (Yarış koşulu ve bozuk okuma kalkanı)
        import tempfile
        with tempfile.NamedTemporaryFile("w", dir=save_dir, delete=False, encoding="utf-8", suffix=".csv") as tf:
            df.to_csv(tf.name, index=False)
            temp_name = tf.name
        os.replace(temp_name, file_path)
        
        return ticker, True, f"{len(df)} mum kaydedildi -> {file_path}"
        
    except Exception as e:
        return ticker, False, str(e)

def download_bist_universe(mode: str = "bist100", period: str = "max", interval: str = "1d", max_workers: int = 10):
    """
    Tüm BIST 100 veya BIST 30 hisse listesi için paralel veri indirme işlemi başlatır.
    """
    tickers = get_tickers(mode=mode)
    print(f"\n[BASLADI] {mode.upper()} Evreni için veri seti indirme başlatılıyor: Toplam {len(tickers)} Sembol | Periyodu: {period} | Interval: {interval}")
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(download_ticker_data, ticker, period, interval): ticker 
            for ticker in tickers
        }
        
        with tqdm(total=len(tickers), desc="BIST Veri Indirme") as pbar:
            for future in as_completed(future_to_ticker):
                ticker, status, msg = future.result()
                if status:
                    success_count += 1
                else:
                    fail_count += 1
                    tqdm.write(f"[UYARI] {ticker}: {msg}")
                pbar.update(1)
                
    print(f"\n[TAMAMLANDI] Indirme Tamamlandi! Basarili: {success_count} Hisse | Basarisiz: {fail_count} Hisse.")
    print(f"[BILGI] Veriler klasöre yazıldı: {os.path.abspath(RAW_DATA_DIR)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 100 Otomatik Yahoo Finance Veri Çekme Motoru")
    parser.add_argument("--mode", type=str, default="bist100", choices=["bist100", "bist30", "single"], help="Indirilecek hisse grubu")
    parser.add_argument("--ticker", type=str, default=None, help="Single modunda indirilecek tek sembol (Örn: THYAO.IS)")
    parser.add_argument("--period", type=str, default="10y", help="Veri periyodu (max, 10y, 5y vb.)")
    parser.add_argument("--interval", type=str, default="1d", help="Mum periyodu (1d, 1h vb.)")
    parser.add_argument("--workers", type=int, default=8, help="Paralel işçi sayısı")
    
    args = parser.parse_args()
    if args.mode == "single" and args.ticker:
        res = download_ticker_data(args.ticker, period=args.period, interval=args.interval)
        print("Sonuc:", res)
    else:
        download_bist_universe(mode=args.mode, period=args.period, interval=args.interval, max_workers=args.workers)
