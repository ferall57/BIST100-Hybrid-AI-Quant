#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️ BIST 100 ENDEKS KAPISI VE PİYASA REJİMİ MOTORU (INDEX GATEKEEPER)
Borsa Workout Kuralı: "Endeks 50 Günlük SMA altındaysa veya düşüş trendindeyse tekil hisselerde agresif LONG açılmaz."
1. XU100.IS günlük trendini (SMA50, EMA21, ADX/DMI, RSI) analiz eder.
2. Piyasa Rejimi Çıkarır: BULL_REGIME (Boğa), NEUTRAL_CHOP (Yatay/Testere), BEAR_REGIME (Ayı/Düşüş).
3. Katı Filtre (Hard Gatekeeper):
   - BULL_REGIME: Tüm VİOP LONG / Spot sinyallerine tam ağırlıkla (1.0x) izin verir.
   - NEUTRAL_CHOP: Sadece A+ SSL Sweep ve FVG C.E. teyitli sinyallere yarım ağırlıkla (0.5x) izin verir.
   - BEAR_REGIME: Tekil hisse LONG alımlarını KESİNLİKLE ENGELLER (Veto), sistemi VİOP SHORT / Delta-Hedge moduna kilitler.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.bist_downloader import download_ticker_data, RAW_DATA_DIR

MIN_REGIME_BARS = 50


def _neutral_regime(desc: str) -> dict:
    """Veri yetersiz ya da hesap yapılamadığında kullanılan temkinli varsayılan."""
    return {
        "regime": "NEUTRAL_CHOP",
        "is_long_allowed": True,
        "is_short_allowed": True,
        "long_size_multiplier": 0.5,
        "regime_score": 50.0,
        "desc": desc
    }


def compute_market_regime(df: pd.DataFrame) -> dict:
    """
    Verilen endeks mumlarının son gününe göre piyasa rejimini hesaplar (saf fonksiyon).
    Yalnızca kendisine verilen mumları kullanır; backtest'te geçmiş bir güne kadar kesilmiş veriyle çağrılabilir.
    """
    if len(df) < MIN_REGIME_BARS:
        return _neutral_regime("XU100 geçmiş veri yetersiz.")

    df = df.reset_index(drop=True)
    current_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    daily_change = ((current_close - prev_close) / prev_close) * 100.0

    # Hareketli Ortalamalar (SMA50 & EMA21)
    sma_50 = float(df["close"].tail(50).mean())
    ema_21_series = df["close"].ewm(span=21, adjust=False).mean()
    ema_21 = float(ema_21_series.iloc[-1])

    above_sma50 = current_close >= sma_50
    above_ema21 = current_close >= ema_21
    ema_slope_bullish = ema_21 > float(ema_21_series.iloc[-5])

    # RSI 14
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    # ADX & DMI Trend Gücü
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0.0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0.0)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = (pd.Series(plus_dm).rolling(14).mean() / (atr14 + 1e-9)) * 100
    minus_di = (pd.Series(minus_dm).rolling(14).mean() / (atr14 + 1e-9)) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = float(dx.rolling(14).mean().iloc[-1])
    is_strong_bull_trend = bool((plus_di.iloc[-1] > minus_di.iloc[-1]) and (adx >= 20))

    # Puanlama & Rejim Kararı
    regime_score = 50.0
    regime_score += 20.0 if above_sma50 else -25.0
    regime_score += 15.0 if (above_ema21 and ema_slope_bullish) else -15.0
    if is_strong_bull_trend:
        regime_score += 15.0
    regime_score += 5.0 if rsi >= 50.0 else -10.0
    regime_score = max(0.0, min(100.0, regime_score))

    if regime_score >= 65.0 and above_sma50:
        regime = "BULL_REGIME"
        is_long_allowed = True
        is_short_allowed = False
        multiplier = 1.0
        desc = "XU100 Güçlü Boğa Rejiminde (SMA50 & EMA21 Üzerinde) -> Tüm LONG sinyaller serbest."
    elif regime_score <= 35.0 or (not above_sma50 and not above_ema21):
        regime = "BEAR_REGIME"
        is_long_allowed = False  # KATI KAPI: LONG ALIMLAR ENGELLENDİ
        is_short_allowed = True
        multiplier = 0.0
        desc = "🚨 XU100 Katı Ayı Rejiminde (SMA50 Altında) -> Tekil hissede LONG alımlar ENGELLENDİ / Sadece SHORT & HEDGE serbest."
    else:
        regime = "NEUTRAL_CHOP"
        is_long_allowed = True
        is_short_allowed = True
        multiplier = 0.5
        desc = "XU100 Testere / Yatay Rejimde -> Sadece A+ ICT kurulumlarına yarım riskle (0.5x) izin verilir."

    return {
        "regime": regime,
        "is_long_allowed": is_long_allowed,
        "is_short_allowed": is_short_allowed,
        "long_size_multiplier": multiplier,
        "regime_score": round(regime_score, 1),
        "xu100_close": round(current_close, 2),
        "daily_change_pct": round(daily_change, 2),
        "sma_50": round(sma_50, 2),
        "ema_21": round(ema_21, 2),
        "rsi_14": round(rsi, 1),
        "adx_14": round(adx, 1),
        "above_sma50": bool(above_sma50),
        "desc": desc
    }


class BistIndexGatekeeper:
    """
    Borsa İstanbul XU100 (BIST 100) Endeks Kalkanı ve Rejim Yöneticisi.
    """
    def __init__(self, index_ticker: str = "XU100.IS"):
        self.index_ticker = index_ticker
        self._cached_regime = None

    def get_market_regime(self, force_refresh: bool = False) -> dict:
        """
        BIST 100 Endeksinin güncel teknik ve yapısal trend rejimini hesaplar.
        """
        try:
            download_ticker_data(self.index_ticker, period="1y", interval="1d", save_dir=RAW_DATA_DIR)
            csv_path = os.path.join(RAW_DATA_DIR, f"{self.index_ticker}_1d.csv")
            if not os.path.exists(csv_path):
                return _neutral_regime("XU100 verisi okunamadı, temkinli mod devrede.")

            result = compute_market_regime(pd.read_csv(csv_path))
            self._cached_regime = result
            return result
        except Exception as e:
            print(f"[UYARI] Endeks rejimi hesaplanamadı, temkinli mod devrede: {e}")
            return _neutral_regime(f"Endeks analizi hesaplama istisnası: {e}")

    def generate_gatekeeper_report(self) -> str:
        """
        Komiteye ve Raporlara sunulacak XU100 Endeks Kapısı brifingi.
        """
        r = self.get_market_regime()

        status_icon = "🟢" if r["regime"] == "BULL_REGIME" else ("🟡" if r["regime"] == "NEUTRAL_CHOP" else "🔴")

        report = f"""🏛️ BIST 100 ENDEKS KAPISI VE PİYASA REJİM RAPORU (XU100)
================================================================================
{status_icon} PİYASA REJİMİ: {r['regime']} (Puan: %{r.get('regime_score', 50)}/100)
--------------------------------------------------------------------------------
• XU100 Güncel Kapanış : {r.get('xu100_close', 0.0):.2f} TRY ({r.get('daily_change_pct', 0.0):+.2f}%)
• 50 Günlük SMA Desteği: {r.get('sma_50', 0.0):.2f} TRY [{'ÜSTÜNDE ✅' if r.get('above_sma50') else 'ALTINDA ❌'}]
• 21 Günlük EMA Desteği: {r.get('ema_21', 0.0):.2f} TRY
• RSI & Trend Gücü (ADX): RSI {r.get('rsi_14', 50.0)} | ADX {r.get('adx_14', 20.0)}
--------------------------------------------------------------------------------
🛡️ İŞLEM İZİN PROTOKOLÜ (BORSA WORKOUT GATEKEEPER):
• Tekil Hisse LONG İzni  : {'AÇIK ✅ (Normal Ağırlık)' if r.get('is_long_allowed') and r.get('long_size_multiplier', 1.0) == 1.0 else ('KISITLI 🟡 (0.5x Yarım Ağırlık)' if r.get('is_long_allowed') else 'KAPALI / YASAK 🛑 (Hard Block)')}
• VİOP SHORT / Hedge İzni: {'AKTİF 🛡️' if r.get('is_short_allowed') else 'PASİF'}
• Operasyonel Direktif  : {r.get('desc', '')}
================================================================================"""
        return report


if __name__ == "__main__":
    print("Testing BistIndexGatekeeper...")
    gk = BistIndexGatekeeper()
    print(gk.generate_gatekeeper_report())
