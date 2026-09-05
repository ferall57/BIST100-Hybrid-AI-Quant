#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️ BIST ICT SMART MONEY CONCEPTS & PRICE ACTION MOTORU
1. Likidite Süpürme Radarı (BSL - Buy-Side Liquidity & SSL - Sell-Side Liquidity Sweeps)
2. Fair Value Gap (FVG - 3 Mumluk Dengesizlik) & %50 Consequent Encroachment (C.E.)
3. Sert Yer Değiştirme (Displacement) & Hacim Patlaması Doğrulaması
4. Hacim Uyumsuzluklu Kırılım ve Retest (Break & Retest with Volume Dry-Up)
5. Power of 3 (PO3 / AMD - Akümülasyon, Manipülasyon/Judas Swing, Dağıtım)
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class BistPriceActionEngine:
    """
    Borsa İstanbul için ICT (Inner Circle Trader) ve Klasik Price Action Analiz Motoru.
    """
    def __init__(self, swing_window: int = 5, fvg_min_pct: float = 0.3):
        self.swing_window = swing_window
        self.fvg_min_pct = fvg_min_pct

    # =========================================================================
    # 1. SWING TEPE VE DİPLER (FRACTALS & LIQUIDITY POOLS)
    # =========================================================================
    def find_swing_points(self, df: pd.DataFrame, window: int = None) -> dict:
        """
        Grafikteki Swing High (BSL Havuzları) ve Swing Low (SSL Havuzları) noktalarını tespit eder.
        """
        w = window or self.swing_window
        if len(df) < (w * 2 + 1):
            return {"swing_highs": [], "swing_lows": []}

        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        
        swing_highs = []
        swing_lows = []

        for i in range(w, n - w):
            if all(highs[i] >= highs[i - j] for j in range(1, w + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, w + 1)):
                swing_highs.append({
                    "index": i,
                    "price": float(highs[i]),
                    "date": str(df["timestamps"].iloc[i]) if "timestamps" in df else str(i)
                })

            if all(lows[i] <= lows[i - j] for j in range(1, w + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, w + 1)):
                swing_lows.append({
                    "index": i,
                    "price": float(lows[i]),
                    "date": str(df["timestamps"].iloc[i]) if "timestamps" in df else str(i)
                })

        return {"swing_highs": swing_highs, "swing_lows": swing_lows}

    # =========================================================================
    # 2. LİKİDİTE SÜPÜRME RADARI (BSL & SSL SWEEPS / TURTLE SOUP)
    # =========================================================================
    def detect_liquidity_sweeps(self, df: pd.DataFrame, lookback_swings: int = 10) -> dict:
        """
        Perakende stoplarının süpürülüp süpürülmediğini denetler:
        - SSL Sweep (Boğa Sinyali): Fiyat önceki dip seviyenin altına fitil atar (stopları patlatır) 
          ancak mum kapanışı bu seviyenin üzerinde gerçekleşir (Rejection / Alıcı Tepkisi).
        - BSL Sweep (Ayı Sinyali): Fiyat önceki tepe seviyenin üzerine fitil atar (alıcıları tuzağa çeker)
          ancak mum kapanışı tepe seviyenin altında gerçekleşir.
        """
        swings = self.find_swing_points(df, window=self.swing_window)
        swing_highs = swings["swing_highs"][-lookback_swings:]
        swing_lows = swings["swing_lows"][-lookback_swings:]
        
        if len(df) < 5 or (not swing_highs and not swing_lows):
            return {"ssl_swept": False, "bsl_swept": False, "last_sweep": None, "sweep_score": 0.0, "details": []}

        ssl_swept = False
        bsl_swept = False
        sweep_details = []

        for offset in range(1, min(4, len(df))):
            idx = len(df) - offset
            c_high = float(df["high"].iloc[idx])
            c_low = float(df["low"].iloc[idx])
            c_close = float(df["close"].iloc[idx])

            # SSL Sweep Kontrolü
            for sl in swing_lows:
                if sl["index"] < idx - 1:
                    if c_low < sl["price"] and c_close > sl["price"]:
                        ssl_swept = True
                        sweep_details.append({
                            "type": "SSL_SWEEP_BULLISH",
                            "level": sl["price"],
                            "candle_offset": offset,
                            "desc": f"{offset} bar önce {sl['price']:.2f} TRY dip seviyesindeki perakende stopları süpürüldü (SSL Sweep - Boğa Tepkisi)"
                        })
                        break

            # BSL Sweep Kontrolü
            for sh in swing_highs:
                if sh["index"] < idx - 1:
                    if c_high > sh["price"] and c_close < sh["price"]:
                        bsl_swept = True
                        sweep_details.append({
                            "type": "BSL_SWEEP_BEARISH",
                            "level": sh["price"],
                            "candle_offset": offset,
                            "desc": f"{offset} bar önce {sh['price']:.2f} TRY tepe seviyesindeki alıcılar tuzağa çekildi (BSL Sweep - Ayı Tuzağı)"
                        })
                        break

        sweep_score = 0.0
        if ssl_swept and not bsl_swept:
            sweep_score = 1.0
        elif bsl_swept and not ssl_swept:
            sweep_score = -1.0

        return {
            "ssl_swept": ssl_swept,
            "bsl_swept": bsl_swept,
            "sweep_score": sweep_score,
            "details": sweep_details
        }

    # =========================================================================
    # 3. FAIR VALUE GAP (FVG) & 50% CONSEQUENT ENCROACHMENT (C.E.)
    # =========================================================================
    def detect_fair_value_gaps(self, df: pd.DataFrame, max_gaps: int = 5) -> dict:
        """
        3 mumluk fiyat dengesizliklerini (Imbalance) ve %50 Consequent Encroachment (C.E.) seviyesini hesaplar.
        - Bullish FVG: Low[i+1] > High[i-1]
        - Bearish FVG: High[i+1] < Low[i-1]
        """
        if len(df) < 5:
            return {"active_bullish_fvgs": [], "active_bearish_fvgs": [], "nearest_fvg": None}

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)
        current_close = float(closes[-1])

        bullish_fvgs = []
        bearish_fvgs = []

        start_i = max(2, n - 30)
        for i in range(start_i, n - 1):
            if lows[i + 1] > highs[i - 1]:
                gap_top = float(lows[i + 1])
                gap_bottom = float(highs[i - 1])
                gap_size_pct = ((gap_top - gap_bottom) / gap_bottom) * 100.0
                ce_50 = gap_bottom + (gap_top - gap_bottom) * 0.5
                
                if gap_size_pct >= self.fvg_min_pct:
                    subsequent_lows = lows[i + 2:] if (i + 2) < n else []
                    is_mitigated = any(l <= gap_bottom for l in subsequent_lows) if len(subsequent_lows) > 0 else False
                    
                    if not is_mitigated:
                        bullish_fvgs.append({
                            "index": i,
                            "top": round(gap_top, 2),
                            "bottom": round(gap_bottom, 2),
                            "ce_50": round(ce_50, 2),
                            "gap_pct": round(gap_size_pct, 2),
                            "type": "BULLISH_FVG",
                            "is_active": current_close >= gap_bottom
                        })

            elif highs[i + 1] < lows[i - 1]:
                gap_top = float(lows[i - 1])
                gap_bottom = float(highs[i + 1])
                gap_size_pct = ((gap_top - gap_bottom) / gap_bottom) * 100.0
                ce_50 = gap_bottom + (gap_top - gap_bottom) * 0.5
                
                if gap_size_pct >= self.fvg_min_pct:
                    subsequent_highs = highs[i + 2:] if (i + 2) < n else []
                    is_mitigated = any(h >= gap_top for h in subsequent_highs) if len(subsequent_highs) > 0 else False
                    
                    if not is_mitigated:
                        bearish_fvgs.append({
                            "index": i,
                            "top": round(gap_top, 2),
                            "bottom": round(gap_bottom, 2),
                            "ce_50": round(ce_50, 2),
                            "gap_pct": round(gap_size_pct, 2),
                            "type": "BEARISH_FVG",
                            "is_active": current_close <= gap_top
                        })

        nearest_fvg = None
        min_dist = float('inf')
        
        for f in bullish_fvgs + bearish_fvgs:
            dist = abs(current_close - f["ce_50"])
            if dist < min_dist:
                min_dist = dist
                nearest_fvg = f

        return {
            "active_bullish_fvgs": bullish_fvgs[-max_gaps:],
            "active_bearish_fvgs": bearish_fvgs[-max_gaps:],
            "nearest_fvg": nearest_fvg
        }

    # =========================================================================
    # 4. DISPLACEMENT (SERT YER DEĞİŞTİRME) & HACİM DOĞRULAMASI
    # =========================================================================
    def detect_displacement(self, df: pd.DataFrame) -> dict:
        """
        Son barlarda akıllı paranın agresif girişini gösteren Displacement mumlarını arar.
        """
        if len(df) < 20:
            return {"has_displacement": False, "direction": "NÖTR", "body_atr_ratio": 1.0, "bars_ago": None}

        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1]
        
        avg_vol_20 = df["volume"].tail(20).mean()
        
        has_disp = False
        direction = "NÖTR"
        disp_bar_idx = None
        max_ratio = 1.0

        for offset in range(1, min(4, len(df))):
            idx = len(df) - offset
            c_open = df["open"].iloc[idx]
            c_close = df["close"].iloc[idx]
            c_vol = df["volume"].iloc[idx]
            body = abs(c_close - c_open)
            
            body_ratio = body / (atr_14 + 1e-9)
            vol_ratio = c_vol / (avg_vol_20 + 1e-9)

            if body_ratio >= 1.3 and vol_ratio >= 1.2:
                has_disp = True
                disp_bar_idx = offset
                max_ratio = max(max_ratio, body_ratio)
                direction = "BULLISH" if c_close > c_open else "BEARISH"
                break

        return {
            "has_displacement": has_disp,
            "direction": direction,
            "body_atr_ratio": round(float(max_ratio), 2),
            "bars_ago": disp_bar_idx
        }

    # =========================================================================
    # 5. HACİM UYUMSUZLUKLU KIRILIM VE RETEST (BREAK & RETEST)
    # =========================================================================
    def detect_break_and_retest(self, df: pd.DataFrame) -> dict:
        """
        Borsa Workout ve Price Action Klasik Kırılım-Retest Teyidi.
        """
        swings = self.find_swing_points(df, window=self.swing_window)
        swing_highs = swings["swing_highs"]
        
        if len(df) < 15 or len(swing_highs) < 2:
            return {"is_break_retest": False, "retest_level": None, "score": 0.0, "status": "YOK"}

        current_close = float(df["close"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        avg_vol = df["volume"].tail(20).mean()
        
        last_sh = swing_highs[-1]
        resistance_level = last_sh["price"]
        
        dist_to_res = ((current_close - resistance_level) / resistance_level) * 100.0
        is_testing = (-1.0 <= dist_to_res <= 2.5) and (current_low <= resistance_level * 1.008)
        
        if is_testing:
            last_vol = df["volume"].iloc[-1]
            vol_dry_up = (last_vol < avg_vol * 0.85)
            
            breakout_idx = last_sh["index"]
            breakout_vol = df["volume"].iloc[breakout_idx] if breakout_idx < len(df) else avg_vol
            high_vol_breakout = (breakout_vol >= avg_vol * 1.15)
            
            score = 0.5
            if vol_dry_up:
                score += 0.3
            if high_vol_breakout:
                score += 0.2

            return {
                "is_break_retest": True,
                "retest_level": round(resistance_level, 2),
                "distance_pct": round(dist_to_res, 2),
                "volume_dry_up": vol_dry_up,
                "score": round(score, 2),
                "status": f"{resistance_level:.2f} TRY Direnci Kırıldı -> Düşük Hacimli Retest Onayında ✅"
            }

        return {
            "is_break_retest": False,
            "retest_level": round(resistance_level, 2),
            "distance_pct": round(dist_to_res, 2),
            "score": 0.0,
            "status": "Kırılım / Retest Alanında Değil"
        }

    # =========================================================================
    # 6. POWER OF 3 (PO3 / AMD DÖNGÜSÜ)
    # =========================================================================
    def detect_po3_cycle(self, df: pd.DataFrame) -> dict:
        """
        Seans / Günlük PO3 döngüsünü çözer.
        """
        if len(df) < 5:
            return {"phase": "NÖTR", "judas_detected": False, "desc": "Yetersiz veri"}

        c_open = float(df["open"].iloc[-1])
        c_high = float(df["high"].iloc[-1])
        c_low = float(df["low"].iloc[-1])
        c_close = float(df["close"].iloc[-1])
        
        lower_wick = min(c_open, c_close) - c_low
        upper_wick = c_high - max(c_open, c_close)
        total_range = c_high - c_low + 1e-9
        
        if (lower_wick / total_range) > 0.40 and c_close >= c_open:
            return {
                "phase": "DISTRIBUTION_EXPANSION_BULLISH",
                "judas_detected": True,
                "desc": "Boğa Judas Swing Tamamlandı (Sabah tuzağı süpürüldü, yukarı genişleme evresinde)"
            }
        elif (upper_wick / total_range) > 0.40 and c_close <= c_open:
            return {
                "phase": "DISTRIBUTION_EXPANSION_BEARISH",
                "judas_detected": True,
                "desc": "Ayı Judas Swing Tamamlandı (Tepede alıcı tuzağı atıldı, aşağı genişleme evresinde)"
            }
        else:
            return {
                "phase": "ACCUMULATION_OR_NORMAL",
                "judas_detected": False,
                "desc": "Normal Mum / Konsolidasyon Rejimi"
            }

    # =========================================================================
    # 7. KAPSAMLI FÜZYON RAPORU (XAI - EXPLAINABLE ICT REPORT)
    # =========================================================================
    def generate_price_action_report(self, df: pd.DataFrame, ticker: str) -> str:
        """
        Komiteye ve Raporlara sunulacak detaylı ICT & Price Action brifingi üretir.
        """
        sweeps = self.detect_liquidity_sweeps(df)
        fvgs = self.detect_fair_value_gaps(df)
        disp = self.detect_displacement(df)
        retest = self.detect_break_and_retest(df)
        po3 = self.detect_po3_cycle(df)
        
        current_close = float(df["close"].iloc[-1])

        # ICT Birleşik Kalite Puanı (A+ Setup Score: 0 - 100)
        ict_score = 50.0
        if sweeps["ssl_swept"]:
            ict_score += 20.0
        if sweeps["bsl_swept"]:
            ict_score -= 20.0
            
        if disp["has_displacement"]:
            if disp["direction"] == "BULLISH":
                ict_score += 15.0
            else:
                ict_score -= 15.0
                
        if retest["is_break_retest"]:
            ict_score += 15.0
            
        if fvgs["nearest_fvg"] and fvgs["nearest_fvg"]["type"] == "BULLISH_FVG":
            ict_score += 10.0

        ict_score = max(0.0, min(100.0, ict_score))
        
        # Rapor formatı
        fvg_text = "Aktif FVG Bulunamadı."
        ce_val = current_close
        if fvgs["nearest_fvg"]:
            nf = fvgs["nearest_fvg"]
            ce_val = nf["ce_50"]
            fvg_text = f"{nf['type']} [{nf['bottom']} - {nf['top']} TRY] | %50 C.E. Denge Seviyesi: {nf['ce_50']} TRY (Fiyata Uzaklık: %{abs(current_close - nf['ce_50'])/current_close*100:.1f})"

        report = f"""🏛️ ICT SMART MONEY CONCEPTS & PRICE ACTION BRİFİNGİ ({ticker})
================================================================================
🎯 ICT A+ SETUP KALİTE PUANI: %{ict_score:.1f} / 100 ({'A+ MÜKEMMEL KURUMSAL GİRİŞ' if ict_score >= 75 else ('B STANDART KURULUM' if ict_score >= 50 else 'C RİSKLİ / AYI TUZAĞI')})
--------------------------------------------------------------------------------
1. 🧹 LİKİDİTE SÜPÜRME (BSL / SSL SWEEPS):
   • Sell-Side Liquidity (SSL) Dip Süpürmesi : {'EVET 🟢 (Perakende Stopları Avlandı - Boğa Onayı)' if sweeps['ssl_swept'] else 'YOK ⚪'}
   • Buy-Side Liquidity (BSL) Tepe Tuzağı   : {'EVET 🔴 (Tepede Alıcı Tuzağı - Dikkat!)' if sweeps['bsl_swept'] else 'YOK ⚪'}
   • Likidite Durum Özeti: {sweeps['details'][0]['desc'] if sweeps['details'] else 'Son barlarda kritik likidite havuzu süpürülmedi.'}

2. 📐 FAIR VALUE GAP (FVG) & %50 CONSEQUENT ENCROACHMENT (C.E.):
   • En Yakın FVG Bölgesi: {fvg_text}
   • 💡 İdeal Giriş Taktigi: Tepeden piyasa emri yerine {ce_val:.2f} TRY seviyesindeki %50 C.E. boşluğuna geri çekilme (Discount) beklenmelidir.

3. ⚡ SERT YER DEĞİŞTİRME (DISPLACEMENT):
   • Akıllı Para Atılımı (Displacement): {'EVET 🟢' if disp['has_displacement'] else 'YOK ⚪'} ({disp['direction']} - Gövde/ATR Oranı: {disp['body_atr_ratio']}x)

4. 🧱 KIRILIM VE RETEST (BREAK & RETEST):
   • Durum: {retest['status']} (Puan: {retest['score']*100:.0f}/100)

5. ⏳ POWER OF 3 (PO3 / AMD DÖNGÜSÜ):
   • Mevcut Faz: {po3['phase']}
   • Açılış Yorumu: {po3['desc']}
================================================================================"""
        return report


if __name__ == "__main__":
    print("Testing BistPriceActionEngine...")
    import os
    import pandas as pd
    from bist_quant.bist_downloader import download_ticker_data, RAW_DATA_DIR
    download_ticker_data("AKBNK.IS", period="6mo", interval="1d", save_dir=RAW_DATA_DIR)
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, "AKBNK.IS_1d.csv"))
    engine = BistPriceActionEngine()
    rep = engine.generate_price_action_report(df, "AKBNK.IS")
    print(rep)
    print("\nBistPriceActionEngine test passed!")
