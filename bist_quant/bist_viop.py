#!/usr/bin/env python3
"""
BIST VİOP (Vadeli İşlem ve Opsiyon Piyasası) Çift Yönlü (Long/Short) Türev Motoru
Borsa İstanbul BIST 30 hisseleri ve endeks kontratlarında çift yönlü kaldıraçlı işlem,
Takasbank teminat nemalandırması, ters iz süren stop (Inverted Trailing Stop) ve
piyasa nötr risk yönetimi sağlar.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class BistViopEngine:
    """
    BIST VİOP Çift Yönlü (Long/Short) Kuant Türev İşlem Motoru.
    """

    # Takasbank Resmi Maktu SPAN Teminat Tablosu (TL / Kontrat)
    TAKASBANK_SPAN_MARGINS = {
        "THYAO": 6200.0,
        "ISCTR": 272.0,
        "AKBNK": 1520.0,
        "GARAN": 2650.0,
        "EREGL": 880.0,
        "ASELS": 890.0,
        "FROTO": 24500.0,
        "BIMAS": 9200.0,
        "KCHOL": 3800.0,
        "TUPRS": 3400.0,
        "SAHOL": 1850.0,
        "EKGYO": 420.0,
        "SISE": 890.0,
        "F_XU030": 16500.0
    }

    def __init__(
        self,
        default_leverage: float = 1.5,
        overnight_interest_annual: float = 0.45,
        commission_rate: float = 0.0004
    ):
        self.default_leverage = default_leverage
        self.overnight_interest_annual = overnight_interest_annual
        # Takasbank Gecelik Nemalandırma Günlük Bileşik Oranı (Yıllık %45 varsayımı)
        self.daily_interest_rate = (1.0 + overnight_interest_annual) ** (1.0 / 365.0) - 1.0
        self.commission_rate = commission_rate
        self.contract_multiplier = 100  # 1 Pay Kontratı = 100 Adet Hisse Senedi
        self.default_margin_ratio = 0.22  # Liste dışı hisseler için varsayılan teminat oranı (~%22)

    def calculate_theoretical_futures_price(
        self,
        spot_price: float,
        days_to_expiry: int = 30,
        dividend_yield_annual: float = 0.02
    ) -> float:
        """
        Cost-of-Carry (Taşıma Maliyeti) modeline göre teorik VİOP vadeli fiyatını hesaplar.
        F = S * (1 + (r_f - q) * (T - t) / 365)
        """
        net_carry_rate = self.overnight_interest_annual - dividend_yield_annual
        futures_price = spot_price * (1.0 + (net_carry_rate * (days_to_expiry / 365.0)))
        return round(futures_price, 2)

    def get_contract_code(self, ticker: str) -> str:
        """Hisse sembolünden VİOP kontrat kodunu türetir (Örn: THYAO.IS -> F_THYAO)."""
        clean = ticker.replace(".IS", "").strip().upper()
        return f"F_{clean}"

    def calculate_position_size(
        self,
        capital: float,
        spot_price: float,
        ticker: str = "",
        leverage: float = 1.5,
        allocation_pct: float = 80.0
    ) -> dict:
        """
        Kasa büyüklüğüne, Takasbank SPAN teminatına ve hedef kaldıraca göre güvenli kontrat sayısını hesaplar.
        """
        if spot_price <= 0 or capital <= 0:
            return {"contracts": 0, "notional_value": 0.0, "required_margin": 0.0, "cash_reserve": capital}

        clean = ticker.replace(".IS", "").strip().upper()
        # Maktu SPAN teminatı varsa doğrudan kullan, yoksa yüzde bazlı hesapla
        span_margin_per_contract = self.TAKASBANK_SPAN_MARGINS.get(clean, None)

        contract_value = spot_price * self.contract_multiplier
        target_notional = capital * (allocation_pct / 100.0) * leverage
        num_contracts = max(1, int(target_notional / contract_value))

        if span_margin_per_contract is not None:
            required_margin = num_contracts * span_margin_per_contract
        else:
            required_margin = num_contracts * contract_value * self.default_margin_ratio
        
        # Teminat kasayı aşıyorsa kontratı küçült
        while required_margin > capital * 0.90 and num_contracts > 1:
            num_contracts -= 1
            if span_margin_per_contract is not None:
                required_margin = num_contracts * span_margin_per_contract
            else:
                required_margin = num_contracts * contract_value * self.default_margin_ratio

        actual_notional = num_contracts * contract_value
        cash_reserve = max(0.0, capital - required_margin)

        return {
            "contracts": num_contracts,
            "contract_value": round(contract_value, 2),
            "notional_value": round(actual_notional, 2),
            "required_margin": round(required_margin, 2),
            "cash_reserve": round(cash_reserve, 2),
            "effective_leverage": round(actual_notional / capital, 2)
        }

    def simulate_viop_trade(
        self,
        forward_df: pd.DataFrame,
        entry_price: float,
        direction: str,  # 'LONG' veya 'SHORT'
        contracts: int,
        capital: float,
        stop_loss_pct: float = 3.5,
        take_profit_pct: float = 8.0,
        trailing_stop_pct: float = 4.5,
        max_hold_days: int = 30
    ) -> dict:
        """
        VİOP üzerinde tek bir Long veya Short pozisyonu gün gün simüle eder.
        Mark-to-Market PnL, Takasbank günlük faiz nemalandırması ve ters iz süren stop işletir.
        """
        direction = direction.upper()
        if direction not in ["LONG", "SHORT"]:
            raise ValueError("Direction 'LONG' veya 'SHORT' olmalıdır.")

        contract_value_entry = entry_price * self.contract_multiplier
        notional_entry = contracts * contract_value_entry
        required_margin = notional_entry * self.initial_margin_ratio
        current_cash = capital - required_margin

        # Giriş komisyonu
        entry_commission = notional_entry * self.commission_rate
        current_cash -= entry_commission

        peak_price = entry_price  # Long için en yüksek
        trough_price = entry_price  # Short için en düşük
        total_interest_earned = 0.0

        exit_price = entry_price
        exit_date = None
        exit_reason = "Vade Sonu Kapanış (Time Horizon)"
        days_held = 0
        pnl_pct = 0.0
        final_capital = capital

        for day_i, row in forward_df.iterrows():
            days_held += 1
            current_close = float(row["close"])
            current_high = float(row.get("high", current_close))
            current_low = float(row.get("low", current_close))
            current_date = row.get("timestamps", f"Gün-{day_i}")

            # 1. Takasbank Günlük Nemalandırma Faizi (Boştaki Nakit + Teminata İşler)
            daily_interest = (current_cash + required_margin) * self.daily_interest_rate
            total_interest_earned += daily_interest
            current_cash += daily_interest

            # 2. LONG Pozisyon Yönetimi
            if direction == "LONG":
                if current_high > peak_price:
                    peak_price = current_high

                # Sabit Stop-Loss Kontrolü
                if current_low <= entry_price * (1.0 - stop_loss_pct / 100.0):
                    exit_price = entry_price * (1.0 - stop_loss_pct / 100.0)
                    exit_date = current_date
                    exit_reason = f"🔴 Stop-Loss (%{stop_loss_pct:.1f})"
                    break

                # Sabit Kâr Al (Take-Profit)
                if current_high >= entry_price * (1.0 + take_profit_pct / 100.0):
                    exit_price = entry_price * (1.0 + take_profit_pct / 100.0)
                    exit_date = current_date
                    exit_reason = f"🎯 Kâr Al (%{take_profit_pct:.1f})"
                    break

                # İz Süren Stop (Trailing Stop)
                trailing_level = peak_price * (1.0 - trailing_stop_pct / 100.0)
                if peak_price >= entry_price * 1.015 and current_low <= trailing_level:
                    exit_price = trailing_level
                    exit_date = current_date
                    gain_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    exit_reason = f"🟢 İz Süren Stop (Trailing %{gain_pct:+.1f})"
                    break

            # 3. SHORT Pozisyon Yönetimi (Açığa Satış - Düşüşten Kazanır)
            elif direction == "SHORT":
                if current_low < trough_price:
                    trough_price = current_low

                # Sabit Stop-Loss (Fiyat yükselirse Short zarar eder)
                if current_high >= entry_price * (1.0 + stop_loss_pct / 100.0):
                    exit_price = entry_price * (1.0 + stop_loss_pct / 100.0)
                    exit_date = current_date
                    exit_reason = f"🔴 Short Stop-Loss (%{stop_loss_pct:.1f})"
                    break

                # Sabit Kâr Al (Fiyat düşerse Short kâr eder)
                if current_low <= entry_price * (1.0 - take_profit_pct / 100.0):
                    exit_price = entry_price * (1.0 - take_profit_pct / 100.0)
                    exit_date = current_date
                    exit_reason = f"🎯 Short Kâr Al (%{take_profit_pct:.1f})"
                    break

                # Ters İz Süren Stop (Inverted Trailing Stop): Dip seviyeden yukarı tepki gelirse kârı al
                inverted_trailing_level = trough_price * (1.0 + trailing_stop_pct / 100.0)
                if trough_price <= entry_price * 0.985 and current_high >= inverted_trailing_level:
                    exit_price = inverted_trailing_level
                    exit_date = current_date
                    gain_pct = ((entry_price - exit_price) / entry_price) * 100.0
                    exit_reason = f"🟢 Ters İz Süren Stop (Short Trailing %{gain_pct:+.1f})"
                    break

            # Zaman aşımı (Maksimum Gün)
            if days_held >= max_hold_days:
                exit_price = current_close
                exit_date = current_date
                exit_reason = f"⏳ Vade Sonu Kapanış ({max_hold_days} Gün)"
                break

        if exit_date is None and len(forward_df) > 0:
            exit_price = float(forward_df["close"].iloc[-1])
            exit_date = forward_df["timestamps"].iloc[-1] if "timestamps" in forward_df else "Son Gün"

        # Çıkış Hesaplaması
        notional_exit = contracts * (exit_price * self.contract_multiplier)
        exit_commission = notional_exit * self.commission_rate
        current_cash -= exit_commission

        if direction == "LONG":
            gross_trade_pnl = (exit_price - entry_price) * self.contract_multiplier * contracts
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
        else:  # SHORT
            gross_trade_pnl = (entry_price - exit_price) * self.contract_multiplier * contracts
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0

        net_trade_pnl = gross_trade_pnl - (entry_commission + exit_commission)
        final_capital = current_cash + required_margin + gross_trade_pnl
        total_pnl_pct = ((final_capital - capital) / capital) * 100.0

        return {
            "direction": direction,
            "contracts": contracts,
            "entry_price": entry_price,
            "exit_price": round(exit_price, 2),
            "exit_date": exit_date,
            "days_held": days_held,
            "pnl_pct": round(pnl_pct, 2),
            "net_pnl_try": round(net_trade_pnl, 2),
            "interest_earned_try": round(total_interest_earned, 2),
            "final_capital": round(final_capital, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "exit_reason": exit_reason
        }

    def generate_viop_signals(self, candidate_data: list[dict]) -> list[dict]:
        """
        Taranan hisseler için anlık VİOP Kontrat Sinyallerini (Long vs Short) üretir.
        """
        signals = []
        for c in candidate_data:
            ticker = c.get("ticker", "")
            fused_ret = float(c.get("fused_expected_return", c.get("expected_return", 0.0)))
            trend = c.get("trend", "NÖTR")
            sentiment_score = float(c.get("sentiment_score", 0.0))

            contract_code = self.get_contract_code(ticker)

            # Sinyal Kuralları:
            if fused_ret >= 0.8 and (trend == "BOĞA" or sentiment_score >= 0.3):
                signal = "🚀 GÜÇLÜ LONG (Kaldıraçlı Alış)"
                target_action = "LONG"
                confidence = min(95, 65 + int(fused_ret * 5))
            elif fused_ret <= -0.8 or (trend == "AYI" and sentiment_score <= -0.2):
                signal = "🔻 GÜÇLÜ SHORT (Açığa Satış)"
                target_action = "SHORT"
                confidence = min(95, 65 + int(abs(fused_ret) * 5))
            else:
                signal = "⚪ NÖTR / NAKİT"
                target_action = "FLAT"
                confidence = 50

            signals.append({
                "ticker": ticker,
                "contract": contract_code,
                "close": c.get("close", 0.0),
                "expected_return": fused_ret,
                "trend": trend,
                "signal": signal,
                "target_action": target_action,
                "confidence": f"%{confidence}"
            })

        return signals


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BIST VİOP Çift Yönlü Türev Motoru Test Arayüzü")
    parser.add_argument("ticker", type=str, default="THYAO.IS", nargs="?", help="BIST Sembolü")
    args = parser.parse_args()

    engine = BistViopEngine()
    print(f"\n⚡ [{args.ticker}] için VİOP Kontrat Hesaplaması Başlatılıyor...")
    
    pos = engine.calculate_position_size(capital=100000.0, spot_price=300.0, ticker=args.ticker, leverage=1.5)
    theo_p = engine.calculate_theoretical_futures_price(spot_price=300.0, days_to_expiry=30)
    print("\n" + "="*80)
    print(f"📊 VİOP POZİSYON VE TEMİNAT PLANI: {engine.get_contract_code(args.ticker)}")
    print("="*80)
    print(f"💰 Başlangıç Kasası     : 100,000.00 TRY")
    print(f"📦 Açılacak Kontrat     : {pos['contracts']} Adet ({pos['contracts']*100} Hisse)")
    print(f"📈 Pozisyon Büyüklüğü   : {pos['notional_value']:,.2f} TRY (Efektif Kaldıraç: {pos['effective_leverage']}x)")
    print(f"🛡️ Takasbank Teminatı   : {pos['required_margin']:,.2f} TRY")
    print(f"💵 Boştaki Nakit Rezervi: {pos['cash_reserve']:,.2f} TRY (%45 Nemalandırma Faizinde)")
    print("="*80 + "\n")
