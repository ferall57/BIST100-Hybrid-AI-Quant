#!/usr/bin/env python3
"""
🏛️ BIST VİOP & TÜREV PİYASALARI KANTİTATİF FİYATLAMA VE RİSK MOTORU
Borsa İstanbul BIST 30 hisseleri ve endeks kontratlarında:
1. Black-Scholes-Merton (BSM) Analitik Opsiyon Fiyatlaması (Temettü & Faiz Entegre)
2. 1. ve 2. Derece Opsiyon Duyarlılıkları (Greeks: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga)
3. Zımni Oynaklık (Implied Volatility - IV) Newton-Raphson & Brent Kök Bulucu
4. Put-Call Paritesi & Arbitraj/Zımni Repo Oranı Tespiti
5. Dinamik Portföy Delta-Hedge (Piyasa Nötrleşme) ve VİOP Çift Yönlü Kaldıraç Motoru
6. Takasbank SPAN Teminat ve Günlük Nemalandırma Yönetimi
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats
from scipy.optimize import brentq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class BistViopEngine:
    """
    BIST VİOP ve Opsiyon Piyasası Kantitatif Fiyatlama ve Çift Yönlü Türev Motoru.
    """

    # Takasbank & Midas Resmi Risk Ağırlıklı Dinamik Teminat Oranları (% / Nominal Değer)
    SPAN_MARGIN_RATIOS = {
        # BIST Bankacılık
        "ISCTR": 0.2008,   # Spot 12.50 TL -> 1 Kontrat = 250.99 TL (Midas ile birebir)
        "AKBNK": 0.24115,  # Spot 73.95 TL -> 1 Kontrat = 1,783.21 TL (Midas ile kuruşu kuruşuna)
        "GARAN": 0.2000,   # Spot 120.00 TL -> 1 Kontrat = 2,400.00 TL
        "YKBNK": 0.2000,   # Spot 30.00 TL -> 1 Kontrat = 600.00 TL
        "HALKB": 0.2200,
        "VAKBN": 0.2200,
        
        # BIST Havacılık & Ulaştırma
        "PGSUS": 0.2627,   # Spot 153.30 TL -> 1 Kontrat = 4,027.80 TL (Midas ile kuruşu kuruşuna)
        "THYAO": 0.2000,   # Spot 310.00 TL -> 1 Kontrat = 6,200.00 TL
        "TAVHL": 0.2000,
        
        # BIST Sanayi, Enerji, Otomotiv & Holding
        "TUPRS": 0.2000,   # Spot 380.00 TL -> 1 Kontrat = 7,600.00 TL
        "EREGL": 0.2000,
        "ASELS": 0.2000,
        "FROTO": 0.2000,
        "BIMAS": 0.18105,  # Spot 411.50 TL -> 1 Kontrat = 7,450.00 TL (Midas ile kuruşu kuruşuna)
        "KCHOL": 0.2000,
        "SAHOL": 0.2000,
        "SISE":  0.2000,
        "PETKM": 0.2200,
        "EKGYO": 0.2200,
        "ENKAI": 0.2000,
        "TOASO": 0.2000,
        "KOZAL": 0.2200,
        "KRDMD": 0.2200,
        "ASTOR": 0.2200,
        "KONTR": 0.2500,
        
        # Endeks Kontratı
        "F_XU030": 0.1650,
        "DEFAULT": 0.2100  # Liste dışı hisseler için Takasbank standart teminat oranı (%21)
    }

    # Geriye dönük uyumluluk için referans maktu sözlük
    TAKASBANK_SPAN_MARGINS = {
        "THYAO": 6200.0,
        "ISCTR": 251.0,
        "AKBNK": 1100.0,
        "GARAN": 2400.0,
        "EREGL": 880.0,
        "ASELS": 890.0,
        "FROTO": 24500.0,
        "BIMAS": 9200.0,
        "KCHOL": 3800.0,
        "TUPRS": 7600.0,
        "SAHOL": 1850.0,
        "EKGYO": 420.0,
        "SISE": 890.0,
        "PGSUS": 4027.8,
        "F_XU030": 16500.0
    }

    @classmethod
    def get_span_margin_per_contract(cls, ticker: str, spot_price: float) -> float:
        """
        Takasbank & Midas dinamik SPAN teminatını güncel spot fiyata göre anlık hesaplar.
        1 Kontrat = 100 Hisse * Spot Fiyat * Teminat Oranı
        """
        clean = ticker.replace(".IS", "").strip().upper()
        ratio = cls.SPAN_MARGIN_RATIOS.get(clean, cls.SPAN_MARGIN_RATIOS["DEFAULT"])
        
        # Endeks 30 için çarpan 10'dur
        if clean in ["F_XU030", "XU030"]:
            return round(spot_price * 10.0 * ratio, 2)
            
        contract_nominal = max(0.1, spot_price) * 100.0
        return round(contract_nominal * ratio, 2)

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
        self.initial_margin_ratio = 0.20

    # =========================================================================
    # 1. VADELİ İŞLEM (FUTURES) COST-OF-CARRY & ARBİTRAJ FİYATLAMASI
    # =========================================================================
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

    def calculate_implied_repo_rate(
        self,
        spot_price: float,
        futures_market_price: float,
        days_to_expiry: int = 30,
        dividend_yield_annual: float = 0.02
    ) -> dict:
        """
        Piyasadaki vadeli kontrat fiyatından zımni repo (taşıma) oranını tersine çözer.
        r_implied = ((F / S) - 1) * (365 / days) + q
        """
        if spot_price <= 0 or days_to_expiry <= 0:
            return {"implied_repo_rate_pct": 0.0, "arbitrage_signal": "NÖTR"}
        
        implied_r = ((futures_market_price / spot_price) - 1.0) * (365.0 / days_to_expiry) + dividend_yield_annual
        theo_p = self.calculate_theoretical_futures_price(spot_price, days_to_expiry, dividend_yield_annual)
        basis = futures_market_price - spot_price
        basis_pct = (basis / spot_price) * 100.0

        if implied_r > self.overnight_interest_annual + 0.05:
            arb_signal = "🔥 TAŞIMA ARBİTRAJI (Cash & Carry: Spot Al, Vadeli Sat)"
        elif implied_r < self.overnight_interest_annual - 0.08:
            arb_signal = "⚡ TERS TAŞIMA ARBİTRAJI (Reverse Cash & Carry: Vadeli Al, Spot Sat)"
        else:
            arb_signal = "✅ PİYASA DENGEDE (Adil Fiyatlama)"

        return {
            "spot_price": spot_price,
            "futures_market_price": futures_market_price,
            "theoretical_price": theo_p,
            "basis_try": round(basis, 2),
            "basis_pct": round(basis_pct, 2),
            "implied_repo_rate_pct": round(implied_r * 100.0, 2),
            "risk_free_rate_pct": round(self.overnight_interest_annual * 100.0, 2),
            "arbitrage_signal": arb_signal
        }

    # =========================================================================
    # 2. BLACK-SCHOLES-MERTON (BSM) OPSİYON FİYATLAMA & GREEKS DUYARLILIKLARI
    # =========================================================================
    def calculate_bsm_option_greeks(
        self,
        spot: float,
        strike: float,
        days_to_expiry: float,
        volatility: float,
        risk_free_rate: float = None,
        dividend_yield: float = 0.02
    ) -> dict:
        """
        Avrupa Tipi Borsa İstanbul Pay ve Endeks Opsiyonları için Black-Scholes-Merton (1973) 
        analitik fiyatlama formülü ve 1. / 2. Derece Duyarlılıkları (Greeks) hesaplar.
        """
        r = self.overnight_interest_annual if risk_free_rate is None else risk_free_rate
        q = dividend_yield
        T = max(1e-4, days_to_expiry / 365.0)
        sigma = max(1e-4, volatility)

        d1 = (np.log(spot / strike) + (r - q + 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        norm_d1 = float(stats.norm.cdf(d1))
        norm_d2 = float(stats.norm.cdf(d2))
        norm_neg_d1 = float(stats.norm.cdf(-d1))
        norm_neg_d2 = float(stats.norm.cdf(-d2))
        pdf_d1 = float(stats.norm.pdf(d1))

        exp_qT = np.exp(-q * T)
        exp_rT = np.exp(-r * T)

        # Teorik Fiyatlar
        call_price = float(spot * exp_qT * norm_d1 - strike * exp_rT * norm_d2)
        put_price = float(strike * exp_rT * norm_neg_d2 - spot * exp_qT * norm_neg_d1)

        # 1. Derece Greeks:
        call_delta = float(exp_qT * norm_d1)
        put_delta = float(-exp_qT * norm_neg_d1)

        gamma = float((exp_qT * pdf_d1) / (spot * sigma * np.sqrt(T)))
        vega = float((spot * exp_qT * np.sqrt(T) * pdf_d1) / 100.0) # 1% vol değişimine duyarlılık

        # Theta (1 Günlük Zaman Erimesi)
        theta_call_annual = -(spot * sigma * exp_qT * pdf_d1) / (2.0 * np.sqrt(T)) - r * strike * exp_rT * norm_d2 + q * spot * exp_qT * norm_d1
        theta_put_annual = -(spot * sigma * exp_qT * pdf_d1) / (2.0 * np.sqrt(T)) + r * strike * exp_rT * norm_neg_d2 - q * spot * exp_qT * norm_neg_d1
        theta_call_daily = float(theta_call_annual / 365.0)
        theta_put_daily = float(theta_put_annual / 365.0)

        # Rho (Faiz Duyarlılığı - %1 faiz değişimi)
        call_rho = float((strike * T * exp_rT * norm_d2) / 100.0)
        put_rho = float((-strike * T * exp_rT * norm_neg_d2) / 100.0)

        # 2. Derece İleri Greeks:
        # Vanna = d(Delta) / d(Vol) = -exp(-qT) * pdf(d1) * d2 / sigma
        vanna = float(-exp_qT * pdf_d1 * (d2 / sigma))
        # Volga (Vomma) = d(Vega) / d(Vol) = Vega * d1 * d2 / sigma
        volga = float(vega * (d1 * d2) / sigma)

        return {
            "spot": spot,
            "strike": strike,
            "days_to_expiry": days_to_expiry,
            "volatility_pct": round(sigma * 100.0, 2),
            "call_price": round(max(0.0, call_price), 3),
            "put_price": round(max(0.0, put_price), 3),
            "call_delta": round(call_delta, 4),
            "put_delta": round(put_delta, 4),
            "gamma": round(gamma, 6),
            "vega": round(vega, 4),
            "call_theta_daily": round(theta_call_daily, 4),
            "put_theta_daily": round(theta_put_daily, 4),
            "call_rho": round(call_rho, 4),
            "put_rho": round(put_rho, 4),
            "vanna": round(vanna, 6),
            "volga": round(volga, 6)
        }

    # =========================================================================
    # 3. ZIMNİ OYNAKLIK (IMPLIED VOLATILITY) KÖK BULUCU
    # =========================================================================
    def solve_implied_volatility(
        self,
        market_option_price: float,
        spot: float,
        strike: float,
        days_to_expiry: float,
        option_type: str = "CALL",
        dividend_yield: float = 0.02
    ) -> float:
        """
        Piyasa opsiyon fiyatından Zımni Oynaklığı (Implied Volatility - IV) Brent kök bulucu ile çözer.
        """
        if market_option_price <= 0:
            return 0.0

        is_call = option_type.upper().startswith("C")

        def objective(vol):
            res = self.calculate_bsm_option_greeks(
                spot=spot,
                strike=strike,
                days_to_expiry=days_to_expiry,
                volatility=vol,
                dividend_yield=dividend_yield
            )
            theo = res["call_price"] if is_call else res["put_price"]
            return theo - market_option_price

        try:
            # Brent kök bulucu ile %1 ile %300 volatilite aralığında arama
            iv = brentq(objective, 0.01, 3.00, xtol=1e-5, maxiter=100)
            return round(float(iv * 100.0), 2)
        except Exception:
            return 30.0

    # =========================================================================
    # 4. DİNAMİK PORTFÖY DELTA-HEDGE & PİYASA NÖTRLEŞME MATRİSİ
    # =========================================================================
    def calculate_portfolio_delta_hedge(
        self,
        spot_positions: dict[str, dict], # {"THYAO": {"lots": 2000, "price": 310.0, "beta": 1.25}}
        index_spot_price: float = 11000.0, # XU030
        days_to_expiry: int = 30
    ) -> dict:
        """
        Hisse portföyünü piyasa düşüşlerine karşı tam nötrleştirmek (Delta / Beta Hedging) için
        kaç adet BIST 30 Vadeli Kontratı (F_XU030) veya Tek Pay Vadeli Kontratı satılması (Short) gerektiğini hesaplar.
        """
        total_port_value = 0.0
        weighted_portfolio_beta = 0.0

        single_stock_hedges = {}
        for ticker, pos in spot_positions.items():
            clean = ticker.replace(".IS", "").strip().upper()
            lots = pos.get("lots", 0)
            px = pos.get("price", 10.0)
            beta = pos.get("beta", 1.0)
            
            stock_val = lots * px
            total_port_value += stock_val
            weighted_portfolio_beta += stock_val * beta

            # Tek hisse bazında vadeli kontrat sayısı (1 kontrat = 100 hisse)
            single_stock_contracts = math.ceil(lots / self.contract_multiplier)
            single_stock_hedges[clean] = {
                "lots": lots,
                "value_try": round(stock_val, 2),
                "required_short_contracts": single_stock_contracts,
                "contract_code": f"F_{clean}"
            }

        port_beta = (weighted_portfolio_beta / total_port_value) if total_port_value > 0 else 1.0
        
        # BIST 30 Endeks Vadeli Kontratı (F_XU030 - Sözleşme büyüklüğü: 10)
        index_contract_multiplier = 10
        index_contract_val = index_spot_price * index_contract_multiplier # Örn: 11.000 * 10 = 110.000 TL
        
        # Optimal Beta Hedge Formülü: N_hedge = (Portfolio_Value * Beta_Portfolio) / Value_Index_Contract
        index_contracts_needed = math.ceil((total_port_value * port_beta) / index_contract_val)
        required_index_margin = index_contracts_needed * self.TAKASBANK_SPAN_MARGINS.get("F_XU030", 16500.0)

        return {
            "total_portfolio_value_try": round(total_port_value, 2),
            "portfolio_beta": round(port_beta, 2),
            "index_spot_price": index_spot_price,
            "index_contract_value_try": round(index_contract_val, 2),
            "required_f_xu030_short_contracts": index_contracts_needed,
            "required_index_hedge_margin_try": round(required_index_margin, 2),
            "single_stock_hedges": single_stock_hedges,
            "hedge_efficiency": "Tam Beta Koruma (Market Neutral %100)"
        }

    # =========================================================================
    # 5. VİOP KALDIRAÇLI POZİSYON BOYUTLANDIRMA & SİMÜLASYON
    # =========================================================================
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
        span_margin_per_contract = self.get_span_margin_per_contract(clean, spot_price)

        contract_value = spot_price * self.contract_multiplier
        target_notional = capital * (allocation_pct / 100.0) * leverage
        num_contracts = max(1, int(target_notional / contract_value))

        required_margin = num_contracts * span_margin_per_contract
        
        while required_margin > capital * 0.90 and num_contracts > 1:
            num_contracts -= 1
            required_margin = num_contracts * span_margin_per_contract

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
        direction: str,
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

        entry_commission = notional_entry * self.commission_rate
        current_cash -= entry_commission

        peak_price = entry_price
        trough_price = entry_price
        total_interest_earned = 0.0

        exit_price = entry_price
        exit_date = None
        exit_reason = "Vade Sonu Kapanış (Time Horizon)"
        days_held = 0

        for day_i, row in forward_df.iterrows():
            days_held += 1
            current_close = float(row["close"])
            current_high = float(row.get("high", current_close))
            current_low = float(row.get("low", current_close))
            current_date = row.get("timestamps", f"Gün-{day_i}")

            daily_interest = (current_cash + required_margin) * self.daily_interest_rate
            total_interest_earned += daily_interest
            current_cash += daily_interest

            if direction == "LONG":
                if current_high > peak_price:
                    peak_price = current_high

                if current_low <= entry_price * (1.0 - stop_loss_pct / 100.0):
                    exit_price = entry_price * (1.0 - stop_loss_pct / 100.0)
                    exit_date = current_date
                    exit_reason = f"🔴 Stop-Loss (%{stop_loss_pct:.1f})"
                    break

                if current_high >= entry_price * (1.0 + take_profit_pct / 100.0):
                    exit_price = entry_price * (1.0 + take_profit_pct / 100.0)
                    exit_date = current_date
                    exit_reason = f"🎯 Kâr Al (%{take_profit_pct:.1f})"
                    break

                trailing_level = peak_price * (1.0 - trailing_stop_pct / 100.0)
                if peak_price >= entry_price * 1.015 and current_low <= trailing_level:
                    exit_price = trailing_level
                    exit_date = current_date
                    gain_pct = ((exit_price - entry_price) / entry_price) * 100.0
                    exit_reason = f"🟢 İz Süren Stop (Trailing %{gain_pct:+.1f})"
                    break

            elif direction == "SHORT":
                if current_low < trough_price:
                    trough_price = current_low

                if current_high >= entry_price * (1.0 + stop_loss_pct / 100.0):
                    exit_price = entry_price * (1.0 + stop_loss_pct / 100.0)
                    exit_date = current_date
                    exit_reason = f"🔴 Short Stop-Loss (%{stop_loss_pct:.1f})"
                    break

                if current_low <= entry_price * (1.0 - take_profit_pct / 100.0):
                    exit_price = entry_price * (1.0 - take_profit_pct / 100.0)
                    exit_date = current_date
                    exit_reason = f"🎯 Short Kâr Al (%{take_profit_pct:.1f})"
                    break

                inverted_trailing_level = trough_price * (1.0 + trailing_stop_pct / 100.0)
                if trough_price <= entry_price * 0.985 and current_high >= inverted_trailing_level:
                    exit_price = inverted_trailing_level
                    exit_date = current_date
                    gain_pct = ((entry_price - exit_price) / entry_price) * 100.0
                    exit_reason = f"🟢 Ters İz Süren Stop (Short Trailing %{gain_pct:+.1f})"
                    break

            if days_held >= max_hold_days:
                exit_price = current_close
                exit_date = current_date
                exit_reason = f"⏳ Vade Sonu Kapanış ({max_hold_days} Gün)"
                break

        if exit_date is None and len(forward_df) > 0:
            exit_price = float(forward_df["close"].iloc[-1])
            exit_date = forward_df["timestamps"].iloc[-1] if "timestamps" in forward_df else "Son Gün"

        notional_exit = contracts * (exit_price * self.contract_multiplier)
        exit_commission = notional_exit * self.commission_rate
        current_cash -= exit_commission

        if direction == "LONG":
            gross_trade_pnl = (exit_price - entry_price) * self.contract_multiplier * contracts
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
        else:
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
    print("⚡ BIST VİOP & BSM Greeks Motoru Test Ediliyor...")
    engine = BistViopEngine()
    
    # 1. BSM Greeks Testi
    greeks = engine.calculate_bsm_option_greeks(spot=12.38, strike=13.00, days_to_expiry=30, volatility=0.32)
    print("\n" + "="*80)
    print("📊 BSM OPSİYON FİYATLAMA VE GREEKS RAPORU (ISCTR Örnek Opsiyonu)")
    print("="*80)
    for k, v in greeks.items():
        print(f"  • {k:22}: {v}")

    # 2. Delta Hedge Testi
    sample_port = {
        "THYAO": {"lots": 1000, "price": 310.0, "beta": 1.25},
        "ISCTR": {"lots": 20000, "price": 12.38, "beta": 1.15},
        "AKBNK": {"lots": 10000, "price": 55.0, "beta": 1.10}
    }
    hedge_plan = engine.calculate_portfolio_delta_hedge(sample_port, index_spot_price=11200.0)
    print("\n" + "="*80)
    print("🛡️ DİNAMİK BIST 30 ENDEKS HEDGE PLANI")
    print("="*80)
    print(f"  • Toplam Portföy Değeri   : {hedge_plan['total_portfolio_value_try']:,.2f} TL")
    print(f"  • Portföy Ağırlıklı Beta : {hedge_plan['portfolio_beta']}")
    print(f"  • Açılacak F_XU030 Short : {hedge_plan['required_f_xu030_short_contracts']} Kontrat")
    print(f"  • Gereken Teminat        : {hedge_plan['required_index_hedge_margin_try']:,.2f} TL")
    print("="*80 + "\n")
