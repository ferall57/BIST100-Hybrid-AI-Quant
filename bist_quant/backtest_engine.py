"""Backtest çekirdeği: çıkış simülasyonu, maliyet modeli ve performans ölçüleri (saf fonksiyonlar)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
LONG = "LONG"
SHORT = "SHORT"

REASON_STOP = "Stop-Loss"
REASON_GAP_STOP = "Stop-Loss (Boşluklu Açılış)"
REASON_TRAILING = "İz Süren Stop"
REASON_TAKE_PROFIT = "Take-Profit"
REASON_HORIZON = "Vade Sonu Zararda Kapanış"
REASON_MAX_HOLD = "Maksimum Vade Sonu"

LEAK_NOT_APPLICABLE = "MODEL_YOK"
LEAK_CLEAN = "TEMIZ"
LEAK_UNKNOWN = "BILINMIYOR"
LEAK_DETECTED = "SIZINTI"


@dataclass(frozen=True)
class CostModel:
    """İşlem maliyeti varsayımları: her bacak için komisyon ve aleyhe fiyat kayması (oran)."""
    commission_rate: float = 0.0010
    slippage_rate: float = 0.0005


@dataclass(frozen=True)
class ExitRules:
    initial_stop_pct: float
    trail_distance_pct: float
    horizon_days: int
    use_trailing_stop: bool = True
    take_profit_pct: float | None = None


@dataclass(frozen=True)
class TradeExit:
    price: float
    day: int
    reason: str


def _direction_sign(direction: str) -> int:
    return 1 if direction == LONG else -1


def _stop_level(entry_price: float, best_price: float, sign: int, rules: ExitRules) -> float:
    initial = entry_price * (1.0 - sign * rules.initial_stop_pct / 100.0)
    if not rules.use_trailing_stop:
        return initial
    trailing = best_price * (1.0 - sign * rules.trail_distance_pct / 100.0)
    return max(initial, trailing) if sign == 1 else min(initial, trailing)


def simulate_exit(forward: pd.DataFrame, entry_price: float, direction: str, rules: ExitRules) -> TradeExit:
    """
    Girişten sonraki mumlarda pozisyonun nerede kapanacağını belirler.
    - Stop seviyesi yalnızca önceki mumların tepe/dibinden hesaplanır (bar içi sıra bilinemez).
    - Fiyat stop seviyesini boşlukla geçerse dolum açılış fiyatından yapılır.
    - Aynı mumda hem stop hem hedef görülürse önce stop çalışmış sayılır.
    """
    if forward.empty:
        raise ValueError("Çıkış simülasyonu için en az bir mum gerekir.")

    sign = _direction_sign(direction)
    fixed_target = (
        entry_price * (1.0 + sign * rules.take_profit_pct / 100.0)
        if rules.take_profit_pct is not None and not rules.use_trailing_stop
        else None
    )
    best_price = entry_price

    for day, bar in enumerate(forward.itertuples(index=False), 1):
        stop = _stop_level(entry_price, best_price, sign, rules)
        adverse, favorable = (bar.low, bar.high) if sign == 1 else (bar.high, bar.low)

        if (adverse - stop) * sign <= 0:
            if (bar.open - stop) * sign <= 0:
                return TradeExit(float(bar.open), day, REASON_GAP_STOP)
            in_profit = (stop - entry_price) * sign > 0
            return TradeExit(float(stop), day, REASON_TRAILING if in_profit else REASON_STOP)

        if fixed_target is not None and (favorable - fixed_target) * sign >= 0:
            gapped = (bar.open - fixed_target) * sign >= 0
            return TradeExit(float(bar.open if gapped else fixed_target), day, REASON_TAKE_PROFIT)

        if day >= rules.horizon_days and (bar.close - entry_price) * sign < 0:
            return TradeExit(float(bar.close), day, REASON_HORIZON)

        best_price = max(best_price, bar.high) if sign == 1 else min(best_price, bar.low)

    return TradeExit(float(forward["close"].iloc[-1]), len(forward), REASON_MAX_HOLD)


def gross_return_pct(entry_price: float, exit_price: float, direction: str, leverage: float) -> float:
    """Maliyetsiz, kaldıraçlı fiyat değişimi (tahsis edilen sermayenin yüzdesi)."""
    return _direction_sign(direction) * (exit_price - entry_price) / entry_price * leverage * 100.0


def net_return_pct(entry_price: float, exit_price: float, direction: str, leverage: float, costs: CostModel) -> float:
    """Komisyon ve kayma düşülmüş kaldıraçlı getiri (tahsis edilen sermayenin yüzdesi)."""
    sign = _direction_sign(direction)
    entry_fill = entry_price * (1.0 + sign * costs.slippage_rate)
    exit_fill = exit_price * (1.0 - sign * costs.slippage_rate)
    price_return = sign * (exit_fill - entry_fill) / entry_fill
    commission = costs.commission_rate * (1.0 + exit_fill / entry_fill)
    return (price_return - commission) * leverage * 100.0


def idle_cash_fraction(use_viop: bool, leverage: float, margin_ratio: float) -> float:
    """Pozisyondayken nemalanabilecek sermaye payı: yalnızca teminata bağlanmayan nakit."""
    if not use_viop:
        return 0.0
    return max(0.0, 1.0 - leverage * margin_ratio)


def daily_rate(annual_rate: float) -> float:
    """Yıllık bileşik oranın işlem günü başına karşılığı."""
    return (1.0 + annual_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


def performance_from_equity(equity: pd.Series, risk_free_annual: float) -> dict:
    """Günlük sermaye eğrisinden toplam getiri, maksimum çekilme ve risksiz faize göre Sharpe/Sortino."""
    values = equity.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(values)
    max_drawdown_pct = float(np.max((peaks - values) / peaks) * 100.0)
    total_return_pct = float((values[-1] / values[0] - 1.0) * 100.0)

    excess = values[1:] / values[:-1] - 1.0 - daily_rate(risk_free_annual)
    annualizer = np.sqrt(TRADING_DAYS_PER_YEAR)
    volatility = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    sharpe = float(np.mean(excess) / volatility * annualizer) if volatility > 1e-12 else 0.0
    downside = excess[excess < 0]
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 0.0
    sortino = float(np.mean(excess) / downside_dev * annualizer) if downside_dev > 1e-12 else 0.0

    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
    }


def compute_training_cutoff(timestamps: pd.Series, train_ratio: float) -> str:
    """
    Kronos ince ayar veri kümesinin eğitim payındaki son tarih (YYYY-MM-DD).
    Veri kümesi tüm satırları tarihe göre sıralar ve ilk train_ratio payını eğitime ayırır.
    """
    days = timestamps.astype(str).str[:10].sort_values(kind="stable").reset_index(drop=True)
    train_rows = int(len(days) * train_ratio)
    if train_rows < 1:
        raise ValueError("Eğitim payı boş: train_ratio veya veri uzunluğu yetersiz.")
    return str(days.iloc[train_rows - 1])


def leakage_status(uses_model: bool, training_cutoff: str | None, test_start: str) -> tuple[str, str]:
    """Modelin test dönemini eğitimde görmüş olma durumunu (kod, açıklama) olarak döndürür."""
    if not uses_model:
        return LEAK_NOT_APPLICABLE, "Öğrenen model kullanılmadı; tahminler yalnızca geçmiş mumlardan kural tabanlı üretildi."
    if not training_cutoff:
        return LEAK_UNKNOWN, ("Modelin eğitim kesim tarihi kayıtlı değil; model test dönemini görmüş olabilir. "
                              "Sonuçlar iyimser olabilir.")
    if str(training_cutoff)[:10] < str(test_start)[:10]:
        return LEAK_CLEAN, f"Model eğitimi {str(training_cutoff)[:10]} tarihinde bitiyor; test dönemi eğitim verisinin dışında."
    return LEAK_DETECTED, (f"VERİ SIZINTISI: model {str(training_cutoff)[:10]} tarihine kadar eğitildi, test "
                           f"{str(test_start)[:10]} tarihinde başlıyor. Model test dönemini gördü; sonuçlar geçersiz sayılmalı.")
