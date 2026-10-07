import numpy as np
import pandas as pd
import pytest

from bist_quant.backtest_engine import (
    LEAK_CLEAN,
    LEAK_DETECTED,
    LEAK_NOT_APPLICABLE,
    LEAK_UNKNOWN,
    LONG,
    SHORT,
    CostModel,
    ExitRules,
    compute_training_cutoff,
    idle_cash_fraction,
    leakage_status,
    net_return_pct,
    performance_from_equity,
    simulate_exit,
)

NO_COSTS = CostModel(commission_rate=0.0, slippage_rate=0.0)


def _bars(*ohlc):
    return pd.DataFrame(ohlc, columns=["open", "high", "low", "close"])


def _rules(**overrides):
    base = dict(initial_stop_pct=5.0, trail_distance_pct=5.0, horizon_days=10, use_trailing_stop=True, take_profit_pct=None)
    return ExitRules(**{**base, **overrides})


# --- çıkış simülasyonu -------------------------------------------------------

def test_long_stop_fills_at_stop_price_when_touched_intraday():
    exit_ = simulate_exit(_bars((99, 100, 94, 96)), entry_price=100.0, direction=LONG, rules=_rules())

    assert exit_.price == pytest.approx(95.0)
    assert exit_.day == 1


def test_long_stop_fills_at_open_when_price_gaps_through_it():
    exit_ = simulate_exit(_bars((90, 92, 88, 91)), entry_price=100.0, direction=LONG, rules=_rules())

    assert exit_.price == pytest.approx(90.0)


def test_trailing_stop_is_not_raised_by_the_same_bar_that_would_hit_it():
    # Bar içinde önce tepe mi dip mi geldiği bilinemez: bar başındaki stop (95) geçerlidir.
    bars = _bars((100, 120, 96, 118), (118, 119, 117, 118))

    exit_ = simulate_exit(bars, entry_price=100.0, direction=LONG, rules=_rules(horizon_days=50))

    assert exit_.day == 2
    assert exit_.reason.startswith("Maksimum")


def test_trailing_stop_follows_the_peak_of_earlier_bars():
    bars = _bars((100, 110, 100, 109), (108, 108, 104, 105))

    exit_ = simulate_exit(bars, entry_price=100.0, direction=LONG, rules=_rules())

    assert exit_.day == 2
    assert exit_.price == pytest.approx(104.5)


def test_fixed_take_profit_is_used_only_without_trailing_stop():
    bars = _bars((100, 109, 99, 108))

    fixed = simulate_exit(bars, 100.0, LONG, _rules(use_trailing_stop=False, take_profit_pct=8.0))
    trailing = simulate_exit(bars, 100.0, LONG, _rules(use_trailing_stop=True, take_profit_pct=8.0))

    assert fixed.price == pytest.approx(108.0)
    assert trailing.price == pytest.approx(108.0) and trailing.reason.startswith("Maksimum")


def test_stop_wins_when_stop_and_target_are_hit_in_the_same_bar():
    bars = _bars((100, 109, 94, 100))

    exit_ = simulate_exit(bars, 100.0, LONG, _rules(use_trailing_stop=False, take_profit_pct=8.0))

    assert exit_.price == pytest.approx(95.0)


def test_losing_position_is_closed_at_horizon():
    bars = _bars((100, 101, 98, 99), (99, 100, 98, 99))

    exit_ = simulate_exit(bars, 100.0, LONG, _rules(horizon_days=2))

    assert (exit_.day, exit_.price) == (2, 99.0)


def test_short_stop_fills_at_open_on_gap_up():
    exit_ = simulate_exit(_bars((110, 112, 109, 111)), entry_price=100.0, direction=SHORT, rules=_rules())

    assert exit_.price == pytest.approx(110.0)


def test_short_trailing_stop_follows_the_trough():
    bars = _bars((100, 100, 90, 91), (92, 95, 91, 94))

    exit_ = simulate_exit(bars, 100.0, SHORT, _rules())

    assert exit_.price == pytest.approx(94.5)


def test_empty_forward_window_is_rejected():
    with pytest.raises(ValueError):
        simulate_exit(_bars(), 100.0, LONG, _rules())


# --- maliyet -----------------------------------------------------------------

def test_without_costs_net_return_equals_price_change():
    assert net_return_pct(100.0, 110.0, LONG, leverage=1.0, costs=NO_COSTS) == pytest.approx(10.0)
    assert net_return_pct(100.0, 90.0, SHORT, leverage=2.0, costs=NO_COSTS) == pytest.approx(20.0)


def test_flat_round_trip_loses_commission_and_slippage():
    costs = CostModel(commission_rate=0.001, slippage_rate=0.0005)

    result = net_return_pct(100.0, 100.0, LONG, leverage=1.0, costs=costs)

    assert result == pytest.approx(-0.30, abs=0.01)


def test_costs_scale_with_leverage():
    costs = CostModel(commission_rate=0.001, slippage_rate=0.0005)

    assert net_return_pct(100.0, 100.0, SHORT, 3.0, costs) == pytest.approx(3 * net_return_pct(100.0, 100.0, SHORT, 1.0, costs))


# --- nemalandırma ------------------------------------------------------------

def test_spot_position_leaves_no_idle_cash():
    assert idle_cash_fraction(use_viop=False, leverage=1.0, margin_ratio=0.2) == 0.0


def test_viop_position_earns_interest_only_on_cash_not_posted_as_margin():
    assert idle_cash_fraction(use_viop=True, leverage=1.5, margin_ratio=0.2) == pytest.approx(0.7)


def test_idle_cash_is_never_negative():
    assert idle_cash_fraction(use_viop=True, leverage=5.0, margin_ratio=0.25) == 0.0


# --- performans --------------------------------------------------------------

def test_max_drawdown_is_measured_on_the_daily_curve():
    equity = pd.Series([100.0, 120.0, 90.0, 130.0])

    perf = performance_from_equity(equity, risk_free_annual=0.0)

    assert perf["max_drawdown_pct"] == pytest.approx(25.0)
    assert perf["total_return_pct"] == pytest.approx(30.0)


def test_flat_equity_has_zero_sharpe():
    perf = performance_from_equity(pd.Series([100.0] * 30), risk_free_annual=0.0)

    assert perf["sharpe_ratio"] == 0.0
    assert perf["max_drawdown_pct"] == 0.0


def test_sharpe_is_measured_against_the_risk_free_rate():
    daily_returns = np.tile([0.011, -0.009], 250)  # günlük ortalama %0.1 -> yıllık ~%28
    equity = pd.Series(100.0 * np.cumprod(1 + daily_returns))

    assert performance_from_equity(equity, risk_free_annual=0.0)["sharpe_ratio"] > 0
    assert performance_from_equity(equity, risk_free_annual=0.45)["sharpe_ratio"] < 0


# --- veri sızıntısı ----------------------------------------------------------

def test_no_model_means_leakage_does_not_apply():
    assert leakage_status(uses_model=False, training_cutoff=None, test_start="2026-01-05")[0] == LEAK_NOT_APPLICABLE


def test_unknown_cutoff_is_not_reported_as_clean():
    assert leakage_status(uses_model=True, training_cutoff=None, test_start="2026-01-05")[0] == LEAK_UNKNOWN


def test_cutoff_before_test_window_is_clean():
    assert leakage_status(True, "2025-12-31", "2026-01-05")[0] == LEAK_CLEAN


def test_cutoff_inside_test_window_is_a_leak():
    assert leakage_status(True, "2026-03-01", "2026-01-05")[0] == LEAK_DETECTED
    assert leakage_status(True, "2026-01-05", "2026-01-05")[0] == LEAK_DETECTED


def test_training_cutoff_is_the_last_date_of_the_time_sorted_training_share():
    # Kronos eğitim kümesi tüm satırları tarihe göre sıralar ve ilk train_ratio payını alır.
    timestamps = pd.Series(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"] * 2)

    assert compute_training_cutoff(timestamps, train_ratio=0.6) == "2024-01-03"
