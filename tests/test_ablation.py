import numpy as np
import pandas as pd
import pytest

from bist_quant.bist_ablation import paired_vs_baseline, summarise_variant
from bist_quant.bist_index_gatekeeper import compute_market_regime
from bist_quant.bist_price_action import ict_setup_score
from bist_quant.entry_filters import FILTER_ICT, FILTER_INDEX_GATE, FILTER_MONEY_FLOW, EntryFilters


def _index(closes, start="2023-01-02"):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "timestamps": pd.date_range(start, periods=len(closes), freq="B").strftime("%Y-%m-%d"),
        "open": closes, "high": closes * 1.005, "low": closes * 0.995, "close": closes, "volume": 1_000_000,
    })


RISING = 100.0 * np.cumprod(np.full(300, 1.004))
FALLING = 100.0 * np.cumprod(np.full(300, 0.996))


# --- endeks rejimi -----------------------------------------------------------

def test_rising_index_allows_long_entries():
    regime = compute_market_regime(_index(RISING))

    assert regime["regime"] == "BULL_REGIME"
    assert regime["is_long_allowed"] is True


def test_falling_index_blocks_long_entries():
    regime = compute_market_regime(_index(FALLING))

    assert regime["regime"] == "BEAR_REGIME"
    assert regime["is_long_allowed"] is False


def test_short_index_history_is_neutral():
    regime = compute_market_regime(_index(RISING[:30]))

    assert regime["regime"] == "NEUTRAL_CHOP"
    assert regime["is_long_allowed"] is True


# --- ICT puanı ---------------------------------------------------------------

def test_ict_score_adds_bullish_setups_and_penalises_the_bsl_trap():
    assert ict_setup_score({"ssl_swept": True}, {"nearest_fvg": {"type": "BULLISH_FVG"}}, {"is_break_retest": True}) == 40.0
    assert ict_setup_score({"bsl_swept": True}, {"nearest_fvg": None}, {}) == -20.0
    assert ict_setup_score({}, {}, {}) == 0.0


# --- giriş filtreleri --------------------------------------------------------

def test_unknown_filter_name_is_rejected():
    with pytest.raises(ValueError, match="bilinmeyen"):
        EntryFilters(("sihir",))


def test_index_gate_requires_index_history():
    with pytest.raises(ValueError, match="XU100"):
        EntryFilters((FILTER_INDEX_GATE,))


def test_index_gate_only_sees_index_bars_before_the_entry_day():
    history = _index(np.concatenate([RISING, RISING[-1] * np.cumprod(np.full(120, 0.99))]))
    filters = EntryFilters((FILTER_INDEX_GATE,), index_history=history)
    last_rising_day = history["timestamps"].iloc[len(RISING) - 1]
    final_day = history["timestamps"].iloc[-1]

    assert filters.blocking_filter(_index(RISING), as_of_day=last_rising_day) is None
    assert filters.blocking_filter(_index(RISING), as_of_day=final_day) == FILTER_INDEX_GATE


def test_money_flow_filter_follows_where_bars_close_within_their_range():
    closes = np.linspace(100, 130, 40)
    accumulation = pd.DataFrame({"open": closes - 1, "high": closes, "low": closes - 2, "close": closes, "volume": 1_000_000})
    distribution = pd.DataFrame({"open": closes[::-1] + 1, "high": closes[::-1] + 2, "low": closes[::-1],
                                 "close": closes[::-1], "volume": 1_000_000})
    filters = EntryFilters((FILTER_MONEY_FLOW,))

    assert filters.blocking_filter(accumulation, as_of_day="2026-01-05") is None
    assert filters.blocking_filter(distribution, as_of_day="2026-01-05") == FILTER_MONEY_FLOW


def test_ict_filter_requires_a_bullish_setup():
    class _Engine:
        def __init__(self, ssl_swept):
            self._ssl_swept = ssl_swept

        def detect_liquidity_sweeps(self, df):
            return {"ssl_swept": self._ssl_swept, "bsl_swept": False}

        def detect_fair_value_gaps(self, df):
            return {"nearest_fvg": None}

        def detect_break_and_retest(self, df):
            return {"is_break_retest": False}

    filters = EntryFilters((FILTER_ICT,))
    hist = _index(RISING)

    filters._price_action = _Engine(ssl_swept=True)
    assert filters.blocking_filter(hist, as_of_day="2026-01-05") is None

    filters._price_action = _Engine(ssl_swept=False)
    assert filters.blocking_filter(hist, as_of_day="2026-01-05") == FILTER_ICT


def test_no_filters_never_block():
    assert EntryFilters(()).blocking_filter(_index(RISING), as_of_day="2026-01-05") is None


# --- özetleme ----------------------------------------------------------------

def _rows(returns, bnh=5.0, trades=4):
    return [{"ticker": f"T{i}.IS", "total_return_pct": r, "bnh_return_pct": bnh, "alpha": r - bnh,
             "total_trades": trades, "max_drawdown": 10.0, "sharpe_ratio": 0.1, "blocked_entries": {}}
            for i, r in enumerate(returns)]


def test_variant_summary_reports_medians_and_share_beating_buy_and_hold():
    summary = summarise_variant("Deneme", _rows([10.0, 2.0, -4.0]))

    assert summary["variant"] == "Deneme"
    assert summary["median_return_pct"] == 2.0
    assert summary["median_alpha"] == -3.0
    assert summary["share_beating_bnh_pct"] == pytest.approx(100 / 3)
    assert summary["total_trades"] == 12


def test_paired_comparison_matches_tickers_and_counts_improvements():
    baseline = _rows([1.0, 2.0, 3.0, 4.0])
    variant = _rows([3.0, 5.0, 3.0, 2.0])

    paired = paired_vs_baseline(baseline, variant)

    assert (paired["improved"], paired["worsened"]) == (2, 1)
    assert paired["median_diff_vs_baseline"] == pytest.approx(1.0)
    assert 0.0 < paired["p_value"] <= 1.0


def test_paired_comparison_ignores_tickers_missing_from_either_side():
    paired = paired_vs_baseline(_rows([1.0, 2.0]), _rows([5.0]))

    assert (paired["improved"], paired["worsened"]) == (1, 0)
