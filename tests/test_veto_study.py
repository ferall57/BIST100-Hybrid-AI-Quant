import numpy as np
import pandas as pd
import pytest

from bist_quant.bist_veto_study import (
    CONDITIONS,
    evaluation_points,
    forward_return_pct,
    run_veto_study,
    summarise_condition,
)


def _frame(closes, start="2025-01-01"):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "timestamps": pd.bdate_range(start, periods=len(closes)),
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": np.full(len(closes), 1000.0),
    })


def test_evaluation_points_leave_room_for_history_and_longest_horizon():
    points = evaluation_points(n_bars=200, first=120, step=5, max_horizon=15)
    assert points[0] == 120
    assert points[-1] + 15 <= 199
    assert all(b - a == 5 for a, b in zip(points, points[1:]))


def test_forward_return_uses_close_at_t_and_t_plus_h():
    df = _frame([100, 101, 102, 110])
    assert forward_return_pct(df, 0, 3) == pytest.approx(10.0)
    assert forward_return_pct(df, 2, 3) is None


def test_conditions_include_live_veto_and_each_part_alone():
    assert set(CONDITIONS) == {"veto", "mc_low", "cmf_out"}
    assert CONDITIONS["veto"](30.0, -0.2) is True
    assert CONDITIONS["veto"](30.0, 0.0) is False
    assert CONDITIONS["veto"](50.0, -0.2) is False
    assert CONDITIONS["mc_low"](30.0, 0.0) is True
    assert CONDITIONS["cmf_out"](50.0, -0.2) is True


def test_summary_compares_fired_and_quiet_rows():
    rows = [
        {"ticker": "A", "veto": True, "fwd_5": -4.0, "excess_5": -3.0},
        {"ticker": "A", "veto": False, "fwd_5": 2.0, "excess_5": 1.0},
        {"ticker": "B", "veto": True, "fwd_5": -2.0, "excess_5": -1.0},
        {"ticker": "B", "veto": False, "fwd_5": 0.0, "excess_5": 0.0},
        {"ticker": "B", "veto": False, "fwd_5": None, "excess_5": None},
    ]
    s = summarise_condition(rows, "veto", horizon=5)
    assert s["fired"] == 2
    assert s["quiet"] == 2
    assert s["fire_rate_pct"] == pytest.approx(50.0)
    assert s["mean_fired"] == pytest.approx(-3.0)
    assert s["mean_quiet"] == pytest.approx(1.0)
    assert s["mean_excess_fired"] == pytest.approx(-2.0)
    assert s["mean_excess_quiet"] == pytest.approx(0.5)
    # both tickers: fired mean below quiet mean
    assert s["tickers_fired_worse"] == 2
    assert s["tickers_compared"] == 2


def test_summary_without_fired_rows_reports_none():
    rows = [{"ticker": "A", "veto": False, "fwd_5": 1.0, "excess_5": 0.5}]
    s = summarise_condition(rows, "veto", horizon=5)
    assert s["fired"] == 0
    assert s["mean_fired"] is None
    assert s["p_value"] is None


def test_run_study_uses_only_data_up_to_each_point():
    seen_lengths = []

    def fake_mc(df):
        seen_lengths.append(len(df))
        return 30.0  # always low

    def fake_cmf(df):
        return -0.2  # always outflow

    stock = _frame(np.linspace(100, 80, 160))
    index = _frame(np.full(160, 1000.0))
    rows = run_veto_study({"TEST": stock}, index, horizons=(5,), step=10, first=120,
                          mc_fn=fake_mc, cmf_fn=fake_cmf)

    assert seen_lengths[0] == 121  # bars 0..120 inclusive, nothing after
    assert all(r["veto"] for r in rows)
    first = rows[0]
    assert first["fwd_5"] == pytest.approx((stock.close[125] / stock.close[120] - 1) * 100)
    assert first["excess_5"] == pytest.approx(first["fwd_5"])


def test_study_measures_the_same_condition_the_committee_reports():
    from hybrid_agents.bist_committee import risk_condition_met
    for prob, cmf in [(30.0, -0.2), (30.0, 0.0), (50.0, -0.2)]:
        assert CONDITIONS["veto"](prob, cmf) == risk_condition_met({"prob_positive": prob}, {"cmf_20": cmf})
