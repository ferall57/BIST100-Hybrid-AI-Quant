import pandas as pd
import pytest

from bist_quant.bist_decision_scorer import (
    first_holding_bar,
    level_problem,
    score_decision,
    summarize_scores,
    wilson_interval,
)


def _prices(closes, start="2026-09-01", highs=None, lows=None):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "timestamps": dates,
        "high": highs if highs is not None else closes,
        "low": lows if lows is not None else closes,
        "close": closes,
    })


def _record(verdict="AL", date="2026-09-01 15:00:00", spot=100.0, target_1w=105.0,
            target_15d=110.0, stop=95.0, confidence=75, ticker="TEST"):
    return {
        "record_id": f"{ticker}_{date}", "ticker": ticker, "date": date,
        "spot_price": spot, "verdict": verdict, "confidence": confidence,
        "target_1w": target_1w, "target_15d": target_15d, "stop_loss": stop,
    }


# --- first_holding_bar -------------------------------------------------------

def test_in_session_decision_starts_holding_on_next_bar():
    prices = _prices([100, 101, 102])  # 09-01, 09-02, 09-03
    assert first_holding_bar(prices, "2026-09-01 15:56:37") == 1


def test_pre_open_decision_holds_the_same_day():
    prices = _prices([100, 101, 102])
    assert first_holding_bar(prices, "2026-09-02 09:05:52") == 1


def test_weekend_decision_starts_on_monday():
    prices = _prices([100, 101, 102], start="2026-09-04")  # Fri, Mon, Tue
    assert first_holding_bar(prices, "2026-09-06 00:11:20") == 1


def test_decision_after_last_bar_has_no_holding_bar():
    prices = _prices([100, 101])
    assert first_holding_bar(prices, "2026-09-02 20:00:00") is None


# --- level_problem -----------------------------------------------------------

def test_buy_levels_must_bracket_spot():
    assert level_problem(_record("AL", target_1w=105, stop=95)) is None
    assert level_problem(_record("AL", target_1w=99, stop=95)) is not None


def test_sell_with_target_equal_to_stop_is_flagged():
    record = _record("SAT", spot=286.25, target_1w=283.99, stop=283.99)
    assert level_problem(record) is not None


def test_confidence_outside_0_100_is_flagged():
    assert "güven" in level_problem(_record(confidence=419))


# --- score_decision ----------------------------------------------------------

def test_buy_scores_return_from_spot_to_horizon_close():
    # decision 09-01 in session -> holding bars 09-02.. ; 5th holding bar is index 5
    prices = _prices([100, 101, 102, 103, 104, 106, 90])
    index = _prices([1000] * 7)
    score = score_decision(_record("AL"), prices, index, horizon=5)
    assert score["status"] == "SCORED"
    assert score["return_pct"] == pytest.approx(6.0)
    assert score["hit"] is True
    assert score["exit_date"] == "2026-09-08"


def test_sell_hit_when_price_falls_and_excess_is_against_index():
    prices = _prices([100, 99, 98, 97, 96, 95])
    index = _prices([1000, 1000, 1000, 1000, 1000, 1020])
    score = score_decision(_record("SAT", target_1w=90, stop=105), prices, index, horizon=5)
    assert score["hit"] is True
    assert score["index_return_pct"] == pytest.approx(2.0)
    # short side: excess = index return - stock return
    assert score["excess_pct"] == pytest.approx(2.0 - (-5.0))


def test_hold_verdict_is_not_scored_for_direction():
    prices = _prices([100, 101, 102, 103, 104, 106])
    score = score_decision(_record("TUT"), prices, _prices([1000] * 6), horizon=5)
    assert score["status"] == "SCORED"
    assert score["hit"] is None
    assert score["excess_pct"] is None


def test_decision_without_enough_bars_is_pending():
    prices = _prices([100, 101, 102])
    score = score_decision(_record("AL"), prices, _prices([1000] * 3), horizon=5)
    assert score["status"] == "PENDING"
    assert score["return_pct"] is None


def test_stop_touched_before_target_counts_as_stop_first():
    lows = [100, 94, 100, 100, 100, 100]
    highs = [100, 100, 106, 100, 100, 100]
    prices = _prices([100, 96, 104, 100, 100, 100], highs=highs, lows=lows)
    score = score_decision(_record("AL"), prices, _prices([1000] * 6), horizon=5)
    assert score["first_touch"] == "STOP"


def test_target_and_stop_on_same_bar_is_ambiguous():
    prices = _prices([100, 100, 100, 100, 100, 100],
                     highs=[100, 106, 100, 100, 100, 100],
                     lows=[100, 94, 100, 100, 100, 100])
    score = score_decision(_record("AL"), prices, _prices([1000] * 6), horizon=5)
    assert score["first_touch"] == "AMBIGUOUS"


def test_invalid_levels_skip_first_touch_but_keep_return():
    prices = _prices([100, 99, 98, 97, 96, 95])
    record = _record("SAT", target_1w=99, stop=99)
    score = score_decision(record, prices, _prices([1000] * 6), horizon=5)
    assert score["first_touch"] is None
    assert score["hit"] is True
    assert score["level_problem"]


def test_ten_day_horizon_uses_fifteen_day_target():
    closes = [100] + [101] * 10
    highs = [100] + [101] * 9 + [111]
    prices = _prices(closes, highs=highs)
    score = score_decision(_record("AL"), prices, _prices([1000] * 11), horizon=10)
    assert score["first_touch"] == "TARGET"


# --- summarize ---------------------------------------------------------------

def test_summary_counts_hits_per_verdict_and_ignores_pending():
    scores = [
        {"verdict": "AL", "status": "SCORED", "hit": True, "return_pct": 2.0, "excess_pct": 1.0, "ticker": "A"},
        {"verdict": "AL", "status": "SCORED", "hit": False, "return_pct": -1.0, "excess_pct": -2.0, "ticker": "B"},
        {"verdict": "AL", "status": "PENDING", "hit": None, "return_pct": None, "excess_pct": None, "ticker": "C"},
        {"verdict": "TUT", "status": "SCORED", "hit": None, "return_pct": 3.0, "excess_pct": None, "ticker": "A"},
    ]
    summary = summarize_scores(scores)
    assert summary["AL"]["n"] == 2
    assert summary["AL"]["hits"] == 1
    assert summary["AL"]["mean_return_pct"] == pytest.approx(0.5)
    assert summary["AL"]["mean_excess_pct"] == pytest.approx(-0.5)
    assert summary["AL"]["pending"] == 1
    assert summary["TUT"]["hits"] is None
    assert summary["TUT"]["mean_return_pct"] == pytest.approx(3.0)


def test_cli_scores_memory_file_and_reports_flags(tmp_path, capsys):
    import json

    import main

    _prices([100, 101, 102, 103, 104, 106]).to_csv(tmp_path / "TEST.IS_1d.csv", index=False)
    _prices([1000] * 6).to_csv(tmp_path / "XU100.IS_1d.csv", index=False)
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"history": [
        _record("AL"), _record("SAT", target_1w=99, stop=99, date="2026-09-01 16:00:00"),
    ]}), encoding="utf-8")

    main.handle_score_decisions(memory_path=str(memory), data_dir=str(tmp_path))

    out = capsys.readouterr().out
    assert "5 gün" in out
    assert "AL" in out and "1/1" in out
    assert "SAT için hedef < spot < stop değil" in out


def test_wilson_interval_is_wide_for_small_samples():
    low, high = wilson_interval(3, 4)
    assert low < 0.35 and high > 0.95
    assert wilson_interval(0, 0) == (None, None)
