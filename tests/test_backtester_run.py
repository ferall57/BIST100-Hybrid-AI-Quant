import numpy as np
import pandas as pd
import pytest

from bist_quant import bist_backtester
from bist_quant.backtest_engine import LEAK_NOT_APPLICABLE, CostModel
from bist_quant.bist_backtester import BistBacktester

NO_COSTS = CostModel(commission_rate=0.0, slippage_rate=0.0)
HEAVY_COSTS = CostModel(commission_rate=0.005, slippage_rate=0.005)


def _write_history(directory, ticker, closes):
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B").strftime("%Y-%m-%d 00:00:00+03:00")
    closes = np.asarray(closes, dtype=float)
    pd.DataFrame(
        {
            "timestamps": dates,
            "open": closes * 0.999,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": 1_000_000,
            "amount": closes * 1_000_000,
        }
    ).to_csv(directory / f"{ticker}_1d.csv", index=False)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """İndirme yapmayan, tüm çıktıları geçici klasöre yazan backtest ortamı."""
    monkeypatch.setattr(bist_backtester, "RAW_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(bist_backtester, "CHARTS_DIR", str(tmp_path))
    monkeypatch.setattr(bist_backtester, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(bist_backtester, "download_ticker_data", lambda *a, **k: None)
    return tmp_path


def _uptrend(n=400):
    rng = np.random.default_rng(5)
    return 100.0 * np.cumprod(1 + rng.normal(0.004, 0.008, n))


def _run(costs, **kwargs):
    metrics, report_path, _ = BistBacktester(use_kronos=False, costs=costs).run_walk_forward_backtest("UP.IS", months=6, **kwargs)
    return metrics, report_path


def test_trading_costs_reduce_the_result(sandbox):
    _write_history(sandbox, "UP.IS", _uptrend())

    free, _ = _run(NO_COSTS)
    costly, _ = _run(HEAVY_COSTS)

    assert free["total_trades"] > 0
    assert free["total_costs_try"] == 0.0
    assert costly["total_costs_try"] > 0.0
    assert costly["total_return_pct"] < free["total_return_pct"]


def test_metrics_expose_the_keys_the_cli_prints(sandbox):
    _write_history(sandbox, "UP.IS", _uptrend())

    metrics, _ = _run(NO_COSTS)

    for key in ["total_return_pct", "bnh_return_pct", "alpha", "win_rate", "winning_trades", "total_trades",
                "profit_factor", "max_drawdown", "sharpe_ratio", "avg_holding_days", "total_costs_try", "leakage_status"]:
        assert key in metrics


def test_flat_viop_cash_never_earns_more_than_the_risk_free_rate(sandbox, monkeypatch):
    monkeypatch.setattr(bist_backtester, "risk_free_rate", lambda: 0.45)
    _write_history(sandbox, "FLAT.IS", np.full(400, 50.0))

    metrics, _, _ = BistBacktester(use_kronos=False, costs=NO_COSTS).run_walk_forward_backtest(
        "FLAT.IS", months=6, use_viop=True, leverage=1.5
    )

    half_year_risk_free_pct = ((1.45 ** 0.5) - 1) * 100
    assert metrics["total_trades"] == 0
    assert 0.0 < metrics["total_return_pct"] <= half_year_risk_free_pct + 0.5


def test_report_states_costs_and_leakage_instead_of_claiming_zero_bias(sandbox):
    _write_history(sandbox, "UP.IS", _uptrend())

    metrics, report_path = _run(HEAVY_COSTS)
    report = open(report_path, encoding="utf-8").read()

    assert metrics["leakage_status"] == LEAK_NOT_APPLICABLE
    assert "Lookahead Bias %0" not in report
    assert "Komisyon" in report and "Kayma" in report


def test_training_cutoff_is_recorded_for_the_leakage_check(tmp_path):
    from bist_quant.bist_trainer import write_training_cutoff

    csv_path = tmp_path / "unified.csv"
    pd.DataFrame({"timestamps": pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d %H:%M:%S"), "close": 1.0}).to_csv(csv_path, index=False)
    out_path = tmp_path / "models" / "training_cutoff.json"

    cutoff = write_training_cutoff(processed_csv=str(csv_path), train_ratio=0.5, out_path=str(out_path))

    assert cutoff == "2024-01-05"
    assert bist_backtester.load_training_cutoff(str(out_path)) == "2024-01-05"
    assert bist_backtester.load_training_cutoff(str(tmp_path / "yok.json")) is None


def test_index_gate_blocks_long_entries_while_the_index_is_in_a_downtrend(sandbox):
    _write_history(sandbox, "UP.IS", _uptrend())
    _write_history(sandbox, "XU100.IS", 100.0 * np.cumprod(np.full(400, 0.995)))

    baseline, _ = _run(NO_COSTS)
    gated, _ = _run(NO_COSTS, entry_filters=("index_gate",))

    assert baseline["total_trades"] > 0
    assert gated["total_trades"] == 0
    assert gated["blocked_entries"]["index_gate"] > 0


def test_index_gate_lets_entries_through_while_the_index_is_rising(sandbox):
    _write_history(sandbox, "UP.IS", _uptrend())
    _write_history(sandbox, "XU100.IS", _uptrend())

    gated, _ = _run(NO_COSTS, entry_filters=("index_gate",))

    assert gated["total_trades"] > 0


def test_backtest_can_skip_report_files(sandbox):
    _write_history(sandbox, "UP.IS", _uptrend())

    metrics, report_path, chart_path = BistBacktester(use_kronos=False, costs=NO_COSTS).run_walk_forward_backtest(
        "UP.IS", months=6, write_artifacts=False
    )

    assert report_path is None and chart_path is None
    assert metrics["total_trades"] > 0


def test_universe_backtest_summarises_every_ticker(sandbox):
    _write_history(sandbox, "UP.IS", _uptrend())
    _write_history(sandbox, "FLAT.IS", np.full(400, 50.0))

    rows, summary = BistBacktester(use_kronos=False, costs=NO_COSTS).run_universe_backtest(["UP.IS", "FLAT.IS"], months=6)

    assert [r["ticker"] for r in rows] == ["UP.IS", "FLAT.IS"]
    assert summary["tickers_tested"] == 2
    assert 0.0 <= summary["share_beating_buy_and_hold_pct"] <= 100.0
    assert "median_alpha" in summary


def test_universe_backtest_reports_failed_tickers_without_stopping(sandbox):
    _write_history(sandbox, "UP.IS", _uptrend())

    rows, summary = BistBacktester(use_kronos=False, costs=NO_COSTS).run_universe_backtest(["YOK.IS", "UP.IS"], months=6)

    assert [r["ticker"] for r in rows] == ["UP.IS"]
    assert summary["failed_tickers"] == ["YOK.IS"]
