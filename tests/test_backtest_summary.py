import pandas as pd

import main
from bist_quant.backtest_engine import LEAK_NOT_APPLICABLE, CostModel
from bist_quant.bist_backtester import BistBacktester

NO_COSTS = CostModel(commission_rate=0.0, slippage_rate=0.0)


def _metrics(trades, equity):
    backtester = BistBacktester(use_kronos=False)
    metrics = backtester._calculate_performance_metrics(
        trades=trades,
        equity_curve=pd.Series(equity),
        test_df=pd.DataFrame({"close": [100.0, 103.0, 105.0]}),
        initial_capital=equity[0],
        total_costs=0.0,
        costs=NO_COSTS,
        rf_annual=0.0,
    )
    return {**metrics, "leakage_status": LEAK_NOT_APPLICABLE, "leakage_note": "model yok"}


def _real_metrics():
    trades = [
        {"return_pct": 10.0, "pnl": 10000.0, "duration_days": 12},
        {"return_pct": -4.0, "pnl": -4400.0, "duration_days": 4},
        {"return_pct": 6.0, "pnl": 6336.0, "duration_days": 8},
    ]
    return _metrics(trades, [100000.0, 110000.0, 105600.0, 111936.0])


def test_cli_summary_prints_metrics_the_backtester_actually_returns(monkeypatch, capsys):
    metrics = _real_metrics()

    class FakeBacktester:
        def __init__(self, use_kronos=False, costs=None):
            pass

        def run_walk_forward_backtest(self, **kwargs):
            return metrics, "rapor.md", None

    monkeypatch.setattr(main, "BistBacktester", FakeBacktester)

    main.handle_backtest("TEST.IS")

    out = capsys.readouterr().out
    assert "%+11.94" in out          # strateji getirisi
    assert "%+5.00" in out           # al ve tut
    assert "%+6.94" in out           # alfa
    assert "%66.7 (2 / 3" in out     # kazanma oranı
    assert "%4.00" in out            # maksimum çekilme
    assert "Ort. Süre: 8.0 Gün" in out
    assert LEAK_NOT_APPLICABLE in out


def test_metrics_report_average_holding_days_when_no_trades():
    metrics = _metrics([], [1000.0, 1000.0])

    assert metrics["avg_holding_days"] == 0.0
    assert metrics["total_trades"] == 0
