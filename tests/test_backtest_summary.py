import pandas as pd

import main
from bist_quant.bist_backtester import BistBacktester


def _real_metrics():
    trades = [
        {"return_pct": 10.0, "capital": 110000.0, "duration_days": 12},
        {"return_pct": -4.0, "capital": 105600.0, "duration_days": 4},
        {"return_pct": 6.0, "capital": 111936.0, "duration_days": 8},
    ]
    test_df = pd.DataFrame({"close": [100.0, 103.0, 105.0]})
    backtester = BistBacktester(use_kronos=False)
    return backtester._calculate_performance_metrics(
        trades=trades, test_df=test_df, initial_capital=100000.0, final_capital=111936.0
    )


def test_cli_summary_prints_metrics_the_backtester_actually_returns(monkeypatch, capsys):
    metrics = _real_metrics()

    class FakeBacktester:
        def __init__(self, use_kronos=False):
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


def test_metrics_report_average_holding_days_when_no_trades():
    backtester = BistBacktester(use_kronos=False)

    metrics = backtester._calculate_performance_metrics(
        trades=[], test_df=pd.DataFrame({"close": [1.0, 1.1]}), initial_capital=1000.0, final_capital=1000.0
    )

    assert metrics["avg_holding_days"] == 0.0
