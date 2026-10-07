import pytest

from bist_quant.bist_trade_memory import BistTradeMemory


@pytest.fixture
def memory(tmp_path):
    return BistTradeMemory(memory_file=str(tmp_path / "vault.json"))


def _record(memory, ticker="THYAO.IS", setup="SSL_SWEEP_FVG_RETEST", outcome="WIN"):
    return memory.record_trade(
        ticker=ticker, direction="LONG", setup_type=setup, market_regime="BULL_REGIME",
        entry_price=100.0, exit_price=104.0, pnl_try=400.0, pnl_pct=4.0,
        holding_days=3, outcome=outcome, lessons_learned="Stop seviyesine sadık kalındı.",
    )


def test_new_vault_starts_without_invented_trades(memory):
    assert memory.vault["records"] == []
    assert memory.vault["total_trades_recorded"] == 0
    assert memory.vault["win_rate_pct"] is None


def test_report_on_empty_vault_claims_no_track_record(memory):
    report = memory.generate_trade_memory_report("THYAO.IS", current_setup="SSL_SWEEP")

    assert "kayıt yok" in report.lower()
    assert "%100" not in report
    assert "KAZANÇ" not in report


def test_unrelated_trades_are_not_returned_as_similar(memory):
    _record(memory, ticker="BIMAS.IS", setup="MOMENTUM_BREAKOUT")

    assert memory.query_similar_setups("THYAO.IS", "SSL_SWEEP") == []


def test_trades_match_by_ticker_or_setup(memory):
    _record(memory, ticker="BIMAS.IS", setup="SSL_SWEEP_FVG_RETEST")
    _record(memory, ticker="THYAO.IS", setup="MOMENTUM_BREAKOUT")

    by_setup = memory.query_similar_setups("ASELS.IS", "SSL_SWEEP")
    by_ticker = memory.query_similar_setups("THYAO.IS", "")

    assert [r["ticker"] for r in by_setup] == ["BIMAS"]
    assert [r["ticker"] for r in by_ticker] == ["THYAO"]


def test_report_without_detected_setup_matches_only_by_ticker(memory):
    _record(memory, ticker="BIMAS.IS", setup="SSL_SWEEP_FVG_RETEST")

    report = memory.generate_trade_memory_report("THYAO.IS", current_setup="")

    assert "BIMAS" not in report


def test_win_rate_is_computed_from_recorded_trades(memory):
    _record(memory, outcome="WIN")
    _record(memory, outcome="LOSS")

    assert memory.vault["win_rate_pct"] == 50.0
    assert "%50.0" in memory.generate_trade_memory_report("THYAO.IS", current_setup="")
