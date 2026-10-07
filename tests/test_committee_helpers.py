import math

from bist_quant.market_assumptions import POLICY_RATE_ENV
from hybrid_agents.bist_committee import band_upper, describe_current_setup, pick_live_price, policy_rate_line, risk_veto_reason
from hybrid_agents.verdict_parser import VERDICT_BUY, VERDICT_HOLD, VERDICT_SELL, VERDICT_UNKNOWN


def test_memory_stores_unread_fields_as_missing_and_still_evaluates(tmp_path):
    from bist_quant.bist_memory import BistCommitteeMemory

    memory = BistCommitteeMemory(memory_filepath=str(tmp_path / "memory.json"))
    entry = memory.record_decision(ticker="TEST.IS", spot_price=100.0, verdict="AL", confidence=None,
                                   target_1w=None, target_15d=None, stop_loss=None, forecast_date="2026-01-05 10:00:00")

    evaluations = memory.evaluate_past_decisions("TEST.IS", current_price=95.0)

    assert (entry["confidence"], entry["target_1w"], entry["target_15d"], entry["stop_loss"]) == (None, None, None, None)
    assert evaluations[0]["is_mature"] is True
    assert evaluations[0]["is_success"] is False


def test_band_upper_returns_the_high_end_or_nothing():
    assert band_upper((268.42, 283.99)) == 283.99
    assert band_upper(None) is None


def test_fundamental_prompt_states_the_real_news_window():
    from hybrid_agents import prompts

    assert "24/48" not in prompts.BIST_FUNDAMENTAL_ANALYST_PROMPT
    assert "{news_window_days}" in prompts.BIST_FUNDAMENTAL_ANALYST_PROMPT


WEAK_ODDS = {"prob_positive": 30.0}
OUTFLOW = {"cmf_20": -0.20}


def test_risk_veto_fires_for_a_buy_when_odds_are_poor_and_money_is_leaving():
    reason = risk_veto_reason(VERDICT_BUY, WEAK_ODDS, OUTFLOW)

    assert reason is not None
    assert "30.0" in reason and "-0.200" in reason


def test_risk_veto_needs_both_conditions():
    assert risk_veto_reason(VERDICT_BUY, WEAK_ODDS, {"cmf_20": 0.05}) is None
    assert risk_veto_reason(VERDICT_BUY, {"prob_positive": 55.0}, OUTFLOW) is None


def test_risk_veto_treats_an_unparsed_verdict_like_a_buy():
    assert risk_veto_reason(VERDICT_UNKNOWN, WEAK_ODDS, OUTFLOW) is not None


def test_risk_veto_never_overrides_hold_or_sell():
    assert risk_veto_reason(VERDICT_HOLD, WEAK_ODDS, OUTFLOW) is None
    assert risk_veto_reason(VERDICT_SELL, WEAK_ODDS, OUTFLOW) is None


def test_risk_veto_does_not_fire_without_data():
    assert risk_veto_reason(VERDICT_BUY, {}, {}) is None


def test_index_regime_is_presented_as_context_not_as_a_ban(monkeypatch):
    from bist_quant.bist_index_gatekeeper import BistIndexGatekeeper

    gatekeeper = BistIndexGatekeeper()
    bear = {"regime": "BEAR_REGIME", "is_long_allowed": False, "is_short_allowed": True, "long_size_multiplier": 0.0,
            "regime_score": 20.0, "xu100_close": 9000.0, "daily_change_pct": -1.0, "sma_50": 9500.0, "ema_21": 9300.0,
            "rsi_14": 38.0, "adx_14": 25.0, "above_sma50": False, "desc": "XU100 ayı rejiminde."}
    monkeypatch.setattr(gatekeeper, "get_market_regime", lambda: bear)

    report = gatekeeper.generate_gatekeeper_report()

    assert "BEAR_REGIME" in report
    for banned_wording in ["YASAK", "ENGELLENDİ", "Hard Block", "KAPALI"]:
        assert banned_wording not in report
    assert "bağlam bilgisi" in report


def test_manager_prompt_no_longer_forbids_buying_in_a_bear_index_regime():
    from hybrid_agents import prompts

    assert "kesinlikle LONG alım verme" not in prompts.BIST_PORTFOLIO_MANAGER_PROMPT
    assert "bağlam bilgisi" in prompts.BIST_PORTFOLIO_MANAGER_PROMPT


def test_policy_rate_is_reported_as_missing_when_not_configured(monkeypatch):
    monkeypatch.delenv(POLICY_RATE_ENV, raising=False)

    line = policy_rate_line()

    assert "VERİ YOK" in line
    assert "%50" not in line


def test_configured_policy_rate_is_reported_with_its_source(monkeypatch):
    monkeypatch.setenv(POLICY_RATE_ENV, "42.5")

    line = policy_rate_line()

    assert "%42.50" in line
    assert POLICY_RATE_ENV in line


def test_setup_is_described_from_detected_price_action():
    assert describe_current_setup({"ssl_swept": True}, {"is_break_retest": True}) == "SSL_SWEEP"
    assert describe_current_setup({"ssl_swept": False}, {"is_break_retest": True}) == "RETEST"
    assert describe_current_setup({"bsl_swept": True}, {}) == "BSL_SWEEP"


def test_no_setup_is_claimed_when_none_is_detected():
    assert describe_current_setup({}, {}) == ""


def test_live_price_is_used_when_valid():
    assert pick_live_price(last_price=12.5, previous_close=12.0, fallback=11.0) == 12.5


def test_nan_live_price_falls_back_to_previous_close():
    assert pick_live_price(last_price=math.nan, previous_close=12.0, fallback=11.0) == 12.0


def test_missing_live_data_keeps_csv_close():
    assert pick_live_price(last_price=None, previous_close=None, fallback=11.0) == 11.0


def test_non_positive_prices_are_rejected():
    assert pick_live_price(last_price=0.0, previous_close=-1.0, fallback=11.0) == 11.0
