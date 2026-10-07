import math

from bist_quant.market_assumptions import POLICY_RATE_ENV
from hybrid_agents.bist_committee import describe_current_setup, pick_live_price, policy_rate_line


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
