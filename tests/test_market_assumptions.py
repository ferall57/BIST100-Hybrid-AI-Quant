import pytest

from bist_quant.market_assumptions import (
    DEFAULT_RISK_FREE_RATE,
    POLICY_RATE_ENV,
    RISK_FREE_RATE_ENV,
    policy_rate_pct,
    risk_free_rate,
)


def test_risk_free_rate_falls_back_to_documented_default(monkeypatch):
    monkeypatch.delenv(RISK_FREE_RATE_ENV, raising=False)

    assert risk_free_rate() == DEFAULT_RISK_FREE_RATE


def test_risk_free_rate_reads_percent_from_environment(monkeypatch):
    monkeypatch.setenv(RISK_FREE_RATE_ENV, "38.5")

    assert risk_free_rate() == pytest.approx(0.385)


def test_policy_rate_is_none_when_not_configured(monkeypatch):
    monkeypatch.delenv(POLICY_RATE_ENV, raising=False)

    assert policy_rate_pct() is None


def test_policy_rate_reads_percent_from_environment(monkeypatch):
    monkeypatch.setenv(POLICY_RATE_ENV, "42,5")

    assert policy_rate_pct() == pytest.approx(42.5)


@pytest.mark.parametrize("bad", ["abc", "-5", "1500"])
def test_invalid_rate_fails_fast_with_variable_name(monkeypatch, bad):
    monkeypatch.setenv(RISK_FREE_RATE_ENV, bad)

    with pytest.raises(ValueError, match=RISK_FREE_RATE_ENV):
        risk_free_rate()
