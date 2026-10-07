import numpy as np
import pandas as pd
import pytest

from bist_quant.bist_econometrics import NO_DATA, BistEconometrics

TRUE_BETA = 1.5


def _prices(returns, dates):
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"timestamps": dates, "close": close, "volume": 1_000_000})


@pytest.fixture
def stock_and_market():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=400, freq="B").strftime("%Y-%m-%d 00:00:00+03:00")
    market_ret = rng.normal(0.0005, 0.012, 400)
    stock_ret = TRUE_BETA * market_ret + rng.normal(0.0, 0.002, 400)
    return _prices(stock_ret, dates), _prices(market_ret, dates)


def test_without_market_data_beta_is_not_estimated_from_random_numbers(stock_and_market):
    stock, _ = stock_and_market

    result = BistEconometrics().calculate_asset_pricing_metrics(stock)

    assert result["market_data_available"] is False
    assert result["beta"] is None
    assert result["jensen_alpha_annual_pct"] is None
    assert result["information_ratio"] is None
    assert result["roll_effective_spread_try"] > 0  # piyasa verisi gerektirmeyen ölçüler kalır


def test_with_market_data_beta_is_recovered(stock_and_market):
    stock, market = stock_and_market

    result = BistEconometrics().calculate_asset_pricing_metrics(stock, df_market=market)

    assert result["market_data_available"] is True
    assert result["beta"] == pytest.approx(TRUE_BETA, abs=0.05)


def test_series_are_aligned_by_date_not_by_row_position(stock_and_market):
    stock, market = stock_and_market
    longer_market = pd.concat(
        [_prices(np.full(50, 0.001), pd.date_range("2023-10-02", periods=50, freq="B").strftime("%Y-%m-%d")), market],
        ignore_index=True,
    )

    result = BistEconometrics().calculate_asset_pricing_metrics(stock, df_market=longer_market)

    assert result["beta"] == pytest.approx(TRUE_BETA, abs=0.05)


def test_capm_summary_says_no_data_when_market_is_missing(stock_and_market):
    stock, _ = stock_and_market
    econ = BistEconometrics()

    summary = econ.format_capm_summary(econ.calculate_asset_pricing_metrics(stock))

    assert NO_DATA in summary
    assert "β: 1.00" not in summary
