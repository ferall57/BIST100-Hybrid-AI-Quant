import math

from hybrid_agents.bist_committee import pick_live_price


def test_live_price_is_used_when_valid():
    assert pick_live_price(last_price=12.5, previous_close=12.0, fallback=11.0) == 12.5


def test_nan_live_price_falls_back_to_previous_close():
    assert pick_live_price(last_price=math.nan, previous_close=12.0, fallback=11.0) == 12.0


def test_missing_live_data_keeps_csv_close():
    assert pick_live_price(last_price=None, previous_close=None, fallback=11.0) == 11.0


def test_non_positive_prices_are_rejected():
    assert pick_live_price(last_price=0.0, previous_close=-1.0, fallback=11.0) == 11.0
