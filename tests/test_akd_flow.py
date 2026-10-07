import numpy as np
import pandas as pd
import pytest

from bist_quant.bist_akd_flow import NO_DATA, BistAkdFlowEngine

BROKER_NAMES = ["Bank of America", "BofA", "İş Yatırım", "QNB", "Garanti", "Yapı Kredi"]


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.3, 1.0, 40))
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(1_000_000, 2_000_000, 40),
        }
    )


def test_without_broker_file_no_broker_figures_are_invented(tmp_path, ohlcv):
    engine = BistAkdFlowEngine(akd_dir=str(tmp_path))

    profile = engine.analyze_akd_profile("TEST.IS", ohlcv)

    assert profile["is_real_feed"] is False
    assert profile["top5_buy_pct"] is None
    assert profile["top5_sell_pct"] is None
    assert profile["net_concentration_pct"] is None
    assert profile["lead_buyer"] == NO_DATA
    assert profile["lead_seller"] == NO_DATA


def test_without_broker_file_summary_names_no_institution(tmp_path, ohlcv):
    engine = BistAkdFlowEngine(akd_dir=str(tmp_path))

    summary = engine.get_akd_summary_text("TEST.IS", ohlcv)

    assert NO_DATA in summary
    for name in BROKER_NAMES:
        assert name not in summary


def test_volume_based_indicators_are_still_reported_without_broker_file(tmp_path, ohlcv):
    engine = BistAkdFlowEngine(akd_dir=str(tmp_path))

    profile = engine.analyze_akd_profile("TEST.IS", ohlcv)

    assert -1.0 <= profile["cmf_20"] <= 1.0
    assert 0.0 <= profile["mfi_14"] <= 100.0
    assert -1.0 <= profile["whale_score"] <= 1.0


def test_real_broker_file_is_reported_with_its_own_names(tmp_path, ohlcv):
    pd.DataFrame(
        {"Kurum": ["Kurum A", "Kurum B", "Kurum C"], "NetLot": [900, 100, -1000]}
    ).to_csv(tmp_path / "TEST_akd.csv", index=False)
    engine = BistAkdFlowEngine(akd_dir=str(tmp_path))

    profile = engine.analyze_akd_profile("TEST.IS", ohlcv)
    summary = engine.get_akd_summary_text("TEST.IS", ohlcv)

    assert profile["is_real_feed"] is True
    assert profile["top5_buy_pct"] == 50.0
    assert "Kurum A" in profile["lead_buyer"]
    assert "Kurum C" in summary
