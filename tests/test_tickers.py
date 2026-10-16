from bist_quant.bist_100_tickers import BIST_100_TICKERS, BIST_30_TICKERS, get_tickers

# 2025'te Koza grubu kod değiştirdi; KNYAS ne KAP'ta ne Yahoo'da bulunuyor (16 Ekim 2026 kontrolü).
RENAMED = {"KOZAL.IS": "TRALT.IS", "KOZAA.IS": "TRMET.IS", "IPEKE.IS": "TRENJ.IS"}
REMOVED = {"KNYAS.IS"}


def test_renamed_codes_replace_the_old_ones():
    for old, new in RENAMED.items():
        assert old not in BIST_100_TICKERS
        assert new in BIST_100_TICKERS


def test_codes_without_any_listing_are_removed():
    assert not REMOVED & set(BIST_100_TICKERS)


def test_lists_are_sorted_unique_yahoo_codes():
    for tickers in (BIST_100_TICKERS, BIST_30_TICKERS):
        assert tickers == sorted(set(tickers))
        assert all(t.endswith(".IS") and t[:-3].isalnum() and t[:-3].isupper() for t in tickers)


def test_get_tickers_returns_the_requested_universe():
    assert get_tickers("bist30") == BIST_30_TICKERS
    assert get_tickers("bist100") == BIST_100_TICKERS
