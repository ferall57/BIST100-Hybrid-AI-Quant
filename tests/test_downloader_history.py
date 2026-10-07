import pandas as pd

from bist_quant.bist_downloader import histories_are_consistent, merge_candles, trim_to_period

COLS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


def _frame(days, close=10.0):
    return pd.DataFrame(
        {
            "timestamps": [f"{d} 00:00:00+03:00" for d in days],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000,
            "amount": close * 1000,
        }
    )[COLS]


def test_short_download_does_not_truncate_longer_history():
    existing = _frame(pd.date_range("2021-01-04", periods=1000, freq="B").strftime("%Y-%m-%d"))
    fresh = existing.tail(20).copy()

    merged = merge_candles(existing, fresh)

    assert len(merged) == 1000
    assert merged["timestamps"].iloc[0].startswith("2021-01-04")


def test_fresh_candle_replaces_stale_candle_of_same_day():
    existing = _frame(["2026-10-05", "2026-10-06"], close=10.0)
    fresh = _frame(["2026-10-06", "2026-10-07"], close=11.0)

    merged = merge_candles(existing, fresh)

    assert merged["timestamps"].str[:10].tolist() == ["2026-10-05", "2026-10-06", "2026-10-07"]
    assert merged["close"].tolist() == [10.0, 11.0, 11.0]


def test_same_day_matches_across_timezone_formats():
    existing = _frame(["2026-10-06"]).assign(timestamps=["2026-10-06"])
    fresh = _frame(["2026-10-06"], close=12.0)

    merged = merge_candles(existing, fresh)

    assert len(merged) == 1
    assert merged["close"].iloc[0] == 12.0


def test_daily_candles_are_stored_without_timezone_suffix():
    # 2016 öncesi Türkiye yaz/kış saati uyguluyordu: aynı dosyada +02:00 ve +03:00 ekleri karışır
    # ve pandas karışık ekli sütunu tarih olarak ayrıştıramaz.
    existing = _frame(["2015-01-05", "2015-07-06"]).assign(
        timestamps=["2015-01-05 00:00:00+02:00", "2015-07-06 00:00:00+03:00"]
    )
    fresh = _frame(["2026-10-07"])

    merged = merge_candles(existing, fresh)

    assert merged["timestamps"].tolist() == ["2015-01-05", "2015-07-06", "2026-10-07"]
    assert pd.to_datetime(merged["timestamps"]).dt.year.tolist() == [2015, 2015, 2026]


def test_intraday_candles_keep_their_full_timestamp():
    existing = _frame(["2026-10-07"]).assign(timestamps=["2026-10-07 10:00:00+03:00"])
    fresh = _frame(["2026-10-07"]).assign(timestamps=["2026-10-07 11:00:00+03:00"])

    merged = merge_candles(existing, fresh, daily=False)

    assert merged["timestamps"].tolist() == ["2026-10-07 10:00:00+03:00", "2026-10-07 11:00:00+03:00"]


def test_merge_with_empty_existing_returns_fresh():
    fresh = _frame(["2026-10-06"])

    merged = merge_candles(pd.DataFrame(columns=COLS), fresh)

    assert len(merged) == 1


def test_rescaled_history_is_detected_as_inconsistent():
    # Bedelsiz sonrası Yahoo geçmiş kapanışları yeniden ölçekler: eski 10.0 artık 5.0 görünür.
    existing = _frame(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"], close=10.0)
    fresh = _frame(["2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"], close=5.0)

    assert histories_are_consistent(existing, fresh) is False


def test_partial_last_candle_does_not_count_as_rescale():
    existing = _frame(["2026-10-05", "2026-10-06"], close=10.0)
    existing.loc[1, "close"] = 9.0  # seans içinde kaydedilmiş yarım mum
    fresh = _frame(["2026-10-05", "2026-10-06"], close=10.0)

    assert histories_are_consistent(existing, fresh) is True


def test_non_overlapping_histories_are_consistent():
    assert histories_are_consistent(_frame(["2026-01-05"]), _frame(["2026-10-06"])) is True


def test_trim_keeps_only_requested_window():
    df = _frame(pd.date_range("2021-01-04", "2026-10-06", freq="B").strftime("%Y-%m-%d"))

    trimmed = trim_to_period(df, "6mo")

    assert trimmed["timestamps"].iloc[0][:10] >= "2026-04-06"
    assert trimmed["timestamps"].iloc[-1][:10] == "2026-10-06"
    assert list(trimmed.index) == list(range(len(trimmed)))


def test_trim_with_max_or_unknown_period_returns_everything():
    df = _frame(pd.date_range("2021-01-04", periods=300, freq="B").strftime("%Y-%m-%d"))

    assert len(trim_to_period(df, "max")) == 300
    assert len(trim_to_period(df, "garip")) == 300
