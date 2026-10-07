import pytest

from hybrid_agents.verdict_parser import (
    VERDICT_BUY,
    VERDICT_HOLD,
    VERDICT_SELL,
    VERDICT_UNKNOWN,
    parse_verdict_type,
)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("* **YATIRIM KARARI:** Güçlü AL (Strong Buy)", VERDICT_BUY),
        ("* **YATIRIM KARARI:** **AL (Buy)**", VERDICT_BUY),
        ("* **YATIRIM KARARI:** TUT (Hold)", VERDICT_HOLD),
        ("* **YATIRIM KARARI:** SAT (Sell)", VERDICT_SELL),
        ("* **YATIRIM KARARI:** Güçlü SAT (Strong Sell)", VERDICT_SELL),
        ("Yatırım Kararı: [TUT]", VERDICT_HOLD),
    ],
)
def test_reads_verdict_from_decision_line(line, expected):
    assert parse_verdict_type(line) == expected


def test_hold_verdict_is_not_read_as_buy_because_of_other_words():
    # "HAFTALIK", "ANALİZ", "KALKANI" hepsi "AL" alt dizisini içerir.
    text = (
        "## NİHAİ ANALİZ\n"
        "* **YATIRIM KARARI:** TUT (Hold)\n"
        "* **1 Haftalık (Kısa Vade) Hedef Bandı:** [12.10 TRY - 12.80 TRY]\n"
        "Hacimsiz kırılım alım için yeterli değildir.\n"
    )

    assert parse_verdict_type(text) == VERDICT_HOLD


def test_sell_verdict_is_not_read_as_buy():
    text = "* **YATIRIM KARARI:** SAT (Sell)\n* 1 Haftalık hedef: 10.00 - 11.00"

    assert parse_verdict_type(text) == VERDICT_SELL


def test_unfilled_template_with_all_options_is_unknown():
    text = "* **YATIRIM KARARI:** [Güçlü AL (Strong Buy) / AL (Buy) / TUT (Hold) / SAT (Sell)]"

    assert parse_verdict_type(text) == VERDICT_UNKNOWN


def test_missing_decision_line_is_unknown():
    assert parse_verdict_type("Haftalık görünüm olumlu, alım fırsatı.") == VERDICT_UNKNOWN


def test_empty_text_is_unknown():
    assert parse_verdict_type("") == VERDICT_UNKNOWN
