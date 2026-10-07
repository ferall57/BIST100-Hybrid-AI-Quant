import pytest

from hybrid_agents.verdict_parser import (
    VERDICT_BUY,
    VERDICT_HOLD,
    VERDICT_SELL,
    VERDICT_UNKNOWN,
    parse_decision_fields,
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


REAL_VERDICT = """# 🏆 NİHAİ YATIRIM KOMİTESİ KARAR RAPORU (THYAO.IS)

## 1. 🎯 Karar ve Derece (Rating & Verdict)
*   **YATIRIM KARARI:** **Güçlü SAT (Strong Sell)**
*   **Güven Katsayısı (Confidence):** %85
*   **1 Haftalık (Kısa Vade) Hedef Bandı:** **268.42 TRY - 283.99 TRY** (Potansiyel %-6.23 ila %-0.8 getiri)
*   **15-30 Günlük (Orta Vade) Hedef Bandı:** **249.70 TRY - 279.86 TRY** (Potansiyel %-12.77 ila %-2.23 getiri)

## 3. 🎯 Kesin Şartlı Tetikleyiciler (If-Then Execution Triggers)
*   **Boğa Tetik Seviyesi:** **300.72 TRY** (Monte Carlo üst sınırı)
*   **Ayı Savunma / Stop Seviyesi:** **283.99 TRY** (günlük kapanış bazında)

## 4. 🔬 Ekonometrik & Matematiksel Doğrulama (XAI - Açıklanabilirlik)
*   **Monte Carlo Yükseliş Olasılığı (Win Rate):** 1 Haftalık: %42.7 | Orta Vadeli: %38.5

## 5. ⚡ VİOP (Türev) & Hedge Stratejisi
*   **İz Süren / Ters Stop Mesafesi:** %3.5
"""


def test_fields_are_read_from_a_real_bold_formatted_verdict():
    fields = parse_decision_fields(REAL_VERDICT, reference_price=286.25)

    assert fields["confidence"] == 85
    assert fields["target_1w"] == (268.42, 283.99)
    assert fields["target_mid"] == (249.70, 279.86)
    assert fields["stop_loss"] == 283.99


def test_fields_are_read_from_the_bracketed_template_format():
    text = ("* **Güven Katsayısı (Confidence):** % 70\n"
            "* **1 Haftalık (Kısa Vade) Hedef Bandı:** [12.10 TRY - 12.80 TRY] (Örn: %+3.1)\n"
            "* **Stop-Loss:** 11.75 TRY\n")

    fields = parse_decision_fields(text, reference_price=12.39)

    assert fields["confidence"] == 70
    assert fields["target_1w"] == (12.10, 12.80)
    assert fields["stop_loss"] == 11.75


def test_turkish_number_format_is_resolved_with_the_reference_price():
    text = "* 1 Haftalık Hedef Bandı: 1.234,50 TRY - 1.300,00 TRY\n* 15-30 Günlük Hedef Bandı: 268,42 - 283,99"

    assert parse_decision_fields(text, reference_price=1250.0)["target_1w"] == (1234.5, 1300.0)
    assert parse_decision_fields(text, reference_price=280.0)["target_mid"] == (268.42, 283.99)


def test_missing_fields_are_none_instead_of_invented_defaults():
    fields = parse_decision_fields("* **YATIRIM KARARI:** TUT (Hold)", reference_price=100.0)

    assert fields == {"confidence": None, "target_1w": None, "target_mid": None, "stop_loss": None}


def test_probability_and_trailing_distance_lines_are_not_mistaken_for_levels():
    text = ("* **Monte Carlo Yükseliş Olasılığı (Win Rate):** 1 Haftalık: %42.7 | Orta Vadeli: %38.5\n"
            "* **İz Süren / Ters Stop Mesafesi:** %3.5\n")

    fields = parse_decision_fields(text, reference_price=286.25)

    assert fields["target_1w"] is None
    assert fields["target_mid"] is None
    assert fields["stop_loss"] is None


def test_out_of_range_confidence_is_rejected():
    assert parse_decision_fields("* Güven Katsayısı: %850", reference_price=10.0)["confidence"] is None


def test_empty_text_is_unknown():
    assert parse_verdict_type("") == VERDICT_UNKNOWN
