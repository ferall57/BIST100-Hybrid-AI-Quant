"""Baş Portföy Müdürü çıktısından yapılandırılmış yatırım kararını ayıklar."""

import math
import re

VERDICT_BUY = "AL"
VERDICT_SELL = "SAT"
VERDICT_HOLD = "TUT"
VERDICT_UNKNOWN = "BELIRSIZ"

_DECISION_LINE = re.compile(r"yat[ıi]r[ıi]m\s+karar[ıi][^:\n]*:(.*)", re.IGNORECASE)
_WORD = re.compile(r"[A-ZÇĞÖŞÜ]+")

_KEYWORDS = {
    VERDICT_BUY: {"AL", "BUY"},
    VERDICT_SELL: {"SAT", "SELL"},
    VERDICT_HOLD: {"TUT", "HOLD"},
}


_MARKDOWN_MARKS = re.compile(r"[*\[\]`]")
_NUMBER = re.compile(r"\d[\d.,]*\d|\d")
_CONFIDENCE_LINE = re.compile(r"g[üu]ven[^:\n]*:(.*)", re.IGNORECASE)
_TARGET_1W_LINE = re.compile(r"1\s*haftal[ıi]k[^:\n]*hedef[^:\n]*:(.*)", re.IGNORECASE)
_TARGET_MID_LINE = re.compile(r"(?:15\s*-\s*30\s*g[üu]nl[üu]k|orta\s+vade)[^:\n]*hedef[^:\n]*:(.*)", re.IGNORECASE)
_STOP_LINE = re.compile(r"(?:stop[\s\-_]*loss|stop\s+seviyesi|zarar\s+kes)[^:\n]*:(.*)", re.IGNORECASE)
MAX_CONFIDENCE_PCT = 100


def parse_price(token: str, reference_price: float) -> float | None:
    """
    Fiyat metnini sayıya çevirir. Nokta ve virgül hem ondalık hem binlik ayırıcı olabildiği için
    ('268.42', '268,42', '1.234,50') iki yorum da denenir ve referans fiyata yakın olan seçilir.
    """
    candidates = []
    for decimal_separator, thousands_separator in ((".", ","), (",", ".")):
        normalized = token.replace(thousands_separator, "")
        if normalized.count(decimal_separator) > 1:
            continue
        try:
            value = float(normalized.replace(decimal_separator, "."))
        except ValueError:
            continue
        if value > 0:
            candidates.append(value)
    if not candidates:
        return None
    if reference_price <= 0:
        return candidates[0]
    return min(candidates, key=lambda value: abs(math.log(value / reference_price)))


def _prices_after_label(pattern: re.Pattern, text: str, reference_price: float) -> list[float]:
    match = pattern.search(text)
    if not match:
        return []
    value_part = match.group(1).split("(")[0]  # parantez içi açıklamalardaki yüzdeler fiyat değildir
    prices = (parse_price(token, reference_price) for token in _NUMBER.findall(value_part))
    return [price for price in prices if price is not None]


def _band(pattern: re.Pattern, text: str, reference_price: float) -> tuple[float, float] | None:
    prices = _prices_after_label(pattern, text, reference_price)
    if not prices:
        return None
    first_two = prices[:2] if len(prices) >= 2 else [prices[0], prices[0]]
    return (min(first_two), max(first_two))


def parse_decision_fields(text: str, reference_price: float) -> dict:
    """
    Karar raporundan güven katsayısını, hedef bantlarını ve stop seviyesini ayıklar.
    Bulunamayan alan None döner; varsayılan değer uydurulmaz.
    """
    clean = _MARKDOWN_MARKS.sub("", text or "")

    confidence = None
    confidence_match = _CONFIDENCE_LINE.search(clean)
    if confidence_match:
        digits = re.search(r"\d+", confidence_match.group(1))
        if digits and int(digits.group(0)) <= MAX_CONFIDENCE_PCT:
            confidence = int(digits.group(0))

    stop_prices = _prices_after_label(_STOP_LINE, clean, reference_price)
    return {
        "confidence": confidence,
        "target_1w": _band(_TARGET_1W_LINE, clean, reference_price),
        "target_mid": _band(_TARGET_MID_LINE, clean, reference_price),
        "stop_loss": stop_prices[0] if stop_prices else None,
    }


def parse_verdict_type(text: str) -> str:
    """
    'YATIRIM KARARI:' satırındaki kararı AL / SAT / TUT olarak döndürür.
    Satır yoksa veya birden fazla karar içeriyorsa (doldurulmamış şablon) BELIRSIZ döner.
    Yalnızca tam kelime eşleşir; 'HAFTALIK' içindeki 'AL' karar sayılmaz.
    """
    match = _DECISION_LINE.search(text or "")
    if not match:
        return VERDICT_UNKNOWN

    value = match.group(1).replace("ı", "I").replace("i", "I").replace("İ", "I").upper()
    words = set(_WORD.findall(value))
    found = [verdict for verdict, keywords in _KEYWORDS.items() if words & keywords]
    return found[0] if len(found) == 1 else VERDICT_UNKNOWN
