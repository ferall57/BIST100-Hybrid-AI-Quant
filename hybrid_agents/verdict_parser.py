"""Baş Portföy Müdürü çıktısından yapılandırılmış yatırım kararını ayıklar."""

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
