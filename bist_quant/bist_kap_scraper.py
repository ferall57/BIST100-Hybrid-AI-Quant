#!/usr/bin/env python3
"""
📂 KRONOS KAP (KAMUYU AYDINLATMA PLATFORMU) & BIST BİLDİRİM KAZIYICI

Özellikler:
1. Kamuyu Aydınlatma Platformu (KAP) bildirim akışını sorgular.
2. Kural tabanlı anahtar kelime analizi ile bildirimleri sınıflandırır:
   - 🟢 POZİTİF (Pay Geri Alımı, İhale Kazanımı, Yüksek Kâr)
   - 🔴 NEGATİF (SPK Cezası, Zarar, Yönetici İstifası)
   - ⚪ NÖTR (Rutin Genel Kurul, Kayıtlı Sermaye Tavanı vb.)
3. Kaynağa ulaşılamazsa bildirim uydurmaz; komiteye açıkça "VERİ YOK" bildirir.
"""

import sys
import time
import requests
from datetime import datetime, timedelta, timezone

# Windows konsol Unicode uyumluluğu
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

# kap.org.tr'nin kendi "Detaylı Sorgulama" sayfasının kullandığı uç noktalar (Ekim 2026'da tarayıcıdan izlendi).
# Eski GET /tr/api/disclosures uç noktası yanıt vermiyor.
KAP_BASE_URL = "https://www.kap.org.tr"
KAP_SEARCH_URL = f"{KAP_BASE_URL}/tr/api/search/smart"
KAP_DISCLOSURES_URL = f"{KAP_BASE_URL}/tr/api/disclosure/members/byCriteria"
KAP_DISCLOSURE_PAGE = f"{KAP_BASE_URL}/tr/Bildirim/{{index}}"
KAP_PUBLISH_FORMAT = "%d.%m.%Y %H:%M:%S"
ISTANBUL_TZ = timezone(timedelta(hours=3))
LISTED_COMPANIES = "IGS"  # İşlem gören şirketler
SEARCH_COMPANY_TYPE = "C"
LOOKBACK_DAYS = 30
# DKB: Borsa İstanbul, SPK gibi başka kurumların hisse hakkındaki duyuruları (devre kesici, işlem yasağı vb.).
# Şirketin kendi açıklaması değildir; anahtar kelime duygu skoru verilmez.
OTHER_INSTITUTION_CLASS = "DKB"
OTHER_INSTITUTION_SENTIMENT = "NÖTR / DİĞER KURUM DUYURUSU"
REQUEST_TIMEOUT_SECONDS = 10
NO_DATA = "VERİ YOK"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"

_CODE_FOLD = str.maketrans("ıİ", "iI")


def _fold_code(code: str) -> str:
    """KAP kodları küçük harf ve noktasız ı ile döner ('bımas'); hisse koduyla karşılaştırmak için sadeleştirir."""
    return str(code).translate(_CODE_FOLD).strip().upper()


def match_member_oid(search_results, ticker: str) -> str | None:
    """search/smart yanıtında hisse koduyla birebir eşleşen şirketin KAP kimliği; sıralamaya güvenilmez."""
    target = _fold_code(ticker)
    for category in search_results or []:
        for result in category.get("results", []):
            if result.get("searchType") != SEARCH_COMPANY_TYPE:
                continue
            codes = {_fold_code(c) for c in str(result.get("cmpOrFundCode") or "").split(",")}
            if target in codes:
                return result.get("memberOrFundOid")
    return None


def build_criteria(from_date: str, to_date: str, member_oid: str) -> dict:
    """Detaylı Sorgulama formunun gönderdiği gövde; yalnızca tarih aralığı ve şirket doldurulur."""
    return {
        "fromDate": from_date, "toDate": to_date, "memberType": LISTED_COMPANIES,
        "mkkMemberOidList": [member_oid], "inactiveMkkMemberOidList": [], "disclosureClass": "",
        "subjectList": [], "isLate": "", "mainSector": "", "sector": "", "subSector": "",
        "marketOid": "", "index": "", "bdkReview": "", "bdkMemberOidList": [], "year": "",
        "term": "", "ruleType": "", "period": "", "fromSrc": False, "srcCategory": "",
        "disclosureIndexList": [],
    }


def parse_publish_date(raw: str) -> datetime | None:
    """KAP yayım zamanı ('16.10.2026 09:35:22', İstanbul saati); okunamazsa None."""
    try:
        return datetime.strptime(str(raw).strip(), KAP_PUBLISH_FORMAT).replace(tzinfo=ISTANBUL_TZ)
    except ValueError:
        return None


class BistKapScraper:
    def __init__(self, cache_ttl_seconds: int = 600, lookback_days: int = LOOKBACK_DAYS):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr",
            "Content-Type": "application/json",
            "Referer": f"{KAP_BASE_URL}/tr/bildirim-sorgu"
        })
        self.cache_ttl = cache_ttl_seconds
        self.lookback_days = lookback_days
        self._cache = {}
        self._member_oids = {}

    def _clean_ticker(self, ticker: str) -> str:
        """'BIMAS.IS' veya 'BIMAS' formatını 'BIMAS' haline getirir."""
        return ticker.upper().replace(".IS", "").strip()

    def fetch_disclosures(self, ticker: str, max_items: int = 5) -> list[dict]:
        """Hissenin güncel KAP bildirimlerini döndürür; kaynağa ulaşılamazsa boş liste."""
        disclosures, _ = self._fetch_cached(ticker, max_items)
        return disclosures

    def _fetch_cached(self, ticker: str, max_items: int) -> tuple[list[dict], bool]:
        """(bildirimler, kaynak_yanıt_verdi_mi) çiftini önbellekli döndürür."""
        clean_t = self._clean_ticker(ticker)
        cache_key = f"{clean_t}_{max_items}"

        now = time.time()
        if cache_key in self._cache:
            result, exp = self._cache[cache_key]
            if now < exp:
                return result

        result = self._fetch_from_source(clean_t, max_items)
        self._cache[cache_key] = (result, now + self.cache_ttl)
        return result

    def _post_json(self, url: str, body: dict, clean_t: str):
        """POST atar; HTTP hatası, zaman aşımı veya JSON olmayan yanıtta None döner."""
        try:
            res = self.session.post(url, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
            if res.status_code != 200:
                print(f"[UYARI] KAP kaynağı {clean_t} için HTTP {res.status_code} döndürdü.")
                return None
            return res.json()
        except Exception as e:
            print(f"[UYARI] KAP bildirimi alınamadı ({clean_t}): {e}")
            return None

    def _member_oid(self, clean_t: str) -> str | None:
        if clean_t not in self._member_oids:
            results = self._post_json(KAP_SEARCH_URL, {"keyword": clean_t}, clean_t)
            if results is None:
                return None  # bağlantı hatası önbelleğe alınmaz
            oid = match_member_oid(results, clean_t)
            if oid is None:
                print(f"[UYARI] {clean_t} KAP şirket aramasında bulunamadı.")
            self._member_oids[clean_t] = oid
        return self._member_oids[clean_t]

    def _fetch_from_source(self, clean_t: str, max_items: int) -> tuple[list[dict], bool]:
        member_oid = self._member_oid(clean_t)
        if member_oid is None:
            return [], False

        today = datetime.now(ISTANBUL_TZ).date()
        body = build_criteria((today - timedelta(days=self.lookback_days)).isoformat(), today.isoformat(), member_oid)
        raw_data = self._post_json(KAP_DISCLOSURES_URL, body, clean_t)
        if not isinstance(raw_data, list):
            if raw_data is not None:
                print(f"[UYARI] KAP kaynağı {clean_t} için beklenmeyen biçimde yanıt verdi.")
            return [], False

        items = [self._parse_api_item(item, clean_t) for item in raw_data]
        oldest = datetime.min.replace(tzinfo=ISTANBUL_TZ)
        items.sort(key=lambda d: parse_publish_date(d["raw_date"]) or oldest, reverse=True)
        return items[:max_items], True

    def _parse_api_item(self, item: dict, ticker: str) -> dict:
        """KAP JSON bildirimini ayıklar ve anahtar kelime duygu skorunu atar."""
        title = item.get("subject") or "Özel Durum Açıklaması"
        summary = item.get("summary") or ""
        raw_date = item.get("publishDate", "")
        moment = parse_publish_date(raw_date)
        index = item.get("disclosureIndex")

        disclosure_class = item.get("disclosureClass") or "Genel"
        if disclosure_class == OTHER_INSTITUTION_CLASS:
            sentiment, impact, score = OTHER_INSTITUTION_SENTIMENT, "NEUTRAL", 0.0
        else:
            sentiment, impact, score = self._classify_sentiment(title + " " + summary)

        return {
            "ticker": ticker,
            "title": title,
            "summary": summary.strip(),
            "date": moment.isoformat() if moment else raw_date,
            "raw_date": raw_date,
            "type": disclosure_class,
            "url": KAP_DISCLOSURE_PAGE.format(index=index) if index is not None else None,
            "sentiment": sentiment,
            "impact_label": impact,
            "sentiment_score": score
        }

    def _classify_sentiment(self, text: str) -> tuple[str, str, float]:
        """Bildirim metninde anahtar kelimelere dayalı hızlı NLP duygu analizi."""
        t_low = text.lower()

        bullish_keywords = [
            "geri alım", "pay geri alım", "ihale", "yeni iş ilişkisi", "sözleşme",
            "kâr", "artış", "temettü", "bedelsiz", "rekor", "onaylandı", "satış hasılatı"
        ]
        bearish_keywords = [
            "ceza", "spk", "zarar", "dava", "iptal", "istifa", "soruşturma",
            "düşüş", "tahkim", "tedbir", "kısıtlama", "iflas"
        ]

        b_score = sum(1 for kw in bullish_keywords if kw in t_low)
        be_score = sum(1 for kw in bearish_keywords if kw in t_low)

        if b_score > be_score:
            return "POZİTİF (BOĞA)", "BULLISH_DISCLOSURE", round(min(1.0, 0.4 + (b_score * 0.2)), 2)
        elif be_score > b_score:
            return "NEGATİF (AYI)", "BEARISH_DISCLOSURE", round(max(-1.0, -0.4 - (be_score * 0.2)), 2)
        else:
            return "NÖTR / BİLGİLENDİRME", "NEUTRAL", 0.0

    def generate_kap_report(self, ticker: str) -> str:
        """Komite toplantısı için yapılandırılmış markdown KAP brifingi oluşturur."""
        disclosures, source_ok = self._fetch_cached(ticker, max_items=3)
        clean_t = self._clean_ticker(ticker)

        lines = [
            f"📰 **KAP (KAMUYU AYDINLATMA PLATFORMU) VE KURUMSAL BİLDİRİM BRİFİNGİ ({clean_t}):**",
            f"  • **İncelenen Şirket:** {clean_t} | **Tarama Zamanı:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ]

        if not source_ok:
            lines.append(f"  • **Durum:** {NO_DATA} — KAP kaynağına ulaşılamadı. Bu, olumlu ya da olumsuz bildirim "
                         "olmadığı anlamına gelmez; KAP bildirimleri hakkında çıkarım yapmayınız.")
            return "\n".join(lines)

        if not disclosures:
            lines.append("  • **Durum:** Kaynak yanıt verdi; bu hisse için bildirim listelenmedi.")
            return "\n".join(lines)

        for idx, d in enumerate(disclosures, 1):
            badge = "🟢" if "POZİTİF" in d["sentiment"] else ("🔴" if "NEGATİF" in d["sentiment"] else "⚪")
            lines.append(f"  {idx}. {badge} **[{d['type']}]** {d['title']}")
            if d.get("summary"):
                lines.append(f"     ↳ Özet: {d['summary'][:140]}...")
            lines.append(f"     ↳ Duygu Analizi: **{d['sentiment']}** (Skor: {d['sentiment_score']:+.2f})")

        return "\n".join(lines)

if __name__ == "__main__":
    scraper = BistKapScraper()
    print("BIMAS için KAP Bildirim Raporu:")
    print(scraper.generate_kap_report("BIMAS.IS"))
    print("\nTHYAO için KAP Bildirim Raporu:")
    print(scraper.generate_kap_report("THYAO.IS"))
