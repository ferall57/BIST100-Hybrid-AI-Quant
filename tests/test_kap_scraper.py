from datetime import datetime, timezone

import pytest

from bist_quant import bist_kap_crawler
from bist_quant.bist_kap_crawler import BistKapCrawler, filter_recent_news, fold_turkish, is_relevant_headline, name_phrase
from bist_quant.bist_kap_scraper import (
    NO_DATA,
    BistKapScraper,
    build_criteria,
    match_member_oid,
    parse_publish_date,
)
from bist_quant.bist_sentiment import format_news_line


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("JSON değil")
        return self._payload


BIMAS_OID = "4028e4a140f2ed710140f2a7a59e0004"
SMART_SEARCH = [{"category": "smart", "results": [
    {"searchType": "C", "cmpOrFundCode": "sıse", "memberOrFundOid": "SISE_OID", "searchValue": "ŞİŞECAM"},
    {"searchType": "C", "cmpOrFundCode": "bımas", "memberOrFundOid": BIMAS_OID, "searchValue": "BİM BİRLEŞİK MAĞAZALAR A.Ş."},
]}]


def _disclosure(index=1683000, subject="Özel Durum Açıklaması (Genel)", summary="Pay Geri Alım Programı",
                publish="16.10.2026 09:35:22", disclosure_class="ODA"):
    return {"publishDate": publish, "disclosureClass": disclosure_class, "subject": subject,
            "summary": summary, "disclosureIndex": index, "stockCodes": "BIMAS"}


class _Session:
    """search/smart ve byCriteria çağrılarını ayrı ayrı yanıtlayan sahte oturum."""

    def __init__(self, search=None, criteria=None, error=None):
        self._search = search if search is not None else _Response(payload=SMART_SEARCH)
        self._criteria = criteria
        self._error = error
        self.posts = []

    def post(self, url, json=None, **kwargs):
        self.posts.append((url, json))
        if self._error:
            raise self._error
        return self._search if "search/smart" in url else self._criteria

    def get(self, *args, **kwargs):
        raise AssertionError("KAP istemcisi GET kullanmamalı")


def _scraper(**session_kwargs):
    scraper = BistKapScraper()
    scraper.session = _Session(**session_kwargs)
    return scraper


@pytest.mark.parametrize("ticker, oid", [("BIMAS", BIMAS_OID), ("SISE", "SISE_OID"), ("THYAO", None), ("BIM", None)])
def test_member_oid_needs_an_exact_code_match_after_turkish_folding(ticker, oid):
    assert match_member_oid(SMART_SEARCH, ticker) == oid


def test_member_oid_matches_one_code_in_a_comma_separated_list():
    results = [{"category": "smart", "results": [
        {"searchType": "C", "cmpOrFundCode": "ısatr,ısbtr,ısctr,ıskur,tıb", "memberOrFundOid": "ISCTR_OID"}]}]
    assert match_member_oid(results, "ISCTR") == "ISCTR_OID"


def test_criteria_body_filters_by_company_and_date_range():
    body = build_criteria("2026-09-16", "2026-10-16", "OID1")
    assert body["fromDate"] == "2026-09-16" and body["toDate"] == "2026-10-16"
    assert body["mkkMemberOidList"] == ["OID1"]
    assert body["memberType"] == "IGS"


def test_publish_date_is_read_as_istanbul_time():
    moment = parse_publish_date("16.10.2026 09:35:22")
    assert moment.isoformat() == "2026-10-16T09:35:22+03:00"
    assert parse_publish_date("bozuk") is None


def test_disclosures_come_newest_first_with_link_and_iso_date():
    payload = [_disclosure(index=1, publish="01.10.2026 10:00:00", summary="Eski"),
               _disclosure(index=2, publish="16.10.2026 09:35:22", summary="Yeni")]
    scraper = _scraper(criteria=_Response(payload=payload))

    items = scraper.fetch_disclosures("BIMAS.IS", max_items=5)

    assert [d["summary"] for d in items] == ["Yeni", "Eski"]
    assert items[0]["url"] == "https://www.kap.org.tr/tr/Bildirim/2"
    assert items[0]["date"] == "2026-10-16T09:35:22+03:00"
    assert items[0]["type"] == "ODA"
    url, body = scraper.session.posts[-1]
    assert url.endswith("/api/disclosure/members/byCriteria")
    assert body["mkkMemberOidList"] == [BIMAS_OID]


def test_member_oid_is_looked_up_once_per_ticker():
    scraper = _scraper(criteria=_Response(payload=[]))
    scraper.fetch_disclosures("BIMAS.IS", max_items=3)
    scraper.fetch_disclosures("BIMAS.IS", max_items=5)

    searches = [url for url, _ in scraper.session.posts if "search/smart" in url]
    assert len(searches) == 1


def test_company_missing_from_kap_search_counts_as_no_data():
    scraper = _scraper(search=_Response(payload=[{"category": "smart", "results": []}]), criteria=_Response(payload=[]))

    assert NO_DATA in scraper.generate_kap_report("BIMAS.IS")


def test_unreachable_source_reports_no_data_instead_of_all_clear():
    scraper = _scraper(error=ConnectionError("zaman aşımı"))

    report = scraper.generate_kap_report("BIMAS.IS")

    assert NO_DATA in report
    assert "olağan seyrinde" not in report
    assert "bulunmamaktadır" not in report
    assert "POZİTİF" not in report


def test_unreachable_source_returns_no_disclosures():
    scraper = _scraper(error=ConnectionError("zaman aşımı"))

    assert scraper.fetch_disclosures("BIMAS.IS") == []


def test_non_json_or_error_status_counts_as_no_data():
    assert NO_DATA in _scraper(criteria=_Response(status_code=200, payload=None)).generate_kap_report("BIMAS.IS")
    assert NO_DATA in _scraper(criteria=_Response(status_code=403, payload=[])).generate_kap_report("BIMAS.IS")


def test_reachable_source_with_no_items_is_not_reported_as_good_news():
    scraper = _scraper(criteria=_Response(payload=[]))

    report = scraper.generate_kap_report("BIMAS.IS")

    assert NO_DATA not in report
    assert "bildirim listelenmedi" in report
    assert "POZİTİF" not in report


NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


def test_stale_news_is_dropped_and_recent_news_comes_newest_first():
    items = [
        {"title": "Temmuz haberi", "published": "Thu, 23 Jul 2026 08:00:00 GMT"},
        {"title": "Geçen hafta", "published": "Thu, 01 Oct 2026 08:00:00 GMT"},
        {"title": "Dün", "published": "Tue, 06 Oct 2026 08:00:00 GMT"},
    ]

    recent = filter_recent_news(items, now=NOW, max_age_days=14)

    assert [item["title"] for item in recent] == ["Dün", "Geçen hafta"]
    assert [item["age_days"] for item in recent] == [1, 6]


def test_news_with_unreadable_date_is_kept_last_with_unknown_age():
    items = [{"title": "Tarihsiz", "published": "bilinmiyor"}, {"title": "Dün", "published": "2026-10-06T08:00:00+00:00"}]

    recent = filter_recent_news(items, now=NOW, max_age_days=14)

    assert [item["title"] for item in recent] == ["Dün", "Tarihsiz"]
    assert recent[1]["age_days"] is None


def test_news_line_shows_age_before_the_headline():
    assert format_news_line({"title": "Başlık", "source": "Kaynak", "age_days": 3}) == "- [3 gün önce] (Kaynak) Başlık"
    assert "yaşı bilinmiyor" in format_news_line({"title": "Başlık", "source": "Kaynak", "age_days": None})


def test_turkish_text_is_folded_for_matching():
    assert fold_turkish("Türk Hava Yolları İŞ Bankası, Koç!") == "turk hava yollari is bankasi koc"


@pytest.mark.parametrize("name, phrase", [
    ("Turk Hava Yollari Anonim Ortakligi", "turk hava"),
    ("Koç Holding A.Ş.", "koc holding"),
    ("Aselsan Elektronik Sanayi ve Ticaret A.S.", "aselsan elektronik"),
    ("BIMAS", "bimas"),
    (None, None),
    ("", None),
])
def test_company_name_is_reduced_to_a_distinctive_phrase(name, phrase):
    assert name_phrase(name) == phrase


def test_headline_is_relevant_when_it_names_the_ticker_or_the_company():
    assert is_relevant_headline("TÜRK HAVA YOLLARI (THYAO) 05 Ekim Günlük Teknik Analiz", "THYAO", "turk hava")
    assert is_relevant_headline("İş Yatırım İşlemleri: ASELS, ASTOR, THYAO Takip Edildi", "THYAO", None)
    assert is_relevant_headline("Türk Hava Yolları, CFM ile mutabakat imzaladı", "THYAO", "turk hava")


def test_headline_without_ticker_or_company_is_not_relevant():
    assert not is_relevant_headline("Yurtdışına 3 milyarlık transfer!", "THYAO", "turk hava")
    assert not is_relevant_headline("Borsa güne yükselişle başladı", "THYAO", "turk hava")
    assert not is_relevant_headline("THYAOX diye bir kod yok", "THYAO", None)


def _rss(*titles):
    published = (datetime.now(timezone.utc)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    items = "".join(f"<item><title>{t} - Kaynak</title><pubDate>{published}</pubDate></item>" for t in titles)
    return f"<rss><channel>{items}</channel></rss>".encode("utf-8")


def test_news_search_keeps_only_headlines_about_the_company(monkeypatch):
    requested = []

    def fake_get(url, *args, **kwargs):
        requested.append(url)
        assert "kap.org.tr" not in url, "KAP artık GET ile sorgulanmıyor"
        response = _Response(status_code=200)
        response.content = _rss("THYAO yeni rekor kırdı", "Yurtdışına 3 milyarlık transfer!", "Türk Hava Yolları CFM ile anlaştı")
        return response

    monkeypatch.setattr(bist_kap_crawler.requests, "get", fake_get)

    crawler = BistKapCrawler(kap_scraper=_scraper(criteria=_Response(status_code=503, payload=[])))
    items = crawler.fetch_disclosures_direct("THYAO.IS", company_name="Turk Hava Yollari Anonim Ortakligi")

    assert [item["title"] for item in items] == ["THYAO yeni rekor kırdı", "Türk Hava Yolları CFM ile anlaştı"]
    assert "%22THYAO%22" in requested[-1] and "turk%20hava" in requested[-1]


def test_real_disclosures_are_listed():
    scraper = _scraper(criteria=_Response(payload=[_disclosure(subject="Payların Geri Alınmasına İlişkin Bildirim", summary="Pay Geri Alım Programı")]))

    report = scraper.generate_kap_report("BIMAS.IS")

    assert "Pay Geri Alım Programı" in report
    assert NO_DATA not in report


def test_crawler_uses_official_kap_disclosures_before_news(monkeypatch):
    def fake_get(*args, **kwargs):
        raise AssertionError("KAP bildirimi varken haber aramasına düşülmemeli")

    monkeypatch.setattr(bist_kap_crawler.requests, "get", fake_get)
    recent = datetime.now(timezone.utc).astimezone().strftime("%d.%m.%Y %H:%M:%S")
    scraper = _scraper(criteria=_Response(payload=[_disclosure(index=7, publish=recent, summary="Yeni iş ilişkisi")]))

    items = BistKapCrawler(kap_scraper=scraper).fetch_disclosures_direct("BIMAS.IS")

    assert len(items) == 1
    assert items[0]["is_official_kap"] is True
    assert "Yeni iş ilişkisi" in items[0]["title"]
    assert items[0]["age_days"] is not None


def test_other_institution_notices_are_listed_without_a_sentiment_score():
    # DKB: Borsa/SPK gibi başka kurumların duyurusu. "SPK İşlem Yasağı" duyurusu aynı gün birçok hissede çıkar;
    # şirket hakkında olumsuz haber değildir, anahtar kelime skoru almamalı.
    notice = _disclosure(subject="SPK İşlem Yasağı Nedeniyle Pay Duyurusu",
                         summary="SPK İşlem Yasağı Nedeniyle Pay Duyurusu", disclosure_class="DKB")
    scraper = _scraper(criteria=_Response(payload=[notice]))

    item = scraper.fetch_disclosures("BIMAS.IS")[0]
    report = scraper.generate_kap_report("BIMAS.IS")

    assert item["sentiment_score"] == 0.0
    assert "NEGATİF" not in report
    assert "SPK İşlem Yasağı" in report
