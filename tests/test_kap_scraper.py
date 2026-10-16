from datetime import datetime, timezone

import pytest

from bist_quant import bist_kap_crawler
from bist_quant.bist_kap_crawler import BistKapCrawler, filter_recent_news, fold_turkish, is_relevant_headline, name_phrase
from bist_quant.bist_kap_scraper import NO_DATA, BistKapScraper
from bist_quant.bist_sentiment import format_news_line


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("JSON değil")
        return self._payload


class _Session:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def get(self, *args, **kwargs):
        if self._error:
            raise self._error
        return self._response


def _scraper(**session_kwargs):
    scraper = BistKapScraper()
    scraper.session = _Session(**session_kwargs)
    return scraper


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
    assert NO_DATA in _scraper(response=_Response(status_code=200, payload=None)).generate_kap_report("BIMAS.IS")
    assert NO_DATA in _scraper(response=_Response(status_code=403, payload=[])).generate_kap_report("BIMAS.IS")


def test_reachable_source_with_no_items_is_not_reported_as_good_news():
    scraper = _scraper(response=_Response(payload=[]))

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
        if "kap.org.tr" in url:
            return _Response(status_code=503, payload=[])
        response = _Response(status_code=200)
        response.content = _rss("THYAO yeni rekor kırdı", "Yurtdışına 3 milyarlık transfer!", "Türk Hava Yolları CFM ile anlaştı")
        return response

    monkeypatch.setattr(bist_kap_crawler.requests, "get", fake_get)

    items = BistKapCrawler().fetch_disclosures_direct("THYAO.IS", company_name="Turk Hava Yollari Anonim Ortakligi")

    assert [item["title"] for item in items] == ["THYAO yeni rekor kırdı", "Türk Hava Yolları CFM ile anlaştı"]
    assert "%22THYAO%22" in requested[-1] and "turk%20hava" in requested[-1]


def test_real_disclosures_are_listed():
    scraper = _scraper(response=_Response(payload=[{"title": "Pay Geri Alım Programı", "summary": "", "disclosureType": "ÖDA"}]))

    report = scraper.generate_kap_report("BIMAS.IS")

    assert "Pay Geri Alım Programı" in report
    assert NO_DATA not in report
