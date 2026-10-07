from bist_quant.bist_kap_scraper import NO_DATA, BistKapScraper


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


def test_real_disclosures_are_listed():
    scraper = _scraper(response=_Response(payload=[{"title": "Pay Geri Alım Programı", "summary": "", "disclosureType": "ÖDA"}]))

    report = scraper.generate_kap_report("BIMAS.IS")

    assert "Pay Geri Alım Programı" in report
    assert NO_DATA not in report
