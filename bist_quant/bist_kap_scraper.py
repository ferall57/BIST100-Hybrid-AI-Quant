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
from datetime import datetime

# Windows konsol Unicode uyumluluğu
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

KAP_API_URL = "https://www.kap.org.tr/tr/api/disclosures"
REQUEST_TIMEOUT_SECONDS = 6
NO_DATA = "VERİ YOK"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"

class BistKapScraper:
    def __init__(self, cache_ttl_seconds: int = 600):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.kap.org.tr/"
        })
        self.cache_ttl = cache_ttl_seconds
        self._cache = {}

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

    def _fetch_from_source(self, clean_t: str, max_items: int) -> tuple[list[dict], bool]:
        try:
            res = self.session.get(f"{KAP_API_URL}?stockCode={clean_t}", timeout=REQUEST_TIMEOUT_SECONDS)
            if res.status_code != 200:
                print(f"[UYARI] KAP kaynağı {clean_t} için HTTP {res.status_code} döndürdü.")
                return [], False
            raw_data = res.json()
        except Exception as e:
            print(f"[UYARI] KAP bildirimi alınamadı ({clean_t}): {e}")
            return [], False

        if isinstance(raw_data, dict):
            raw_data = raw_data.get("disclosures", [])
        if not isinstance(raw_data, list):
            print(f"[UYARI] KAP kaynağı {clean_t} için beklenmeyen biçimde yanıt verdi.")
            return [], False

        return [self._parse_api_item(item, clean_t) for item in raw_data[:max_items]], True

    def _parse_api_item(self, item: dict, ticker: str) -> dict:
        """KAP JSON bildirimini ayıklar ve anahtar kelime duygu skorunu atar."""
        title = item.get("title", item.get("disclosureTitle", "Özel Durum Açıklaması"))
        summary = item.get("summary", item.get("disclosureSummary", ""))
        publish_date = item.get("publishDate", item.get("ruleName", "Bugün"))
        disc_type = item.get("disclosureType", "Genel")

        sentiment, impact, score = self._classify_sentiment(title + " " + summary)

        return {
            "ticker": ticker,
            "title": title,
            "summary": summary,
            "date": publish_date,
            "type": disc_type,
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
