#!/usr/bin/env python3
"""
BIST KAP Doğrudan Bildirim Dinleyicisi & Crawler (BistKapCrawler)
Kamuyu Aydınlatma Platformu (kap.gov.tr) resmi uç noktalarına doğrudan bağlanarak
şirket bildirimlerini (Yeni İş İlişkisi, Pay Geri Alımı, Finansal Rapor, Bedelsiz vb.)
en taze ve filtrelenmiş şekilde çeker.
"""

import os
import re
import sys
import json
import time
import requests
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any

NEWS_MAX_AGE_DAYS = 14
SECONDS_PER_DAY = 86400.0


TURKISH_FOLD = str.maketrans("ıİşŞğĞüÜöÖçÇâÂîÎûÛ", "iissgguuooccaaiiuu")
# Şirket unvanındaki hukuki ekler ayırt edici değildir; haber başlıklarında da geçmez.
LEGAL_NAME_WORDS = frozenset({"anonim", "ortakligi", "sirketi", "as", "a", "s", "tas", "t", "sanayi", "ve", "ticaret", "inc", "corp"})
NAME_PHRASE_WORDS = 2


def fold_turkish(text: str) -> str:
    """Eşleştirme için metni sadeleştirir: Türkçe karakterleri ASCII'ye indirir, küçültür, noktalamayı atar."""
    folded = str(text).translate(TURKISH_FOLD).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", folded).split())


def name_phrase(company_name: str | None) -> str | None:
    """Şirket unvanının ilk ayırt edici kelimeleri ('Turk Hava Yollari A.O.' -> 'turk hava'); unvan yoksa None."""
    if not company_name:
        return None
    words = [word for word in fold_turkish(company_name).split() if word not in LEGAL_NAME_WORDS]
    return " ".join(words[:NAME_PHRASE_WORDS]) or None


def is_relevant_headline(title: str, clean_ticker: str, company_phrase: str | None) -> bool:
    """Başlıkta hisse kodu (tam kelime olarak) ya da şirket adı geçiyorsa haber o hisseyle ilgilidir."""
    folded = fold_turkish(title)
    if re.search(rf"\b{re.escape(clean_ticker.lower())}\b", folded):
        return True
    return bool(company_phrase) and company_phrase in folded


def published_at(item: dict) -> datetime | None:
    """Haberin yayım zamanını (UTC) döndürür; biçim tanınmıyorsa None."""
    raw = str(item.get("published", "")).strip()
    if not raw:
        return None
    for parse in (parsedate_to_datetime, datetime.fromisoformat):
        try:
            moment = parse(raw)
        except (TypeError, ValueError):
            continue
        return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)
    return None


def filter_recent_news(items: List[Dict[str, Any]], now: datetime, max_age_days: int = NEWS_MAX_AGE_DAYS) -> List[Dict[str, Any]]:
    """
    Yalnızca son max_age_days gün içindeki haberleri, en yeniden eskiye sıralı ve 'age_days' alanıyla döndürür.
    Tarihi okunamayan haberler yaşı bilinmediği için sona, 'age_days': None ile eklenir.
    """
    dated, undated = [], []
    for item in items:
        moment = published_at(item)
        if moment is None:
            undated.append({**item, "age_days": None})
            continue
        age_days = (now - moment).total_seconds() / SECONDS_PER_DAY
        if age_days <= max_age_days:
            dated.append({**item, "age_days": max(0, int(age_days))})
    return sorted(dated, key=lambda item: item["age_days"]) + undated

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')


class BistKapCrawler:
    """
    KAP (kap.gov.tr) ve resmi finansal veri kaynaklarından doğrudan bildirim toplayıcı.
    """

    def __init__(self, timeout: int = 8, kap_scraper=None):
        self.timeout = timeout
        self._kap_scraper = kap_scraper
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.kap.org.tr/"
        }

    def fetch_disclosures_direct(self, ticker: str, max_items: int = 15, company_name: str | None = None) -> List[Dict[str, Any]]:
        """
        KAP resmi arama ve bildirim API'sine doğrudan bağlanarak hisseye ait en güncel bildirimleri çeker.
        """
        clean_ticker = ticker.replace(".IS", "").upper()
        disclosures = []

        # 1. KAP'ın kendi bildirim sorgusu (bist_kap_scraper); yanıt yoksa boş liste döner
        for item in self._kap().fetch_disclosures(clean_ticker, max_items=max_items):
            title = f"{item['type']} - {item['title']}" + (f": {item['summary']}" if item["summary"] else "")
            disclosures.append({
                "title": title,
                "source": "KAP Doğrudan Bildirim",
                "published": item["date"],
                "summary": item["summary"] or item["title"],
                "url": item["url"],
                "is_official_kap": True
            })

        # 2. Eğer KAP API doğrudan yanıt vermezse, Google News TR KAP aramasıyla yedekle
        if not disclosures:
            disclosures = self._fetch_google_news_kap_fallback(clean_ticker, max_items=max_items, company_name=company_name)

        # Bayat haber güncel olay gibi sunulmasın: yalnızca son NEWS_MAX_AGE_DAYS gün, en yeni önce
        disclosures = filter_recent_news(disclosures, datetime.now(timezone.utc))[:max_items]

        return disclosures

    def _kap(self):
        if self._kap_scraper is None:
            from bist_quant.bist_kap_scraper import BistKapScraper
            self._kap_scraper = BistKapScraper()
        return self._kap_scraper

    def _fetch_google_news_kap_fallback(self, clean_ticker: str, max_items: int = 15, company_name: str | None = None) -> List[Dict[str, Any]]:
        """KAP doğrudan API'si kapalı olduğunda taze Google News KAP filtrelerini çeker."""
        import urllib.parse
        import xml.etree.ElementTree as ET

        # Eski sorgu ("KOD KAP OR hisse OR borsa") hisseyle ilgisiz borsa haberlerini de getiriyordu
        company_phrase = name_phrase(company_name)
        query = f'"{clean_ticker}"' + (f' OR "{company_phrase}"' if company_phrase else "")
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=tr&gl=TR&ceid=TR:tr"

        news_items = []
        try:
            resp = requests.get(rss_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item"):
                    title_elem = item.find("title")
                    pub_date_elem = item.find("pubDate")
                    desc_elem = item.find("description")

                    title = title_elem.text if title_elem is not None else ""
                    pub_date = pub_date_elem.text if pub_date_elem is not None else ""
                    desc = desc_elem.text if desc_elem is not None else ""

                    clean_title = title.split(" - ")[0] if " - " in title else title
                    source = title.split(" - ")[-1] if " - " in title else "Finansal Haber"

                    news_items.append({
                        "title": clean_title.strip(),
                        "source": source.strip(),
                        "published": pub_date,
                        "summary": desc.strip() if desc else clean_title.strip(),
                        "is_official_kap": "kap" in title.lower()
                    })
        except Exception:
            pass

        relevant = [item for item in news_items if is_relevant_headline(item["title"], clean_ticker, company_phrase)]
        if len(relevant) < len(news_items):
            print(f"[BİLGİ] {clean_ticker}: {len(news_items)} başlıktan {len(news_items) - len(relevant)} tanesi "
                  f"hisse kodu veya şirket adı geçmediği için elendi.")
        return relevant


if __name__ == "__main__":
    crawler = BistKapCrawler()
    print("\n🔍 [ISCTR] için Doğrudan KAP & Finansal Haber Dinleyicisi Test Ediliyor...")
    res = crawler.fetch_disclosures_direct("ISCTR.IS", max_items=10)
    print(f"✅ Toplam {len(res)} adet taze bildirim çekildi:")
    for idx, item in enumerate(res, 1):
        badge = "🏛️ [KAP RESMİ]" if item.get("is_official_kap") else "📰 [HABER]"
        print(f"{idx}. {badge} {item['title']} ({item['source']}) - {item['published']}")
