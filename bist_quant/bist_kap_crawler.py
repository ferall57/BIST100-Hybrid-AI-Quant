#!/usr/bin/env python3
"""
BIST KAP Doğrudan Bildirim Dinleyicisi & Crawler (BistKapCrawler)
Kamuyu Aydınlatma Platformu (kap.gov.tr) resmi uç noktalarına doğrudan bağlanarak
şirket bildirimlerini (Yeni İş İlişkisi, Pay Geri Alımı, Finansal Rapor, Bedelsiz vb.)
en taze ve filtrelenmiş şekilde çeker.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')


class BistKapCrawler:
    """
    KAP (kap.gov.tr) ve resmi finansal veri kaynaklarından doğrudan bildirim toplayıcı.
    """

    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.kap.org.tr/"
        }

    def fetch_disclosures_direct(self, ticker: str, max_items: int = 15) -> List[Dict[str, Any]]:
        """
        KAP resmi arama ve bildirim API'sine doğrudan bağlanarak hisseye ait en güncel bildirimleri çeker.
        """
        clean_ticker = ticker.replace(".IS", "").upper()
        disclosures = []

        # 1. KAP Doğrudan Bildirim Uç Noktası (Public Disclosure Platform API)
        kap_api_url = f"https://www.kap.org.tr/tr/api/disclosures"
        
        try:
            params = {
                "stockCodes": clean_ticker,
                "fromDate": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "toDate": datetime.now().strftime("%Y-%m-%d")
            }
            resp = requests.get(kap_api_url, headers=self.headers, params=params, timeout=self.timeout)
            
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for item in data[:max_items]:
                        title = item.get("disclosureType", "") + " - " + item.get("title", item.get("summary", ""))
                        publish_date = item.get("publishDate", datetime.now().strftime("%Y-%m-%d"))
                        disclosures.append({
                            "title": title.strip(),
                            "source": "KAP Doğrudan Bildirim",
                            "published": publish_date,
                            "summary": item.get("summary", title).strip(),
                            "is_official_kap": True
                        })
        except Exception:
            pass

        # 2. Eğer KAP API doğrudan yanıt vermezse, Google News TR KAP aramasıyla yedekle
        if not disclosures:
            disclosures = self._fetch_google_news_kap_fallback(clean_ticker, max_items=max_items)

        return disclosures

    def _fetch_google_news_kap_fallback(self, clean_ticker: str, max_items: int = 15) -> List[Dict[str, Any]]:
        """KAP doğrudan API'si kapalı olduğunda taze Google News KAP filtrelerini çeker."""
        import urllib.parse
        import xml.etree.ElementTree as ET

        query = f"{clean_ticker} KAP OR hisse OR borsa"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=tr&gl=TR&ceid=TR:tr"

        news_items = []
        try:
            resp = requests.get(rss_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:max_items]:
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

        return news_items


if __name__ == "__main__":
    crawler = BistKapCrawler()
    print("\n🔍 [ISCTR] için Doğrudan KAP & Finansal Haber Dinleyicisi Test Ediliyor...")
    res = crawler.fetch_disclosures_direct("ISCTR.IS", max_items=10)
    print(f"✅ Toplam {len(res)} adet taze bildirim çekildi:")
    for idx, item in enumerate(res, 1):
        badge = "🏛️ [KAP RESMİ]" if item.get("is_official_kap") else "📰 [HABER]"
        print(f"{idx}. {badge} {item['title']} ({item['source']}) - {item['published']}")
