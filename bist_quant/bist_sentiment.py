#!/usr/bin/env python3
"""
BIST 100 Çok Modlu (Multi-Modal) NLP Haber ve KAP Duyarlılık Füzyon Motoru
Canlı KAP bildirimleri ve finansal haber akışını analiz edip sayısal duyarlılık skoruna (-1.0 ile +1.0)
dönüştürür ve teknik modellerle matematiksel olarak birleştirir.
"""

import os
import sys
import json
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from hybrid_agents.gemini_rotator import GeminiRotator
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


class BistSentimentEngine:
    """
    Borsa İstanbul hisseleri için Canlı KAP ve Finansal Haber NLP Duyarlılık Motoru.
    Haber akışını sayısal sinyale dönüştürür ve teknik modellerle füzyon (birleşim) sağlar.
    """

    def __init__(self, gemini_model: str = "gemini-2.5-flash", temperature: float = 0.2):
        self.llm = None
        if GEMINI_AVAILABLE:
            try:
                self.llm = GeminiRotator(model_name=gemini_model, temperature=temperature)
            except Exception as e:
                print(f"[BİLGİ] Gemini Rotator sentiment motoruna bağlanamadı: {e}")

        # BIST Türkçe Finansal Anahtar Kelime Ağırlık Sözlüğü (Heuristic Fallback)
        self.bullish_keywords = {
            "yeni iş ilişkisi": 1.0, "ihale kazandı": 1.0, "sözleşme imzaladı": 0.9,
            "rekor kâr": 0.9, "kârını artırdı": 0.8, "pay geri alımı": 0.85,
            "bedelsiz sermaye": 0.8, "temettü dağıtımı": 0.75, "kapasite artışı": 0.8,
            "yeni fabrika": 0.85, "ihracat rekoru": 0.8, "hedef fiyat yükseltti": 0.7,
            "al tavsiyesi": 0.7, "iş birliği": 0.65, "sipariş aldı": 0.8
        }
        self.bearish_keywords = {
            "zarar açıkladı": -0.9, "kârı düştü": -0.75, "üretim durdurma": -0.95,
            "ceza kesildi": -0.85, "spk incelemesi": -0.9, "dava açıldı": -0.7,
            "iflas": -1.0, "konkordato": -1.0, "hedef fiyat düşürdü": -0.7,
            "sat tavsiyesi": -0.75, "soruşturma": -0.8, "grev kararı": -0.85,
            "pazar payı kaybı": -0.7, "borç yapılandırması": -0.75
        }

    def fetch_live_news(self, ticker: str, limit: int = 25) -> list[dict]:
        """
        Google News TR ve KAP RSS akışlarından belirtilen hisse için en güncel haberleri çeker.
        """
        clean_ticker = ticker.replace(".IS", "").strip().upper()
        query = f'{clean_ticker} KAP OR {clean_ticker} hisse OR {clean_ticker} borsa'
        safe_query = urllib.parse.quote(query)
        url = f'https://news.google.com/rss/search?q={safe_query}&hl=tr&gl=TR&ceid=TR:tr'

        news_items = []
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            items = root.findall('./channel/item')

            for item in items[:limit]:
                title = item.find('title').text if item.find('title') is not None else ""
                pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ""
                link = item.find('link').text if item.find('link') is not None else ""
                source_elem = item.find('source')
                source = source_elem.text if source_elem is not None else "Google News TR"

                if title:
                    title = title.replace("&#39;", "'").replace("&quot;", '"').replace("&amp;", '&')
                    news_items.append({
                        "title": title,
                        "pub_date": pubDate,
                        "link": link,
                        "source": source
                    })
        except Exception as e:
            print(f"[UYARI] {ticker} canlı haber akışı çekilirken hata: {e}")

        return news_items

    def analyze_sentiment(self, ticker: str, news_items: list[dict] = None) -> dict:
        """
        Hisseye ait haberleri finansal NLP süzgecinden geçirir.
        Duyarlılık Skoru (-1.0 ile +1.0), Etki Şiddeti (0.0 ile 1.0) ve Katalizör Tespiti üretir.
        """
        clean_ticker = ticker.replace(".IS", "").strip().upper()
        if news_items is None:
            news_items = self.fetch_live_news(ticker, limit=20)

        if not news_items:
            return {
                "ticker": ticker,
                "sentiment_score": 0.0,
                "impact_intensity": 0.0,
                "sentiment_label": "NÖTR (Haber Yok)",
                "catalyst_detected": False,
                "bearish_catalyst_detected": False,
                "news_count": 0,
                "summary": f"{ticker} için son dönemde manşet oluşturan kritik bir KAP veya finans haberi bulunamadı.",
                "key_catalysts": []
            }

        # 1. Öncelik: Gemini LLM Tabanlı Derin Finansal NLP Analizi
        if self.llm is not None:
            try:
                news_text = "\n".join([f"- [{item.get('pub_date', '')}] ({item.get('source', '')}) {item.get('title', '')}" for item in news_items[:15]])
                
                prompt = f"""Sen kurumsal bir Kuant Fonunda görev yapan Baş Finansal NLP ve KAP Duyarlılık Analistisin.
Görevin, aşağıdaki Borsa İstanbul ({clean_ticker}) hissesine ait en güncel KAP bildirimleri ve haber başlıklarını analiz edip kesinlikle geçerli bir JSON nesnesi döndürmektir.

Haber Başlıkları:
{news_text}

JSON Çıktı Formatı (Yalnızca bu JSON'ı döndür, markdown bloğu veya başka metin ekleme):
{{
  "sentiment_score": <float: -1.0 (Aşırı Negatif/Kriz) ile +1.0 (Aşırı Pozitif/Mega Katalizör) arasında>,
  "impact_intensity": <float: 0.0 (Önemsiz/Gürültü) ile 1.0 (Piyasa Yapıcı/Dev İhale/Rekor Bilanço) arasında>,
  "sentiment_label": "<ÇOK POZİTİF | POZİTİF | NÖTR | NEGATİF | ÇOK NEGATİF>",
  "catalyst_detected": <true/false: Şirket cirosunu/kârını ciddi artıracak yeni sözleşme, dev ihale, pay geri alımı var mı?>,
  "bearish_catalyst_detected": <true/false: Üretim durdurma, SPK cezası, dev zarar, iflas/dava gibi çöküş tetikleyicisi var mı?>,
  "summary": "<2 cümlelik Türkçe yönetici özeti>",
  "key_catalysts": ["<Öne çıkan en kritik 1-3 haber maddesi>"]
}}"""
                response = self.llm.invoke(prompt)
                if hasattr(response, "content"):
                    raw_text = response.content if isinstance(response.content, str) else str(response.content)
                else:
                    raw_text = str(response)
                
                # JSON bloğunu ayıkla
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    data["ticker"] = ticker
                    data["news_count"] = len(news_items)
                    data["sentiment_score"] = max(-1.0, min(1.0, float(data.get("sentiment_score", 0.0))))
                    data["impact_intensity"] = max(0.0, min(1.0, float(data.get("impact_intensity", 0.0))))
                    return data
            except Exception as e:
                print(f"[UYARI] Gemini sentiment analizi fallback'e yönlendiriliyor: {e}")

        # 2. Öncelik: Kural Tabanlı Finansal Heuristic NLP Fallback (Gemini Kota/İnternet Yoksa)
        return self._heuristic_sentiment_analysis(ticker, news_items)

    def _heuristic_sentiment_analysis(self, ticker: str, news_items: list[dict]) -> dict:
        """Gemini API kota dolumu veya çevrimdışı durumlarda çalışan finansal kural tabanlı NLP motoru."""
        total_score = 0.0
        match_count = 0
        detected_catalysts = []
        is_bearish_cat = False
        is_bullish_cat = False

        for item in news_items:
            title_lower = item["title"].lower()
            
            for kw, score in self.bullish_keywords.items():
                if kw in title_lower:
                    total_score += score
                    match_count += 1
                    detected_catalysts.append(f"🟢 {item['title']}")
                    if score >= 0.85:
                        is_bullish_cat = True

            for kw, score in self.bearish_keywords.items():
                if kw in title_lower:
                    total_score += score
                    match_count += 1
                    detected_catalysts.append(f"🔴 {item['title']}")
                    if score <= -0.85:
                        is_bearish_cat = True

        if match_count > 0:
            avg_score = max(-1.0, min(1.0, total_score / match_count))
            impact = min(1.0, 0.3 + (match_count * 0.15))
        else:
            avg_score = 0.0
            impact = 0.1

        if avg_score >= 0.4:
            label = "ÇOK POZİTİF" if avg_score >= 0.7 else "POZİTİF"
        elif avg_score <= -0.4:
            label = "ÇOK NEGATİF" if avg_score <= -0.7 else "NEGATİF"
        else:
            label = "NÖTR"

        return {
            "ticker": ticker,
            "sentiment_score": round(avg_score, 2),
            "impact_intensity": round(impact, 2),
            "sentiment_label": label,
            "catalyst_detected": is_bullish_cat,
            "bearish_catalyst_detected": is_bearish_cat,
            "news_count": len(news_items),
            "summary": f"{ticker} için incelenen {len(news_items)} haberde duyarlılık {label} ({avg_score:+.2f}) olarak hesaplandı.",
            "key_catalysts": detected_catalysts[:3]
        }

    def fuse_with_technical_signal(
        self,
        tech_expected_return: float,
        sentiment_data: dict,
        volatility_atr_pct: float = 3.5,
        base_news_weight: float = 0.35
    ) -> dict:
        """
        Çok Modlu (Multi-Modal) Hibrit Füzyon Formülü:
        Teknik zaman serisi tahmini ile KAP/Haber duyarlılığını matematiksel olarak birleştirir.
        Dinamik Alım Eşiği (min_thresh) ve İz Süren Stop mesafesini modüle eder.
        """
        s_news = float(sentiment_data.get("sentiment_score", 0.0))
        i_impact = float(sentiment_data.get("impact_intensity", 0.0))
        is_bull_cat = sentiment_data.get("catalyst_detected", False)
        is_bear_cat = sentiment_data.get("bearish_catalyst_detected", False)

        # Efektif haber ağırlığı (Haberin etki şiddeti yüksekse modele etkisi artar)
        effective_weight = base_news_weight * (0.5 + 0.5 * i_impact)
        effective_weight = max(0.10, min(0.60, effective_weight))

        # Haber kaynaklı getiri ivmesi (%): Sentiment * Impact * Volatilite
        news_return_momentum = s_news * i_impact * (volatility_atr_pct * 1.5)

        # 📐 Nihai Hibrit Beklenen Getiri Formülü:
        # R_fused = (1 - w) * R_tech + w * R_news
        fused_expected_return = ((1.0 - effective_weight) * tech_expected_return) + (effective_weight * news_return_momentum)

        # 🎯 Dinamik Giriş Eşiği Modülasyonu:
        # Güçlü pozitif katalizör varsa giriş barajı düşürülür (ralli kaçırılmaz)
        # Negatif haber varsa giriş barajı yükseltilir
        base_threshold = 0.8
        threshold_offset = -0.5 * s_news * i_impact
        modulated_threshold = max(0.1, base_threshold + threshold_offset)

        # 🛡️ İz Süren Stop Mesafesi Modülasyonu (Kârı Koşturma Kalkanı):
        # Güçlü katalizörlü hissede erken silkelenmeyi önlemek için stop mesafesi genişletilir
        base_trailing_pct = 4.5
        trailing_multiplier = 1.0 + (0.6 * max(0.0, s_news) * i_impact)
        modulated_trailing_pct = min(8.5, base_trailing_pct * trailing_multiplier)

        # 🛑 Rejim Override: Eğer devasa negatif katalizör varsa teknik alım sinyali iptal edilir
        force_cash = is_bear_cat or (s_news < -0.65 and i_impact > 0.6)

        # Karar Tavsiyesi
        if force_cash:
            recommendation = "🚨 KESİN NAKİTTE KAL (Yüksek Negatif Katalizör Riski)"
        elif fused_expected_return > modulated_threshold and s_news >= 0.0:
            recommendation = "🚀 GÜÇLÜ AL (KAP/Haber Katalizörü Destekli)" if is_bull_cat else "🟢 AL (Pozitif Hibrit Görünüm)"
        elif fused_expected_return > modulated_threshold:
            recommendation = "🟡 TEMKİNLİ AL (Teknik İyi, Haber Nötr/Zayıf)"
        else:
            recommendation = "⚪ NAKİTTE BEKLE (Yetersiz Hibrit Sinyal)"

        return {
            "tech_expected_return": round(tech_expected_return, 2),
            "news_momentum_return": round(news_return_momentum, 2),
            "fused_expected_return": round(fused_expected_return, 2),
            "effective_news_weight": round(effective_weight, 2),
            "modulated_threshold": round(modulated_threshold, 2),
            "modulated_trailing_pct": round(modulated_trailing_pct, 2),
            "force_cash_defense": force_cash,
            "recommendation": recommendation,
            "sentiment_label": sentiment_data.get("sentiment_label", "NÖTR")
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BIST Canlı KAP ve Finans Haberleri NLP Duyarlılık Motoru")
    parser.add_argument("ticker", type=str, default="THYAO.IS", nargs="?", help="BIST Sembolü (Örn: ASELS.IS)")
    args = parser.parse_args()

    engine = BistSentimentEngine()
    print(f"\n🔍 [{args.ticker}] için Canlı KAP & Haber NLP Analizi Başlatılıyor...")
    sentiment_result = engine.analyze_sentiment(args.ticker)

    print("\n" + "="*80)
    print(f"📊 KAP & HABER DUYARLILIK KARNESİ: {sentiment_result['ticker']}")
    print("="*80)
    print(f"🎯 Duyarlılık Skoru (Sentiment) : {sentiment_result['sentiment_score']:+.2f} [-1.0 ile +1.0]")
    print(f"💥 Etki Şiddeti (Impact)        : %{sentiment_result['impact_intensity']*100:.0f}")
    print(f"🏷️ Duyarlılık Derecesi          : {sentiment_result['sentiment_label']}")
    print(f"🚀 Pozitif Katalizör Tespiti     : {'EVET 🟢' if sentiment_result['catalyst_detected'] else 'YOK ⚪'}")
    print(f"🚨 Negatif Kriz Katalizörü       : {'EVET 🔴' if sentiment_result['bearish_catalyst_detected'] else 'YOK ⚪'}")
    print(f"📰 İncelenen Haber Sayısı        : {sentiment_result['news_count']} Adet")
    print(f"📝 Yönetici Özeti                : {sentiment_result['summary']}")

    if sentiment_result.get("key_catalysts"):
        print("\n📌 Öne Çıkan Başlıklar:")
        for cat in sentiment_result["key_catalysts"]:
            print(f"   {cat}")

    # Örnek Füzyon Testi (Teknik model +%0.5 bekliyorken)
    fusion = engine.fuse_with_technical_signal(tech_expected_return=0.5, sentiment_data=sentiment_result)
    print("\n" + "-"*80)
    print("🧠 HİBRİT TEKNİK + NLP FÜZYON ÇIKTISI:")
    print(f"  • Teknik Model Beklentisi : %{fusion['tech_expected_return']:+.2f}")
    print(f"  • Haber İvmesi Katkısı    : %{fusion['news_momentum_return']:+.2f}")
    print(f"  • 🏆 Nihai Füzyon Getirisi: %{fusion['fused_expected_return']:+.2f}")
    print(f"  • Dinamik Alım Barajı     : %{fusion['modulated_threshold']:.2f}")
    print(f"  • Kârı Koşturma Stop Mes. : %{fusion['modulated_trailing_pct']:.2f}")
    print(f"  • Karar Önerisi           : {fusion['recommendation']}")
    print("="*80 + "\n")
