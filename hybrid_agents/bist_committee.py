import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime

from hybrid_agents.gemini_rotator import GeminiRotator
from hybrid_agents.prompts import (
    BIST_FUNDAMENTAL_ANALYST_PROMPT,
    BIST_TECHNICAL_MACRO_PROMPT,
    BIST_BULL_RESEARCHER_PROMPT,
    BIST_BEAR_RESEARCHER_PROMPT,
    BIST_PORTFOLIO_MANAGER_PROMPT
)
from bist_quant.bist_econometrics import BistEconometrics

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(ROOT_DIR, "outputs", "reports")

try:
    from bist_quant.bist_kronos_quant import BistKronosQuant
    QUANT_AVAILABLE = True
except Exception as e:
    QUANT_AVAILABLE = False
    print(f"[UYARI] BistKronosQuant motoru bağlanamadi: {e}")

class BistHybridCommittee:
    """
    Kronos-Base Quant tahmini ile TradingAgents felsefesindeki çoklu yapay zeka komitesini
    (Temel, Teknik, Boğa, Ayı ve Portföy Müdürü) buluşturan ana merkez sinir ağı.
    """
    def __init__(self, gemini_model: str = "gemini-2.5-pro", temperature: float = 0.3):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        print("🏛️ BIST Hibrit Yapay Zeka Komitesi Toplanıyor...")
        
        # 3'lü Gemini rotasyon motorunu başlat
        self.llm = GeminiRotator(model_name=gemini_model, temperature=temperature)
        self.econometric_engine = BistEconometrics()
        
        if QUANT_AVAILABLE:
            self.quant_engine = BistKronosQuant(use_base_model=True)
        else:
            self.quant_engine = None

    def _format_financial_ratios(self, info: dict) -> str:
        """Yahoo Finance meta verisinden temel finansal ve bilanço rasyolarını düzenli bir tabloya dönüştürür."""
        def _val(key, fmt="{:.2f}", mul=1.0, suffix=""):
            v = info.get(key)
            if v is None or v == "":
                return "N/A"
            try:
                val = float(v) * mul
                return fmt.format(val) + suffix
            except (ValueError, TypeError):
                return str(v)

        def _cap_val(v):
            if not v or v == "N/A":
                return "N/A"
            try:
                num = float(v)
                if num >= 1e12:
                    return f"{num / 1e12:.2f} Trilyon TRY"
                elif num >= 1e9:
                    return f"{num / 1e9:.2f} Milyar TRY"
                elif num >= 1e6:
                    return f"{num / 1e6:.2f} Milyon TRY"
                return f"{num:,.0f} TRY"
            except (ValueError, TypeError):
                return str(v)

        pe = _val("trailingPE", "{:.2f}")
        fwd_pe = _val("forwardPE", "{:.2f}")
        pb = _val("priceToBook", "{:.2f}")
        ev_ebitda = _val("enterpriseToEbitda", "{:.2f}")
        roe = _val("returnOnEquity", "{:.2f}", mul=100.0, suffix="%")
        div_yield = _val("dividendYield", "{:.2f}", mul=100.0, suffix="%")
        beta = _val("beta", "{:.2f}")
        high_52 = _val("fiftyTwoWeekHigh", "{:.2f}", suffix=" TRY")
        low_52 = _val("fiftyTwoWeekLow", "{:.2f}", suffix=" TRY")
        mcap = _cap_val(info.get("marketCap"))
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")

        table = f"""| Finansal Gösterge / Rasyo | Değer | Sektör / Tanım |
| :--- | :--- | :--- |
| **Sektör / Endüstri** | {sector} | {industry} |
| **F/K (Fiyat/Kazanç - P/E)** | {pe} (İleri F/K: {fwd_pe}) | Hissenin kârlılık çarpanı |
| **PD/DD (Piyasa/Defter Değeri - P/B)** | {pb} | Özkaynak değerleme çarpanı |
| **FD/FAVÖK (EV/EBITDA)** | {ev_ebitda} | Operasyonel nakit kârlılığı çarpanı |
| **Özsermaye Kârlılığı (ROE)** | {roe} | Hissedar sermayesinin getiri verimi |
| **Temettü Verimi (Dividend Yield)** | {div_yield} | Yıllık kâr payı dağıtım oranı |
| **Piyasa Değeri (Market Cap)** | {mcap} | Toplam şirket büyüklüğü |
| **52 Haftalık Zirve / Dip** | {high_52} / {low_52} | Yıllık fiyat salınım koridoru |
| **Beta (Piyasa Hassasiyeti)** | {beta} | BIST 100 korelasyon katsayısı |"""
        return table

    def _fetch_macro_indicators(self) -> str:
        """BIST 100 endeks trendi ve USD/TRY kurunu canlı çekerek özetler."""
        try:
            df_xu = yf.Ticker("XU100.IS").history(period="5d")
            df_usd = yf.Ticker("TRY=X").history(period="5d")
            
            xu_close = "N/A"
            xu_change = "N/A"
            if not df_xu.empty and len(df_xu) >= 2:
                c1 = df_xu["Close"].iloc[-1]
                c0 = df_xu["Close"].iloc[-2]
                chg = ((c1 - c0) / c0) * 100.0
                xu_close = f"{c1:,.2f}"
                xu_change = f"%{chg:+.2f}"
            elif not df_xu.empty:
                xu_close = f"{df_xu['Close'].iloc[-1]:,.2f}"
                
            usd_close = "N/A"
            usd_change = "N/A"
            if not df_usd.empty and len(df_usd) >= 2:
                u1 = df_usd["Close"].iloc[-1]
                u0 = df_usd["Close"].iloc[-2]
                chg_u = ((u1 - u0) / u0) * 100.0
                usd_close = f"{u1:.4f} TRY"
                usd_change = f"%{chg_u:+.2f}"
            elif not df_usd.empty:
                usd_close = f"{df_usd['Close'].iloc[-1]:.4f} TRY"

            macro_text = f"""* **BIST 100 Endeksi (XU100):** {xu_close} puan (Son Gün Değişimi: {xu_change})
* **Dolar / TL Kuru (USD/TRY):** {usd_close} (Son Gün Değişimi: {usd_change})"""
            return macro_text
        except Exception as e:
            return f"* Makro göstergeler çekilemedi: {e}"

    def analyze_ticker(self, ticker: str, forecast_days: int = 15):
        if not ticker.endswith(".IS"):
            ticker += ".IS"
            
        print(f"\n🚀 === [{ticker}] İÇİN HİBRİT YAPAY ZEKA YATIRIM KOMİTESİ ANALİZİ BAŞLADI ===")
        
        # 0. Yahoo Finance Canlı Veri Setini İndir ve Güncelle
        print(f"📥 [CANLI PİYASA] {ticker} için en güncel OHLCV verileri Yahoo Finance'ten çekiliyor...")
        from bist_quant.bist_downloader import download_ticker_data
        download_ticker_data(ticker, period="5y", interval="1d", save_dir=os.path.join(ROOT_DIR, "bist_data", "raw"))

        ticker_obj = yf.Ticker(ticker)
        info = {}
        try:
            info = ticker_obj.info or {}
            company_name = info.get("longName", info.get("shortName", ticker))
        except Exception:
            company_name = ticker

        # Finansal rasyoları ve makro verileri hazırla
        financial_ratios = self._format_financial_ratios(info)
        macro_indicators = self._fetch_macro_indicators()

        raw_csv = os.path.join(ROOT_DIR, "bist_data", "raw", f"{ticker}_1d.csv")
        current_price = 0.0
        recent_history = "Veri okunamadı"
        econometric_report = "Ekonometrik veri hazır değil."
        
        if os.path.exists(raw_csv):
            df = pd.read_csv(raw_csv)
            current_price = float(df["close"].iloc[-1])
            recent_history = df.tail(5)[["timestamps", "close", "volume"]].to_string(index=False)
            try:
                econometric_report = self.econometric_engine.generate_econometric_report(df, ticker, forecast_days=forecast_days)
            except Exception as ee:
                econometric_report = f"Ekonometrik analiz hatası: {ee}"
            
        # 1. Aşama: Kronos-base Quant Raporunun Çıkartılması
        print(f"📊 [AŞAMA 1/4] Kronos-base Quant Yapay Zekası Mum Formasyonlarını Hesaplıyor...")
        kronos_report, chart_path = ("Kronos Quant verisi hazir degil.", None)
        
        if self.quant_engine:
            kronos_report, chart_path = self.quant_engine.generate_quant_report(ticker, pred_len=forecast_days)

        def extract_text(res):
            if hasattr(res, "content"):
                if isinstance(res.content, list) and len(res.content) > 0 and isinstance(res.content[0], dict) and "text" in res.content[0]:
                    return res.content[0]["text"]
                elif isinstance(res.content, str):
                    return res.content
            return str(res)

        # 2. Aşama: Analistler (Temel & Teknik-Makro)
        print(f"💼 [AŞAMA 2/4] Temel ve Teknik Stratejist Ajanlar Rapor Yazıyor (Gemini Rotator)...")
        
        # 2.1 Canlı İnternet Araması (KAP ve Güncel Haberler)
        print(f"🌍 [CANLI BAĞLANTI] {ticker} için güncel RSS/XML beslemeleri okunuyor (Google News TR, KAP)...")
        live_news = ""
        try:
            import urllib.request
            import urllib.parse
            import xml.etree.ElementTree as ET
            
            # Google News RSS (Türkiye) üzerinden özel hisse senedi araması
            clean_ticker = ticker.replace(".IS", "")
            # Şirketin uzun resmi adı yerine sadece borsa kodu (Örn: ISCTR) üzerinden geniş arama
            query = f'{clean_ticker} KAP OR {clean_ticker} hisse OR {clean_ticker} haber'
            safe_query = urllib.parse.quote(query)
            url = f'https://news.google.com/rss/search?q={safe_query}&hl=tr&gl=TR&ceid=TR:tr'
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
                
            root = ET.fromstring(xml_data)
            items = root.findall('./channel/item')
            
            # En güncel 25 haberi al (KAP bildirimleri dahil tüm kritik haberler)
            for item in items[:25]:
                title = item.find('title').text if item.find('title') is not None else ""
                pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ""
                if title:
                    # Rss içeriğini daha okunabilir yap
                    title = title.replace("&#39;", "'").replace("&quot;", '"')
                    live_news += f"- [{pubDate}] {title}\n"
                    
        except Exception as e:
            print(f"⚠️ RSS Haber okuması başarısız: {e}")
            
        if not live_news.strip():
            live_news = "Güncel piyasa/KAP haberi bulunamadı."

        current_date_str = datetime.now().strftime("%d %B %Y")
        
        prompt_fund = BIST_FUNDAMENTAL_ANALYST_PROMPT.format(
            ticker=ticker, 
            company_name=company_name, 
            financial_ratios=financial_ratios,
            macro_indicators=macro_indicators,
            live_news=live_news,
            current_date=current_date_str
        )
        res_fund = self.llm.invoke(prompt_fund)
        fundamental_report = extract_text(res_fund)
        
        prompt_tech = BIST_TECHNICAL_MACRO_PROMPT.format(
            ticker=ticker,
            current_price=current_price,
            recent_history=recent_history,
            macro_indicators=macro_indicators,
            econometric_report=econometric_report,
            kronos_report=kronos_report
        )
        res_tech = self.llm.invoke(prompt_tech)
        technical_report = extract_text(res_tech)
        
        # 3. Aşama: Boğa - Ayı Tartışması (The Debate)
        print(f"⚔️ [AŞAMA 3/4] Boğa (Bull) ve Ayı (Bear) Yapay Zekaları Masada Tartışıyor...")
        prompt_bull = BIST_BULL_RESEARCHER_PROMPT.format(
            ticker=ticker,
            fundamental_report=fundamental_report,
            technical_report=technical_report
        )
        res_bull = self.llm.invoke(prompt_bull)
        bull_thesis = extract_text(res_bull)
        
        prompt_bear = BIST_BEAR_RESEARCHER_PROMPT.format(
            ticker=ticker,
            bull_thesis=bull_thesis,
            fundamental_report=fundamental_report,
            technical_report=technical_report
        )
        res_bear = self.llm.invoke(prompt_bear)
        bear_thesis = extract_text(res_bear)
        
        # 4. Aşama: Portföy Yönetim Müdürü Kararı
        print(f"🏆 [AŞAMA 4/4] Baş Portföy Müdürü (Executive Manager) Nihai Kararı Açıklıyor...")
        prompt_mgr = BIST_PORTFOLIO_MANAGER_PROMPT.format(
            ticker=ticker,
            current_price=current_price,
            fundamental_report=fundamental_report,
            technical_report=technical_report,
            econometric_report=econometric_report,
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis
        )
        res_mgr = self.llm.invoke(prompt_mgr)
        executive_verdict = extract_text(res_mgr)
        
        # 5. Dev Kapsamlı Dosyayı Derle ve Kaydet
        full_dossier = f"""# 🏛️ BIST 100 HİBRİT YAPAY ZEKA KOMİTE RAPORU
**Tarih:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | **Sembol:** {ticker} | **Şirket:** {company_name}
**Aktif Model:** Kronos-Base Quant + Klasik Ekonometri & Monte Carlo + Gemini Rotational Multi-Agent Debate

---

{executive_verdict}

---

## 📊 ŞİRKET BİLANÇO & DEĞERLEME RASYOLARI
{financial_ratios}

### 🌐 Makroekonomik Piyasa Görünümü
{macro_indicators}

---

## 🔬 KRONOS-BASE KANTİTATİF VE TEKNİK ÖNGÖRÜLER
{kronos_report}
*(Görsel Grafik Kayıt Yeri: `{chart_path}`)*

---

## 📐 KLASİK EKONOMETRİ & 1.000 YOLLU MONTE CARLO SİMÜLASYONU
{econometric_report}

---

## 📋 KOMİTE ÜYELERİNİN DETAYLI ÇALIŞMA RAPORLARI

### 💼 1. Temel Analist Raporu
{fundamental_report}

---

### 📈 2. Teknik ve Makroekonomi Raporu
{technical_report}

---

### 🐂 3. Boğa (Bull) Araştırmacısı Savunması
{bull_thesis}

---

### 🐻 4. Ayı (Bear) Araştırmacısı Eleştiri ve Riskler
{bear_thesis}
"""
        save_file = os.path.join(REPORTS_DIR, f"{ticker.replace('.', '_')}_committee_report.md")
        with open(save_file, "w", encoding="utf-8") as f:
            f.write(full_dossier)
            
        print(f"\n✅ Kapsamlı Hibrit Rapor Tamamlanarak Kaydedildi -> {save_file}")
        return executive_verdict, save_file, chart_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BIST Hibrit Komite Çalıştırıcı")
    parser.add_argument("--ticker", type=str, default="THYAO.IS", help="İşteklenecek BIST sembolü")
    parser.add_argument("--days", type=int, default=15, help="Kronos tahmin gün sayısı")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Gemini modeli")
    
    args = parser.parse_args()
    committee = BistHybridCommittee(gemini_model=args.model)
    verdict, doc_path, img_path = committee.analyze_ticker(args.ticker, forecast_days=args.days)
    print("\n" + verdict)
