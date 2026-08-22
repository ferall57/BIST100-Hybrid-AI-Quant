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
from bist_quant.bist_sentiment import BistSentimentEngine
from bist_quant.bist_viop import BistViopEngine
from bist_quant.bist_akd_flow import BistAkdFlowEngine

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
    def __init__(self, gemini_model: str = "gemini-3.5-flash", temperature: float = 0.3):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        print("🏛️ BIST Hibrit Yapay Zeka Komitesi Toplanıyor...")
        
        # 3'lü Gemini rotasyon motorunu başlat
        self.llm = GeminiRotator(model_name=gemini_model, temperature=temperature)
        self.econometric_engine = BistEconometrics()
        self.sentiment_engine = BistSentimentEngine(gemini_model=gemini_model, temperature=0.2)
        self.viop_engine = BistViopEngine()
        self.akd_engine = BistAkdFlowEngine()
        
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
        
        # Temettü verimi: yfinance BIST hisselerinde bazen 4.36 (yüzde), bazen 0.0436 (oran) döndürür
        raw_div = info.get("dividendYield")
        div_yield = "N/A"
        if raw_div is not None and raw_div != "":
            try:
                d_val = float(raw_div)
                if d_val > 1.0:
                    div_yield = f"%{d_val:.2f}"
                elif d_val > 0:
                    div_yield = f"%{d_val * 100.0:.2f}"
                else:
                    div_yield = "%0.00"
            except (ValueError, TypeError):
                div_yield = str(raw_div)

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
        akd_report = "AKD ve Para Akışı verisi hazır değil."
        
        if os.path.exists(raw_csv):
            df = pd.read_csv(raw_csv)
            current_price = float(df["close"].iloc[-1])
            recent_history = df.tail(5)[["timestamps", "close", "volume"]].to_string(index=False)
            try:
                econometric_report = self.econometric_engine.generate_econometric_report(df, ticker, forecast_days=forecast_days)
            except Exception as ee:
                econometric_report = f"Ekonometrik analiz hatası: {ee}"
            try:
                akd_report = self.akd_engine.get_akd_summary_text(ticker, df)
            except Exception as e_akd:
                akd_report = f"AKD para akışı analiz hatası: {e_akd}"
            
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

        # 2. Aşama: Analistler (Temel, NLP Sentiment, AKD & Teknik-Makro)
        print(f"💼 [AŞAMA 2/4] Temel, NLP Sentiment, AKD Para Akışı ve Teknik Stratejist Ajanlar Rapor Yazıyor (Gemini Rotator)...")
        
        # 2.1 Canlı NLP KAP ve Haber Duyarlılık Analizi
        print(f"🌍 [CANLI BAĞLANTI] {ticker} için Canlı KAP ve Finans Haberleri NLP ile skorlanıyor...")
        sentiment_data = self.sentiment_engine.analyze_sentiment(ticker)
        
        live_news = f"""* **NLP Duyarlılık Skoru (Sentiment):** {sentiment_data.get('sentiment_score', 0.0):+.2f} ({sentiment_data.get('sentiment_label', 'NÖTR')}) | Etki Şiddeti: %{sentiment_data.get('impact_intensity', 0.0)*100:.0f}
* **Pozitif Katalizör:** {'EVET 🟢' if sentiment_data.get('catalyst_detected') else 'YOK ⚪'} | **Negatif Risk:** {'EVET 🔴' if sentiment_data.get('bearish_catalyst_detected') else 'YOK ⚪'}
* **Haber Analiz Özeti:** {sentiment_data.get('summary', '')}
"""
        if sentiment_data.get("key_catalysts"):
            live_news += "\n**Öne Çıkan KAP ve Haber Başlıkları:**\n"
            for cat in sentiment_data["key_catalysts"]:
                live_news += f"- {cat}\n"

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
            akd_report=akd_report,
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
        
        # 🛡️ DETERMINİSTİK HARD-GATE VETO KAPISI (Fix 5.1 - LLM Bullish Bias Guardrail)
        mc_data = self.econometric_engine.run_monte_carlo_simulation(df, days=forecast_days, num_sims=1000) if 'df' in locals() else {}
        akd_data = self.akd_engine.analyze_akd_profile(ticker, df) if 'df' in locals() else {}
        
        veto_triggered = False
        veto_reason = ""
        if mc_data.get("prob_positive", 50.0) < 38.0 and akd_data.get("cmf_20", 0.0) < -0.12:
            if "AL" in executive_verdict.upper() or "BUY" in executive_verdict.upper():
                veto_triggered = True
                veto_reason = f"Merton MC Kazanma Olasılığı (%{mc_data.get('prob_positive', 0.0):.1f} < %38) ve CMF Para Çıkışı ({akd_data.get('cmf_20', 0.0):.3f})"
                executive_verdict = f"""> [!WARNING]
> 🛡️ **DETERMİNİSTİK HARD-GATE VETO KALKANI DEVREDE:**
> Yapay zeka delegasyonu yükseliş yönlü tezler sunsa da; **matematiksel risk eşikleri** ({veto_reason}) nedeniyle komite kararı programatik olarak **"TUT / GÖZLEMLE (Beklemede Kal)"** seviyesine revize edilmiştir.
""" + executive_verdict

        # 5. VİOP Türev & Dinamik SPAN Teminat Hesaplaması (Fix 4.3 & 4.1)
        contract_code = self.viop_engine.get_contract_code(ticker)
        viop_pos = self.viop_engine.calculate_position_size(capital=100000.0, spot_price=current_price, ticker=ticker, leverage=1.5)
        theo_futures_p = self.viop_engine.calculate_theoretical_futures_price(spot_price=current_price, days_to_expiry=30)
        
        # 6. Dev Kapsamlı Dosyayı Derle ve Kaydet
        full_dossier = f"""# 🏛️ BIST 100 HİBRİT YAPAY ZEKA KOMİTE RAPORU
**Tarih:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | **Sembol:** {ticker} | **Şirket:** {company_name}
**Aktif Model:** Kronos-Base Quant + Merton Jump Diffusion & GARCH + VİOP Cost-of-Carry Motoru + Takasbank AKD Köprüsü + Gemini Rotational Multi-Agent Debate

---

{executive_verdict}

---

## ⚡ VİOP (VADELİ İŞLEM VE OPSİYON PİYASASI) TÜREV & HEDGE MATRİSİ
* **VİOP Kontrat Kodu:** `{contract_code}` (1 Kontrat = 100 Pay)
* **Spot Fiyat:** {current_price:.2f} TRY | **Teorik Vadeli Fiyat (Cost-of-Carry):** **{theo_futures_p:.2f} TRY**
* **1 Kontrat Büyüklüğü:** {viop_pos['contract_value']:,.2f} TRY | **Takasbank Maktu SPAN Teminatı:** **{viop_pos['required_margin']/max(1, viop_pos['contracts']):,.2f} TRY / Kontrat**
* **100.000 TL Kasa İçin Pozisyon:** {viop_pos['contracts']} Kontrat ({viop_pos['contracts']*100} Pay) | **Toplam Notional Değer:** {viop_pos['notional_value']:,.2f} TRY (Efektif Kaldıraç: {viop_pos['effective_leverage']}x)
* **Takasbank Nemalandırma Faizi:** Boşta kalan {viop_pos['cash_reserve']:,.2f} TRY nakit rezervi gecelik yıllık ~%45 bileşik faiz getirisi üretir.
* **Ters / İz Süren Stop:** Long pozisyonlar için zirveden %4.5 aşağı, Short pozisyonlar için dipten %4.5 yukarı tepkide kâr koruma kalkanı devrededir.

---

{akd_report}

---

## 📰 ÇOK MODLU (MULTI-MODAL) NLP HABER VE KAP DUYARLILIK ANALİZİ
* **Duyarlılık Skoru (Sentiment):** {sentiment_data.get('sentiment_score', 0.0):+.2f} [-1.0 ile +1.0] ({sentiment_data.get('sentiment_label', 'NÖTR')})
* **Etki Şiddeti (Impact):** %{sentiment_data.get('impact_intensity', 0.0)*100:.0f} | **İncelenen Haber:** {sentiment_data.get('news_count', 0)} Adet
* **Katalizör Durumu:** {'🚀 Pozitif Katalizör Tespit Edildi 🟢' if sentiment_data.get('catalyst_detected') else ('🚨 Negatif Kriz Katalizörü 🔴' if sentiment_data.get('bearish_catalyst_detected') else '⚪ Nötr Haber Akışı')}
* **Haber & KAP Özeti:** {sentiment_data.get('summary', '')}

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
