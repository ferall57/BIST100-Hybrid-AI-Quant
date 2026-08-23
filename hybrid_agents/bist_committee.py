import os
import sys
import re
import pandas as pd
import yfinance as yf
from datetime import datetime

from hybrid_agents.gemini_rotator import GeminiRotator
from hybrid_agents.prompts import (
    BIST_FUNDAMENTAL_ANALYST_PROMPT,
    BIST_TECHNICAL_MACRO_PROMPT,
    BIST_BULL_RESEARCHER_PROMPT,
    BIST_BEAR_RESEARCHER_PROMPT,
    BIST_BULL_REBUTTAL_PROMPT,
    BIST_BEAR_REBUTTAL_PROMPT,
    BIST_PORTFOLIO_MANAGER_PROMPT
)
from bist_quant.bist_econometrics import BistEconometrics
from bist_quant.bist_sentiment import BistSentimentEngine
from bist_quant.bist_viop import BistViopEngine
from bist_quant.bist_akd_flow import BistAkdFlowEngine
from bist_quant.bist_microstructure import BistMarketMicrostructure
from bist_quant.bist_memory import BistCommitteeMemory

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(ROOT_DIR, "outputs", "reports")

try:
    from bist_quant.bist_kronos_quant import BistKronosQuant
    QUANT_AVAILABLE = True
except Exception as e:
    QUANT_AVAILABLE = False
    print(f"[UYARI] BistKronosQuant motoru bağlanamadı: {e}")

class BistHybridCommittee:
    """
    Kronos-Base Quant tahmini ile TradingAgents çoklu yapay zeka komitesini
    (Temel, Teknik, 2 Turlu Boğa/Ayı Münazarası, Baş Portföy Müdürü ve Öz-Yansıtma Hafızasını)
    buluşturan kurumsal yatırım yönetim motoru.
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
        self.microstructure_engine = BistMarketMicrostructure()
        self.memory_engine = BistCommitteeMemory()
        
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
        target_mean = _val("targetMeanPrice", "{:.2f}", suffix=" TRY")
        num_analysts = _val("numberOfAnalystOpinions", "{:.0f}", suffix=" Kurum")
        rec_key = info.get("recommendationKey", "N/A").upper()

        market_cap = _cap_val(info.get("marketCap"))
        total_rev = _cap_val(info.get("totalRevenue"))
        net_inc = _cap_val(info.get("netIncomeToCommon"))
        ebitda_val = _cap_val(info.get("ebitda"))
        total_cash = _cap_val(info.get("totalCash"))
        total_debt = _cap_val(info.get("totalDebt"))

        txt = f"""* **Piyasa Değeri (Market Cap):** {market_cap} | **Fiyat / Kazanç (F/K):** {pe} (İleri F/K: {fwd_pe})
* **Piyasa Değeri / Defter Değeri (PD/DD):** {pb} | **FD / FAVÖK (EV/EBITDA):** {ev_ebitda}
* **Özkaynak Kârlılığı (ROE):** {roe} | **Temettü Verimi (Dividend Yield):** {div_yield}
* **Finansal Büyüklükler:** Yıllık Ciro: {total_rev} | Net Kâr: {net_inc} | FAVÖK: {ebitda_val}
* **Nakit & Borçluluk:** Toplam Nakit: {total_cash} | Toplam Finansal Borç: {total_debt}
* **52 Haftalık Fiyat Aralığı:** {low_52} - {high_52} | **Hisse Betası (BIST 100):** {beta}
* **Konsensüs Analist Hedef Fiyatı:** {target_mean} ({num_analysts} analist) | **Konsensüs Tavsiyesi:** {rec_key}
"""
        return txt

    def _fetch_macro_indicators(self) -> str:
        """Türkiye ve küresel makro göstergeleri derler."""
        try:
            usd = yf.Ticker("USDTRY=X").history(period="5d")
            brent = yf.Ticker("BZ=F").history(period="5d")
            gold = yf.Ticker("GC=F").history(period="5d")
            xu100 = yf.Ticker("XU100.IS").history(period="5d")

            usd_try = f"{float(usd['Close'].iloc[-1]):.4f}" if not usd.empty else "N/A"
            brent_p = f"{float(brent['Close'].iloc[-1]):.2f} USD" if not brent.empty else "N/A"
            gold_p = f"{float(gold['Close'].iloc[-1]):.2f} USD" if not gold.empty else "N/A"
            xu100_p = f"{float(xu100['Close'].iloc[-1]):,.2f}" if not xu100.empty else "N/A"

            return f"""* **USD/TRY Kuru:** {usd_try}
* **TCMB Politika / Gösterge Faizi (Varsayılan):** %50.00 (Gecelik Fonlama: ~%53.00)
* **BIST 100 Endeksi:** {xu100_p}
* **Brent Petrol (Varil):** {brent_p} | **Ons Altın (USD):** {gold_p}
"""
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

        financial_ratios = self._format_financial_ratios(info)
        macro_indicators = self._fetch_macro_indicators()

        raw_csv = os.path.join(ROOT_DIR, "bist_data", "raw", f"{ticker}_1d.csv")
        current_price = 0.0
        recent_history = "Veri okunamadı"
        econometric_report = "Ekonometrik veri hazır değil."
        akd_report = "AKD ve Para Akışı verisi hazır değil."
        microstructure_report = "Mikro-yapı verisi hazır değil."
        
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
            try:
                microstructure_report = self.microstructure_engine.generate_microstructure_report(df, ticker)
            except Exception as e_ms:
                microstructure_report = f"Mikro-yapı analiz hatası: {e_ms}"

        # 🧠 Ajan Öz-Yansıtma ve Zaman Duyarlı Geçmiş Hafıza Brifingi
        print(f"🧠 [HAFIZA MOTORU] {ticker} için geçmiş kararlar ve zaman duyarlı post-mortem denetimi yapılıyor...")
        self_reflection_context = self.memory_engine.get_self_reflection_context(ticker, current_price)
            
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
            current_date=current_date_str,
            self_reflection_context=self_reflection_context
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
            kronos_report=kronos_report,
            self_reflection_context=self_reflection_context
        )
        res_tech = self.llm.invoke(prompt_tech)
        technical_report = extract_text(res_tech)
        
        # 3. Aşama: ⚔️ 2 TURLU ÇOKLU AJAN DİYALEKTİK MÜNAZARASI (THE 2-ROUND DEBATE)
        print(f"⚔️ [AŞAMA 3/4] 2 TURLU DİYALEKTİK BOĞA vs AYI MÜNAZARASI BAŞLADI...")
        
        # 1. Tur: Açılış Tezleri
        print(f"  • [1. Tur] Boğa Açılış Tezi Sunuluyor...")
        prompt_bull_r1 = BIST_BULL_RESEARCHER_PROMPT.format(
            ticker=ticker,
            fundamental_report=fundamental_report,
            technical_report=technical_report,
            self_reflection_context=self_reflection_context
        )
        res_bull_r1 = self.llm.invoke(prompt_bull_r1)
        bull_opening = extract_text(res_bull_r1)
        
        print(f"  • [1. Tur] Ayı Kontra-Tezi Sunuluyor...")
        prompt_bear_r1 = BIST_BEAR_RESEARCHER_PROMPT.format(
            ticker=ticker,
            bull_thesis=bull_opening,
            fundamental_report=fundamental_report,
            technical_report=technical_report,
            self_reflection_context=self_reflection_context
        )
        res_bear_r1 = self.llm.invoke(prompt_bear_r1)
        bear_opening = extract_text(res_bear_r1)

        # 2. Tur: Çapraz Savunma & Çürütme (Rebuttal Round)
        print(f"  • [2. Tur] Boğa Çapraz Savunması (Rebuttal) Yapılıyor...")
        prompt_bull_r2 = BIST_BULL_REBUTTAL_PROMPT.format(
            ticker=ticker,
            bull_opening=bull_opening,
            bear_opening=bear_opening
        )
        res_bull_r2 = self.llm.invoke(prompt_bull_r2)
        bull_rebuttal = extract_text(res_bull_r2)

        print(f"  • [2. Tur] Ayı Nihai Meydan Okuması (Counter-Rebuttal) Yapılıyor...")
        prompt_bear_r2 = BIST_BEAR_REBUTTAL_PROMPT.format(
            ticker=ticker,
            bull_rebuttal=bull_rebuttal
        )
        res_bear_r2 = self.llm.invoke(prompt_bear_r2)
        bear_rebuttal = extract_text(res_bear_r2)
        
        # 4. Aşama: Portföy Yönetim Müdürü Hakem Kararı
        print(f"🏆 [AŞAMA 4/4] Baş Portföy Müdürü (Executive Manager) 2 Turlu Münazarayı Yargılıyor...")
        prompt_mgr = BIST_PORTFOLIO_MANAGER_PROMPT.format(
            ticker=ticker,
            current_price=current_price,
            fundamental_report=fundamental_report,
            technical_report=technical_report,
            econometric_report=econometric_report,
            bull_opening=bull_opening,
            bear_opening=bear_opening,
            bull_rebuttal=bull_rebuttal,
            bear_rebuttal=bear_rebuttal,
            self_reflection_context=self_reflection_context
        )
        res_mgr = self.llm.invoke(prompt_mgr)
        executive_verdict = extract_text(res_mgr)
        
        # 🛡️ DETERMINİSTİK HARD-GATE VETO KAPISI
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

        # 5. VİOP Türev & Dinamik SPAN Teminat & BSM Greeks Hesaplaması
        contract_code = self.viop_engine.get_contract_code(ticker)
        viop_pos = self.viop_engine.calculate_position_size(capital=100000.0, spot_price=current_price, ticker=ticker, leverage=1.5)
        theo_futures_p = self.viop_engine.calculate_theoretical_futures_price(spot_price=current_price, days_to_expiry=30)
        
        strike_atm = round(current_price * 1.05, 2)
        greeks = self.viop_engine.calculate_bsm_option_greeks(spot=current_price, strike=strike_atm, days_to_expiry=30, volatility=0.32)
        
        # 6. Kararı Hafıza Veritabanına Kaydet (Gelecek Öz-Yansıtma İçin)
        t1w_match = re.search(r"1\s*Haftal[ıi]k.*?:\s*\[?([0-9]+\.?[0-9]*)\s*-\s*([0-9]+\.?[0-9]*)", executive_verdict)
        t15d_match = re.search(r"15-30\s*G[üu]nl[üu]k.*?:\s*\[?([0-9]+\.?[0-9]*)\s*-\s*([0-9]+\.?[0-9]*)", executive_verdict)
        stop_match = re.search(r"Stop-Loss.*?:\s*\[?([0-9]+\.?[0-9]*)", executive_verdict)
        conf_match = re.search(r"G[üu]ven.*?:\s*%?\s*([0-9]+)", executive_verdict)
        
        t1w_val = float(t1w_match.group(2)) if t1w_match else current_price * 1.03
        t15d_val = float(t15d_match.group(2)) if t15d_match else current_price * 1.08
        stop_val = float(stop_match.group(1)) if stop_match else current_price * 0.96
        conf_val = int(conf_match.group(1)) if conf_match else 75
        verdict_type = "TUT" if veto_triggered else ("AL" if "AL" in executive_verdict.upper() else ("SAT" if "SAT" in executive_verdict.upper() else "TUT"))

        self.memory_engine.record_decision(
            ticker=ticker,
            spot_price=current_price,
            verdict=verdict_type,
            confidence=conf_val,
            target_1w=t1w_val,
            target_15d=t15d_val,
            stop_loss=stop_val,
            bull_thesis_summary=bull_rebuttal[:250],
            bear_thesis_summary=bear_rebuttal[:250],
            cmf_score=akd_data.get("cmf_20", 0.0),
            vwap_price=akd_data.get("vwap_20", current_price)
        )

        # 7. Dev Kapsamlı Dosyayı Derle ve Kaydet
        full_dossier = f"""# 🏛️ BIST 100 HİBRİT YAPAY ZEKA KOMİTE RAPORU
**Tarih:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | **Sembol:** {ticker} | **Şirket:** {company_name}
**Aktif Model:** Kronos-Base Quant + Merton Jump Diffusion & GARCH + VİOP BSM Motoru + Takasbank AKD Köprüsü + 2 Turlu Diyalektik Münazara + Episodik Öz-Yansıtma Hafızası

---

{executive_verdict}

---

## 🧠 KOMİTE ÖZ-YANSITMA & GEÇMİŞ HAFIZA DENETİMİ
{self_reflection_context}

---

## ⚔️ 2 TURLU BOĞA vs AYI DİYALEKTİK MÜNAZARA TUTANAĞI

### 🐂 1. Tur: Boğa Açılış Tezi
{bull_opening}

---

### 🐻 1. Tur: Ayı Kontra-Tezi
{bear_opening}

---

### 🐂 2. Tur: Boğa Çapraz Savunması & Riskleri Çürütme (Rebuttal)
{bull_rebuttal}

---

### 🐻 2. Tur: Ayı Nihai Karşı Atağı & Meydan Okuması (Counter-Rebuttal)
{bear_rebuttal}

---

## ⚡ VİOP (VADELİ İŞLEM VE OPSİYON PİYASASI) TÜREV & BSM GREEKS MATRİSİ
* **VİOP Kontrat Kodu:** `{contract_code}` (1 Kontrat = 100 Pay)
* **Spot Fiyat:** {current_price:.2f} TRY | **Teorik Vadeli Fiyat (Cost-of-Carry):** **{theo_futures_p:.2f} TRY**
* **1 Kontrat Büyüklüğü:** {viop_pos['contract_value']:,.2f} TRY | **Takasbank Maktu SPAN Teminatı:** **{viop_pos['required_margin']/max(1, viop_pos['contracts']):,.2f} TRY / Kontrat**
* **100.000 TL Kasa İçin Pozisyon:** {viop_pos['contracts']} Kontrat ({viop_pos['contracts']*100} Pay) | **Toplam Notional Değer:** {viop_pos['notional_value']:,.2f} TRY (Efektif Kaldıraç: {viop_pos['effective_leverage']}x)
* **Takasbank Nemalandırma Faizi:** Boşta kalan {viop_pos['cash_reserve']:,.2f} TRY nakit rezervi gecelik yıllık ~%45 bileşik faiz getirisi üretir.

### 📐 Black-Scholes-Merton (BSM) 30G Opsiyon Fiyatlama & Greeks Duyarlılıkları (Strike: {strike_atm:.2f} TRY)
* **Teorik Call Primi:** {greeks['call_price']:.3f} TRY (Delta Δ: {greeks['call_delta']:+.4f}, Theta Θ: {greeks['call_theta_daily']:.4f} TL/gün, Rho ρ: {greeks['call_rho']:+.4f})
* **Teorik Put Primi:** {greeks['put_price']:.3f} TRY (Delta Δ: {greeks['put_delta']:+.4f}, Theta Θ: {greeks['put_theta_daily']:.4f} TL/gün, Rho ρ: {greeks['put_rho']:+.4f})
* **Gamma (Γ):** {greeks['gamma']:.6f} | **Vega (𝒱):** {greeks['vega']:.4f} | **Vanna:** {greeks['vanna']:.6f} | **Volga:** {greeks['volga']:.6f}

---

{akd_report}

---

{microstructure_report}

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
    parser.add_argument("--model", type=str, default="gemini-3.5-flash", help="Gemini modeli")
    
    args = parser.parse_args()
    committee = BistHybridCommittee(gemini_model=args.model)
    verdict, doc_path, img_path = committee.analyze_ticker(args.ticker, forecast_days=args.days)
    print("\n" + verdict)
