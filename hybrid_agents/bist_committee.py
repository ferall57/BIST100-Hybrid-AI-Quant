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
        
        if QUANT_AVAILABLE:
            self.quant_engine = BistKronosQuant(use_base_model=True)
        else:
            self.quant_engine = None

    def analyze_ticker(self, ticker: str, forecast_days: int = 15):
        if not ticker.endswith(".IS"):
            ticker += ".IS"
            
        print(f"\n🚀 === [{ticker}] İÇİN HİBRİT YAPAY ZEKA YATIRIM KOMİTESİ ANALİZİ BAŞLADI ===")
        
        # 0. Yahoo Finance Anlık Meta Verisi & Güncel Fiyatlar
        ticker_obj = yf.Ticker(ticker)
        try:
            info = ticker_obj.info
            company_name = info.get("longName", info.get("shortName", ticker))
        except:
            company_name = ticker
            
        # 1. Aşama: Kronos-base Quant Raporunun Çıkartılması
        print(f"📊 [AŞAMA 1/4] Kronos-base Quant Yapay Zekası Mum Formasyonlarını Hesaplıyor...")
        kronos_report, chart_path = ("Kronos Quant verisi hazir degil.", None)
        current_price = 0.0
        recent_history = "Veri okunamadı"
        
        if self.quant_engine:
            kronos_report, chart_path = self.quant_engine.generate_quant_report(ticker, pred_len=forecast_days)
            # Fiyat ve geçmiş özetti alıp formatla
            raw_csv = os.path.join(ROOT_DIR, "bist_data", "raw", f"{ticker}_1d.csv")
            if os.path.exists(raw_csv):
                df = pd.read_csv(raw_csv)
                current_price = df["close"].iloc[-1]
                recent_history = df.tail(5)[["timestamps", "close", "volume"]].to_string(index=False)

        # 2. Aşama: Analistler (Temel & Teknik-Makro)
        print(f"💼 [AŞAMA 2/4] Temel ve Teknik Stratejist Ajanlar Rapor Yazıyor (Gemini Rotator)...")
        
        # 2.1 Canlı İnternet Araması (KAP ve Güncel Haberler)
        print(f"🌍 [CANLI BAĞLANTI] {ticker} için internetteki son haberler taranıyor...")
        live_news = "Güncel haber bulunamadı."
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                search_query = f"{company_name} {ticker.replace('.IS', '')} hisse KAP haber"
                results = list(ddgs.text(search_query, max_results=5))
                if results:
                    live_news = ""
                    for r in results:
                        live_news += f"- {r.get('title')}: {r.get('body')}\n"
        except Exception as e:
            live_news = f"İnternet aramasında hata oluştu: {e}"

        prompt_fund = BIST_FUNDAMENTAL_ANALYST_PROMPT.format(ticker=ticker, company_name=company_name, live_news=live_news)
        res_fund = self.llm.invoke(prompt_fund)
        fundamental_report = res_fund.content if hasattr(res_fund, "content") else str(res_fund)
        
        prompt_tech = BIST_TECHNICAL_MACRO_PROMPT.format(
            ticker=ticker,
            current_price=current_price,
            recent_history=recent_history,
            kronos_report=kronos_report
        )
        res_tech = self.llm.invoke(prompt_tech)
        technical_report = res_tech.content if hasattr(res_tech, "content") else str(res_tech)
        
        # 3. Aşama: Boğa - Ayı Tartışması (The Debate)
        print(f"⚔️ [AŞAMA 3/4] Boğa (Bull) ve Ayı (Bear) Yapay Zekaları Masada Tartışıyor...")
        prompt_bull = BIST_BULL_RESEARCHER_PROMPT.format(
            ticker=ticker,
            fundamental_report=fundamental_report,
            technical_report=technical_report
        )
        res_bull = self.llm.invoke(prompt_bull)
        bull_thesis = res_bull.content if hasattr(res_bull, "content") else str(res_bull)
        
        prompt_bear = BIST_BEAR_RESEARCHER_PROMPT.format(
            ticker=ticker,
            bull_thesis=bull_thesis,
            fundamental_report=fundamental_report,
            technical_report=technical_report
        )
        res_bear = self.llm.invoke(prompt_bear)
        bear_thesis = res_bear.content if hasattr(res_bear, "content") else str(res_bear)
        
        # 4. Aşama: Portföy Yönetim Müdürü Kararı
        print(f"🏆 [AŞAMA 4/4] Baş Portföy Müdürü (Executive Manager) Nihai Kararı Açıklıyor...")
        prompt_mgr = BIST_PORTFOLIO_MANAGER_PROMPT.format(
            ticker=ticker,
            current_price=current_price,
            fundamental_report=fundamental_report,
            technical_report=technical_report,
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis
        )
        res_mgr = self.llm.invoke(prompt_mgr)
        executive_verdict = res_mgr.content if hasattr(res_mgr, "content") else str(res_mgr)
        
        # 5. Dev Kapsamlı Dosyayı Derle ve Kaydet
        full_dossier = f"""# 🏛️ BIST 100 HİBRİT YAPAY ZEKA KOMİTE RAPORU
**Tarih:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | **Sembol:** {ticker} | **Şirket:** {company_name}
**Aktif Model:** Kronos-Base Quant + Gemini Rotational Agents (TradingAgents Framework)

---

{executive_verdict}

---

## 🔬 KRONOS-BASE KANTİTATİF VE TEKNİK ÖNGÖRÜLER
{kronos_report}
*(Görsel Grafik Kayıt Yeri: `{chart_path}`)*

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
