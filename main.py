#!/usr/bin/env python3
"""
🏛️ BIST 100 HİBRİT YAPAY ZEKA VE KANTİTATİF FİNANS SİSTEMİ
Kronos-Base (Quant Forecasting) + TradingAgents (Multi-Agent Committee) + 
İleri Ekonometri & Stokastik Simülasyon + VİOP BSM Greeks + HRP & Black-Litterman Portföy + Öz-Yansıtma Hafızası
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd

# Windows konsollarında Unicode/Emoji kilitlenmelerini önleme:
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

# Proje yollarını ve bağımlılıkları ekle
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.bist_downloader import download_bist_universe, download_ticker_data, trim_to_period, RAW_DATA_DIR
from bist_quant.bist_preprocess import preprocess_bist_for_kronos
from bist_quant.bist_trainer import generate_bist_config, run_training
from hybrid_agents.bist_committee import BistHybridCommittee, load_market_history
from bist_quant.bist_scanner import BistScanner
from bist_quant.bist_backtester import BistBacktester, SPOT_COSTS, VIOP_COSTS
from bist_quant.backtest_engine import CostModel
from bist_quant.bist_sentiment import BistSentimentEngine
from bist_quant.bist_viop import BistViopEngine
from bist_quant.bist_akd_flow import BistAkdFlowEngine
from bist_quant.bist_econometrics import BistEconometrics
from bist_quant.bist_portfolio_opt import BistPortfolioOptimizer
from bist_quant.bist_microstructure import BistMarketMicrostructure
from bist_quant.bist_memory import BistCommitteeMemory

def banner():
    print("""
================================================================================
   🏛️ BIST 100 HİBRİT YAPAY ZEKA KANTİTATİF VE EKONOMETRİK YATIRIM SİSTEMİ
--------------------------------------------------------------------------------
 [*] Çekirdek 1 : Kronos-Base Foundation Model (102.3M Parametre - Mum Tahmincisi)
 [*] Çekirdek 2 : İleri Ekonometri, GARCH & Merton Jump Diffusion Stokastik Motoru
 [*] Çekirdek 3 : 5 Temel Ekonometrik Tanı Testi (JB, DW, BG-LM, White, RESET, VIF)
 [*] Çekirdek 4 : TradingAgents Çoklu Yapay Zeka Komitesi (Bull vs Bear Debate & XAI)
 [*] Çekirdek 5 : Hiyerarşik Risk Paritesi (HRP) & Black-Litterman Portföy Motoru
 [*] Çekirdek 6 : BSM Opsiyon Fiyatlama, Greeks (Δ,Γ,𝒱,Θ,ρ,Vanna) & Delta-Hedge
 [*] Çekirdek 7 : Piyasa Mikro-Yapısı (Corwin-Schultz, Roll, Amihud, VPIN, POC)
 [*] Çekirdek 8 : İstatistiksel Arbitraj & Eşbütünleşme (Engle-Granger Pairs Trading)
 [*] Çekirdek 9 : Takasbank & AKD Para Giriş/Çıkış Radarı (BofA & Balina Akışı)
 [*] Çekirdek 10: Çok Modlu (Multi-Modal) NLP Haber & KAP Duyarlılık Füzyon Motoru
 [*] Çekirdek 11: Episodik Öz-Yansıtma & Recursive Sürekli Öğrenen Ajan Hafızası
 [*] Motor      : 3'lü Gemini API Akıllı Rotasyon & Deterministik Veto Kalkanı
================================================================================
""")

def handle_download(mode="bist100", period="max", workers=8):
    print("\n[STEP 1] Yahoo Finance Üzerinden BIST Tarihsel Veri Setleri İndiriliyor...")
    download_bist_universe(mode=mode, period=period, max_workers=workers)
    print("\n[STEP 2] Kronos-base Özel Formatı İçin Ön İşleme ve Birleştirme Çalışıyor...")
    preprocess_bist_for_kronos()
    print("\n[BAŞARILI] Veri Hazırlığı Bitti! Artık '--train-kronos' ile derin eğitim başlatabilir veya '--analyze <SEMBOL>' kullanabilirsiniz.")

def handle_train(epochs_tok=15, epochs_pred=25, batch_size=2, accum=16, lr=1e-6, skip_tok=False, fresh=False, train_steps=None, val_steps=None):
    print("\n[EGITIM YONETICISI] 4GB VRAM Optimize Derin BIST 100 İnce Ayar (Fine-Tuning) Başlatılıyor...")
    generate_bist_config(epochs_tokenizer=epochs_tok, epochs_predictor=epochs_pred, batch_size=batch_size, accum_steps=accum, lr_predictor=lr, train_tokenizer=not skip_tok)
    run_training(skip_tokenizer=skip_tok, fresh=fresh, train_steps=train_steps, val_steps=val_steps)

def handle_analyze(ticker: str, days: int = 15, model: str = "gemini-2.5-flash", temp: float = 0.3):
    if not ticker:
        print("[HATA] Lütfen analiz edilecek BIST sembolü girin. Örn: '--analyze THYAO.IS'")
        return
        
    try:
        committee = BistHybridCommittee(gemini_model=model, temperature=temp)
        verdict, report_file, chart_file = committee.analyze_ticker(ticker, forecast_days=days)
        print("\n" + "="*80)
        print("ANALİZ GERÇEKLEŞTİRİLDİ - ÇIKTI ÖZETİ:")
        print("="*80)
        print(verdict)
        print("="*80)
        print(f"Tam Komite Tartışma Raporu : {report_file}")
        if chart_file:
            print(f"Fiyat Projeksiyon Grafiği  : {chart_file}")
    except Exception as e:
        print(f"\n[KOMİTE HATASI] Analiz sırasında problem oluştu: {e}")

def handle_econometrics(ticker: str, days: int = 15):
    if not ticker:
        print("[HATA] Lütfen ekonometrik analiz yapılacak BIST sembolü girin. Örn: '--econometrics ISCTR.IS'")
        return
    try:
        t = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
        download_ticker_data(t, period="5y", interval="1d", save_dir=RAW_DATA_DIR)
        csv_file = os.path.join(RAW_DATA_DIR, f"{t}_1d.csv")
        if not os.path.exists(csv_file):
            print(f"[HATA] {t} verisi indirilemedi.")
            return
            
        df = trim_to_period(pd.read_csv(csv_file), "5y")
        econ = BistEconometrics()
        print("\n" + "="*85)
        print(econ.generate_econometric_report(df, t, forecast_days=days, df_market=load_market_history("5y")))
        print("="*85 + "\n")
    except Exception as e:
        print(f"\n[EKONOMETRİ HATASI] Analiz sırasında problem oluştu: {e}")

def handle_portfolio_opt(tickers_str: str, target: str = "max_sharpe", rf: float = None):
    try:
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        formatted_tickers = [t if t.endswith(".IS") else f"{t}.IS" for t in tickers]
        
        print(f"\n💼 [PORTFÖY OPTİMİZASYONU] {len(formatted_tickers)} Hisse İçin Veriler Yükleniyor...")
        import yfinance as yf
        df_prices = pd.DataFrame()
        for t in formatted_tickers:
            data = yf.download(t, period="1y", progress=False)
            if not data.empty and "Close" in data.columns:
                df_prices[t] = data["Close"]

        optimizer = BistPortfolioOptimizer(risk_free_rate_annual=rf)
        mean_ret, cov, corr = optimizer.compute_returns_and_covariance(df_prices)
        
        mvo_res = optimizer.optimize_markowitz(mean_ret, cov, target=target)
        hrp_res = optimizer.optimize_hrp(df_prices)
        bl_res = optimizer.optimize_black_litterman(df_prices)
        kelly_res = optimizer.calculate_kelly_portfolio(mean_ret, cov, fraction=0.5)

        print("\n" + "="*90)
        print(f"📊 KURUMSAL ÇOKLU MODEL PORTFÖY TAHSİS RAPORU (Risksiz Faiz Varsayımı: %{optimizer.rf_annual*100:.1f})")
        print("="*90)
        print(f"{'Hisse':<12} {'Markowitz (MVO)':<18} {'HRP (Lopez de Prado)':<24} {'Black-Litterman':<20} {'Kelly (0.5x)':<15}")
        print("-" * 90)
        for t in formatted_tickers:
            w_mvo = mvo_res['weights'].get(t, 0.0) * 100.0
            w_hrp = hrp_res['weights'].get(t, 0.0) * 100.0
            w_bl = bl_res['weights'].get(t, 0.0) * 100.0
            w_k = kelly_res.get('normalized_portfolio_weights', {}).get(t, 0.0) * 100.0
            print(f"{t:<12} %{w_mvo:<17.1f} %{w_hrp:<23.1f} %{w_bl:<19.1f} %{w_k:<14.1f}")
        print("-" * 90)
        print(f"📈 Beklenen Getiri: Markowitz: %{mvo_res['expected_return_pct']} | HRP: %{hrp_res['expected_return_pct']} | Black-Litterman: %{bl_res['expected_return_pct']}")
        print(f"📉 Portföy Risk/Vol: Markowitz: %{mvo_res['expected_volatility_pct']} | HRP: %{hrp_res['expected_volatility_pct']} | Black-Litterman: %{bl_res['expected_volatility_pct']}")
        print(f"🏆 Sharpe Oranları: Markowitz: {mvo_res['sharpe_ratio']:.3f} | HRP: {hrp_res['sharpe_ratio']:.3f} | Black-Litterman: {bl_res['sharpe_ratio']:.3f}")
        print("="*90 + "\n")
    except Exception as e:
        print(f"\n[PORTFÖY OPTİMİZASYON HATASI] {e}")

def handle_microstructure(ticker: str):
    if not ticker:
        print("[HATA] Lütfen analiz edilecek BIST sembolü girin. Örn: '--microstructure ISCTR.IS'")
        return
    try:
        t = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
        download_ticker_data(t, period="6mo", interval="1d", save_dir=RAW_DATA_DIR)
        csv_file = os.path.join(RAW_DATA_DIR, f"{t}_1d.csv")
        if not os.path.exists(csv_file):
            print(f"[HATA] {t} verisi indirilemedi.")
            return
            
        df = trim_to_period(pd.read_csv(csv_file), "6mo")
        ms = BistMarketMicrostructure()
        print("\n" + "="*85)
        print(ms.generate_microstructure_report(df, t))
        print("="*85 + "\n")
    except Exception as e:
        print(f"\n[MİKRO-YAPI HATASI] Analiz sırasında problem oluştu: {e}")

def handle_pairs_trade(pairs_str: str):
    try:
        tickers = [t.strip().upper() for t in pairs_str.split(",") if t.strip()]
        if len(tickers) != 2:
            print("[HATA] Lütfen 2 hisse girin. Örn: '--pairs-trade ISCTR.IS,AKBNK.IS'")
            return
        t1 = tickers[0] if tickers[0].endswith(".IS") else f"{tickers[0]}.IS"
        t2 = tickers[1] if tickers[1].endswith(".IS") else f"{tickers[1]}.IS"
        
        download_ticker_data(t1, period="2y", interval="1d", save_dir=RAW_DATA_DIR)
        download_ticker_data(t2, period="2y", interval="1d", save_dir=RAW_DATA_DIR)
        
        df1 = trim_to_period(pd.read_csv(os.path.join(RAW_DATA_DIR, f"{t1}_1d.csv")), "2y")
        df2 = trim_to_period(pd.read_csv(os.path.join(RAW_DATA_DIR, f"{t2}_1d.csv")), "2y")
        
        s1 = df1.set_index("timestamps")["close"]
        s2 = df2.set_index("timestamps")["close"]
        
        econ = BistEconometrics()
        res = econ.test_cointegration_pair(s1, s2, name1=t1, name2=t2)
        
        print("\n" + "="*85)
        print(f"📊 İSTATİSTİKSEL ARBİTRAJ & EŞBÜTÜNLEŞME (PAIRS TRADING): [{t1}] vs [{t2}]")
        print("="*85)
        coint_badge = "✅ EŞBÜTÜNLEŞİK (Cointegrated - Mean-Reverting)" if res['is_cointegrated'] else "❌ EŞBÜTÜNLEŞİK DEĞİL (Iraksak Seri)"
        print(f"  • Eşbütünleşme Durumu     : {coint_badge} (p={res['p_value']:.4f}, t-stat={res['t_stat']:.3f})")
        print(f"  • Optimal Hedge Oranı (β) : {res['hedge_ratio']:.4f} (1 Adet {t1} için {res['hedge_ratio']:.4f} Adet {t2})")
        print(f"  • Mevcut Spread Z-Score   : {res['current_z_score']:+.2f} σ")
        print(f"  • Yarılanma Ömrü (Half-Life): {res['half_life_days']:.1f} Gün (Ortalamaya Dönüş Hızı)")
        print(f"  • İstatistiksel Sinyal    : 🎯 {res['trading_signal']}")
        print("="*85 + "\n")
    except Exception as e:
        print(f"\n[PAIRS TRADING HATASI] {e}")

def handle_greeks(ticker: str, spot: float = None, strike: float = None, days: float = 30, vol: float = 0.32):
    try:
        engine = BistViopEngine()
        if spot is None:
            t = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
            download_ticker_data(t, period="1mo", interval="1d", save_dir=RAW_DATA_DIR)
            df = pd.read_csv(os.path.join(RAW_DATA_DIR, f"{t}_1d.csv"))
            spot = float(df["close"].iloc[-1])
        if strike is None:
            strike = round(spot * 1.05, 2)
            
        greeks = engine.calculate_bsm_option_greeks(spot=spot, strike=strike, days_to_expiry=days, volatility=vol)
        theo_fut = engine.calculate_theoretical_futures_price(spot_price=spot, days_to_expiry=int(days))
        
        print("\n" + "="*85)
        print(f"📊 BLACK-SCHOLES-MERTON (BSM) OPSİYON VE GREEKS RAPORU ({ticker.upper()})")
        print("="*85)
        print(f"  • Spot Fiyat: {spot:.2f} TRY | Kullanım Fiyatı (Strike): {strike:.2f} TRY | Vadeye Kalan: {days:.0f} Gün | Vol: %{vol*100:.1f}")
        print(f"  • Teorik Vadeli Fiyat (Futures): {theo_fut:.2f} TRY")
        print("-" * 85)
        print(f"  • 📈 CALL Opsiyon Primi : {greeks['call_price']:.3f} TRY  | Delta (Δ): {greeks['call_delta']:+.4f} | Theta (Θ): {greeks['call_theta_daily']:.4f} TL/Gün")
        print(f"  • 📉 PUT Opsiyon Primi  : {greeks['put_price']:.3f} TRY  | Delta (Δ): {greeks['put_delta']:+.4f} | Theta (Θ): {greeks['put_theta_daily']:.4f} TL/Gün")
        print(f"  • ⚡ Gamma (Γ)          : {greeks['gamma']:.6f}   | Vega (𝒱): {greeks['vega']:.4f} | Rho (ρ): {greeks['call_rho']:+.4f}")
        print(f"  • 🔬 Vanna / Volga      : Vanna: {greeks['vanna']:.6f} | Volga: {greeks['volga']:.6f}")
        print("="*85 + "\n")
    except Exception as e:
        print(f"\n[GREEKS HATASI] {e}")

def handle_memory(ticker: str):
    """Seçilen hissenin komite hafıza geçmişini ve öz-yansıtma denetimini görüntüler."""
    if not ticker:
        print("[HATA] Lütfen hafıza geçmişi incelenecek BIST sembolü girin. Örn: '--memory ISCTR.IS'")
        return
    try:
        t = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
        download_ticker_data(t, period="1mo", interval="1d", save_dir=RAW_DATA_DIR)
        csv_file = os.path.join(RAW_DATA_DIR, f"{t}_1d.csv")
        current_p = 12.38
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            current_p = float(df["close"].iloc[-1])

        memory = BistCommitteeMemory()
        evals = memory.evaluate_past_decisions(t, current_price=current_p)
        
        print("\n" + "="*85)
        print(f"🧠 KOMİTE EPİSODİK HAFIZA VE ÖZ-YANSITMA KAYITLARI: [{t.upper()}] (Canlı: {current_p:.2f} TRY)")
        print("="*85)
        if not evals:
            print(f"  ℹ️ [{t}] için henüz kaydedilmiş geçmiş analiz kararı bulunmuyor.")
        else:
            for idx, e in enumerate(evals, 1):
                print(f"  📌 Kayıt #{idx} | Tarih: {e['entry_date']} | Giriş Fiyatı: {e['entry_price']:.2f} TRY | Karar: {e['verdict']}")
                print(f"     • Gerçekleşen Getiri : %{e['actual_return_pct']:+.2f} ({e['current_price']:.2f} TRY)")
                print(f"     • Post-Mortem Notu   : {e['evaluation_note']}")
                print("-" * 85)
        print(memory.get_self_reflection_context(t, current_price=current_p))
        print("="*85 + "\n")
    except Exception as e:
        print(f"\n[HAFIZA HATASI] {e}")

def handle_scan(mode: str = "bist30", top_n: int = 5, days: int = 15, model: str = "gemini-2.5-flash", temp: float = 0.2):
    try:
        scanner = BistScanner(gemini_model=model, temperature=temp)
        scanner.scan_and_report(mode=mode, top_n=top_n, forecast_days=days)
    except Exception as e:
        print(f"\n[TARAMA HATASI] Tarama sırasında problem oluştu: {e}")

BPS_PER_UNIT = 10000.0

def build_cost_model(commission_bps: float = None, slippage_bps: float = None, use_viop: bool = False) -> CostModel:
    """CLI'dan gelen baz puan değerlerini maliyet modeline çevirir; verilmeyen değer varsayılanda kalır."""
    defaults = VIOP_COSTS if use_viop else SPOT_COSTS
    return CostModel(
        commission_rate=defaults.commission_rate if commission_bps is None else commission_bps / BPS_PER_UNIT,
        slippage_rate=defaults.slippage_rate if slippage_bps is None else slippage_bps / BPS_PER_UNIT,
    )

def handle_backtest_universe(mode: str = "bist30", months: int = 6, sl: float = 3.5, tp: float = 8.0, use_kronos: bool = False,
                             use_viop: bool = False, leverage: float = 1.5, commission_bps: float = None, slippage_bps: float = None,
                             fixed_tp: bool = False):
    try:
        from bist_quant.bist_100_tickers import get_tickers
        tickers = get_tickers(mode=mode)
        backtester = BistBacktester(use_kronos=use_kronos, costs=build_cost_model(commission_bps, slippage_bps, use_viop))
        rows, summary = backtester.run_universe_backtest(
            tickers, months=months, stop_loss_pct=sl, take_profit_pct=tp,
            use_viop=use_viop, leverage=leverage, use_trailing_stop=not fixed_tp
        )
        print("\n" + "="*90)
        print(f"EVREN BACKTEST SONUCU - {mode.upper()} ({summary['tickers_tested']} hisse, son {months} ay, maliyetler dahil)")
        print("="*90)
        print(f"{'Hisse':<12} {'Strateji %':<12} {'Al-Tut %':<12} {'Fark %':<12} {'İşlem':<8} {'MaxDD %':<10} {'Sharpe':<8}")
        print("-" * 90)
        for r in sorted(rows, key=lambda x: x["alpha"], reverse=True):
            print(f"{r['ticker']:<12} {r['total_return_pct']:<+12.2f} {r['bnh_return_pct']:<+12.2f} {r['alpha']:<+12.2f} {r['total_trades']:<8} {r['max_drawdown']:<10.2f} {r['sharpe_ratio']:<8.2f}")
        print("-" * 90)
        if rows:
            print(f"  • Medyan Strateji Getirisi   : %{summary['median_total_return_pct']:+.2f}")
            print(f"  • Medyan Al-Tut Getirisi     : %{summary['median_bnh_return_pct']:+.2f}")
            print(f"  • Medyan Fark (Alfa)         : %{summary['median_alpha']:+.2f}")
            print(f"  • Al-Tut'u Geçen Hisse Oranı : %{summary['share_beating_buy_and_hold_pct']:.1f}")
            print(f"  • Toplam İşlem Sayısı        : {summary['total_trades']}")
        if summary["failed_tickers"]:
            print(f"  • Test Edilemeyen Hisseler   : {', '.join(summary['failed_tickers'])}")
        print("="*90)
    except Exception as e:
        print(f"\n[EVREN BACKTEST HATASI] Simülasyon sırasında problem oluştu: {e}")

def handle_backtest(ticker: str, months: int = 6, sl: float = 3.5, tp: float = 8.0, use_kronos: bool = False, use_viop: bool = False, leverage: float = 1.5,
                    commission_bps: float = None, slippage_bps: float = None, fixed_tp: bool = False):
    if not ticker:
        print("[HATA] Lütfen backtest edilecek BIST sembolü girin. Örn: '--backtest ISCTR.IS'")
        return
    try:
        backtester = BistBacktester(use_kronos=use_kronos, costs=build_cost_model(commission_bps, slippage_bps, use_viop))
        metrics, rep_file, chart_file = backtester.run_walk_forward_backtest(
            ticker=ticker,
            months=months,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            use_viop=use_viop,
            leverage=leverage,
            use_trailing_stop=not fixed_tp
        )
        print("\n" + "="*80)
        print(f"BACKTEST TAMAMLANDI - [{ticker}] İÇİN FİNANSAL PERFORMANS METRİKLERİ:")
        print("="*80)
        print(f"  • Strateji Toplam Getirisi : %{metrics['total_return_pct']:+.2f}")
        print(f"  • Al ve Tut (Buy & Hold)   : %{metrics['bnh_return_pct']:+.2f}")
        print(f"  • Üretilen Alfa (Alpha)    : %{metrics['alpha']:+.2f}")
        print(f"  • Kazanma Oranı (Win Rate) : %{metrics['win_rate']:.1f} ({metrics['winning_trades']} / {metrics['total_trades']} İşlem)")
        print(f"  • Kâr / Zarar Oranı (P/L)  : {metrics['profit_factor']:.2f}")
        print(f"  • Maksimum Çekilme (MaxDD) : %{metrics['max_drawdown']:.2f}")
        print(f"  • Yıllık Sharpe Oranı      : {metrics['sharpe_ratio']:.2f}")
        print(f"  • Gerçekleşen İşlem Sayısı : {metrics['total_trades']} Adet (Ort. Süre: {metrics['avg_holding_days']:.1f} Gün)")
        print(f"  • Toplam İşlem Maliyeti    : {metrics['total_costs_try']:,.2f} TRY")
        print(f"  • Veri Sızıntısı Denetimi  : [{metrics['leakage_status']}] {metrics['leakage_note']}")
        print("="*80)
        print(f"Detaylı Performans Dosyası : {rep_file}")
        if chart_file:
            print(f"Kümülatif Getiri Grafiği   : {chart_file}")
    except Exception as e:
        print(f"\n[BACKTEST HATASI] Simülasyon sırasında problem oluştu: {e}")

def handle_viop_signals(top_n: int = 10):
    try:
        from bist_quant.bist_scanner import BistScanner
        scanner = BistScanner()
        engine = BistViopEngine()
        
        print("\n" + "="*80)
        print("⚡ BIST 30 VİOP ÇİFT YÖNLÜ (LONG / SHORT) KONTRAT SİNYAL RADARI")
        print("="*80)
        print("📊 Taranan Evren: BIST 30 Kontratları | Veri: Kronos-Base Quant + Trend + NLP Sentiment")
        print("-" * 80)
        
        candidates = scanner.scan_universe(mode="bist30", top_n=top_n, forecast_days=15)
        signals = engine.generate_viop_signals(candidates)
        
        print(f"{'Kontrat':<12} {'Spot Fiyat':<12} {'Quant Getiri %':<16} {'Trend':<10} {'Sinyal Kararı':<30} {'Güven':<8}")
        print("-" * 80)
        for s in signals:
            print(f"{s['contract']:<12} {s['close']:<12.2f} %{s['expected_return']:<15.2f} {s['trend']:<10} {s['signal']:<30} {s['confidence']:<8}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n[VİOP SİNYAL HATASI] Tarama sırasında problem oluştu: {e}")

def handle_sentiment(ticker: str, model: str = "gemini-2.5-flash"):
    if not ticker:
        print("[HATA] Lütfen analiz edilecek BIST sembolü girin. Örn: '--sentiment ASELS.IS'")
        return
    try:
        engine = BistSentimentEngine(gemini_model=model, temperature=0.2)
        print(f"\n🌍 [{ticker}] için Canlı KAP ve Finans Haberleri Taranıyor...")
        data = engine.analyze_sentiment(ticker)
        
        print("\n" + "="*85)
        print(f"📰 CANLI NLP HABER & KAP DUYARLILIK ANALİZİ: {ticker.upper()}")
        print("="*85)
        print(f"  • Genel Duyarlılık Skoru : {data['sentiment_score']:+.2f} [-1.0 ile +1.0]")
        print(f"  • Duyarlılık Etiketi     : {data['sentiment_label']}")
        print(f"  • Etki Şiddeti (Impact)  : %{data['impact_intensity']*100:.0f}")
        print(f"  • Pozitif Katalizör      : {'EVET 🟢' if data['catalyst_detected'] else 'YOK ⚪'}")
        print(f"  • Negatif Risk Faktörü   : {'EVET 🔴' if data['bearish_catalyst_detected'] else 'YOK ⚪'}")
        print(f"  • İncelenen Haber Sayısı : {data['news_count']} Adet")
        print("-" * 85)
        print(f"  • NLP Haber Analiz Özeti :\n    {data['summary']}")
        if data["key_catalysts"]:
            print("-" * 85)
            print("  • Öne Çıkan Başlıklar / Maddeler:")
            for cat in data["key_catalysts"]:
                print(f"    - {cat}")
        print("-" * 85)
        
        fusion = engine.modulate_quant_threshold(base_expected_return=3.5, sentiment_data=data)
        print(f"  • Dinamik Alım Barajı     : %{fusion['modulated_threshold']:.2f}")
        print(f"  • Kârı Koşturma Stop Mes. : %{fusion['modulated_trailing_pct']:.2f}")
        print(f"  • Karar Önerisi           : {fusion['recommendation']}")
        print("="*85 + "\n")
    except Exception as e:
        print(f"\n[SENTIMENT HATASI] Analiz sırasında problem oluştu: {e}")

def handle_akd(ticker: str):
    if not ticker:
        print("[HATA] Lütfen AKD analizi yapılacak BIST sembolü girin. Örn: '--akd ISCTR.IS'")
        return
    try:
        t = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
        download_ticker_data(t, period="6mo", interval="1d", save_dir=RAW_DATA_DIR)
        csv_file = os.path.join(RAW_DATA_DIR, f"{t}_1d.csv")
        if not os.path.exists(csv_file):
            print(f"[HATA] {t} verisi indirilemedi.")
            return
            
        df = trim_to_period(pd.read_csv(csv_file), "6mo")
        engine = BistAkdFlowEngine()
        print("\n" + "="*85)
        print(engine.get_akd_summary_text(t, df))
        print("="*85 + "\n")
    except Exception as e:
        print(f"\n[AKD HATASI] Analiz sırasında problem oluştu: {e}")

def handle_akd_scan(mode: str = "bist30", top_n: int = 15):
    try:
        from bist_quant.bist_100_tickers import get_tickers
        tickers = get_tickers(mode=mode)
        engine = BistAkdFlowEngine()
        
        print("\n" + "="*85)
        print(f"🔍 BIST {mode.upper()} TAKASBANK & AKD KURUMSAL PARA GİRİŞ/ÇIKIŞ TARAMASI")
        print("="*85)
        print(f"📊 Taranan Evren: {len(tickers)} Hisse | Sıralama Kriteri: Kurumsal Balina Skoru (CMF + MFI + VWAP)")
        print("-" * 85)
        
        results = engine.scan_akd_universe(tickers[:top_n])
        
        print(f"{'Sıra':<5} {'Sembol':<10} {'Fiyat':<10} {'Balina Skoru':<14} {'CMF(20G)':<11} {'MFI(14G)':<10} {'İlk 5 Alıcı %':<15} {'Durum':<20}")
        print("-" * 85)
        for idx, r in enumerate(results, 1):
            top5_text = f"%{r['top5_buy_pct']:.1f}" if r['top5_buy_pct'] is not None else "VERİ YOK"
            print(f"{idx:<5} {r['ticker']:<10} {r['close']:<10.2f} {r['whale_score']:<+14.2f} {r['cmf_20']:<+11.3f} {r['mfi_14']:<10.1f} {top5_text:<15} {r['status_badge']:<20}")
        print("="*85 + "\n")
    except Exception as e:
        print(f"\n[AKD TARAMA HATASI] Tarama sırasında problem oluştu: {e}")

def handle_compare_models(mode: str = "bist30", origins_per_ticker: int = 12):
    try:
        from bist_quant.bist_100_tickers import get_tickers
        from bist_quant.bist_model_eval import run_comparison
        metrics, pairs, report_path = run_comparison(get_tickers(mode=mode), origins_per_ticker=origins_per_ticker)
        print("\n" + "="*100)
        print(f"KRONOS MODEL KARŞILAŞTIRMASI - {mode.upper()}")
        print("="*100)
        print(f"{'Model':<18} {'Ufuk':<6} {'Yön %':<8} {'Hata %':<9} {'Naif %':<9} {'Beceri %':<10} {'IC':<8}")
        print("-" * 100)
        for row in metrics.itertuples(index=False):
            print(f"{row.model:<18} {row.horizon:<6} {row.directional_accuracy_pct:<8.1f} {row.mae_pct:<9.2f} {row.naive_mae_pct:<9.2f} {row.skill_vs_naive_pct:<+10.1f} {row.information_coefficient:<+8.3f}")
        print("-" * 100)
        for pair in pairs:
            closer = "—" if pair["first_closer_pct"] is None else f"%{pair['first_closer_pct']:.1f}"
            print(f"  • {pair['first']} vs {pair['second']}: daha yakın {closer}, hata farkı {pair['mean_abs_error_diff']:+.2f} puan, p={pair['p_value']:.3f}")
        print("="*100)
        print(f"Rapor: {report_path}")
    except Exception as e:
        print(f"\n[MODEL KARŞILAŞTIRMA HATASI] {e}")

def handle_bot():
    from private_interactive_telegram import InteractiveTelegramBot
    import time
    bot = InteractiveTelegramBot()
    if not bot.is_configured:
        print("[HATA] Telegram Bot Token veya Chat ID yapılandırılmamış!")
        return
    print("\n" + "="*80)
    print("🤖 [KRONOS] Çift Yönlü İnteraktif Telegram Bot Dinleyicisi Başlatıldı...")
    print("📲 Telegram'dan '/status', '/scan' komutlarını veya sinyal onay butonlarını gönderebilirsiniz.")
    print("Durdurmak için Ctrl+C tuşlarına basınız.")
    print("="*80 + "\n")
    try:
        while True:
            bot.poll_and_process_updates(timeout=10)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[BİLGİ] Telegram Bot dinleyicisi durduruldu.")

def handle_sync_db():
    from bist_quant.bist_duckdb_engine import BistDuckDbEngine
    engine = BistDuckDbEngine()
    print("\n[DUCKDB] BIST mum verileri yerel DuckDB veritabanına aktarılıyor...")
    n = engine.sync_csv_to_duckdb()
    print(f"✅ Toplam {n} hissenin tüm geçmiş mumları 'bist_data/kronos_market.duckdb' içine senkronize edildi.")

def main():
    banner()
    parser = argparse.ArgumentParser(description="BIST 100 Hibrit AI Komitesi & Ekonometri Ana İletişim Arayüzü")
    
    # Komut Modları
    parser.add_argument("--download-all", action="store_true", help="Tüm BIST 100 geçmiş günlük verilerini indir ve hazırla")
    parser.add_argument("--download-mode", default="bist100", choices=["bist100", "bist30"], help="İndirilecek hisse evreni")
    parser.add_argument("--train-kronos", action="store_true", help="Kronos-base modelini BIST 100 üzerinde Derin Eğit")
    parser.add_argument("--train-predictor", action="store_true", help="Tokenizer eğitimini atlayıp doğrudan Tahminci (Predictor) eğit")
    parser.add_argument("--analyze", type=str, metavar="SEMBOL", help="Seçilen hissede (Örn: THYAO.IS) hibrit Quant + Ajan Komitesi raporu üret")
    parser.add_argument("--econometrics", type=str, metavar="SEMBOL", help="Seçilen hissede 15 maddelik Tam Ekonometrik Tanı & Merton MC Raporu çalıştır")
    parser.add_argument("--portfolio-opt", type=str, metavar="SEMBÖLLER", help="Virgülle ayrılmış hisselerde Markowitz, HRP ve Black-Litterman portföy tahsisi yap (Örn: 'THYAO.IS,ISCTR.IS,AKBNK.IS,ASELS.IS,BIMAS.IS')")
    parser.add_argument("--microstructure", type=str, metavar="SEMBOL", help="Seçilen hissede Corwin-Schultz Spread, Roll Spread, Amihud, VPIN ve Hacim Profili analizini çalıştır")
    parser.add_argument("--pairs-trade", type=str, metavar="SEMBOL1,SEMBOL2", help="İki hisse arasında Engle-Granger Eşbütünleşme, Hedge Oranı, Z-Score ve Half-Life hesapla")
    parser.add_argument("--greeks", type=str, metavar="SEMBOL", help="Seçilen hissede BSM Opsiyon Fiyatlama, Greeks ve Delta-Hedge matrisini hesapla")
    parser.add_argument("--memory", type=str, metavar="SEMBOL", help="Seçilen hissede geçmiş komite kararlarını, gerçekleşen getirileri ve öz-yansıtma denetimini göster")
    parser.add_argument("--sentiment", type=str, metavar="SEMBOL", help="Seçilen hissede Canlı KAP ve Haber NLP Duyarlılık analizini çalıştır")
    parser.add_argument("--akd", type=str, metavar="SEMBOL", help="Seçilen hissede Takasbank & AKD Para Giriş/Çıkış ve Balina analizini çalıştır")
    parser.add_argument("--akd-scan", type=str, nargs="?", const="bist30", default=None, choices=["bist30", "bist100"], help="BIST hisselerini Kurumsal Balina Para Akışına göre tara")
    parser.add_argument("--scan", type=str, nargs="?", const="bist30", default=None, choices=["bist30", "bist100"], help="BIST hisselerini otomatik tara ve en iyi fırsatları keşfet")
    parser.add_argument("--backtest", type=str, metavar="SEMBOL", help="Seçilen hissede geçmiş N aylık Walk-Forward Backtest simülasyonu çalıştır")
    parser.add_argument("--backtest-universe", type=str, nargs="?", const="bist30", default=None, choices=["bist30", "bist100"], help="Aynı backtest kurallarını tüm evrende koş; medyan fark ve Al-Tut'u geçen hisse oranını raporla")
    parser.add_argument("--compare-models", type=str, nargs="?", const="bist30", default=None, choices=["bist30", "bist100"], help="Temel Kronos ile ince ayarlı modelleri aynı hisse ve tarihlerde tahmin isabetine göre karşılaştır")
    parser.add_argument("--origins", type=int, default=12, help="Model karşılaştırmasında hisse başına tahmin noktası sayısı")
    parser.add_argument("--viop-signals", action="store_true", help="BIST 30 kontratları için Canlı VİOP (Long / Short) sinyal ve pozisyon taraması yap")
    parser.add_argument("--bot", action="store_true", help="Çift yönlü interaktif Telegram komuta botunu arka planda dinleyici olarak başlat")
    parser.add_argument("--sync-db", action="store_true", help="Tüm CSV mum verilerini yerel DuckDB analitik tablosuna senkronize et")
    
    # Opsiyonel parametreler
    parser.add_argument("--viop", action="store_true", help="Backtest içinde Çift Yönlü (Long & Short) VİOP türev motorunu çalıştır")
    parser.add_argument("--leverage", type=float, default=1.5, help="VİOP kaldıraç katsayısı (Varsayılan: 1.5x)")
    parser.add_argument("--top", type=int, default=5, help="Tarama modunda derin analize girecek hisse sayısı")
    parser.add_argument("--days", type=int, default=15, help="Projeksiyon gün sayısı (Varsayılan: 15)")
    parser.add_argument("--months", type=int, default=6, help="Backtest test periyodu (Ay)")
    parser.add_argument("--sl", type=float, default=3.5, help="Stop-Loss yüzdesi")
    parser.add_argument("--tp", type=float, default=8.0, help="Take-Profit yüzdesi (yalnızca --fixed-tp ile kullanılır)")
    parser.add_argument("--fixed-tp", action="store_true", help="Backtest'te iz süren stop yerine sabit Take-Profit hedefi kullan")
    parser.add_argument("--commission-bps", type=float, default=None, help="Bacak başına komisyon (baz puan). Varsayılan: spot 10, VİOP 4")
    parser.add_argument("--slippage-bps", type=float, default=None, help="Bacak başına fiyat kayması (baz puan). Varsayılan: 5")
    parser.add_argument("--use-kronos-backtest", action="store_true", help="Backtest içinde derin Kronos modelini çalıştır")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Kullanılacak Gemini modeli")
    parser.add_argument("--tok-epochs", type=int, default=15, help="Fine-tuning: Tokenizer epok sayısı")
    parser.add_argument("--pred-epochs", type=int, default=25, help="Fine-tuning: Predictor epok sayısı")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--fresh-train", action="store_true", help="Fine-tuning: mevcut modeli arşivleyip önceden eğitilmiş Kronos ağırlıklarından başla")
    parser.add_argument("--train-steps", type=int, default=None, help="Fine-tuning: epoch başına eğitim adımı sınırı (tanımsızsa tüm pencereler)")
    parser.add_argument("--val-steps", type=int, default=None, help="Fine-tuning: epoch başına doğrulama adımı sınırı")
    parser.add_argument("--workers", type=int, default=8, help="Veri indirmedeki paralel thread sayısı")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.sync_db:
        handle_sync_db()
    if args.bot:
        handle_bot()
    if args.download_all:
        handle_download(mode=args.download_mode, period="max", workers=args.workers)
    if args.train_kronos:
        handle_train(epochs_tok=args.tok_epochs, epochs_pred=args.pred_epochs, batch_size=args.batch_size, skip_tok=False,
                     fresh=args.fresh_train, train_steps=args.train_steps, val_steps=args.val_steps)
    if args.train_predictor:
        handle_train(epochs_tok=args.tok_epochs, epochs_pred=args.pred_epochs, batch_size=args.batch_size, skip_tok=True,
                     fresh=args.fresh_train, train_steps=args.train_steps, val_steps=args.val_steps)
    if args.analyze:
        handle_analyze(args.analyze, days=args.days, model=args.model)
    if args.econometrics:
        handle_econometrics(args.econometrics, days=args.days)
    if args.portfolio_opt:
        handle_portfolio_opt(args.portfolio_opt)
    if args.microstructure:
        handle_microstructure(args.microstructure)
    if args.pairs_trade:
        handle_pairs_trade(args.pairs_trade)
    if args.greeks:
        handle_greeks(args.greeks, days=args.days)
    if args.memory:
        handle_memory(args.memory)
    if args.sentiment:
        handle_sentiment(args.sentiment, model=args.model)
    if args.akd:
        handle_akd(args.akd)
    if args.akd_scan:
        handle_akd_scan(mode=args.akd_scan, top_n=args.top)
    if args.viop_signals:
        handle_viop_signals(top_n=args.top)
    if args.scan:
        handle_scan(mode=args.scan, top_n=args.top, days=args.days, model=args.model)
    if args.backtest:
        handle_backtest(args.backtest, months=args.months, sl=args.sl, tp=args.tp, use_kronos=args.use_kronos_backtest, use_viop=args.viop, leverage=args.leverage,
                        commission_bps=args.commission_bps, slippage_bps=args.slippage_bps, fixed_tp=args.fixed_tp)
    if args.compare_models:
        handle_compare_models(mode=args.compare_models, origins_per_ticker=args.origins)
    if args.backtest_universe:
        handle_backtest_universe(mode=args.backtest_universe, months=args.months, sl=args.sl, tp=args.tp, use_kronos=args.use_kronos_backtest,
                                 use_viop=args.viop, leverage=args.leverage, commission_bps=args.commission_bps,
                                 slippage_bps=args.slippage_bps, fixed_tp=args.fixed_tp)

if __name__ == "__main__":
    main()
