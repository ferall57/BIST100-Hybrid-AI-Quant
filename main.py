#!/usr/bin/env python3
"""
BIST 100 HİBRİT YAPAY ZEKA YATIRIM SİSTEMİ
Kronos-Base (Quant Forecasting) + TradingAgents (Multi-Agent Committee) + Gemini Rotational API Engine
"""

import os
import sys
import argparse
from datetime import datetime

# Windows konsollarında Unicode/Emoji kilitlenmelerini önleme:
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

# Proje yollarını ve bağımlılıkları ekle
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.bist_downloader import download_bist_universe, download_ticker_data
from bist_quant.bist_preprocess import preprocess_bist_for_kronos
from bist_quant.bist_trainer import generate_bist_config, run_training
from hybrid_agents.bist_committee import BistHybridCommittee
from bist_quant.bist_scanner import BistScanner

def banner():
    print("""
================================================================================
   BIST 100 HIBRIT YAPAY ZEKA KANTITATIF VE KOMITE YATIRIM SISTEMI
--------------------------------------------------------------------------------
 [*] Cekirdek 1 : Kronos-Base Foundation Model (102.3M Parametre - Mum Tahmincisi)
 [*] Cekirdek 2 : TradingAgents Coklu Yapay Zeka Komitesi (Bull vs Bear Debate)
 [*] Cekirdek 3 : BIST 30/100 Otomatik Tarama ve Keşif Motoru (Screener)
 [*] Motor      : 3'lu Gemini API Akilli Rotasyon & Kota Koruma Kalkani
================================================================================
""")

def handle_download(mode="bist100", period="max", workers=8):
    print("\n[STEP 1] Yahoo Finance Uzerinden BIST Tarihsel Veri Setleri Indiriliyor...")
    download_bist_universe(mode=mode, period=period, max_workers=workers)
    print("\n[STEP 2] Kronos-base Ozel Formati Icin On Isleme ve Birlestirme Calisiyor...")
    preprocess_bist_for_kronos()
    print("\n[BASARILI] Veri Hazirligi Bitti! Artik '--train-kronos' ile derin egitim baslatabilir veya '--analyze <SEMBOL>' kullanabilirsiniz.")

def handle_train(epochs_tok=15, epochs_pred=25, batch_size=2, accum=16, lr=1e-6, skip_tok=False):
    print("\n[EGITIM YONETICISI] 4GB VRAM Optimize Derin BIST 100 Ince Ayar (Fine-Tuning) Baslatiliyor...")
    generate_bist_config(epochs_tokenizer=epochs_tok, epochs_predictor=epochs_pred, batch_size=batch_size, accum_steps=accum, lr_predictor=lr, train_tokenizer=not skip_tok)
    run_training(skip_tokenizer=skip_tok)

def handle_analyze(ticker: str, days: int = 15, model: str = "gemini-2.5-pro", temp: float = 0.3):
    if not ticker:
        print("[HATA] Lutfen analiz edilecek BIST sembolu girin. Orn: '--analyze THYAO.IS'")
        return
        
    try:
        committee = BistHybridCommittee(gemini_model=model, temperature=temp)
        verdict, report_file, chart_file = committee.analyze_ticker(ticker, forecast_days=days)
        print("\n" + "="*80)
        print("ANALIZ GERCEKLESTIRILDI - CIKTI OZETI:")
        print("="*80)
        print(verdict)
        print("="*80)
        print(f"Tam Komite Tartisma Raporu : {report_file}")
        if chart_file:
            print(f"Fiyat Projeksiyon Grafigi  : {chart_file}")
    except Exception as e:
        print(f"\n[KOMITE HATASI] Analiz sirasinda problem olustu: {e}")
        if "api anahtar" in str(e).lower() or "not found" in str(e).lower():
            print("[BILGI] Lutfen .env dosyanizdaki GOOGLE_API_KEY_1, _2, _3 degerlerini kontrol ettiginizden emin olun!")

def handle_scan(mode: str = "bist30", top_n: int = 5, days: int = 15, model: str = "gemini-2.5-pro", temp: float = 0.2):
    try:
        scanner = BistScanner(gemini_model=model, temperature=temp)
        scanner.scan_and_report(mode=mode, top_n=top_n, forecast_days=days)
    except Exception as e:
        print(f"\n[TARAMA HATASI] Tarama sirasinda problem olustu: {e}")

def main():
    banner()
    parser = argparse.ArgumentParser(description="BIST 100 Hibrit AI Komitesi Ana Iletisim Arayuzu")
    
    # Komut Modları
    parser.add_argument("--download-all", action="store_true", help="Tum BIST 100 gecmis gunluk/saatlik verilerini indir ve hazirla")
    parser.add_argument("--download-mode", default="bist100", choices=["bist100", "bist30"], help="Indirilecek hisse evreni (Varsayilan: bist100)")
    parser.add_argument("--train-kronos", action="store_true", help="Kronos-base modelini BIST 100 uzerinde uygulanacak Derin Egitimi baslat")
    parser.add_argument("--train-predictor", action="store_true", help="Tokenizer egitimini atlayıp dogrudan Tahminci (Predictor) motorunun derin egitimine basla")
    parser.add_argument("--analyze", type=str, metavar="SEMBOL", help="Secilen BIST hissesinde (Orn: THYAO.IS) hibrit Quant + Ajan Komitesi raporu uret")
    parser.add_argument("--scan", type=str, nargs="?", const="bist30", default=None, choices=["bist30", "bist100"], help="Tum BIST 30 veya BIST 100 hisselerini otomatik tara ve en iyi firsatlari kesfet (Varsayilan: bist30)")
    
    # Opsiyonel parametreler
    parser.add_argument("--top", type=int, default=5, help="Tarama modunda derin analize girecek hisse sayisi (Varsayilan: 5)")
    parser.add_argument("--days", type=int, default=15, help="Kronos-base quant projeksiyon gun sayisi (Varsayilan: 15)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Sistemin sozel akil yurutmede kullanacigi Gemini modeli (Varsayilan: gemini-2.5-pro)")
    parser.add_argument("--tok-epochs", type=int, default=15, help="Fine-tuning: Tokenizer epok sayisi")
    parser.add_argument("--pred-epochs", type=int, default=25, help="Fine-tuning: Predictor (base) epok sayisi")
    parser.add_argument("--batch-size", type=int, default=2, help="Fine-tuning: 4GB VRAM icin batch size (Varsayilan: 2)")
    parser.add_argument("--workers", type=int, default=8, help="Veri indirmedeki paralel thread sayisi")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.download_all:
        handle_download(mode=args.download_mode, period="max", workers=args.workers)
        
    if args.train_kronos:
        handle_train(epochs_tok=args.tok_epochs, epochs_pred=args.pred_epochs, batch_size=args.batch_size, skip_tok=False)
        
    if args.train_predictor:
        print("\n🚀 [DOĞRUDAN PREDICTOR AŞAMASI] Usta Tokenizer rekorunuz hafızaya eklenerek BIST Tahminci modeli eğitimi başlatılıyor!")
        handle_train(epochs_tok=args.tok_epochs, epochs_pred=args.pred_epochs, batch_size=args.batch_size, skip_tok=True)
        
    if args.analyze:
        handle_analyze(args.analyze, days=args.days, model=args.model)

    if args.scan:
        handle_scan(mode=args.scan, top_n=args.top, days=args.days, model=args.model)

if __name__ == "__main__":
    main()
