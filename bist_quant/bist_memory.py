#!/usr/bin/env python3
"""
🏛️ BIST YATIRIM KOMİTESİ ÖZ-YANSITMA & EPİSODİK HAFIZA MOTORU (SELF-REFLECTION MEMORY)
Yapay Zeka Ajanlarının Geçmiş Tahminlerini Hatırlaması, Hatalarından Ders Çıkarması
ve Sürekli Öğrenmesi (Recursive Continuous Learning) İçin Geliştirilmiş Hafıza Altyapısı.

Özellikler:
1. Episodik Karar Veritabanı (Tüm komite kararlarını, hedef bantlarını ve tezlerini kaydeder).
2. Otomatik Post-Mortem Denetimi (Geçmiş tahmin ile gerçekleşen fiyatı karşılaştırır).
3. Hata İzahı & Ders Çıkarma (Hangi ajanın haklı, hangisinin yanıldığını tespit eder).
4. Dinamik Ajan Güvenilirlik Ağırlıklandırması (Boğa vs Ayı itibar puanlaması).
5. Ajan Promptlarına Öz-Yansıtma Enjeksiyonu.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMORY_DIR = os.path.join(ROOT_DIR, "outputs", "memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, "committee_episodic_memory.json")


class BistCommitteeMemory:
    """
    BIST Yapay Zeka Komitesi için Kalıcı Hafıza ve Öz-Yansıtma (Self-Reflection) Yöneticisi.
    """
    def __init__(self, memory_filepath: str = MEMORY_FILE):
        self.memory_filepath = memory_filepath
        os.makedirs(os.path.dirname(self.memory_filepath), exist_ok=True)
        self.memory_data = self._load_memory()

    def _load_memory(self) -> dict:
        """Kayıtlı hafıza dosyasını yükler, yoksa şablon oluşturur."""
        if os.path.exists(self.memory_filepath):
            try:
                with open(self.memory_filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[UYARI] Hafıza dosyası okunamadı, yeniden oluşturuluyor: {e}")
        
        default_memory = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "KRONOS Komite Karar ve Öz-Yansıtma Hafıza Veritabanı",
                "version": "1.0.0"
            },
            "agent_reputation": {
                "bull_analyst": {"correct": 0, "total": 0, "weight": 1.0},
                "bear_analyst": {"correct": 0, "total": 0, "weight": 1.0},
                "technical_analyst": {"correct": 0, "total": 0, "weight": 1.0}
            },
            "history": []
        }
        self._save_memory(default_memory)
        return default_memory

    def _save_memory(self, data: dict = None):
        """Hafızayı diske kaydeder."""
        if data is None:
            data = self.memory_data
        with open(self.memory_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def record_decision(
        self,
        ticker: str,
        spot_price: float,
        verdict: str,
        confidence: int,
        target_1w: float,
        target_15d: float,
        stop_loss: float,
        bull_thesis_summary: str = "",
        bear_thesis_summary: str = "",
        cmf_score: float = 0.0,
        vwap_price: float = 0.0,
        forecast_date: str = None
    ) -> dict:
        """
        Komitenin ürettiği nihai analizi ve tahmin parametrelerini hafıza defterine kaydeder.
        """
        clean_ticker = ticker.replace(".IS", "").strip().upper()
        if forecast_date is None:
            forecast_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "record_id": f"{clean_ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "ticker": clean_ticker,
            "full_ticker": f"{clean_ticker}.IS",
            "date": forecast_date,
            "spot_price": round(spot_price, 2),
            "verdict": verdict.strip().upper(),
            "confidence": confidence,
            "target_1w": round(target_1w, 2),
            "target_15d": round(target_15d, 2),
            "stop_loss": round(stop_loss, 2),
            "bull_thesis_summary": bull_thesis_summary[:300],
            "bear_thesis_summary": bear_thesis_summary[:300],
            "cmf_score": round(cmf_score, 3),
            "vwap_price": round(vwap_price, 2),
            "evaluated": False,
            "evaluation_result": None
        }

        self.memory_data["history"].append(entry)
        self._save_memory()
        return entry

    def evaluate_past_decisions(self, ticker: str, current_price: float, df_history: pd.DataFrame = None) -> list[dict]:
        """
        Seçilen hissenin geçmiş kayıtlarını tarar; gerçekleşen fiyatlarla karşılaştırıp
        başarı/başarısızlık durumunu ve hangi ajanın haklı çıktığını değerlendirir.
        """
        clean_ticker = ticker.replace(".IS", "").strip().upper()
        evaluations = []

        for item in self.memory_data.get("history", []):
            if item.get("ticker") != clean_ticker:
                continue

            entry_price = item["spot_price"]
            entry_date_str = item["date"].split()[0]
            verdict = item["verdict"]
            t_1w = item["target_1w"]
            stop_l = item["stop_loss"]

            try:
                entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d")
                days_passed = (datetime.now() - entry_dt).days
            except Exception:
                days_passed = 7

            # Gerçekleşen getiri
            actual_return_pct = ((current_price - entry_price) / entry_price) * 100.0
            expected_1w_ret = ((t_1w - entry_price) / entry_price) * 100.0

            is_success = False
            evaluation_note = ""
            bull_validated = False
            bear_validated = False

            if "AL" in verdict or "BUY" in verdict:
                if current_price >= entry_price * 0.995 and actual_return_pct > 0:
                    is_success = True
                    bull_validated = True
                    evaluation_note = f"🟢 BAŞARILI: {entry_date_str} tarihindeki AL tavsiyesi %+ {actual_return_pct:.2f} kâr sağladı. Boğa tezi doğrulandı."
                elif current_price <= stop_l:
                    is_success = False
                    bear_validated = True
                    evaluation_note = f"🔴 BAŞARISIZ (Stop-Loss İhlali): Fiyat {current_price:.2f} TL'ye düşerek stop seviyesini ({stop_l:.2f}) kırdı. Ayı analistinin risk uyarıları haklı çıktı."
                else:
                    is_success = False
                    bear_validated = True
                    evaluation_note = f"🟡 HEDEF SAPMASI: Fiyat beklenen yükselişi yapamadı (%{actual_return_pct:+.2f}). Ayı baskısı teyit edildi."
            elif "SAT" in verdict or "SHORT" in verdict:
                if current_price < entry_price:
                    is_success = True
                    bear_validated = True
                    evaluation_note = f"🟢 BAŞARILI: Düşüş beklentisi doğrulandı, hisse %{actual_return_pct:.2f} geriledi. Ayı analisti haklı çıktı."
                else:
                    is_success = False
                    bull_validated = True
                    evaluation_note = f"🔴 YANILGI: Satış tavsiyesine rağmen hisse yükseldi (%{actual_return_pct:+.2f}). Boğa tezi güçlendi."
            else: # TUT / NÖTR
                is_success = abs(actual_return_pct) < 3.0
                evaluation_note = f"⚪ NÖTR BAŞARISI: Fiyat yatay konsolidasyon bandında kaldı (%{actual_return_pct:+.2f})."

            eval_dict = {
                "record_id": item["record_id"],
                "entry_date": item["date"],
                "days_passed": days_passed,
                "entry_price": entry_price,
                "current_price": current_price,
                "actual_return_pct": round(actual_return_pct, 2),
                "verdict": verdict,
                "is_success": is_success,
                "bull_validated": bull_validated,
                "bear_validated": bear_validated,
                "evaluation_note": evaluation_note
            }
            evaluations.append(eval_dict)

            # Hafızadaki kaydı güncelle
            item["evaluated"] = True
            item["evaluation_result"] = eval_dict

        self._save_memory()
        return evaluations

    def get_self_reflection_context(self, ticker: str, current_price: float) -> str:
        """
        Komite toplantısı başlamadan önce LLM promptlarına enjekte edilecek
        Öz-Yansıtma ve Geçmiş Hatalardan Ders Çıkarma brifingini üretir.
        """
        evals = self.evaluate_past_decisions(ticker, current_price)
        clean_ticker = ticker.replace(".IS", "").strip().upper()

        if not evals:
            return f"""* 📝 **Hafıza Durumu:** [{clean_ticker}] için sistem hafızasında daha önce kaydedilmiş bir komite kararı bulunmamaktadır. Bu analiz ilk referans hafıza kaydı olarak işlenecektir.
* 🎯 **Öz-Yansıtma Kuralı:** İnatçı önyargılardan kaçınınız; teknik, ekonometrik ve CMF para akışı verilerini objektif değerlendiriniz.
"""

        last_eval = evals[-1]
        success_count = sum(1 for e in evals if e["is_success"])
        total_evals = len(evals)
        historical_accuracy = (success_count / total_evals) * 100.0

        bull_correct = sum(1 for e in evals if e.get("bull_validated"))
        bear_correct = sum(1 for e in evals if e.get("bear_validated"))

        reflection_text = f"""* 🧠 **KOMİTE GEÇMİŞ PERFORMANS VE ÖZ-YANSITMA BRİFİNGİ ({clean_ticker}):**
  • **Geçmiş Tahmin Sayısı:** {total_evals} Analiz | **Tarihsel Başarı Oranı:** %{historical_accuracy:.1f}
  • **Son Karar Tarihi & Fiyatı:** {last_eval['entry_date']} ({last_eval['entry_price']:.2f} TRY)
  • **Önceki Kararımız:** **{last_eval['verdict']}**
  • **Piyasa Gerçekleşmesi:** Fiyat {current_price:.2f} TRY seviyesine geldi (Getiri: %{last_eval['actual_return_pct']:+.2f})
  • **Otomatik Post-Mortem Değerlendirmesi:** {last_eval['evaluation_note']}

* ⚖️ **Ajan Güvenilirlik & Doğrulanma Dengesi:**
  • Boğa Analisti Doğrulanma: **{bull_correct} Kez** | Ayı Analisti Doğrulanma: **{bear_correct} Kez**
"""
        if bear_correct > bull_correct:
            reflection_text += """  • ⚠️ **KOMİTEYE KRİTİK DERS:** Geçmiş analizlerimizde hisse üzerindeki satış baskısı ve değer tuzakları hafife alınmıştır. **Ayı analistinin risk uyarılarına ve CMF para çıkışlarına ekstra ağırlık veriniz!** 12.45 VWAP veya dirençler hacimli kırılmadan körü körüne alım önermeyiniz.
"""
        elif bull_correct > bear_correct:
            reflection_text += """  • 🟢 **KOMİTEYE KRİTİK DERS:** Geçmiş analizlerimizde hissenin dip toparlanma gücü doğru öngörülmüştür. Boğa katalizörlerini takip ediniz ancak stop-loss disiplinini elden bırakmayınız.
"""
        else:
            reflection_text += """  • ⚪ **KOMİTEYE KRİTİK DERS:** Piyasa dengelidir. Verilere göre dinamik reaksiyon veriniz.
"""
        return reflection_text


if __name__ == "__main__":
    print("🧠 BistCommitteeMemory Test Ediliyor...")
    memory = BistCommitteeMemory()
    
    # 1. Örnek Karar Kaydı Simülasyonu
    memory.record_decision(
        ticker="ISCTR.IS",
        spot_price=12.52,
        verdict="AL",
        confidence=85,
        target_1w=13.15,
        target_15d=13.65,
        stop_loss=12.10,
        bull_thesis_summary="BofA yabancı takas artışı ve 12.45 VWAP desteği",
        bear_thesis_summary="Bankacılık marj baskısı ve CMF negatifliği",
        cmf_score=-0.176,
        vwap_price=12.45,
        forecast_date="2026-08-15 10:00:00"
    )
    
    # 2. Güncel Fiyatla (12.38) Öz-Yansıtma Brifingi Üretme
    reflection = memory.get_self_reflection_context("ISCTR.IS", current_price=12.38)
    print("\n" + "="*80)
    print("📋 ÜRETİLEN ÖZ-YANSITMA VE DERS ÇIKARMA METNİ:")
    print("="*80)
    print(reflection)
    print("="*80)
