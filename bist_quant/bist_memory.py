#!/usr/bin/env python3
"""
🏛️ BIST YATIRIM KOMİTESİ ZAMAN VE VADE DUYARLI ÖZ-YANSITMA HAFIZA MOTORU
(Horizon & Time-Aware Self-Reflection Memory)

Özellikler:
1. Zaman & Vade Farkındalığı (Geçen saat/gün hesabına göre vadesi dolmamış analizleri 'TAKİPTE' tutar).
2. Dakikalar veya saatler önce yapılan analizleri 'Yeni / Güncel Analiz' olarak tanır ve erken başarı/hata damgalamaz.
3. Vadesi dolan (>=3-5 gün) geçmiş analizlerde gerçekçi Post-Mortem hedef/stop denetimi yapar.
4. Boğa vs Ayı itibar puanlamasını sadece vadesi tamamlanmış gerçek döngülerden hesaplar.
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


import tempfile

class BistCommitteeMemory:
    """
    BIST Yapay Zeka Komitesi için Zaman ve Vade Duyarlı Kalıcı Hafıza Yöneticisi.
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
                "version": "1.1.0"
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
        """Hafızayı atomik olarak diske kaydeder (race condition kalkanı)."""
        if data is None:
            data = self.memory_data
        dir_name = os.path.dirname(self.memory_filepath)
        os.makedirs(dir_name, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                json.dump(data, tf, ensure_ascii=False, indent=2)
                temp_name = tf.name
            os.replace(temp_name, self.memory_filepath)
        except Exception:
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
        now_dt = datetime.now()
        if forecast_date is None:
            forecast_date = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Son 15 dakika içinde aynı hissede mükerrer kayıt açılmasını engelle (güncelle)
        history = self.memory_data.get("history", [])
        for item in reversed(history):
            if item.get("ticker") == clean_ticker:
                try:
                    last_dt = datetime.strptime(item["date"], "%Y-%m-%d %H:%M:%S")
                    if (now_dt - last_dt).total_seconds() < 900:  # 15 dakika
                        item["spot_price"] = round(spot_price, 2)
                        item["verdict"] = verdict.strip().upper()
                        item["confidence"] = confidence
                        item["target_1w"] = round(target_1w, 2)
                        item["target_15d"] = round(target_15d, 2)
                        item["stop_loss"] = round(stop_loss, 2)
                        item["bull_thesis_summary"] = bull_thesis_summary[:300]
                        item["bear_thesis_summary"] = bear_thesis_summary[:300]
                        item["cmf_score"] = round(cmf_score, 3)
                        item["vwap_price"] = round(vwap_price, 2)
                        item["date"] = forecast_date
                        self._save_memory()
                        return item
                except Exception:
                    pass

        entry = {
            "record_id": f"{clean_ticker}_{now_dt.strftime('%Y%m%d_%H%M%S')}",
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
            "status": "ACTIVE_IN_PROGRESS",
            "evaluated": False,
            "evaluation_result": None
        }

        self.memory_data["history"].append(entry)
        self._save_memory()
        return entry

    def evaluate_past_decisions(self, ticker: str, current_price: float) -> list[dict]:
        """
        Geçmiş analizleri zaman damgası ve vade dolumuna göre değerlendirir.
        - < 24 saat: 'GÜNCEL_TAKİPTE' (Erken başarısızlık damgalamaz)
        - 1 - 3 gün: 'KISA_VADE_GELİŞİYOR'
        - >= 3 gün: 'VADE_SONU_POST_MORTEM' (Hedef/Stop resmi değerlendirmesi)
        """
        clean_ticker = ticker.replace(".IS", "").strip().upper()
        evaluations = []
        now_dt = datetime.now()

        for item in self.memory_data.get("history", []):
            if item.get("ticker") != clean_ticker:
                continue

            entry_price = item["spot_price"]
            date_str = item["date"]
            verdict = item["verdict"]
            t_1w = item["target_1w"]
            stop_l = item["stop_loss"]

            try:
                if len(date_str) > 10:
                    entry_dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                else:
                    entry_dt = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                entry_dt = now_dt - timedelta(days=1)

            diff_seconds = (now_dt - entry_dt).total_seconds()
            hours_passed = max(0.0, diff_seconds / 3600.0)
            days_passed = max(0.0, diff_seconds / 86400.0)

            actual_return_pct = ((current_price - entry_price) / max(0.01, entry_price)) * 100.0

            is_mature = days_passed >= 3.0
            is_success = False
            bull_validated = False
            bear_validated = False
            status = "ACTIVE_IN_PROGRESS"
            evaluation_note = ""

            if not is_mature:
                # VADESİ HENÜZ DOLMAMIŞ ÇOK YENİ ANALİZ
                status = "IN_PROGRESS_FRESH"
                if hours_passed < 1.0:
                    time_label = f"{int(diff_seconds // 60)} dakika önce"
                elif hours_passed < 24.0:
                    time_label = f"{hours_passed:.1f} saat önce"
                else:
                    time_label = f"{days_passed:.1f} gün önce"

                evaluation_note = f"⏳ AKTİF TAKİPTE ({time_label} üretildi): Vadesi henüz dolmadı. Anlık Fiyat: {current_price:.2f} TRY (Giriş: {entry_price:.2f} TRY, Getiri: %{actual_return_pct:+.2f}). Hedef/stop seviyeleri izleniyor."
            else:
                # VADESİ DOLMUŞ (>= 3 GÜN) TARİHSEL ANALİZ DEĞERLENDİRMESİ
                status = "MATURED_POST_MORTEM"
                if "AL" in verdict or "BUY" in verdict:
                    if current_price >= entry_price * 1.01:
                        is_success = True
                        bull_validated = True
                        evaluation_note = f"🟢 BAŞARILI: {date_str[:10]} tarihindeki AL tavsiyesi %+ {actual_return_pct:.2f} getiri sağladı. Boğa tezi doğrulandı."
                    elif current_price <= stop_l:
                        is_success = False
                        bear_validated = True
                        evaluation_note = f"🔴 BAŞARISIZ (Stop-Loss İhlali): Fiyat {current_price:.2f} TL'ye inerek stop seviyesini ({stop_l:.2f}) kırdı. Ayı uyarısı haklı çıktı."
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
                else:
                    is_success = abs(actual_return_pct) < 3.0
                    evaluation_note = f"⚪ NÖTR BAŞARISI: Fiyat yatay konsolidasyon bandında kaldı (%{actual_return_pct:+.2f})."

            eval_dict = {
                "record_id": item["record_id"],
                "entry_date": date_str,
                "hours_passed": round(hours_passed, 1),
                "days_passed": round(days_passed, 2),
                "is_mature": is_mature,
                "status": status,
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
            item["evaluated"] = is_mature
            item["evaluation_result"] = eval_dict

        self._save_memory()
        return evaluations

    def get_self_reflection_context(self, ticker: str, current_price: float) -> str:
        """
        Komite toplantısı başlamadan önce LLM promptlarına enjekte edilecek
        Zaman ve Vade Duyarlı Öz-Yansıtma ve Geçmiş Hatalardan Ders Çıkarma brifingini üretir.
        """
        evals = self.evaluate_past_decisions(ticker, current_price)
        clean_ticker = ticker.replace(".IS", "").strip().upper()

        if not evals:
            return f"""* 📝 **Hafıza Durumu:** [{clean_ticker}] için sistem hafızasında daha önce kayıtlı bir komite kararı bulunmamaktadır. Bu analiz ilk referans hafıza kaydı olarak işlenecektir.
* 🎯 **Öz-Yansıtma Kuralı:** İnatçı önyargılardan kaçınınız; teknik formasyonları, ekonometrik riskleri ve CMF para akışını objektif değerlendiriniz.
"""

        last_eval = evals[-1]
        mature_evals = [e for e in evals if e["is_mature"]]
        
        # Eğer son analiz 24 saatten daha yeniyse (örneğin 5 dakika veya birkaç saat önce yapılmışsa)
        if not last_eval["is_mature"]:
            hours_ago = last_eval["hours_passed"]
            time_str = f"{int(hours_ago * 60)} dakika önce" if hours_ago < 1.0 else f"{hours_ago:.1f} saat önce"
            
            return f"""* 🧠 **KOMİTE GÜNCEL TAKİP VE HAFIZA BİLGİSİ ({clean_ticker}):**
  • **Son Analiz Zamanı:** {last_eval['entry_date']} ({time_str})
  • **Son Kararımız:** **{last_eval['verdict']}** (Referans Fiyat: {last_eval['entry_price']:.2f} TRY)
  • **Mevcut Piyasa Durumu:** Fiyat şu an **{current_price:.2f} TRY** (%{last_eval['actual_return_pct']:+.2f} değişim).
  • **Vade Durumu:** ⏳ **Henüz vadesi dolmadı, işlem takip periyodu devam ediyor.**
  • **Komite Direktifi:** Kısa süreli seans içi dalgalanmalara karşı aşırı panik veya inatçılık yapmayınız; belirlenen stop-loss ve hedef seviyelerine sadık kalarak stratejiyi güncelleyiniz.
"""

        # Eğer vadesi dolmuş tarihsel analizler varsa:
        total_mature = len(mature_evals)
        success_count = sum(1 for e in mature_evals if e["is_success"])
        accuracy = (success_count / total_mature) * 100.0 if total_mature > 0 else 0.0

        bull_correct = sum(1 for e in mature_evals if e.get("bull_validated"))
        bear_correct = sum(1 for e in mature_evals if e.get("bear_validated"))

        reflection_text = f"""* 🧠 **KOMİTE GEÇMİŞ VADE SONU (POST-MORTEM) HAFIZA RAPORU ({clean_ticker}):**
  • **Vadesi Dolan Analiz Sayısı:** {total_mature} Adet | **Tarihsel Başarı Oranı:** %{accuracy:.1f}
  • **Önceki Kararımız ({last_eval['entry_date'][:10]}):** **{last_eval['verdict']}** ({last_eval['entry_price']:.2f} TRY)
  • **Piyasa Gerçekleşmesi:** Fiyat {current_price:.2f} TRY seviyesine geldi (Getiri: %{last_eval['actual_return_pct']:+.2f})
  • **Post-Mortem Notu:** {last_eval['evaluation_note']}

* ⚖️ **Tarihsel Ajan Güvenilirlik Dengesi:**
  • Boğa Analisti Doğrulanma: **{bull_correct} Kez** | Ayı Analisti Doğrulanma: **{bear_correct} Kez**
"""
        if bear_correct > bull_correct:
            reflection_text += """  • ⚠️ **KOMİTEYE KRİTİK DERS:** Geçmiş analizlerimizde satış baskısı ve değer tuzakları hafife alınmıştır. **Ayı analistinin risk uyarılarına ve CMF para çıkışlarına ekstra ağırlık veriniz!** Dirençler hacimli kırılmadan körü körüne alım önermeyiniz.
"""
        elif bull_correct > bear_correct:
            reflection_text += """  • 🟢 **KOMİTEYE KRİTİK DERS:** Hissenin toparlanma potansiyeli doğru öngörülmüştür. Boğa katalizörlerini takip ediniz ancak stop-loss disiplinini koruyunuz.
"""
        else:
            reflection_text += """  • ⚪ **KOMİTEYE KRİTİK DERS:** Piyasa dengelidir. Verilere göre dinamik reaksiyon veriniz.
"""
        return reflection_text


if __name__ == "__main__":
    print("🧠 BistCommitteeMemory Zaman Duyarlılık Testi...")
    memory = BistCommitteeMemory()
    
    # 5 dakika önce yapılan analiz simülasyonu
    memory.record_decision(
        ticker="ISCTR.IS",
        spot_price=12.39,
        verdict="TUT",
        confidence=78,
        target_1w=12.45,
        target_15d=12.80,
        stop_loss=12.35,
        bull_thesis_summary="Faiz indirimi beklentisi ve iştirak portföyü",
        bear_thesis_summary="Rekor hacimde tıkanma ve CMF -0.182 para çıkışı",
        cmf_score=-0.182,
        vwap_price=12.39
    )
    
    ctx = memory.get_self_reflection_context("ISCTR.IS", current_price=12.39)
    print("\n" + "="*80)
    print("📋 ÜRETİLEN ZAMAN VE VADE DUYARLI HAFIZA METNİ:")
    print("="*80)
    print(ctx)
    print("="*80)
