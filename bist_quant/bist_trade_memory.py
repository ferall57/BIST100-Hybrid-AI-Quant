#!/usr/bin/env python3
"""
🧠 KRONOS TRADEMEMORY PROTOCOL & EPİSODİK İŞLEM GÜNLÜĞÜ

Özellikler:
1. Her tamamlanan ve aktif VİOP işleminin teknik ve temel kurulum parmak izini (setup fingerprint) saklar.
2. Fiyat Action kurulumlarını (SSL Sweep, FVG C.E., Volume Dry-Up, Break & Retest) ve piyasa rejimini ilişkilendirir.
3. Komite toplantısında aynı hisseye veya aynı kuruluma ait geçmiş işlemleri sorgular ve kayıtlı dersleri sunar.
4. Kayıt yoksa bunu açıkça bildirir; örnek işlem veya başarı oranı uydurmaz.
"""

import os
import sys
import json
from datetime import datetime

# Windows konsol Unicode uyumluluğu
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMORY_DIR = os.path.join(ROOT_DIR, "outputs", "trade_memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, "trade_memory_vault.json")
WINNING_OUTCOMES = ("WIN", "WINNING_ACTIVE")

def _win_rate_pct(records: list[dict]) -> float | None:
    """Kayıtlı işlemlerden kazanma oranı; kayıt yoksa None."""
    if not records:
        return None
    wins = sum(1 for r in records if r.get("outcome") in WINNING_OUTCOMES)
    return round((wins / len(records)) * 100.0, 1)

class BistTradeMemory:
    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.vault = self._load_vault()

    def _load_vault(self) -> dict:
        """Kasa ve işlem hafızasını yükler; yoksa boş kasa açar."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[UYARI] İşlem hafızası okunamadı, boş kasa ile devam ediliyor: {e}")

        initial_vault = {
            "version": "2.0-TradeMemory",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_trades_recorded": 0,
            "win_rate_pct": None,
            "records": []
        }
        self._save_vault(initial_vault)
        return initial_vault

    def _save_vault(self, data: dict = None):
        """Hafıza kasasını diske güvenle yazar."""
        to_save = data or self.vault
        to_save["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2, ensure_ascii=False)

    def record_trade(self, ticker: str, direction: str, setup_type: str, market_regime: str,
                     entry_price: float, exit_price: float, pnl_try: float, pnl_pct: float,
                     holding_days: int, outcome: str, lessons_learned: str) -> str:
        """Yeni bir işlem sonucunu ve çıkarılan dersi hafızaya kaydeder."""
        records = self.vault.get("records", [])
        ticket_id = f"VIOP-{datetime.now().year}-{len(records)+1:03d}"

        entry = {
            "ticket_id": ticket_id,
            "ticker": ticker.replace(".IS", ""),
            "direction": direction,
            "setup_type": setup_type,
            "market_regime": market_regime,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_try": pnl_try,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "outcome": outcome,
            "lessons_learned": lessons_learned,
            "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        records.append(entry)
        self.vault["records"] = records
        self.vault["total_trades_recorded"] = len(records)
        self.vault["win_rate_pct"] = _win_rate_pct(records)

        self._save_vault()
        return ticket_id

    def query_similar_setups(self, ticker: str = "", setup_type: str = "") -> list[dict]:
        """Aynı hisseye veya aynı kurulum tipine ait geçmiş işlemleri döndürür; yoksa boş liste."""
        records = self.vault.get("records", [])
        clean_t = ticker.replace(".IS", "").upper() if ticker else ""

        matches = []
        for r in records:
            if clean_t and r.get("ticker") == clean_t:
                matches.append(r)
            elif setup_type and setup_type.lower() in r.get("setup_type", "").lower():
                matches.append(r)

        return matches

    def generate_trade_memory_report(self, ticker: str, current_setup: str = "") -> str:
        """Komite toplantısında sunulmak üzere hafıza brifingi oluşturur."""
        clean_t = ticker.replace(".IS", "")
        records = self.vault.get("records", [])
        matches = self.query_similar_setups(clean_t, current_setup)
        win_rate = _win_rate_pct(records)
        win_rate_text = f"%{win_rate}" if win_rate is not None else "kayıt yok"

        lines = [
            f"🧠 **TRADEMEMORY PROTOCOL: EPİSODİK İŞLEM HAFIZASI & POST-MORTEM DERSLERİ:**",
            f"  • **Toplam Kayıtlı VİOP İşlemi:** {len(records)} | **Kazanma Oranı (Win Rate):** {win_rate_text}",
            f"  • **Sorgulanan Hisse:** {clean_t} | **Tespit Edilen Kurulum:** {current_setup or 'tespit edilmedi'}"
        ]

        if not matches:
            lines.append("  • **Benzer Geçmiş İşlem:** Kayıt yok. Bu hisse veya kurulum için geçmiş işlem dersi "
                         "bulunmuyor; hafızaya dayalı çıkarım yapmayınız.")
            return "\n".join(lines)

        for idx, m in enumerate(matches, 1):
            badge = "🟢 KAZANÇ" if "WIN" in m["outcome"] else "🔴 ZARAR"
            lines.append(f"  {idx}. [{m['ticket_id']}] **{m['ticker']} ({m['direction']})** ➔ {badge} (Kâr: {m['pnl_try']:+.2f} TL / %{m['pnl_pct']:+.1f})")
            lines.append(f"     ↳ **Kritik Ders:** {m['lessons_learned']}")

        return "\n".join(lines)

if __name__ == "__main__":
    tm = BistTradeMemory()
    print(tm.generate_trade_memory_report("BIMAS.IS", "SSL_SWEEP_FVG_RETEST"))
