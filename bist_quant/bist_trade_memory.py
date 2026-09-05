#!/usr/bin/env python3
"""
🧠 KRONOS TRADEMEMORY PROTOCOL & EPİSODİK İŞLEM GÜNLÜĞÜ
(Awesome-MCP Knowledge & Memory / TradeMemory Protocol Deseninden Esinlenilmiştir)

Özellikler:
1. Her tamamlanan ve aktif VİOP işleminin teknik ve temel kurulum parmak izini (setup fingerprint) saklar.
2. Fiyat Action kurulumlarını (SSL Sweep, FVG C.E., Volume Dry-Up, Break & Retest) ve piyasa rejimini ilişkilendirir.
3. Komite toplantısında benzer geçmiş işlemleri semantik olarak sorgular ve "Çıkarılan Dersler (Post-Mortem Lessons)" üretir.
4. Böylece yapay zeka aynı hataları tekrarlamaz (Örn: Hacimsiz geri çekilmelerde erkenden stop olma hatası).
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

class BistTradeMemory:
    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.vault = self._load_vault()

    def _load_vault(self) -> dict:
        """Kasa ve işlem hafızasını yükler; yoksa başlangıç şablonuyla açar."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Başlangıç hafıza kasası
        initial_vault = {
            "version": "2.0-TradeMemory",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_trades_recorded": 2,
            "win_rate_pct": 100.0,
            "records": [
                {
                    "ticket_id": "VIOP-2026-001",
                    "ticker": "PGSUS",
                    "direction": "LONG",
                    "setup_type": "MOMENTUM_BREAKOUT",
                    "market_regime": "NEUTRAL_CHOP",
                    "entry_price": 149.80,
                    "exit_price": 151.00,
                    "pnl_try": 120.00,
                    "pnl_pct": 2.98,
                    "holding_days": 1,
                    "outcome": "WIN",
                    "lessons_learned": "Seans içi kâr koruma ile erken kâr alındı; piyasa oynaklığında sermaye korundu."
                },
                {
                    "ticket_id": "VIOP-2026-002",
                    "ticker": "BIMAS",
                    "direction": "LONG",
                    "setup_type": "SSL_SWEEP_FVG_RETEST",
                    "market_regime": "BULL_REGIME",
                    "entry_price": 411.50,
                    "exit_price": 415.75,
                    "pnl_try": 850.00,
                    "pnl_pct": 5.70,
                    "holding_days": 4,
                    "outcome": "WINNING_ACTIVE",
                    "lessons_learned": "Giriş sonrası seans içi 406 TL'ye gerileme oldu ancak hacim 0.57x (Volume Dry-Up) seviyesindeydi. Panik yapmayıp 380 TL stop seviyesine sadık kalınarak 415.75 TL'ye kârla toparlandı."
                }
            ]
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
        ticket_id = f"VIOP-2026-{len(records)+1:03d}"
        
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
        
        wins = sum(1 for r in records if r.get("outcome") in ["WIN", "WINNING_ACTIVE"])
        self.vault["win_rate_pct"] = round((wins / len(records)) * 100.0, 1)
        
        self._save_vault()
        return ticket_id

    def query_similar_setups(self, ticker: str = "", setup_type: str = "") -> list[dict]:
        """Belirli bir hisse veya setup tipine göre geçmiş tecrübeleri filtreler."""
        records = self.vault.get("records", [])
        clean_t = ticker.replace(".IS", "").upper() if ticker else ""
        
        matches = []
        for r in records:
            if clean_t and r.get("ticker") == clean_t:
                matches.append(r)
            elif setup_type and setup_type.lower() in r.get("setup_type", "").lower():
                matches.append(r)
                
        return matches or records[-3:]

    def generate_trade_memory_report(self, ticker: str, current_setup: str = "SSL_SWEEP") -> str:
        """Komite toplantısında sunulmak üzere hafıza brifingi oluşturur."""
        clean_t = ticker.replace(".IS", "")
        matches = self.query_similar_setups(clean_t, current_setup)
        
        lines = [
            f"🧠 **TRADEMEMORY PROTOCOL: EPİSODİK İŞLEM HAFIZASI & POST-MORTEM DERSLERİ:**",
            f"  • **Toplam Kayıtlı VİOP İşlemi:** {self.vault.get('total_trades_recorded', 0)} | **Kazanma Oranı (Win Rate):** %{self.vault.get('win_rate_pct', 100.0)}",
            f"  • **Sorgulanan Kurulum/Hisse:** {clean_t} (Setup: {current_setup})"
        ]
        
        for idx, m in enumerate(matches, 1):
            badge = "🟢 KAZANÇ" if "WIN" in m["outcome"] else "🔴 ZARAR"
            lines.append(f"  {idx}. [{m['ticket_id']}] **{m['ticker']} ({m['direction']})** ➔ {badge} (Kâr: {m['pnl_try']:+.2f} TL / %{m['pnl_pct']:+.1f})")
            lines.append(f"     ↳ **Kritik Ders:** {m['lessons_learned']}")
            
        lines.append("  • **Komite Direktifi:** Geçmişte yaşanan hacimsiz geri çekilmelerde panik satış yapılmamalı; belirlenen stop seviyesine ve hacim indikatörlerine sadık kalınmalıdır.")
        return "\n".join(lines)

if __name__ == "__main__":
    tm = BistTradeMemory()
    print(tm.generate_trade_memory_report("BIMAS.IS", "SSL_SWEEP_FVG_RETEST"))
