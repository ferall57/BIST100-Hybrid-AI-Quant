"""
Tek noktadan yönetilen piyasa varsayımları (faiz oranları).
Değerler .env dosyasından veya ortam değişkenlerinden yüzde olarak okunur (örn. 42.5).
"""

import os

from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RISK_FREE_RATE_ENV = "KRONOS_RISK_FREE_RATE_PCT"
POLICY_RATE_ENV = "KRONOS_POLICY_RATE_PCT"

# Ortam değişkeni tanımlı değilse hesaplamalarda kullanılan yıllık risksiz faiz varsayımı.
DEFAULT_RISK_FREE_RATE = 0.45
MAX_RATE_PCT = 1000.0

load_dotenv(os.path.join(ROOT_DIR, ".env"))


def _read_rate_pct(env_name: str) -> float | None:
    raw = os.getenv(env_name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw.strip().rstrip("%").replace(",", "."))
    except ValueError:
        raise ValueError(f"{env_name} sayısal bir yüzde olmalı (örn. 42.5); gelen değer: {raw!r}") from None
    if not 0.0 <= value <= MAX_RATE_PCT:
        raise ValueError(f"{env_name} 0 ile {MAX_RATE_PCT:.0f} arasında bir yüzde olmalı; gelen değer: {raw!r}")
    return value


def risk_free_rate() -> float:
    """Yıllık risksiz faiz (ondalık). Tanımlı değilse DEFAULT_RISK_FREE_RATE varsayımı döner."""
    pct = _read_rate_pct(RISK_FREE_RATE_ENV)
    return DEFAULT_RISK_FREE_RATE if pct is None else pct / 100.0


def policy_rate_pct() -> float | None:
    """TCMB politika faizi (yüzde). Kullanıcı tanımlamadıysa None: varsayılan uydurulmaz."""
    return _read_rate_pct(POLICY_RATE_ENV)
