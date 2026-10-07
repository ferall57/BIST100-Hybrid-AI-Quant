"""
Kronos modellerinin doğrulama durumu.

Model karşılaştırması (bist_model_eval) her modelin tahmin isabetini ölçer ve özetini buraya yazar.
Tahmin üreten ve tüketen kod bu özeti okuyarak modelin çıktısını "doğrulanmış" ya da "doğrulanmamış"
olarak etiketler; doğrulanmamış bir tahmin karar ve sıralamada kullanılmaz.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models", "bist_kronos")
VALIDATION_FILE = os.path.join(MODELS_DIR, "validation.json")
FINETUNED_WEIGHTS_FILE = os.path.join(MODELS_DIR, "bist100_kronos_base", "basemodel", "best_model", "model.safetensors")

MODEL_BASE = "Temel Kronos"
MODEL_FINETUNED = "Yeni ince ayar"
MODEL_LEGACY = "Eski ince ayar"

LABEL_VALIDATED = "DOĞRULANDI"
LABEL_UNVALIDATED = "DOĞRULANMADI"
# Doğrulanmış sayılmak için: hata naif ("fiyat değişmez") tahminden düşük VE yön isabeti yazı-turadan iyi olmalı.
MIN_SKILL_VS_NAIVE_PCT = 0.0
MIN_DIRECTIONAL_ACCURACY_PCT = 50.0
SUMMARY_FIELDS = ("horizon", "n", "directional_accuracy_pct", "mae_pct", "naive_mae_pct",
                  "skill_vs_naive_pct", "information_coefficient")


@dataclass(frozen=True)
class ValidationStatus:
    is_validated: bool
    label: str
    detail: str


def weights_fingerprint(path: str = FINETUNED_WEIGHTS_FILE) -> float | None:
    """İnce ayarlı modelin ağırlık dosyası kimliği (değişiklik zamanı); model yoksa None."""
    return os.path.getmtime(path) if os.path.exists(path) else None


def write_validation(metrics: pd.DataFrame, period_start: str, period_end: str, fingerprints: dict, path: str = VALIDATION_FILE) -> str:
    """Karşılaştırma ölçülerinden her modelin en uzun ufuktaki satırını doğrulama özeti olarak kaydeder."""
    models = {}
    for name, rows in metrics.groupby("model", sort=False):
        long_horizon = rows.iloc[-1]
        models[name] = {
            **{field: (long_horizon[field].item() if hasattr(long_horizon[field], "item") else long_horizon[field]) for field in SUMMARY_FIELDS},
            "weights_fingerprint": fingerprints.get(name),
        }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "period_start": period_start,
            "period_end": period_end,
            "models": models,
        }, f, indent=2, ensure_ascii=False)
    return path


def validation_status(model_name: str, path: str = VALIDATION_FILE, current_fingerprint: float | None = None) -> ValidationStatus:
    """Modelin doğrulama durumunu ve dayandığı ölçümü döndürür; ölçüm yoksa veya eskiyse doğrulanmamış sayılır."""
    def unvalidated(detail: str) -> ValidationStatus:
        return ValidationStatus(False, LABEL_UNVALIDATED, detail)

    if not os.path.exists(path):
        return unvalidated("ölçüm yok; 'python main.py --compare-models' ile ölçülmeli")
    try:
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        record = summary["models"].get(model_name)
    except (ValueError, KeyError, OSError) as e:
        print(f"[UYARI] Doğrulama özeti okunamadı ({path}): {e}")
        return unvalidated("ölçüm yok; doğrulama özeti okunamadı")
    if record is None:
        return unvalidated(f"ölçüm yok; '{model_name}' modeli karşılaştırmada ölçülmedi")

    measured_fingerprint = record.get("weights_fingerprint")
    if current_fingerprint is not None and measured_fingerprint is not None and current_fingerprint != measured_fingerprint:
        return unvalidated(f"model {summary['evaluated_at']} tarihli ölçümden sonra yeniden eğitildi; ölçüm tekrarlanmalı")

    evidence = (f"{record['horizon']} ufukta naife göre beceri %{record['skill_vs_naive_pct']:+.1f}, "
                f"yön isabeti %{record['directional_accuracy_pct']:.1f}, n={record['n']}, "
                f"dönem {summary['period_start']} – {summary['period_end']}")
    passed = (record["skill_vs_naive_pct"] > MIN_SKILL_VS_NAIVE_PCT
              and record["directional_accuracy_pct"] > MIN_DIRECTIONAL_ACCURACY_PCT)
    if passed:
        return ValidationStatus(True, LABEL_VALIDATED, evidence)
    return unvalidated(f"\"fiyat değişmeyecek\" diyen naif tahmini geçemedi ({evidence})")


def format_banner(status: ValidationStatus) -> str:
    """Tahmin raporunun başına konan doğrulama satırı."""
    if status.is_validated:
        return f"**Model Doğrulama Durumu: ✅ {status.label}** — {status.detail}."
    return (f"**Model Doğrulama Durumu: ⚠️ {status.label}** — {status.detail}. "
            "Aşağıdaki tahminler bilgi amaçlıdır; yatırım kararına, hedef fiyata veya güven katsayısına gerekçe yapılmamalıdır.")
