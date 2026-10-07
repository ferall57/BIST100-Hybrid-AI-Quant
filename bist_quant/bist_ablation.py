"""
Bileşen katkı ölçümü (ablation): aynı evren ve aynı dönemde, giriş kurallarını tek tek açıp kapatarak
her bileşenin sonuca etkisini ölçer. Karşılaştırma hisse bazında eşleştirilir: bir varyant kaç hissede
temel kurallardan iyi, kaç hissede kötü sonuç verdi.
"""

import contextlib
import io
import os
import sys
from datetime import datetime

import numpy as np
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.entry_filters import ALL_FILTERS, FILTER_ICT, FILTER_INDEX_GATE, FILTER_MONEY_FLOW, INDEX_TICKER

REPORTS_DIR = os.path.join(ROOT_DIR, "outputs", "reports")
INDEX_HISTORY_PERIOD = "5y"

BASELINE = "Temel kurallar"
VARIANTS = {
    BASELINE: {},
    "Rejim filtresi kapalı": {"enable_regime_filter": False},
    "+ ICT kurulumu": {"entry_filters": (FILTER_ICT,)},
    "+ Para akışı": {"entry_filters": (FILTER_MONEY_FLOW,)},
    "+ XU100 kapısı": {"entry_filters": (FILTER_INDEX_GATE,)},
    "+ Üç filtre birlikte": {"entry_filters": ALL_FILTERS},
}


def summarise_variant(name: str, rows: list[dict]) -> dict:
    """Bir varyantın evren genelindeki özeti."""
    def median(key):
        return float(np.median([row[key] for row in rows]))

    blocked = {}
    for row in rows:
        for filter_name, count in row.get("blocked_entries", {}).items():
            blocked[filter_name] = blocked.get(filter_name, 0) + count
    return {
        "variant": name,
        "tickers": len(rows),
        "median_return_pct": median("total_return_pct"),
        "median_bnh_pct": median("bnh_return_pct"),
        "median_alpha": median("alpha"),
        "share_beating_bnh_pct": sum(1 for row in rows if row["alpha"] > 0) / len(rows) * 100.0,
        "median_max_drawdown": median("max_drawdown"),
        "total_trades": int(sum(row["total_trades"] for row in rows)),
        "blocked_entries": blocked,
    }


def paired_vs_baseline(baseline_rows: list[dict], variant_rows: list[dict]) -> dict:
    """Aynı hisselerde varyant getirisi eksi temel getiri: medyan fark, iyileşen/kötüleşen sayısı, işaret testi."""
    baseline = {row["ticker"]: row["total_return_pct"] for row in baseline_rows}
    diffs = np.array([row["total_return_pct"] - baseline[row["ticker"]] for row in variant_rows if row["ticker"] in baseline])
    improved = int(np.sum(diffs > 0))
    worsened = int(np.sum(diffs < 0))
    decided = improved + worsened
    return {
        "median_diff_vs_baseline": float(np.median(diffs)) if len(diffs) else None,
        "improved": improved,
        "worsened": worsened,
        "p_value": float(stats.binomtest(improved, decided, 0.5).pvalue) if decided else 1.0,
    }


def write_report(summaries: list[dict], months: int) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    lines = [
        "# Bileşen Katkı Ölçümü",
        f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | **Hisse:** {summaries[0]['tickers']} | "
        f"**Dönem:** son {months} ay | Maliyetler dahil, kural tabanlı tahminci, yalnızca LONG",
        "",
        "| Varyant | Medyan getiri % | Medyan al-tut % | Medyan fark % | Al-tut'u geçen % | Medyan MaxDD % | İşlem | Temele göre medyan fark | İyileşen / kötüleşen hisse | p |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        paired = "—" if s.get("median_diff_vs_baseline") is None else f"{s['median_diff_vs_baseline']:+.2f}"
        counts = "—" if s.get("improved") is None else f"{s['improved']} / {s['worsened']}"
        p_value = "—" if s.get("p_value") is None else f"{s['p_value']:.3f}"
        lines.append(f"| {s['variant']} | {s['median_return_pct']:+.2f} | {s['median_bnh_pct']:+.2f} | {s['median_alpha']:+.2f} | "
                     f"{s['share_beating_bnh_pct']:.1f} | {s['median_max_drawdown']:.2f} | {s['total_trades']} | {paired} | {counts} | {p_value} |")
    lines += ["", "## Filtrelerin engellediği giriş sayısı", ""]
    for s in summaries:
        if s["blocked_entries"]:
            detail = ", ".join(f"{name}: {count}" for name, count in s["blocked_entries"].items())
            lines.append(f"* **{s['variant']}**: {detail}")
    lines += [
        "",
        "> Filtreler yalnızca LONG girişi engeller; işlem eklemez. Bir filtrenin getirisi çoğunlukla daha az işlem yapmaktan gelir.",
        "> Aynı tarihteki hisseler birlikte hareket ettiği için p değerleri olduğundan güçlü görünür; tek dönem tek piyasa rejimidir.",
    ]
    report_path = os.path.join(REPORTS_DIR, "ablation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def run_ablation(tickers: list[str], months: int = 12, costs=None, variants: dict = None) -> tuple[list[dict], str]:
    """Her varyantı aynı evrende koşar; özetleri ve temel kurallara göre eşleştirilmiş farkları döndürür."""
    from bist_quant.bist_backtester import BistBacktester
    from bist_quant.bist_downloader import RAW_DATA_DIR, download_ticker_data

    variants = variants or VARIANTS
    download_ticker_data(INDEX_TICKER, period=INDEX_HISTORY_PERIOD, interval="1d", save_dir=RAW_DATA_DIR)
    backtester = BistBacktester(use_kronos=False, costs=costs)

    results = {}
    for position, (name, overrides) in enumerate(variants.items()):
        print(f"▶️ {name} ({position + 1}/{len(variants)})", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _ = backtester.run_universe_backtest(
                tickers, months=months, refresh_data=(position == 0), write_artifacts=False, **overrides
            )
        if not rows:
            raise ValueError(f"'{name}' varyantında hiçbir hisse test edilemedi.")
        results[name] = rows

    baseline_name = next(iter(variants))
    summaries = []
    for name, rows in results.items():
        summary = summarise_variant(name, rows)
        if name == baseline_name:
            summary.update({"median_diff_vs_baseline": None, "improved": None, "worsened": None, "p_value": None})
        else:
            summary.update(paired_vs_baseline(results[baseline_name], rows))
        summaries.append(summary)
    return summaries, write_report(summaries, months)
