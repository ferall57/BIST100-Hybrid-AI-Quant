"""
Monte Carlo + para çıkışı vetosunun geçmiş veride ölçülmesi.

Komite eskiden bir AL kararını, Merton MC'nin 15 günlük yükseliş olasılığı %38'in altında VE
CMF-20 -0.12'nin altındaysa TUT'a çeviriyordu. Bu ölçümün ardından koşul bağlam bilgisine çevrildi
(hybrid_agents.bist_committee.risk_condition_met / risk_context_note).
Bu iki girdi yalnızca fiyat verisinden hesaplandığı için kural geçmişte yeniden oynatılabilir:

- Her hissede, yalnızca o güne kadarki mumlar kullanılarak belirli aralıklarla MC olasılığı ve CMF hesaplanır.
- Vetonun (ve her parçasının tek başına) devreye girdiği günlerin ileri getirisi,
  girmediği günlerle karşılaştırılır (ham ve XU100'e göre fazla getiri).

Veto ancak devreye girdiği günlerin ileri getirisi belirgin biçimde daha kötüyse alım engellemeye değer.
Ölçüm tüm günleri kapsar; komitenin AL dediği günlerin seçimini test etmez.
"""

from __future__ import annotations

import contextlib
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from bist_quant.bist_downloader import RAW_DATA_DIR
from hybrid_agents.bist_committee import (
    CMF_VETO_MAX,
    MC_VETO_MAX_WIN_PROBABILITY_PCT,
    risk_condition_met,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(ROOT_DIR, "outputs", "reports")
INDEX_TICKER = "XU100.IS"

MC_FORECAST_DAYS = 15
MC_HISTORY_BARS = 120
HORIZONS = (5, 10, 15)
DEFAULT_STEP = 5
TRADING_DAYS_PER_MONTH = 21

CONDITIONS = {
    "veto": lambda prob, cmf: risk_condition_met({"prob_positive": prob}, {"cmf_20": cmf}),
    "mc_low": lambda prob, cmf: prob < MC_VETO_MAX_WIN_PROBABILITY_PCT,
    "cmf_out": lambda prob, cmf: cmf < CMF_VETO_MAX,
}
CONDITION_LABELS = {
    "veto": "Veto (MC düşük VE CMF çıkış)",
    "mc_low": f"Yalnız MC < %{MC_VETO_MAX_WIN_PROBABILITY_PCT:.0f}",
    "cmf_out": f"Yalnız CMF < {CMF_VETO_MAX}",
}


def evaluation_points(n_bars: int, first: int, step: int, max_horizon: int) -> list[int]:
    """İleri getirisi tam ölçülebilen değerlendirme mumlarının konumları."""
    return list(range(first, n_bars - max_horizon, step))


def forward_return_pct(df: pd.DataFrame, t: int, horizon: int) -> float | None:
    """t kapanışından t+horizon kapanışına getiri; veri yetmezse None."""
    if t + horizon >= len(df):
        return None
    return (df["close"].iloc[t + horizon] / df["close"].iloc[t] - 1.0) * 100.0


def _index_return_pct(index: pd.DataFrame, start_day, end_day) -> float | None:
    days = pd.to_datetime(index["timestamps"]).dt.normalize()
    start = index[days <= start_day]
    end = index[days <= end_day]
    if start.empty or end.empty:
        return None
    return (end["close"].iloc[-1] / start["close"].iloc[-1] - 1.0) * 100.0


def _default_mc_fn():
    from bist_quant.bist_econometrics import BistEconometrics
    engine = BistEconometrics()

    def mc(df: pd.DataFrame) -> float:
        with contextlib.redirect_stdout(io.StringIO()):
            result = engine.run_monte_carlo_simulation(df, days=MC_FORECAST_DAYS, num_sims=1000)
        return result.get("prob_positive", 50.0)
    return mc


def _default_cmf_fn():
    from bist_quant.bist_akd_flow import BistAkdFlowEngine
    engine = BistAkdFlowEngine()
    return lambda df: engine.calculate_money_flow_metrics(df)["cmf_20"]


def run_veto_study(prices: dict[str, pd.DataFrame], index: pd.DataFrame | None,
                   horizons=HORIZONS, step: int = DEFAULT_STEP, first: int = MC_HISTORY_BARS,
                   mc_fn=None, cmf_fn=None) -> list[dict]:
    """Her hisse ve değerlendirme noktası için koşul bayrakları ve ileri getirileri döndürür."""
    mc_fn = mc_fn or _default_mc_fn()
    cmf_fn = cmf_fn or _default_cmf_fn()
    rows = []
    for ticker, df in prices.items():
        df = df.reset_index(drop=True)
        for t in evaluation_points(len(df), max(first, MC_HISTORY_BARS), step, max(horizons)):
            history = df.iloc[:t + 1]
            prob, cmf = mc_fn(history), cmf_fn(history)
            row = {"ticker": ticker, "date": pd.Timestamp(df["timestamps"].iloc[t]).strftime("%Y-%m-%d"),
                   "prob_positive": prob, "cmf_20": cmf}
            row.update({name: bool(rule(prob, cmf)) for name, rule in CONDITIONS.items()})
            start_day = pd.Timestamp(df["timestamps"].iloc[t]).normalize()
            for h in horizons:
                fwd = forward_return_pct(df, t, h)
                idx = None
                if fwd is not None and index is not None:
                    end_day = pd.Timestamp(df["timestamps"].iloc[t + h]).normalize()
                    idx = _index_return_pct(index, start_day, end_day)
                row[f"fwd_{h}"] = fwd
                row[f"excess_{h}"] = None if fwd is None or idx is None else fwd - idx
            rows.append(row)
    return rows


def _mean(values: list) -> float | None:
    return float(np.mean(values)) if values else None


def summarise_condition(rows: list[dict], condition: str, horizon: int) -> dict:
    """Koşulun devrede olduğu ve olmadığı günlerin ileri getirilerini karşılaştırır."""
    fwd_key, ex_key = f"fwd_{horizon}", f"excess_{horizon}"
    usable = [r for r in rows if r.get(fwd_key) is not None]
    fired = [r for r in usable if r[condition]]
    quiet = [r for r in usable if not r[condition]]

    ex_fired = [r[ex_key] for r in fired if r.get(ex_key) is not None]
    ex_quiet = [r[ex_key] for r in quiet if r.get(ex_key) is not None]
    p_value = None
    if len(ex_fired) >= 2 and len(ex_quiet) >= 2:
        p_value = float(stats.mannwhitneyu(ex_fired, ex_quiet, alternative="two-sided").pvalue)

    compared = worse = 0
    for ticker in sorted({r["ticker"] for r in usable}):
        f = [r[ex_key] for r in fired if r["ticker"] == ticker and r.get(ex_key) is not None]
        q = [r[ex_key] for r in quiet if r["ticker"] == ticker and r.get(ex_key) is not None]
        if f and q:
            compared += 1
            worse += int(np.mean(f) < np.mean(q))

    return {
        "condition": condition, "horizon": horizon,
        "fired": len(fired), "quiet": len(quiet),
        "fire_rate_pct": 100.0 * len(fired) / len(usable) if usable else 0.0,
        "mean_fired": _mean([r[fwd_key] for r in fired]),
        "mean_quiet": _mean([r[fwd_key] for r in quiet]),
        "mean_excess_fired": _mean(ex_fired),
        "mean_excess_quiet": _mean(ex_quiet),
        "share_negative_fired_pct": 100.0 * np.mean([r[fwd_key] < 0 for r in fired]) if fired else None,
        "share_negative_quiet_pct": 100.0 * np.mean([r[fwd_key] < 0 for r in quiet]) if quiet else None,
        "tickers_fired_worse": worse, "tickers_compared": compared,
        "p_value": p_value,
    }


def load_prices(tickers: list[str], months: int, data_dir: str = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Yerel CSV'lerden son `months` ayı (artı MC için geçmiş) okur; dosyası olmayan hisse atlanır."""
    keep = months * TRADING_DAYS_PER_MONTH + MC_HISTORY_BARS + max(HORIZONS)
    prices = {}
    for ticker in tickers:
        path = os.path.join(data_dir, f"{ticker}_1d.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        prices[ticker] = df.sort_values("timestamps").tail(keep).reset_index(drop=True)
    return prices


def _fmt(value, spec: str = "+.2f") -> str:
    return "—" if value is None else format(value, spec)


def write_report(summaries: list[dict], n_tickers: int, months: int, step: int) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    lines = [
        "# Monte Carlo + Para Çıkışı Vetosu Ölçümü",
        f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | **Hisse:** {n_tickers} | "
        f"**Dönem:** son {months} ay | {step} işlem gününde bir değerlendirme",
        "",
        "| Koşul | Vade | Devrede | Devre dışı | Devrede % | Ort. getiri devrede | Ort. getiri devre dışı | "
        "Fazla getiri devrede | Fazla getiri devre dışı | Negatif % devrede | Negatif % devre dışı | Devrede daha kötü hisse | p |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {CONDITION_LABELS[s['condition']]} | {s['horizon']}G | {s['fired']} | {s['quiet']} | {s['fire_rate_pct']:.1f} | "
            f"{_fmt(s['mean_fired'])} | {_fmt(s['mean_quiet'])} | {_fmt(s['mean_excess_fired'])} | {_fmt(s['mean_excess_quiet'])} | "
            f"{_fmt(s['share_negative_fired_pct'], '.1f')} | {_fmt(s['share_negative_quiet_pct'], '.1f')} | "
            f"{s['tickers_fired_worse']} / {s['tickers_compared']} | {_fmt(s['p_value'], '.3f')} |")
    lines += [
        "",
        "> Veto yalnızca AL kararını TUT'a çevirir. Değerli olması için devrede olduğu günlerin ileri getirisi belirgin biçimde daha kötü olmalı.",
        "> Değerlendirme noktaları vade boyunca örtüşür ve aynı tarihteki hisseler birlikte hareket eder; p değerleri olduğundan güçlü görünür.",
        "> Ölçüm tüm günleri kapsar; komitenin AL dediği günlerin seçimini test etmez.",
    ]
    report_path = os.path.join(REPORTS_DIR, "veto_study_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def run_study_for_universe(tickers: list[str], months: int = 24, step: int = DEFAULT_STEP,
                           data_dir: str = RAW_DATA_DIR) -> tuple[list[dict], str, int]:
    """Evreni yerel veriden ölçer; özetleri, rapor yolunu ve ölçülen hisse sayısını döndürür."""
    prices = load_prices(tickers, months, data_dir)
    if not prices:
        raise ValueError("Yerel fiyat verisi bulunamadı; önce --download-all çalıştırın.")
    index = load_prices([INDEX_TICKER], months, data_dir).get(INDEX_TICKER)
    rows = run_veto_study(prices, index, step=step)
    summaries = [summarise_condition(rows, c, h) for c in CONDITIONS for h in HORIZONS]
    return summaries, write_report(summaries, len(prices), months, step), len(prices)
