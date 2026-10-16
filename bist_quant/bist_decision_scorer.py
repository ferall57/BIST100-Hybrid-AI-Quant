"""
Komite kararlarının sabit vadede gerçekleşen getiriyle ölçülmesi.

bist_memory.evaluate_past_decisions bir kararı bir sonraki çalıştırmadaki fiyatla
karşılaştırır; vade her kayıtta farklı olur. Bu modül her kararı aynı kuralla ölçer:

- Giriş fiyatı: komitenin kaydettiği spot fiyat (kararın verildiği fiyat).
- İlk tutma mumu: karar seans açılışından (10:00) önceyse o günün mumu,
  değilse karar gününden sonraki ilk mum.
- Çıkış: ilk tutma mumundan itibaren `horizon` işlem gününün kapanışı.
- AL isabeti: getiri > 0; SAT isabeti: getiri < 0; TUT yön iddiası taşımaz.
- Fazla getiri: aynı mumlarda XU100'e göre (SAT için işaret ters).
- İlk temas: tutma mumlarında hedef mi stop mu önce görüldü (günlük yüksek/düşük).
"""

from __future__ import annotations

import math
import os
from datetime import datetime, time

import pandas as pd

from bist_quant.bist_downloader import RAW_DATA_DIR

SESSION_OPEN = time(10, 0)
HORIZONS = (5, 10)
TARGET_FIELD = {5: "target_1w", 10: "target_15d"}
INDEX_TICKER = "XU100.IS"
WILSON_Z = 1.96

BUY_VERDICTS = ("AL", "BUY")
SELL_VERDICTS = ("SAT", "SHORT", "SELL")


def _side(verdict: str) -> str | None:
    verdict = (verdict or "").upper()
    if any(word in verdict for word in SELL_VERDICTS):
        return "SHORT"
    if any(word in verdict for word in BUY_VERDICTS):
        return "LONG"
    return None


def _parse_date(date_str: str) -> datetime:
    fmt = "%Y-%m-%d %H:%M:%S" if len(date_str) > 10 else "%Y-%m-%d"
    return datetime.strptime(date_str, fmt)


def first_holding_bar(prices: pd.DataFrame, decision_date: str) -> int | None:
    """Kararın ardından pozisyonun ilk taşındığı mumun konumu; yoksa None."""
    decided = _parse_date(decision_date)
    bar_days = pd.to_datetime(prices["timestamps"]).dt.normalize()
    day = pd.Timestamp(decided.date())
    eligible = bar_days >= day if decided.time() < SESSION_OPEN else bar_days > day
    positions = eligible.to_numpy().nonzero()[0]
    return int(positions[0]) if len(positions) else None


def level_problem(record: dict, horizon: int = 5) -> str | None:
    """Hedef/stop seviyeleri veya güven skoru tutarsızsa açıklaması; değilse None."""
    confidence = record.get("confidence")
    if confidence is not None and not 0 <= confidence <= 100:
        return f"güven skoru 0-100 dışında ({confidence})"

    side = _side(record.get("verdict"))
    if side is None:
        return None
    spot = record.get("spot_price")
    target = record.get(TARGET_FIELD.get(horizon, "target_1w"))
    stop = record.get("stop_loss")
    if spot is None or target is None or stop is None:
        return "hedef veya stop kaydı eksik"
    if side == "LONG" and not stop < spot < target:
        return f"AL için stop < spot < hedef değil (stop {stop}, spot {spot}, hedef {target})"
    if side == "SHORT" and not target < spot < stop:
        return f"SAT için hedef < spot < stop değil (hedef {target}, spot {spot}, stop {stop})"
    return None


def _first_touch(window: pd.DataFrame, side: str, target: float, stop: float) -> str:
    for _, bar in window.iterrows():
        if side == "LONG":
            hit_target, hit_stop = bar["high"] >= target, bar["low"] <= stop
        else:
            hit_target, hit_stop = bar["low"] <= target, bar["high"] >= stop
        if hit_target and hit_stop:
            return "AMBIGUOUS"
        if hit_stop:
            return "STOP"
        if hit_target:
            return "TARGET"
    return "NEITHER"


def _index_return(index_prices: pd.DataFrame, entry_day, exit_day) -> float | None:
    days = pd.to_datetime(index_prices["timestamps"]).dt.normalize()
    before = index_prices[days < entry_day]
    at_exit = index_prices[days == exit_day]
    if before.empty or at_exit.empty:
        return None
    start, end = before["close"].iloc[-1], at_exit["close"].iloc[0]
    return (end / start - 1.0) * 100.0


def score_decision(record: dict, prices: pd.DataFrame, index_prices: pd.DataFrame | None,
                   horizon: int) -> dict:
    """Tek bir komite kararını `horizon` işlem günü vadesinde ölçer."""
    side = _side(record.get("verdict"))
    problem = level_problem(record, horizon)
    base = {
        "record_id": record.get("record_id"), "ticker": record.get("ticker"),
        "date": record.get("date"), "verdict": record.get("verdict"),
        "horizon": horizon, "spot_price": record.get("spot_price"),
        "level_problem": problem, "status": "PENDING", "exit_date": None,
        "return_pct": None, "index_return_pct": None, "excess_pct": None,
        "hit": None, "first_touch": None,
    }

    start = first_holding_bar(prices, record["date"])
    if start is None or start + horizon - 1 >= len(prices):
        return base

    window = prices.iloc[start:start + horizon]
    exit_bar = window.iloc[-1]
    spot = record["spot_price"]
    ret = (exit_bar["close"] / spot - 1.0) * 100.0
    exit_day = pd.Timestamp(exit_bar["timestamps"]).normalize()
    entry_day = pd.Timestamp(window.iloc[0]["timestamps"]).normalize()

    index_ret = None if index_prices is None else _index_return(index_prices, entry_day, exit_day)
    excess = None
    if side is not None and index_ret is not None:
        excess = ret - index_ret if side == "LONG" else index_ret - ret

    first_touch = None
    if side is not None and problem is None:
        first_touch = _first_touch(window, side, record[TARGET_FIELD[horizon]], record["stop_loss"])

    hit = None
    if side == "LONG":
        hit = bool(ret > 0)
    elif side == "SHORT":
        hit = bool(ret < 0)

    return {
        **base, "status": "SCORED", "exit_date": exit_day.strftime("%Y-%m-%d"),
        "return_pct": ret, "index_return_pct": index_ret, "excess_pct": excess,
        "hit": hit, "first_touch": first_touch,
    }


def wilson_interval(hits: int, n: int, z: float = WILSON_Z) -> tuple[float | None, float | None]:
    """İsabet oranı için %95 Wilson güven aralığı."""
    if n == 0:
        return (None, None)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize_scores(scores: list[dict]) -> dict:
    """Karar türüne göre isabet, ortalama getiri ve fazla getiri özeti."""
    summary = {}
    for verdict in sorted({s["verdict"] for s in scores}):
        group = [s for s in scores if s["verdict"] == verdict]
        scored = [s for s in group if s["status"] == "SCORED"]
        directional = [s for s in scored if s["hit"] is not None]
        hits = sum(1 for s in directional if s["hit"]) if directional else None
        low, high = wilson_interval(hits or 0, len(directional))
        summary[verdict] = {
            "n": len(scored),
            "pending": len(group) - len(scored),
            "tickers": len({s["ticker"] for s in scored}),
            "hits": hits,
            "hit_rate_ci": (low, high),
            "mean_return_pct": _mean([s["return_pct"] for s in scored]),
            "mean_excess_pct": _mean([s["excess_pct"] for s in scored]),
        }
    return summary


def load_daily_prices(ticker: str, data_dir: str = RAW_DATA_DIR) -> pd.DataFrame | None:
    """Yerel günlük CSV'yi okur; dosya yoksa None."""
    full = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
    path = os.path.join(data_dir, f"{full}_1d.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.sort_values("timestamps").reset_index(drop=True)


def score_memory(history: list[dict], horizons=HORIZONS, data_dir: str = RAW_DATA_DIR) -> list[dict]:
    """Hafızadaki her kararı her vadede ölçer; fiyat verisi olmayanları NO_DATA işaretler."""
    index_prices = load_daily_prices(INDEX_TICKER, data_dir)
    cache: dict[str, pd.DataFrame | None] = {}
    scores = []
    for record in history:
        ticker = record["ticker"]
        if ticker not in cache:
            cache[ticker] = load_daily_prices(ticker, data_dir)
        prices = cache[ticker]
        for horizon in horizons:
            if prices is None:
                scores.append({"record_id": record.get("record_id"), "ticker": ticker,
                               "verdict": record.get("verdict"), "horizon": horizon,
                               "status": "NO_DATA", "hit": None, "return_pct": None,
                               "excess_pct": None, "level_problem": None})
                continue
            scores.append(score_decision(record, prices, index_prices, horizon))
    return scores
