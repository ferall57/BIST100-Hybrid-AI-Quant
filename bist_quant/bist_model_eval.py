"""
Kronos modellerinin tahmin isabetini aynı hisse ve tarihlerde yan yana ölçer.

Her model aynı geçmiş pencerelerinden aynı rastgelelik tohumuyla tahmin üretir; tahmin edilen getiri
gerçekleşen getiriyle karşılaştırılır. Kıyas noktası "fiyat değişmeyecek" diyen naif tahmindir:
bir model ancak bunu geçiyorsa bilgi taşıyor demektir.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KRONOS_DIR = os.path.join(ROOT_DIR, "repos", "Kronos")
for _path in (ROOT_DIR, KRONOS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

RAW_DATA_DIR = os.path.join(ROOT_DIR, "bist_data", "raw")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "bist_kronos")
REPORTS_DIR = os.path.join(ROOT_DIR, "outputs", "reports")
TRAINING_CUTOFF_FILE = os.path.join(MODELS_DIR, "training_cutoff.json")

FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
LOOKBACK_BARS = 256
HORIZON_BARS = 15
SHORT_HORIZON_BARS = 5
ORIGIN_STEP_BARS = 20
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_P = 0.9
BASE_SEED = 1234

MODEL_BASE = "Temel Kronos"
MODEL_FINETUNED = "Yeni ince ayar"
MODEL_LEGACY = "Eski ince ayar"


# --- saf ölçü fonksiyonları ---------------------------------------------------

def forecast_metrics(predicted_pct, actual_pct) -> dict:
    """Tahmin edilen ve gerçekleşen getiri yüzdelerinden isabet ölçülerini hesaplar."""
    predicted = np.asarray(predicted_pct, dtype=float)
    actual = np.asarray(actual_pct, dtype=float)
    if predicted.shape != actual.shape or predicted.size == 0:
        raise ValueError("Tahmin ve gerçekleşen diziler aynı uzunlukta ve dolu olmalı.")

    mae = float(np.mean(np.abs(predicted - actual)))
    naive_mae = float(np.mean(np.abs(actual)))
    has_spread = np.std(predicted) > 0 and np.std(actual) > 0
    return {
        "n": int(predicted.size),
        "directional_accuracy_pct": float(np.mean((predicted > 0) == (actual > 0)) * 100.0),
        "mae_pct": mae,
        "naive_mae_pct": naive_mae,
        "skill_vs_naive_pct": float((1.0 - mae / naive_mae) * 100.0) if naive_mae > 0 else 0.0,
        "information_coefficient": float(stats.spearmanr(predicted, actual).statistic) if has_spread else 0.0,
        "mean_predicted_pct": float(np.mean(predicted)),
        "mean_actual_pct": float(np.mean(actual)),
    }


def paired_comparison(abs_errors_first, abs_errors_second) -> dict:
    """Aynı tahmin noktalarında iki modelin mutlak hatalarını eşleştirir (iki yönlü işaret testi)."""
    first = np.asarray(abs_errors_first, dtype=float)
    second = np.asarray(abs_errors_second, dtype=float)
    first_wins = int(np.sum(first < second))
    second_wins = int(np.sum(second < first))
    decided = first_wins + second_wins
    return {
        "first_closer_pct": (first_wins / decided * 100.0) if decided else None,
        "mean_abs_error_diff": float(np.mean(first - second)),
        "p_value": float(stats.binomtest(first_wins, decided, 0.5).pvalue) if decided else 1.0,
        "decided_pairs": decided,
    }


def forecast_origins(n_bars: int, horizon: int, step: int, count: int, min_index: int) -> list[int]:
    """
    Tahminin yapılacağı son-mum indeksleri (artan sırada). Her birinin ardında tam bir ufuk kadar
    gerçekleşmiş mum bulunur; min_index'ten önceki mumlar (eğitim dönemi / yetersiz geçmiş) kullanılmaz.
    """
    last_origin = n_bars - 1 - horizon
    origins = [last_origin - k * step for k in range(count)]
    return sorted(o for o in origins if o >= min_index)


# --- veri ve model ------------------------------------------------------------

def load_training_cutoff(path: str = TRAINING_CUTOFF_FILE) -> str | None:
    import json
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("train_end_date")


def build_tasks(tickers: list[str], origins_per_ticker: int, cutoff: str | None, raw_dir: str = RAW_DATA_DIR) -> list[dict]:
    """Her hisse için eğitim kesiminden sonraki tahmin noktalarını ve gerçekleşen getirileri hazırlar."""
    tasks = []
    for ticker in tickers:
        csv_path = os.path.join(raw_dir, f"{ticker}_1d.csv")
        if not os.path.exists(csv_path):
            print(f"[UYARI] {ticker}: veri dosyası yok, atlandı.")
            continue
        df = pd.read_csv(csv_path)
        days = df["timestamps"].astype(str).str[:10]
        df = df.assign(timestamps=pd.to_datetime(days)).sort_values("timestamps").reset_index(drop=True)
        first_test_index = int((days.sort_values().reset_index(drop=True) <= cutoff).sum()) if cutoff else 0
        # Tahmin edilen dönem eğitim kesiminden sonra başlamalı; girdi penceresi daha eski mumları içerebilir
        min_index = max(LOOKBACK_BARS - 1, first_test_index)

        for origin in forecast_origins(len(df), HORIZON_BARS, ORIGIN_STEP_BARS, origins_per_ticker, min_index):
            context = df.iloc[origin - LOOKBACK_BARS + 1 : origin + 1]
            future = df.iloc[origin + 1 : origin + 1 + HORIZON_BARS]
            last_close = float(context["close"].iloc[-1])
            tasks.append({
                "ticker": ticker,
                "origin_date": context["timestamps"].iloc[-1].strftime("%Y-%m-%d"),
                "x_df": context[FEATURE_COLUMNS].reset_index(drop=True),
                "x_ts": context["timestamps"].reset_index(drop=True),
                "y_ts": future["timestamps"].reset_index(drop=True),
                "last_close": last_close,
                "actual_short_pct": (float(future["close"].iloc[SHORT_HORIZON_BARS - 1]) / last_close - 1.0) * 100.0,
                "actual_pct": (float(future["close"].iloc[-1]) / last_close - 1.0) * 100.0,
            })
    return tasks


def available_models() -> dict[str, tuple[str, str]]:
    """Karşılaştırılacak modeller: ad -> (tahminci yolu, tokenizer yolu). Yerelde olmayanlar listeye girmez."""
    models = {MODEL_BASE: ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base")}
    candidates = {MODEL_FINETUNED: os.path.join(MODELS_DIR, "bist100_kronos_base")}
    legacy_dirs = sorted(d for d in os.listdir(MODELS_DIR) if "_legacy_" in d) if os.path.isdir(MODELS_DIR) else []
    if legacy_dirs:
        candidates[MODEL_LEGACY] = os.path.join(MODELS_DIR, legacy_dirs[0])
    for name, experiment_dir in candidates.items():
        predictor_dir = os.path.join(experiment_dir, "basemodel", "best_model")
        tokenizer_dir = os.path.join(experiment_dir, "tokenizer", "best_model")
        if os.path.exists(os.path.join(predictor_dir, "model.safetensors")):
            models[name] = (predictor_dir, tokenizer_dir if os.path.exists(tokenizer_dir) else models[MODEL_BASE][1])
    return models


def predict_returns(model_path: str, tokenizer_path: str, tasks: list[dict], sample_count: int, batch_size: int) -> pd.DataFrame:
    """Bir modelle tüm tahmin noktalarını üretir; her parti tüm modellerde aynı tohumla örneklenir."""
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(Kronos.from_pretrained(model_path), KronosTokenizer.from_pretrained(tokenizer_path),
                                device=device, max_context=512)
    rows = []
    try:
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            torch.manual_seed(BASE_SEED + start)
            with torch.inference_mode():
                forecasts = predictor.predict_batch(
                    df_list=[t["x_df"] for t in batch],
                    x_timestamp_list=[t["x_ts"] for t in batch],
                    y_timestamp_list=[t["y_ts"] for t in batch],
                    pred_len=HORIZON_BARS, T=SAMPLING_TEMPERATURE, top_p=SAMPLING_TOP_P,
                    sample_count=sample_count, verbose=False,
                )
            for task, forecast in zip(batch, forecasts):
                closes = forecast["close"].to_numpy(dtype=float)
                rows.append({
                    "ticker": task["ticker"],
                    "origin_date": task["origin_date"],
                    "pred_short_pct": (closes[SHORT_HORIZON_BARS - 1] / task["last_close"] - 1.0) * 100.0,
                    "pred_pct": (closes[-1] / task["last_close"] - 1.0) * 100.0,
                })
            print(f"   {min(start + batch_size, len(tasks))}/{len(tasks)} tahmin", flush=True)
    finally:
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# --- raporlama ----------------------------------------------------------------

def _metrics_row(name: str, horizon_label: str, predicted, actual) -> dict:
    return {"model": name, "horizon": horizon_label, **forecast_metrics(predicted, actual)}


def summarise(results: pd.DataFrame, model_names: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Model başına ölçü tablosu ve yeni ince ayarın diğerleriyle eşleştirilmiş karşılaştırması."""
    rows = []
    for name in model_names:
        rows.append(_metrics_row(name, f"{SHORT_HORIZON_BARS}G", results[f"{name}|short"], results["actual_short_pct"]))
        rows.append(_metrics_row(name, f"{HORIZON_BARS}G", results[f"{name}|long"], results["actual_pct"]))

    pairs = []
    if MODEL_FINETUNED in model_names:
        for other in (n for n in model_names if n != MODEL_FINETUNED):
            comparison = paired_comparison(
                (results[f"{MODEL_FINETUNED}|long"] - results["actual_pct"]).abs(),
                (results[f"{other}|long"] - results["actual_pct"]).abs(),
            )
            pairs.append({"first": MODEL_FINETUNED, "second": other, **comparison})
    return pd.DataFrame(rows), pairs


def write_report(metrics: pd.DataFrame, pairs: list[dict], results: pd.DataFrame, cutoff: str | None, sample_count: int) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    lines = [
        "# Kronos Model Karşılaştırması",
        f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | **Tahmin noktası:** {len(results)} "
        f"({results['ticker'].nunique()} hisse, {results['origin_date'].min()} – {results['origin_date'].max()})",
        f"**Eğitim kesimi:** {cutoff or 'kayıtlı değil'} | **Geçmiş pencere:** {LOOKBACK_BARS} mum | "
        f"**Örnekleme:** {sample_count} yol ortalaması, T={SAMPLING_TEMPERATURE}, top_p={SAMPLING_TOP_P}",
        "",
        "| Model | Ufuk | Yön isabeti % | Ort. mutlak hata % | Naif hata % | Naife göre beceri % | Sıra korelasyonu (IC) | Ort. tahmin % | Ort. gerçekleşen % |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.model} | {row.horizon} | {row.directional_accuracy_pct:.1f} | {row.mae_pct:.2f} | {row.naive_mae_pct:.2f} | "
                     f"{row.skill_vs_naive_pct:+.1f} | {row.information_coefficient:+.3f} | {row.mean_predicted_pct:+.2f} | {row.mean_actual_pct:+.2f} |")
    lines += ["", f"## Eşleştirilmiş karşılaştırma ({HORIZON_BARS} günlük ufuk)", ""]
    for pair in pairs:
        closer = "—" if pair["first_closer_pct"] is None else f"%{pair['first_closer_pct']:.1f}"
        lines.append(f"* **{pair['first']}** vs **{pair['second']}**: daha yakın olduğu tahmin oranı {closer}, "
                     f"ortalama mutlak hata farkı {pair['mean_abs_error_diff']:+.2f} puan, işaret testi p={pair['p_value']:.3f}")
    lines += [
        "",
        "> Naif tahmin \"fiyat değişmeyecek\" der. Naife göre beceri sıfırın altındaysa model, hiç tahmin yapmamaktan kötüdür.",
        "> Test dönemi model seçiminde kullanılan doğrulama dönemiyle örtüşür; ince ayarlı model lehine hafif iyimserlik içerir.",
    ]
    report_path = os.path.join(REPORTS_DIR, "model_comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def run_comparison(tickers: list[str], origins_per_ticker: int = 12, sample_count: int = 5, batch_size: int = 4,
                   model_names: list[str] | None = None) -> tuple[pd.DataFrame, list[dict], str]:
    """Seçilen modelleri aynı tahmin noktalarında çalıştırır, ölçüleri hesaplar ve raporu yazar."""
    cutoff = load_training_cutoff()
    tasks = build_tasks(tickers, origins_per_ticker, cutoff)
    if not tasks:
        raise ValueError("Tahmin noktası üretilemedi: veri yok veya eğitim kesiminden sonra yeterli mum yok.")

    models = available_models()
    names = [n for n in (model_names or list(models)) if n in models]
    print(f"🔬 {len(tasks)} tahmin noktası, {len(names)} model: {', '.join(names)}")

    results = pd.DataFrame([{k: t[k] for k in ("ticker", "origin_date", "actual_short_pct", "actual_pct")} for t in tasks])
    for name in names:
        print(f"▶️ {name}")
        predictions = predict_returns(*models[name], tasks, sample_count, batch_size)
        results[f"{name}|short"] = predictions["pred_short_pct"].to_numpy()
        results[f"{name}|long"] = predictions["pred_pct"].to_numpy()

    metrics, pairs = summarise(results, names)
    return metrics, pairs, write_report(metrics, pairs, results, cutoff, sample_count)
