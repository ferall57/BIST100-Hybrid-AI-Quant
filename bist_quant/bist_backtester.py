import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.bist_downloader import download_ticker_data, RAW_DATA_DIR
from bist_quant.bist_viop import BistViopEngine
from bist_quant.market_assumptions import risk_free_rate
from bist_quant.entry_filters import FILTER_INDEX_GATE, INDEX_TICKER, EntryFilters
from bist_quant.backtest_engine import (
    LONG,
    SHORT,
    CostModel,
    ExitRules,
    daily_rate,
    gross_return_pct,
    idle_cash_fraction,
    leakage_status,
    net_return_pct,
    performance_from_equity,
    simulate_exit,
)

try:
    from bist_quant.bist_kronos_quant import BistKronosQuant
    KRONOS_AVAILABLE = True
except Exception:
    KRONOS_AVAILABLE = False

CHARTS_DIR = os.path.join(ROOT_DIR, "outputs", "charts")
REPORTS_DIR = os.path.join(ROOT_DIR, "outputs", "reports")
TRAINING_CUTOFF_FILE = os.path.join(ROOT_DIR, "models", "bist_kronos", "training_cutoff.json")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Maliyet varsayımları (bacak başına oran). Gerçek aracı kurum tarifesine göre CLI'dan değiştirilebilir.
SPOT_COSTS = CostModel(commission_rate=0.0010, slippage_rate=0.0005)
VIOP_COSTS = CostModel(commission_rate=0.0004, slippage_rate=0.0005)

TRADING_DAYS_PER_MONTH = 21
MAX_HOLD_DAYS = 60
MIN_HISTORY_BARS = 100
ROLLOVER_FRICTION_RATE = 0.0015  # vade taşıma başına sürtünme (tahsis edilen sermayenin oranı)
VIOP_EXPIRY_MONTHS = (2, 4, 6, 8, 10, 12)
ROLLOVER_FROM_DAY = 25
UNIVERSE_COLUMNS = ("total_return_pct", "bnh_return_pct", "alpha", "win_rate", "total_trades",
                    "max_drawdown", "sharpe_ratio", "total_costs_try", "blocked_entries")


def load_training_cutoff(path: str = TRAINING_CUTOFF_FILE) -> str | None:
    """İnce ayarlı modelin eğitim kesim tarihini okur; kayıt yoksa None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("train_end_date")
    except Exception as e:
        print(f"[UYARI] Eğitim kesim tarihi okunamadı ({path}): {e}")
        return None


class BistBacktester:
    """
    Borsa İstanbul (BIST) hisseleri için Walk-Forward Rolling Window Backtesting Motoru.
    Her adımda yalnızca o güne kadarki mumlarla tahmin üretir, komisyon ve kayma maliyetlerini düşer,
    stop/hedef dolumlarını boşluklu açılışlara göre hesaplar ve günlük sermaye eğrisinden metrik çıkarır.
    """
    def __init__(self, use_kronos: bool = True, costs: CostModel = None):
        self.use_kronos = use_kronos
        self.costs = costs
        self.quant_engine = None
        if self.use_kronos and KRONOS_AVAILABLE:
            try:
                self.quant_engine = BistKronosQuant(use_base_model=True)
            except Exception as e:
                print(f"[UYARI] Kronos modeli yüklenemedi, momentum bazlı kuant tahminciye geçiliyor: {e}")
                self.quant_engine = None

    def _check_market_regime(self, hist_df: pd.DataFrame) -> tuple[bool, str]:
        """
        Hissenin Boğa (Yükseliş) veya Ayı (Düşüş) rejiminde olduğunu tespit eder.
        Eğer hisse sert düşüş trendindeyse (EMA20 < SMA50 veya Price < SMA50 ve negatif eğim),
        Long işlem açılması engellenir ve %100 nakitte beklenir.
        """
        if len(hist_df) < 50:
            return True, "Nötr Rejim"

        close = hist_df["close"]
        current_close = float(close.iloc[-1])
        sma_50 = float(close.tail(50).mean())
        ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])

        # 20 günlük SMA eğimi (slope)
        sma_20_prev = float(close.iloc[-25:-5].mean())
        sma_20_curr = float(close.tail(20).mean())
        slope_20 = (sma_20_curr - sma_20_prev) / (sma_20_prev + 1e-6)

        # Ayı Rejimi Koşulları:
        is_bear = (current_close < sma_50 and ema_20 < sma_50) or (current_close < ema_20 and slope_20 < -0.02)

        if is_bear:
            return False, "🐻 Ayı Rejimi (Düşüş Trendi - Nakitte Kal)"
        return True, "🐂 Boğa / Toparlanma Rejimi (İşleme Açık)"

    def _calculate_dynamic_sl_tp(self, hist_df: pd.DataFrame, default_sl: float = 3.5, default_tp: float = 8.0) -> tuple[float, float]:
        """
        Hissenin 14 günlük ATR (Average True Range) ve dalga boyuna göre
        volatiliteye duyarlı asimetrik Stop-Loss ve Take-Profit oranlarını hesaplar.
        """
        if len(hist_df) < 20:
            return default_sl, default_tp

        tail_df = hist_df.tail(15)
        high = tail_df["high"]
        low = tail_df["low"]
        close = tail_df["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).dropna()

        atr = float(tr.mean())
        current_close = float(close.iloc[-1])
        atr_pct = (atr / current_close) * 100.0 if current_close > 0 else default_sl

        # Dinamik Stop-Loss: 1.8 * ATR% (Minimum %3.5, Maksimum %7.0)
        dynamic_sl = max(3.5, min(7.0, atr_pct * 1.8))
        # Dinamik Take-Profit: En az 2.2 katı (Asimetrik 1:2.2 Risk/Ödül)
        dynamic_tp = max(default_tp, dynamic_sl * 2.2)

        return float(dynamic_sl), float(dynamic_tp)

    def _accrue_flat_days(self, equity: dict, dates, capital: float, rate: float, earns_interest: bool) -> float:
        """Pozisyon yokken geçen günleri sermaye eğrisine işler; VİOP modunda nakit nemalanır."""
        for date in dates:
            if earns_interest:
                capital *= (1.0 + rate)
            equity[date] = capital
        return capital

    def run_walk_forward_backtest(
        self,
        ticker: str,
        months: int = 6,
        horizon_days: int = 15,
        step_days: int = 10,
        stop_loss_pct: float = 3.5,
        take_profit_pct: float = 8.0,
        initial_capital: float = 100000.0,
        allocation_pct: float = 100.0,
        enable_regime_filter: bool = True,
        use_trailing_stop: bool = True,
        use_viop: bool = False,
        leverage: float = 1.5,
        entry_filters: tuple = (),
        refresh_data: bool = True,
        write_artifacts: bool = True
    ):
        """
        Geçmiş N aylık veri üzerinde Walk-Forward simülasyonu çalıştırır.
        use_viop=True: Çift yönlü (Long & Short) VİOP türev motorunu çalıştırır.
        use_trailing_stop=True: İz süren stop kullanır. False ise sabit Take-Profit hedefi devreye girer.
        entry_filters: LONG girişe ek onay şartları ("ict", "money_flow", "index_gate"); katkı ölçümü için.
        refresh_data=False: veriyi yeniden indirmez, diskteki dosyayı kullanır.
        write_artifacts=False: grafik ve rapor dosyası üretmez.
        """
        if not ticker.endswith(".IS"):
            ticker += ".IS"

        costs = self.costs or (VIOP_COSTS if use_viop else SPOT_COSTS)
        rf_annual = risk_free_rate()

        print(f"\n" + "="*85)
        print(f"📊 KRONOS WALK-FORWARD BACKTEST MOTORU BAŞLATILDI")
        print(f"🎯 Hedef Hisse    : {ticker}")
        print(f"⏳ Test Periyodu  : Son {months} Ay (Adım: Her {step_days} İşlem Gününde Bir)")
        print(f"🔮 Tahmin Ufku    : {horizon_days} İşlem Günü")
        print(f"🛡️ Risk Yönetimi  : {'İz Süren Stop (Trailing Stop - Kârı Koştur)' if use_trailing_stop else f'Sabit TP: %{take_profit_pct:.1f}'} | Rejim Filtresi: {'Aktif' if enable_regime_filter else 'Pasif'}")
        print(f"💸 Maliyet        : Komisyon %{costs.commission_rate*100:.3f} + Kayma %{costs.slippage_rate*100:.3f} (bacak başına)")
        if use_viop:
            print(f"⚡ VİOP Modu      : Aktif (Çift Yönlü Long & Short | Kaldıraç: {leverage}x | Nemalandırma Varsayımı: %{rf_annual * 100:.1f})")
        print(f"💰 Başlangıç Kasa : {initial_capital:,.2f} TRY")
        print("="*85 + "\n")

        # 1. Canlı veriyi güncelle ve oku
        if refresh_data:
            download_ticker_data(ticker, period="5y", interval="1d", save_dir=RAW_DATA_DIR)
        csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}_1d.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{ticker} veri seti bulunamadı: {csv_path}")

        df = pd.read_csv(csv_path)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.sort_values("timestamps").reset_index(drop=True)

        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]

        # Test aralığını belirle
        total_test_days = int(months * TRADING_DAYS_PER_MONTH)
        if len(df) < total_test_days + MIN_HISTORY_BARS:
            total_test_days = max(30, len(df) - MIN_HISTORY_BARS)
            print(f"[BİLGİ] Veri seti uzunluğuna göre test periyodu {total_test_days} işlem gününe uyarlandı.")

        test_start_idx = len(df) - total_test_days
        test_df = df.iloc[test_start_idx:].copy().reset_index(drop=True)
        dates = df["timestamps"]

        leak_code, leak_note = leakage_status(
            uses_model=self.quant_engine is not None,
            training_cutoff=load_training_cutoff(),
            test_start=str(dates.iloc[test_start_idx].date()),
        )
        print(f"🔎 Veri Sızıntısı Denetimi: [{leak_code}] {leak_note}\n")

        clean_ticker = ticker.replace(".IS", "").upper()
        margin_ratio = BistViopEngine.SPAN_MARGIN_RATIOS.get(clean_ticker, BistViopEngine.SPAN_MARGIN_RATIOS["DEFAULT"])
        rate = daily_rate(rf_annual)

        filters = None
        if entry_filters:
            index_history = self._load_index_history(refresh_data) if FILTER_INDEX_GATE in entry_filters else None
            filters = EntryFilters(entry_filters, index_history=index_history)
        blocked_entries = {name: 0 for name in entry_filters}

        trades = []
        total_costs = 0.0
        current_capital = initial_capital
        equity = {dates.iloc[test_start_idx]: initial_capital}

        # 2. Walk-Forward Döngüsü (Pencere Kaydırma)
        current_idx = test_start_idx
        last_entry_idx = len(df) - horizon_days
        pbar = tqdm(total=max(0, last_entry_idx - test_start_idx), desc=f"{ticker} Backtest")

        while current_idx < last_entry_idx and current_capital > 0:
            # Sadece current_idx öncesindeki geçmiş veriyi gör
            hist_df = df.iloc[:current_idx].copy()
            entry_row = df.iloc[current_idx]
            entry_price = float(entry_row["close"])

            is_bull, regime_str = self._check_market_regime(hist_df)
            active_sl, active_tp = self._calculate_dynamic_sl_tp(hist_df, stop_loss_pct, take_profit_pct)
            expected_return = self._predict_return(hist_df, horizon_days=horizon_days)

            min_thresh = 0.8 if self.quant_engine is not None else 1.2
            should_enter_long = expected_return > min_thresh and (is_bull or not enable_regime_filter)
            should_enter_short = use_viop and (expected_return < -min_thresh or not is_bull)

            # Ek giriş filtreleri yalnızca giriş gününden önceki mumları görür
            if should_enter_long and filters is not None:
                blocker = filters.blocking_filter(hist_df, str(entry_row["timestamps"].date()))
                if blocker is not None:
                    blocked_entries[blocker] += 1
                    should_enter_long = False

            if should_enter_long or should_enter_short:
                direction = LONG if should_enter_long else SHORT
                eff_leverage = leverage if use_viop else 1.0
                trade_allocated = current_capital * (allocation_pct / 100.0)
                cash_reserve = current_capital - trade_allocated

                max_hold_days = min(MAX_HOLD_DAYS, len(df) - current_idx - 1)
                forward_window = df.iloc[current_idx + 1 : current_idx + 1 + max_hold_days].reset_index(drop=True)
                rules = ExitRules(
                    initial_stop_pct=active_sl,
                    trail_distance_pct=max(4.5, min(8.0, active_sl * 1.1)),
                    horizon_days=horizon_days,
                    use_trailing_stop=use_trailing_stop,
                    take_profit_pct=active_tp,
                )
                trade_exit = simulate_exit(forward_window, entry_price, direction, rules)
                held = forward_window.iloc[:trade_exit.day]

                # Pozisyondayken yalnızca teminata bağlanmayan nakit nemalanır (çift sayım yok)
                interest_base = cash_reserve + trade_allocated * idle_cash_fraction(use_viop, eff_leverage, margin_ratio)
                accumulated_interest = 0.0
                rolled_expiries = set()
                for day_i, bar in enumerate(held.itertuples(index=False), 1):
                    if use_viop:
                        accumulated_interest += interest_base * rate
                        expiry = (bar.timestamps.year, bar.timestamps.month)
                        if bar.timestamps.month in VIOP_EXPIRY_MONTHS and bar.timestamps.day >= ROLLOVER_FROM_DAY and expiry not in rolled_expiries:
                            rolled_expiries.add(expiry)
                            accumulated_interest -= trade_allocated * ROLLOVER_FRICTION_RATE
                    if day_i < trade_exit.day:
                        open_pnl = trade_allocated * gross_return_pct(entry_price, float(bar.close), direction, eff_leverage) / 100.0
                        equity[bar.timestamps] = current_capital + open_pnl + accumulated_interest

                trade_return_pct = net_return_pct(entry_price, trade_exit.price, direction, eff_leverage, costs)
                cost_try = trade_allocated * (gross_return_pct(entry_price, trade_exit.price, direction, eff_leverage) - trade_return_pct) / 100.0
                realized_pnl = trade_allocated * (trade_return_pct / 100.0) + accumulated_interest
                total_costs += cost_try
                current_capital = max(0.0, current_capital + realized_pnl)
                exit_date = held["timestamps"].iloc[-1]
                equity[exit_date] = current_capital

                trades.append({
                    "entry_date": entry_row["timestamps"].strftime("%Y-%m-%d"),
                    "entry_price": entry_price,
                    "exit_date": exit_date.strftime("%Y-%m-%d"),
                    "exit_price": trade_exit.price,
                    "direction": direction,
                    "return_pct": trade_return_pct,
                    "pnl": realized_pnl,
                    "cost_try": cost_try,
                    "capital": current_capital,
                    "duration_days": trade_exit.day,
                    "reason": trade_exit.reason,
                    "expected_return": expected_return,
                    "regime": regime_str,
                    "sl_used": active_sl,
                    "tp_used": active_tp,
                    "win": realized_pnl > 0
                })
                step = max(1, trade_exit.day)
            else:
                # Sinyal yok: nakitte bekle
                flat_dates = dates.iloc[current_idx + 1 : min(current_idx + step_days, len(df) - 1) + 1]
                current_capital = self._accrue_flat_days(equity, flat_dates, current_capital, rate, earns_interest=use_viop)
                step = step_days

            current_idx += step
            pbar.update(step)

        pbar.close()
        if current_capital <= 0:
            print("[UYARI] Sermaye sıfırlandı; backtest erken sonlandırıldı.")

        # Son işlenen günden test sonuna kadar nakitte kalınan günler
        remaining_dates = dates[dates > max(equity)]
        current_capital = self._accrue_flat_days(equity, remaining_dates, current_capital, rate, earns_interest=use_viop and current_capital > 0)
        equity_curve = pd.Series(equity).sort_index()

        # 3. Metrikleri Hesapla
        metrics = {
            **self._calculate_performance_metrics(trades, equity_curve, test_df, initial_capital, total_costs, costs, rf_annual),
            "leakage_status": leak_code,
            "leakage_note": leak_note,
            "blocked_entries": blocked_entries,
        }
        if not write_artifacts:
            return metrics, None, None

        # 4. Grafiği Çiz ve Kaydet
        chart_path = self._plot_equity_curve(ticker, test_df, equity_curve, initial_capital)

        # 5. Markdown Raporunu Oluştur ve Kaydet
        report_path = self._generate_markdown_report(ticker, metrics, trades, chart_path, months, stop_loss_pct, take_profit_pct, use_trailing_stop, costs)

        return metrics, report_path, chart_path

    def _load_index_history(self, refresh_data: bool) -> pd.DataFrame:
        """Endeks kapısı filtresi için XU100 geçmişini okur."""
        if refresh_data:
            download_ticker_data(INDEX_TICKER, period="5y", interval="1d", save_dir=RAW_DATA_DIR)
        csv_path = os.path.join(RAW_DATA_DIR, f"{INDEX_TICKER}_1d.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{INDEX_TICKER} veri seti bulunamadı: {csv_path}")
        return pd.read_csv(csv_path)

    def run_universe_backtest(self, tickers: list[str], **backtest_kwargs) -> tuple[list[dict], dict]:
        """
        Aynı kuralları bir hisse evreninin tamamında koşar. Tek hissede iyi görünen bir sonucun
        şans olup olmadığını görmek için medyan alfa ve Al-Tut'u geçen hisse oranını özetler.
        """
        rows, failed = [], []
        for ticker in tickers:
            try:
                metrics, report_path, _ = self.run_walk_forward_backtest(ticker, **backtest_kwargs)
            except Exception as e:
                print(f"[UYARI] {ticker} backtest edilemedi: {e}")
                failed.append(ticker)
                continue
            rows.append({"ticker": ticker, "report_file": report_path, **{k: metrics[k] for k in UNIVERSE_COLUMNS}})

        def _median(key):
            return float(np.median([r[key] for r in rows])) if rows else None

        summary = {
            "tickers_tested": len(rows),
            "failed_tickers": failed,
            "median_total_return_pct": _median("total_return_pct"),
            "median_bnh_return_pct": _median("bnh_return_pct"),
            "median_alpha": _median("alpha"),
            "median_sharpe": _median("sharpe_ratio"),
            "share_beating_buy_and_hold_pct": (sum(1 for r in rows if r["alpha"] > 0) / len(rows) * 100.0) if rows else None,
            "total_trades": sum(r["total_trades"] for r in rows),
        }
        return rows, summary

    def _predict_return(self, hist_df: pd.DataFrame, horizon_days: int = 15) -> float:
        """Lookahead bias olmadan geçmiş veriden beklenen getiri tahmin eder."""
        if len(hist_df) < 50:
            return 0.0

        current_close = float(hist_df["close"].iloc[-1])

        # Eğer Kronos modeli hazırsa derin öğrenme tahmini üret
        if self.quant_engine and hasattr(self.quant_engine, "predictor") and self.quant_engine.predictor is not None:
            try:
                lookback = min(128, len(hist_df))
                sub_df = hist_df.iloc[-lookback:].copy().reset_index(drop=True)
                x_df = sub_df[["open", "high", "low", "close", "volume", "amount"]]
                x_ts = pd.to_datetime(sub_df["timestamps"])

                last_date = x_ts.iloc[-1]
                y_timestamps = []
                cur_d = last_date
                while len(y_timestamps) < horizon_days:
                    cur_d += pd.Timedelta(days=1)
                    if cur_d.weekday() < 5:
                        y_timestamps.append(cur_d)

                pred_df = self.quant_engine.predictor.predict(
                    df=x_df,
                    x_timestamp=x_ts,
                    y_timestamp=pd.Series(y_timestamps),
                    pred_len=horizon_days,
                    T=0.8,
                    top_p=0.9,
                    sample_count=1
                )
                pred_close = float(pred_df["close"].iloc[-1])
                return ((pred_close - current_close) / current_close) * 100.0
            except Exception as e:
                tqdm.write(f"[UYARI] Kronos tahmini üretilemedi, momentum tahminine düşüldü: {e}")

        # Yedek / Hızlı Trend & Momentum Tahmini (EMA 9/21, SMA 50, RSI 14)
        ema_9 = float(hist_df["close"].ewm(span=9, adjust=False).mean().iloc[-1])
        ema_21 = float(hist_df["close"].ewm(span=21, adjust=False).mean().iloc[-1])
        sma_50 = float(hist_df["close"].tail(50).mean())

        # 14 günlük RSI
        deltas = hist_df["close"].diff().tail(14)
        gains = deltas.clip(lower=0).mean()
        losses = -deltas.clip(upper=0).mean()
        rs = (gains / losses) if losses > 0 else 1.0
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Log getiri momentumu
        log_rets = np.log(hist_df["close"] / hist_df["close"].shift(1)).tail(15)
        drift = float(log_rets.mean()) * horizon_days * 100.0

        # Trend Çarpanı: Fiyat 50 günlük ortalamanın veya EMA21'in altındaysa alım eşiğini zorlaştır
        trend_mult = 1.0
        if current_close < sma_50:
            trend_mult *= 0.4
        if ema_9 < ema_21:
            trend_mult *= 0.5
        if rsi < 42:
            trend_mult *= 0.5
        elif rsi > 55:
            trend_mult *= 1.3

        tech_score = (((current_close - ema_9) / ema_9) * 35.0) + (((ema_9 - ema_21) / ema_21) * 65.0)
        return ((drift * 0.5) + (tech_score * 0.5)) * trend_mult

    def _calculate_performance_metrics(
        self,
        trades: list,
        equity_curve: pd.Series,
        test_df: pd.DataFrame,
        initial_capital: float,
        total_costs: float,
        costs: CostModel,
        rf_annual: float,
    ) -> dict:
        """Günlük sermaye eğrisinden risk/getiri, işlem listesinden isabet ölçülerini hesaplar."""
        perf = performance_from_equity(equity_curve, rf_annual)
        final_capital = float(equity_curve.iloc[-1])

        # Kıyaslama: test başında al, sonunda sat (aynı maliyet varsayımıyla)
        bnh_return_pct = net_return_pct(float(test_df["close"].iloc[0]), float(test_df["close"].iloc[-1]), LONG, 1.0, costs)

        pnls = np.array([t["pnl"] for t in trades], dtype=float)
        returns = np.array([t["return_pct"] for t in trades], dtype=float)
        wins = returns[pnls > 0]
        losses = returns[pnls <= 0]
        sum_gains = float(pnls[pnls > 0].sum()) if len(pnls) else 0.0
        sum_losses = float(-pnls[pnls <= 0].sum()) if len(pnls) else 0.0
        avg_gain = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0

        return {
            "total_trades": len(trades),
            "winning_trades": int(len(wins)),
            "losing_trades": int(len(losses)),
            "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
            "total_return_pct": perf["total_return_pct"],
            "bnh_return_pct": bnh_return_pct,
            "alpha": perf["total_return_pct"] - bnh_return_pct,
            "sharpe_ratio": perf["sharpe_ratio"],
            "sortino_ratio": perf["sortino_ratio"],
            "max_drawdown": perf["max_drawdown_pct"],
            "profit_factor": (sum_gains / sum_losses) if sum_losses > 0 else 0.0,
            "payoff_ratio": (avg_gain / avg_loss) if avg_loss > 0 else 0.0,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "avg_trade_return": float(np.mean(returns)) if len(returns) else 0.0,
            "avg_holding_days": float(np.mean([t.get("duration_days", 0) for t in trades])) if trades else 0.0,
            "total_costs_try": float(total_costs),
            "risk_free_rate_pct": rf_annual * 100.0,
            "initial_capital": initial_capital,
            "final_capital": final_capital
        }

    def _plot_equity_curve(self, ticker: str, test_df: pd.DataFrame, equity_curve: pd.Series, initial_capital: float) -> str:
        """Dark-mode günlük sermaye eğrisi ve Drawdown grafiği oluşturur."""
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=False)

        # 1. Sermaye Eğrisi
        ax1.plot(equity_curve.index, equity_curve.values, label="KRONOS Quant Stratejisi (TRY)", color="#00FFCC", linewidth=2.0)

        # Benchmark Kıyaslaması (Buy & Hold Kasa Karşılığı)
        bnh_init = float(test_df["close"].iloc[0])
        bnh_capitals = [initial_capital * (float(p) / bnh_init) for p in test_df["close"]]
        ax1.plot(test_df["timestamps"], bnh_capitals, label=f"Al ve Tut (Buy & Hold: {ticker})", color="#888888", linestyle="--", linewidth=1.5, alpha=0.7)

        ax1.set_title(f"KRONOS Walk-Forward Backtest Sermaye Eğrisi (Equity Curve) -> {ticker}", fontsize=14, fontweight="bold", color="white")
        ax1.set_ylabel("Portfoy Degeri (TRY)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.3)
        ax1.legend(loc="upper left", framealpha=0.3)

        # 2. Drawdown Grafiği
        caps_arr = equity_curve.to_numpy(dtype=float)
        peaks = np.maximum.accumulate(caps_arr)
        dd = (caps_arr - peaks) / peaks * 100.0

        ax2.fill_between(equity_curve.index, dd, 0, color="#FF3366", alpha=0.4, label="Kasa Çekilmesi (Drawdown %)")
        ax2.plot(equity_curve.index, dd, color="#FF3366", linewidth=1.5)
        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.set_xlabel("Tarih", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.3)
        ax2.legend(loc="lower left", framealpha=0.3)

        plt.xticks(rotation=30)
        plt.tight_layout()

        chart_file = os.path.join(CHARTS_DIR, f"{ticker.replace('.', '_')}_backtest_equity_curve.png")
        plt.savefig(chart_file, dpi=150)
        plt.close()
        print(f"📊 Backtest Sermaye Eğrisi Kaydedildi: {chart_file}")
        return chart_file

    def _generate_markdown_report(self, ticker: str, m: dict, trades: list, chart_path: str, months: int, sl: float, tp: float,
                                  use_trailing_stop: bool, costs: CostModel) -> str:
        """Tüm işlem detaylarını, maliyet varsayımlarını ve veri sızıntısı durumunu içeren rapor üretir."""
        trade_rows = ""
        for idx, t in enumerate(trades, 1):
            badge = "🟢 KÂR" if t["win"] else "🔴 ZARAR"
            direction_badge = "🟢 LONG" if t.get("direction", LONG) == LONG else "🔻 SHORT"
            trade_rows += f"| #{idx} | {direction_badge} | {t['entry_date']} | {t['entry_price']:.2f} TRY | {t['exit_date']} | {t['exit_price']:.2f} TRY | **%{t['return_pct']:+.2f}** | {t['pnl']:+,.2f} TRY | {t['cost_try']:,.2f} TRY | {t['capital']:,.2f} TRY | {t['duration_days']}G | {badge} ({t['reason']}) |\n"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_file = os.path.join(REPORTS_DIR, f"{ticker.replace('.', '_')}_backtest_report.md")
        exit_mode = "İz Süren Stop" if use_trailing_stop else f"Sabit Take-Profit: en az %{tp:.1f}"

        report_md = f"""# 🏛️ KRONOS WALK-FORWARD BACKTEST & PERFORMANS RAPORU
**Tarih:** {now_str} | **Hisse:** {ticker} | **Test Periyodu:** Son {months} Ay
**Risk Kuralları:** Stop-Loss: en az %{sl:.1f} | Çıkış: {exit_mode} | **Başlangıç Kasa:** {m['initial_capital']:,.2f} TRY
**Maliyet Varsayımı:** Komisyon %{costs.commission_rate*100:.3f} + Kayma %{costs.slippage_rate*100:.3f} (bacak başına) | **Toplam Maliyet:** {m['total_costs_try']:,.2f} TRY
**Veri Sızıntısı Denetimi:** [{m['leakage_status']}] {m['leakage_note']}

---

## 🏆 1. YÖNETİCİ PERFORMANS VE RİSK ÖZETİ

| Performans Metriği | KRONOS Stratejisi | Kıyaslama (Buy & Hold) | Açıklama |
| :--- | :--- | :--- | :--- |
| **Kümülatif Toplam Getiri (Net)** | **%{m['total_return_pct']:+.2f}** | %{m['bnh_return_pct']:+.2f} | **Fark (Alfa): %{m['alpha']:+.2f}** |
| **Nihai Kasa Değeri** | **{m['final_capital']:,.2f} TRY** | {(m['initial_capital'] * (1.0 + m['bnh_return_pct']/100.0)):,.2f} TRY | Net Kâr: {(m['final_capital'] - m['initial_capital']):+,.2f} TRY |
| **Kazanma Oranı (Win Rate)** | **%{m['win_rate']:.1f}** ({m['winning_trades']}/{m['total_trades']}) | N/A | Maliyet sonrası kârlı işlem oranı |
| **Sharpe Oranı (Yıllık)** | **{m['sharpe_ratio']:.2f}** | N/A | Günlük getiriden, %{m['risk_free_rate_pct']:.1f} risksiz faiz düşülerek |
| **Sortino Oranı** | **{m['sortino_ratio']:.2f}** | N/A | Sadece aşağı yönlü riske göre getiri verimi |
| **Kâr Faktörü (Profit Factor)** | **{m['profit_factor']:.2f}x** | N/A | Toplam Kâr / Toplam Zarar |
| **Payoff Oranı (R:R)** | **{m['payoff_ratio']:.2f}x** | N/A | Ortalama Kâr (%{m['avg_gain']:.2f}) / Ortalama Zarar (%{m['avg_loss']:.2f}) |
| **Maksimum Çekilme (MDD)** | **-%{m['max_drawdown']:.2f}** | N/A | Günlük sermaye eğrisinde zirveden en derin düşüş |
| **Toplam İşlem Sayısı** | **{m['total_trades']} Adet** | 1 Pozisyon | Ort. süre: {m['avg_holding_days']:.1f} gün |

> Tek hisse ve tek dönem sonucu istatistiksel kanıt değildir. Kuralların genel geçerliliği için evren taramasını
> (`--backtest-universe`) kullanın. Temettüler hesaba katılmamıştır.

---

## 📈 2. SERMAYE EĞRİSİ VE ÇEKİLME GRAFİĞİ
*(Görsel Grafik Dosyası: `{chart_path}`)*

---

## 📋 3. İŞLEM GÜNLÜĞÜ (TRADE LOG)

| İşlem | Yön | Giriş Tarihi | Giriş Fiyatı | Çıkış Tarihi | Çıkış Fiyatı | Net Getiri (%) | Net Kâr/Zarar | Maliyet | Bakiye | Süre | Sonuç / Kapanış Nedeni |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{trade_rows if trade_rows else "| - | - | İşlem gerçekleşmedi | - | - | - | - | - | - | - | - | - |\n"}

---
*(Bu rapor KRONOS Walk-Forward Backtesting Motoru tarafından üretilmiştir. Geçmiş performans gelecekteki sonuçların garantisi değildir.)*
"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_md)

        print(f"📑 Backtest Raporu Kaydedildi: {report_file}")
        return report_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KRONOS Walk-Forward Backtest Motoru")
    parser.add_argument("--ticker", type=str, default="ISCTR.IS", help="BIST Sembolü")
    parser.add_argument("--months", type=int, default=6, help="Test periyodu (Ay)")
    parser.add_argument("--sl", type=float, default=3.5, help="Stop-Loss yüzdesi")
    parser.add_argument("--tp", type=float, default=8.0, help="Take-Profit yüzdesi")
    args = parser.parse_args()

    backtester = BistBacktester(use_kronos=False)
    backtester.run_walk_forward_backtest(args.ticker, months=args.months, stop_loss_pct=args.sl, take_profit_pct=args.tp)
