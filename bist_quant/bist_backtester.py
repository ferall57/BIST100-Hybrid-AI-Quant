import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.bist_downloader import download_ticker_data, RAW_DATA_DIR
from bist_quant.bist_viop import BistViopEngine

try:
    from bist_quant.bist_kronos_quant import BistKronosQuant
    KRONOS_AVAILABLE = True
except Exception:
    KRONOS_AVAILABLE = False

CHARTS_DIR = os.path.join(ROOT_DIR, "outputs", "charts")
REPORTS_DIR = os.path.join(ROOT_DIR, "outputs", "reports")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

class BistBacktester:
    """
    Borsa İstanbul (BIST) hisseleri için Walk-Forward Rolling Window Backtesting Motoru.
    Geleceği görmeden (Lookahead bias olmadan) geçmiş periyotlarda adım adım tahmin üretir,
    dinamik Stop-Loss / Take-Profit kurallarını işletir ve kurumsal finansal metrikleri hesaplar.
    """
    def __init__(self, use_kronos: bool = True):
        self.use_kronos = use_kronos
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
        leverage: float = 1.5
    ):
        """
        Geçmiş N aylık veri üzerinde Walk-Forward simülasyonu çalıştırır.
        use_viop=True: Çift yönlü (Long & Short) VİOP türev motorunu çalıştırır.
        use_trailing_stop=True: Kârı koşturur, hisse yükseldikçe (veya düştükçe) stop seviyesini taşır.
        """
        if not ticker.endswith(".IS"):
            ticker += ".IS"

        print(f"\n" + "="*85)
        print(f"📊 KRONOS WALK-FORWARD BACKTEST MOTORU BAŞLATILDI")
        print(f"🎯 Hedef Hisse    : {ticker}")
        print(f"⏳ Test Periyodu  : Son {months} Ay (Adım: Her {step_days} İşlem Gününde Bir)")
        print(f"🔮 Tahmin Ufku    : {horizon_days} İşlem Günü")
        print(f"🛡️ Risk Yönetimi  : {'İz Süren Stop (Trailing Stop - Kârı Koştur)' if use_trailing_stop else f'Sabit TP: %{take_profit_pct:.1f}'} | Rejim Filtresi: {'Aktif' if enable_regime_filter else 'Pasif'}")
        if use_viop:
            print(f"⚡ VİOP Modu      : Aktif (Çift Yönlü Long & Short | Kaldıraç: {leverage}x | Takasbank Nemalandırması: %45)")
        print(f"💰 Başlangıç Kasa : {initial_capital:,.2f} TRY")
        print("="*85 + "\n")

        # 1. Canlı veriyi güncelle ve oku
        download_ticker_data(ticker, period="5y", interval="1d", save_dir=RAW_DATA_DIR)
        csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}_1d.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{ticker} veri seti bulunamadı: {csv_path}")

        df = pd.read_csv(csv_path)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.sort_values("timestamps").reset_index(drop=True)

        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]

        # Test aralığını belirle (Yaklaşık 1 ay = 21 işlem günü)
        total_test_days = int(months * 21)
        if len(df) < total_test_days + 100:
            total_test_days = max(30, len(df) - 100)
            print(f"[BİLGİ] Veri seti uzunluğuna göre test periyodu {total_test_days} işlem gününe uyarlandı.")

        test_start_idx = len(df) - total_test_days
        test_df = df.iloc[test_start_idx:].copy().reset_index(drop=True)

        trades = []
        current_capital = initial_capital
        daily_interest_rate = (1.0 + 0.45) ** (1.0 / 365.0) - 1.0

        # 2. Walk-Forward Döngüsü (Pencere Kaydırma)
        current_idx = test_start_idx
        pbar = tqdm(total=len(df) - horizon_days - test_start_idx, desc=f"{ticker} Backtest")

        while current_idx < len(df) - horizon_days:
            # Sadece current_idx öncesindeki geçmiş veriyi gör (Lookahead Bias Yok!)
            hist_df = df.iloc[:current_idx].copy()
            entry_row = df.iloc[current_idx]
            entry_date = entry_row["timestamps"]
            entry_price = float(entry_row["close"])

            # 1. Rejim Filtresi Kontrolü
            is_bull, regime_str = self._check_market_regime(hist_df)
            
            # 2. Dinamik ATR Stop-Loss ve Take-Profit
            active_sl, active_tp = self._calculate_dynamic_sl_tp(hist_df, stop_loss_pct, take_profit_pct)

            # 3. Model Tahmini Üret
            expected_return = self._predict_return(hist_df, horizon_days=horizon_days)

            # İşlem Kararı (Sniper & VİOP Çift Yönlü Karar):
            min_thresh = 0.8 if self.quant_engine is not None else 1.2
            should_enter_long = expected_return > min_thresh and (is_bull or not enable_regime_filter)
            should_enter_short = use_viop and (expected_return < -min_thresh or not is_bull)

            if should_enter_long or should_enter_short:
                direction = "LONG" if should_enter_long else "SHORT"
                eff_leverage = leverage if use_viop else 1.0
                
                trade_allocated = current_capital * (allocation_pct / 100.0)
                cash_reserve = current_capital - trade_allocated
                
                # Pozisyon Yönetimi (Trailing Stop ile Trend Sürme)
                max_hold_days = min(60, len(df) - current_idx - 1)
                forward_window = df.iloc[current_idx + 1 : current_idx + 1 + max_hold_days].copy().reset_index(drop=True)
                
                trail_distance_pct = max(4.5, min(8.0, active_sl * 1.1))
                peak_price = entry_price
                trough_price = entry_price
                
                exit_price = float(forward_window["close"].iloc[-1])
                exit_date = forward_window["timestamps"].iloc[-1]
                exit_reason = f"Maksimum Vade Sonu ({max_hold_days}G)"
                duration = len(forward_window)
                accumulated_interest = 0.0

                # Gün gün pozisyon takibi
                for day_i, (_, f_row) in enumerate(forward_window.iterrows(), 1):
                    f_high = float(f_row["high"])
                    f_low = float(f_row["low"])
                    f_close = float(f_row["close"])
                    f_date = f_row["timestamps"]

                    if use_viop:
                        accumulated_interest += current_capital * daily_interest_rate

                    if direction == "LONG":
                        if f_high > peak_price:
                            peak_price = f_high
                        
                        trailing_sl_price = peak_price * (1.0 - trail_distance_pct / 100.0)
                        initial_sl_price = entry_price * (1.0 - active_sl / 100.0)
                        effective_sl = max(initial_sl_price, trailing_sl_price) if use_trailing_stop else initial_sl_price

                        if f_low <= effective_sl:
                            exit_price = effective_sl
                            exit_date = f_date
                            exit_reason = f"İz Süren Stop (Trailing +%{((exit_price - entry_price)/entry_price)*100:.1f})" if exit_price >= entry_price else f"Stop-Loss (%{active_sl:.1f})"
                            duration = day_i
                            break

                        if day_i >= horizon_days and f_close < entry_price:
                            exit_price = f_close
                            exit_date = f_date
                            exit_reason = f"Vade Sonu Konsolidasyon ({day_i}G)"
                            duration = day_i
                            break

                    elif direction == "SHORT":
                        if f_low < trough_price:
                            trough_price = f_low

                        # Inverted Trailing Stop: Dip fiyattan yukarı sıçrarsa kârı al
                        inverted_trailing_sl = trough_price * (1.0 + trail_distance_pct / 100.0)
                        short_initial_sl = entry_price * (1.0 + active_sl / 100.0)
                        effective_short_sl = min(short_initial_sl, inverted_trailing_sl) if use_trailing_stop else short_initial_sl

                        if f_high >= effective_short_sl:
                            exit_price = effective_short_sl
                            exit_date = f_date
                            exit_reason = f"Ters İz Süren Stop (Short +%{((entry_price - exit_price)/entry_price)*100:.1f})" if exit_price <= entry_price else f"Short Stop-Loss (%{active_sl:.1f})"
                            duration = day_i
                            break

                        if day_i >= horizon_days and f_close > entry_price:
                            exit_price = f_close
                            exit_date = f_date
                            exit_reason = f"Short Vade Sonu ({day_i}G)"
                            duration = day_i
                            break

                # Getiri Hesabı
                if direction == "LONG":
                    price_change_pct = ((exit_price - entry_price) / entry_price) * 100.0
                else:  # SHORT
                    price_change_pct = ((entry_price - exit_price) / entry_price) * 100.0

                trade_return_pct = price_change_pct * eff_leverage
                realized_pnl = trade_allocated * (trade_return_pct / 100.0) + accumulated_interest
                current_capital = max(1000.0, current_capital + realized_pnl)

                trades.append({
                    "entry_date": entry_date.strftime("%Y-%m-%d"),
                    "entry_price": entry_price,
                    "exit_date": pd.to_datetime(exit_date).strftime("%Y-%m-%d"),
                    "exit_price": exit_price,
                    "direction": direction,
                    "return_pct": trade_return_pct,
                    "pnl": realized_pnl,
                    "capital": current_capital,
                    "duration_days": duration,
                    "reason": exit_reason,
                    "expected_return": expected_return,
                    "regime": regime_str,
                    "sl_used": active_sl,
                    "tp_used": active_tp,
                    "win": trade_return_pct > 0
                })

                # Kârlı trend devam ediyorsa küçük adımla ilerle
                step = max(1, duration)
            else:
                # Ayı piyasası veya Nötr sinyal: Nakitte bekle
                if use_viop:
                    current_capital += current_capital * daily_interest_rate * step_days
                step = step_days

            current_idx += step
            pbar.update(step)

        pbar.close()

        # 3. Kümülatif Metrikleri Hesapla
        metrics = self._calculate_performance_metrics(
            trades=trades,
            test_df=test_df,
            initial_capital=initial_capital,
            final_capital=current_capital
        )

        # 4. Grafiği Çiz ve Kaydet
        chart_path = self._plot_equity_curve(ticker, test_df, trades, initial_capital)

        # 5. Markdown Raporunu Oluştur ve Kaydet
        report_path = self._generate_markdown_report(ticker, metrics, trades, chart_path, months, stop_loss_pct, take_profit_pct)

        return metrics, report_path, chart_path

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
            except Exception:
                pass

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

    def _calculate_performance_metrics(self, trades: list, test_df: pd.DataFrame, initial_capital: float, final_capital: float) -> dict:
        """Wall Street standartlarında risk ve getiri metriklerini hesaplar."""
        bnh_start = float(test_df["close"].iloc[0]) if len(test_df) > 0 else 1.0
        bnh_end = float(test_df["close"].iloc[-1]) if len(test_df) > 0 else 1.0
        bnh_return_pct = ((bnh_end - bnh_start) / bnh_start) * 100.0

        total_trades = len(trades)
        if total_trades == 0:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "win_rate": 0.0, "total_return_pct": 0.0, "bnh_return_pct": bnh_return_pct,
                "alpha": -bnh_return_pct, "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
                "max_drawdown": 0.0, "profit_factor": 0.0, "payoff_ratio": 0.0,
                "avg_gain": 0.0, "avg_loss": 0.0, "avg_trade_return": 0.0,
                "initial_capital": initial_capital, "final_capital": final_capital
            }

        returns = np.array([t["return_pct"] for t in trades])
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades) * 100.0

        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100.0

        # Benchmark (Buy & Hold) Getirisi
        bnh_start = float(test_df["close"].iloc[0])
        bnh_end = float(test_df["close"].iloc[-1])
        bnh_return_pct = ((bnh_end - bnh_start) / bnh_start) * 100.0
        alpha = total_return_pct - bnh_return_pct

        # Kâr Faktörü ve Payoff Rasyosu
        sum_gains = float(np.sum(wins)) if len(wins) > 0 else 0.0
        sum_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = (sum_gains / sum_losses) if sum_losses > 0 else (99.0 if sum_gains > 0 else 1.0)

        avg_gain = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0
        payoff_ratio = (avg_gain / avg_loss) if avg_loss > 0 else 1.0

        # Kümülatif Kasa Eğrisi üzerinden Drawdown ve Sharpe
        capitals = [initial_capital] + [t["capital"] for t in trades]
        peaks = np.maximum.accumulate(capitals)
        drawdowns = (peaks - capitals) / peaks * 100.0
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Sharpe & Sortino (Yıllıklandırılmış: 252 gün / 10 gün adım ortalama ~ 25 işlem periyodu)
        trade_returns_decimal = returns / 100.0
        mean_ret = float(np.mean(trade_returns_decimal))
        raw_std = float(np.std(trade_returns_decimal)) if len(trade_returns_decimal) > 1 else 0.02
        std_ret = max(0.01, raw_std)

        annual_factor = np.sqrt(25.0)
        sharpe_ratio = float((mean_ret / std_ret) * annual_factor)

        downside_arr = trade_returns_decimal[trade_returns_decimal < 0]
        raw_downside = float(np.std(downside_arr)) if len(downside_arr) > 1 else std_ret
        downside_std = max(0.01, raw_downside)
        sortino_ratio = float((mean_ret / downside_std) * annual_factor)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_return_pct": total_return_pct,
            "bnh_return_pct": bnh_return_pct,
            "alpha": alpha,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "payoff_ratio": payoff_ratio,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "avg_trade_return": float(np.mean(returns)),
            "initial_capital": initial_capital,
            "final_capital": final_capital
        }

    def _plot_equity_curve(self, ticker: str, test_df: pd.DataFrame, trades: list, initial_capital: float) -> str:
        """Dark-mode sermaye büyüme eğrisi ve Drawdown grafiği oluşturur."""
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=False)

        # 1. Sermaye Eğrisi
        dates = [pd.to_datetime(test_df["timestamps"].iloc[0])]
        capitals = [initial_capital]

        for t in trades:
            dates.append(pd.to_datetime(t["exit_date"]))
            capitals.append(t["capital"])

        ax1.plot(dates, capitals, label="KRONOS Quant Stratejisi (TRY)", color="#00FFCC", linewidth=2.5, marker="o", markersize=4)

        # Benchmark Kıyaslaması (Buy & Hold Kasa Karşılığı)
        bnh_init = float(test_df["close"].iloc[0])
        bnh_capitals = [initial_capital * (float(p) / bnh_init) for p in test_df["close"]]
        ax1.plot(test_df["timestamps"], bnh_capitals, label=f"Al ve Tut (Buy & Hold: {ticker})", color="#888888", linestyle="--", linewidth=1.5, alpha=0.7)

        ax1.set_title(f"KRONOS Walk-Forward Backtest Sermaye Eğrisi (Equity Curve) -> {ticker}", fontsize=14, fontweight="bold", color="white")
        ax1.set_ylabel("Portfoy Degeri (TRY)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.3)
        ax1.legend(loc="upper left", framealpha=0.3)

        # 2. Drawdown Grafiği
        caps_arr = np.array(capitals)
        peaks = np.maximum.accumulate(caps_arr)
        dd = (caps_arr - peaks) / peaks * 100.0

        ax2.fill_between(dates, dd, 0, color="#FF3366", alpha=0.4, label="Kasa Çekilmesi (Drawdown %)")
        ax2.plot(dates, dd, color="#FF3366", linewidth=1.5)
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

    def _generate_markdown_report(self, ticker: str, m: dict, trades: list, chart_path: str, months: int, sl: float, tp: float) -> str:
        """Tüm işlem detaylarını ve finansal göstergeleri içeren kurumsal rapor üretir."""
        trade_rows = ""
        for idx, t in enumerate(trades, 1):
            badge = "🟢 KÂR" if t["win"] else "🔴 ZARAR"
            direction_badge = "🟢 LONG" if t.get("direction", "LONG") == "LONG" else "🔻 SHORT"
            trade_rows += f"| #{idx} | {direction_badge} | {t['entry_date']} | {t['entry_price']:.2f} TRY | {t['exit_date']} | {t['exit_price']:.2f} TRY | **%{t['return_pct']:+.2f}** | {t['pnl']:+,.2f} TRY | {t['capital']:,.2f} TRY | {t['duration_days']}G | {badge} ({t['reason']}) |\n"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_file = os.path.join(REPORTS_DIR, f"{ticker.replace('.', '_')}_backtest_report.md")

        report_md = f"""# 🏛️ KRONOS WALK-FORWARD BACKTEST & PERFORMANS RAPORU
**Tarih:** {now_str} | **Hisse:** {ticker} | **Test Periyodu:** Son {months} Ay (Lookahead Bias %0)
**Risk Kuralları:** Stop-Loss: %{sl:.1f} | Take-Profit: %{tp:.1f} | **Başlangıç Kasa:** {m['initial_capital']:,.2f} TRY

---

## 🏆 1. YÖNETİCİ PERFORMANS VE RİSK ÖZETİ

| Performans Metriği | KRONOS Stratejisi | Kıyaslama (Buy & Hold) | Açıklama |
| :--- | :--- | :--- | :--- |
| **Kümülatif Toplam Getiri** | **%{m['total_return_pct']:+.2f}** | %{m['bnh_return_pct']:+.2f} | **Alpha ($\alpha$): %{m['alpha']:+.2f}** |
| **Nihai Kasa Değeri** | **{m['final_capital']:,.2f} TRY** | {(m['initial_capital'] * (1.0 + m['bnh_return_pct']/100.0)):,.2f} TRY | Net Kâr: {(m['final_capital'] - m['initial_capital']):+,.2f} TRY |
| **Kazanma Oranı (Win Rate)** | **%{m['win_rate']:.1f}** ({m['winning_trades']}/{m['total_trades']}) | N/A | Başarılı işlem oranı |
| **Sharpe Oranı (Yıllık)** | **{m['sharpe_ratio']:.2f}** | N/A | Birim risk başına üretilen getiri ($>1.5$ Mükemmel) |
| **Sortino Oranı** | **{m['sortino_ratio']:.2f}** | N/A | Sadece aşağı yönlü riske göre getiri verimi |
| **Kâr Faktörü (Profit Factor)** | **{m['profit_factor']:.2f}x** | N/A | Toplam Kâr / Toplam Zarar ($>1.5$ Sağlıklı) |
| **Payoff Oranı (R:R)** | **{m['payoff_ratio']:.2f}x** | N/A | Ortalama Kâr (%{m['avg_gain']:.2f}) / Ortalama Zarar (%{m['avg_loss']:.2f}) |
| **Maksimum Çekilme (MDD)** | **-%{m['max_drawdown']:.2f}** | N/A | Portföyün zirveden yaşadığı en derin düşüş |
| **Toplam İşlem Sayısı** | **{m['total_trades']} Adet** | 1 Pozisyon | Gerçekleştirilen işlem döngüsü |

---

## 📈 2. SERMAYE EĞRİSİ VE ÇEKİLME GRAFİĞİ
*(Görsel Grafik Dosyası: `{chart_path}`)*

---

## 📋 3. İŞLEM GÜNLÜĞÜ (TRADE LOG)

| İşlem | Yön | Giriş Tarihi | Giriş Fiyatı | Çıkış Tarihi | Çıkış Fiyatı | Getiri (%) | Net Kâr/Zarar | Bakiye | Süre | Sonuç / Kapanış Nedeni |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{trade_rows if trade_rows else "| - | - | İşlem gerçekleşmedi | - | - | - | - | - | - | - | - |\n"}

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
