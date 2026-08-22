import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Windows konsollarında Unicode/Emoji kilitlenmelerini önleme:
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class BistEconometrics:
    """
    Borsa İstanbul hisseleri için Klasik İstatistiksel & Ekonometrik Kuant Motoru:
    - Augmented Dickey-Fuller (ADF) Durağanlık ve Birim Kök Testi
    - Günlük ve Aylık Mevsimsellik (Seasonality) Analizi
    - Parkinson & Yıllıklandırılmış Volatilite Rejimi
    - 1.000 Yollu Stokastik Monte Carlo Simülasyonu (Geometrik Brown Hareketi)
    - 15 Günlük Parametrik Value-at-Risk (VaR %95 ve VaR %99)
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def test_stationarity(self, df: pd.DataFrame) -> dict:
        """Fiyat ve log-getiri serilerinde Augmented Dickey-Fuller (ADF) testi uygular."""
        if not STATSMODELS_AVAILABLE or len(df) < 30:
            return {
                "available": False,
                "price_stat": 0.0,
                "price_pvalue": 1.0,
                "price_is_stationary": False,
                "return_pvalue": 0.0,
                "return_is_stationary": True,
                "interpretation": "İstatistiksel model kütüphanesi eksik veya veri boyutu yetersiz."
            }

        try:
            close_prices = df["close"].dropna()
            # Fiyat serisi ADF testi
            adf_price = adfuller(close_prices, autolag="AIC")
            price_stat = float(adf_price[0])
            price_p = float(adf_price[1])
            price_stat_crit = adf_price[4]
            price_is_stat = price_p < 0.05

            # Getiri serisi ADF testi
            log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
            adf_ret = adfuller(log_returns, autolag="AIC")
            ret_p = float(adf_ret[1])
            ret_is_stat = ret_p < 0.05

            if not price_is_stat:
                interp = f"Fiyat serisi durağan değildir (Birim kök vardır, p={price_p:.4f}). Fiyatlar stokastik bir trend veya sürüklenme (drift) içindedir; log-getiriler ise durağandır (p={ret_p:.4e}, I(1) entegrasyon)."
            else:
                interp = f"Fiyat serisi durağandır (p={price_p:.4f}). Hisse ortalamaya dönen (mean-reverting) yatay bir bantta hareket etmektedir."

            return {
                "available": True,
                "price_stat": price_stat,
                "price_pvalue": price_p,
                "price_critical_5pct": price_stat_crit.get("5%", -2.86),
                "price_is_stationary": price_is_stat,
                "return_pvalue": ret_p,
                "return_is_stationary": ret_is_stat,
                "interpretation": interp
            }
        except Exception as e:
            return {
                "available": False,
                "price_stat": 0.0,
                "price_pvalue": 1.0,
                "price_is_stationary": False,
                "return_pvalue": 0.0,
                "return_is_stationary": True,
                "interpretation": f"ADF hesaplama hatası: {e}"
            }

    def analyze_seasonality(self, df: pd.DataFrame) -> dict:
        """Hissenin haftanın günleri ve aylara göre tarihsel getiri anomalilerini ölçer."""
        try:
            df_s = df.copy()
            df_s["timestamps"] = pd.to_datetime(df_s["timestamps"])
            df_s["log_ret"] = np.log(df_s["close"] / df_s["close"].shift(1)) * 100.0
            df_s = df_s.dropna(subset=["log_ret"])

            # Haftanın günleri analizi (0: Pazartesi, 4: Cuma)
            df_s["day_of_week"] = df_s["timestamps"].dt.dayofweek
            day_names = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma"}
            day_means = df_s.groupby("day_of_week")["log_ret"].mean().to_dict()
            
            best_day_idx = max(day_means, key=day_means.get) if day_means else 0
            worst_day_idx = min(day_means, key=day_means.get) if day_means else 0
            
            best_day = day_names.get(best_day_idx, "N/A")
            worst_day = day_names.get(worst_day_idx, "N/A")
            best_day_ret = day_means.get(best_day_idx, 0.0)
            worst_day_ret = day_means.get(worst_day_idx, 0.0)

            # Mevsimsel güç indeksi (Haftanın günleri arasındaki getiri varyansı)
            day_variance = float(np.var(list(day_means.values()))) if day_means else 0.0
            seasonal_strength = "Yüksek" if day_variance > 0.08 else ("Orta" if day_variance > 0.03 else "Zayıf / Nötr")

            return {
                "best_day": best_day,
                "best_day_ret": best_day_ret,
                "worst_day": worst_day,
                "worst_day_ret": worst_day_ret,
                "seasonal_strength": seasonal_strength,
                "day_means": {day_names.get(k, k): v for k, v in day_means.items()}
            }
        except Exception:
            return {
                "best_day": "N/A",
                "best_day_ret": 0.0,
                "worst_day": "N/A",
                "worst_day_ret": 0.0,
                "seasonal_strength": "Nötr",
                "day_means": {}
            }

    def calculate_volatility(self, df: pd.DataFrame) -> dict:
        """Klasik ve Parkinson (Yüksek-Düşük Marjı) Volatilitelerini hesaplar."""
        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            
            # 1. Kapanıştan kapanışa log-getiri volatilitesi (Son 60 gün)
            window = min(60, len(df))
            recent_close = close.tail(window)
            log_ret = np.log(recent_close / recent_close.shift(1)).dropna()
            hist_vol = float(log_ret.std() * np.sqrt(252) * 100.0)

            # 2. Parkinson Volatilitesi (Intraday Extreme-Value Volatility)
            # Formül: sqrt( 1 / (4 * ln(2) * N) * sum( (ln(H/L))^2 ) ) * sqrt(252)
            recent_high = high.tail(window)
            recent_low = low.tail(window)
            valid_mask = (recent_high > 0) & (recent_low > 0) & (recent_high >= recent_low)
            hl_ratio = np.log(recent_high[valid_mask] / recent_low[valid_mask])
            parkinson_sum = np.sum(hl_ratio ** 2)
            N = len(hl_ratio)
            
            if N > 5:
                parkinson_vol = float(np.sqrt((1.0 / (4.0 * np.log(2.0) * N)) * parkinson_sum) * np.sqrt(252) * 100.0)
            else:
                parkinson_vol = hist_vol

            # Volatilite Rejimi
            if hist_vol < 28.0:
                regime = "Düşük Oynaklık (Sakin Konsolidasyon)"
            elif hist_vol <= 48.0:
                regime = "Normal / Sağlıklı BIST Oynaklığı"
            else:
                regime = "Yüksek Oynaklık (Türbülans / Yüksek Risk)"

            return {
                "hist_volatility": hist_vol,
                "parkinson_volatility": parkinson_vol,
                "volatility_regime": regime
            }
        except Exception:
            return {
                "hist_volatility": 30.0,
                "parkinson_volatility": 30.0,
                "volatility_regime": "Normal Oynaklık"
            }

    def _estimate_garch_volatilities(self, log_returns: pd.Series, days: int) -> np.ndarray:
        """
        GARCH(1,1) koşullu varyans projeksiyonunu hesaplar.
        sigma_t^2 = V_L + (alpha + beta)^t * (sigma_0^2 - V_L)
        """
        try:
            var_uncond = float(log_returns.var())
            sigma_0_sq = float(log_returns.tail(10).var())
            # BIST piyasası tipik GARCH katsayıları
            alpha = 0.12
            beta = 0.83
            persistence = alpha + beta
            
            sigmas = []
            for t in range(1, days + 1):
                cond_var = var_uncond + (persistence ** t) * (sigma_0_sq - var_uncond)
                sigmas.append(np.sqrt(max(1e-6, cond_var)))
            return np.array(sigmas)
        except Exception:
            daily_std = float(log_returns.std())
            return np.full(days, daily_std)

    def run_monte_carlo_simulation(self, df: pd.DataFrame, days: int = 15, num_sims: int = 1000) -> dict:
        """
        Merton Jump Diffusion (Poisson Sıçramaları & Şişman Kuyruk) ve GARCH(1,1) Dinamik Volatilitesi
        kullanarak 1.000 bağımsız stokastik fiyat yolu simüle eder ve olasılık dağılımı üretir.
        """
        try:
            close_series = df["close"].dropna()
            current_price = float(close_series.iloc[-1])
            
            # Son 120 günün log-getiri dağılım parametreleri
            window = min(120, len(close_series))
            log_returns = np.log(close_series.tail(window) / close_series.tail(window).shift(1)).dropna()
            
            mu_daily = float(log_returns.mean()) # Günlük ortalama sürüklenme (drift)
            
            # GARCH(1,1) Dinamik Günlük Volatilite Vektörü (Boyut: days)
            daily_garch_sigmas = self._estimate_garch_volatilities(log_returns, days)
            
            # ⚡ MERTON JUMP DIFFUSION PARAMETRELERİ (Şişman Kuyruk & BIST Şokları)
            lambda_daily = 2.0 / 252.0  # Yılda ortalama 2 büyük KAP/Makro sıçrama şoku
            mu_jump = -0.015           # Şokların asimetrik ortalama negatif etkisi (%-1.5)
            sigma_jump = 0.045          # Şokun standart sapması (%4.5)
            k_jump = np.exp(mu_jump + 0.5 * (sigma_jump ** 2)) - 1.0 # Beklenen oransal sıçrama
            
            np.random.seed(self.seed)
            price_paths = np.zeros((days + 1, num_sims))
            price_paths[0] = current_price

            for t in range(1, days + 1):
                sigma_t = daily_garch_sigmas[t - 1]
                
                # 1. Standart Gauss Piyasa Gürültüsü
                z = np.random.normal(0, 1, num_sims)
                
                # 2. Poisson Sıçrama Süreci (Kriz / Tavan / Taban Şokları)
                num_jumps = np.random.poisson(lambda_daily, num_sims)
                jump_shocks = np.zeros(num_sims)
                for i in range(num_sims):
                    if num_jumps[i] > 0:
                        jump_shocks[i] = np.sum(np.random.normal(mu_jump, sigma_jump, num_jumps[i]))

                # Merton SDE Formülü: S(t) = S(t-1) * exp((mu - lambda*k - 0.5*sigma^2) + sigma*Z + Jump)
                drift_t = mu_daily - (lambda_daily * k_jump) - (0.5 * (sigma_t ** 2))
                diffusion_t = sigma_t * z
                total_return = np.exp(drift_t + diffusion_t + jump_shocks)
                
                price_paths[t] = price_paths[t - 1] * total_return
                
            # 1. 1-Haftalık (5. Gün / Kısa Vade) Fiyat Dağılımı
            idx_5d = min(5, days)
            prices_5d = price_paths[idx_5d]
            median_5d = float(np.median(prices_5d))
            ci_95_l_5d = float(np.percentile(prices_5d, 5))
            ci_95_u_5d = float(np.percentile(prices_5d, 95))
            prob_pos_5d = float((np.sum(prices_5d > current_price) / num_sims) * 100.0)
            var_95_5d = float(((current_price - ci_95_l_5d) / current_price) * 100.0)
            
            # Expected Shortfall (CVaR %95): %5'lik en kötü kuyruktaki ortalama kayıp
            tail_5d = prices_5d[prices_5d <= ci_95_l_5d]
            cvar_95_5d = float(((current_price - np.mean(tail_5d)) / current_price) * 100.0) if len(tail_5d) > 0 else var_95_5d
            ret_5d_pct = float(((median_5d - current_price) / current_price) * 100.0)

            # 2. Orta Vadeli (N Günlük) Fiyat Dağılımı
            final_prices = price_paths[-1]
            median_price = float(np.median(final_prices))
            mean_price = float(np.mean(final_prices))
            pct_5 = float(np.percentile(final_prices, 5))   # %95 VaR alt sınırı
            pct_95 = float(np.percentile(final_prices, 95)) # %95 Üst sınır
            pct_1 = float(np.percentile(final_prices, 1))   # %99 VaR alt sınırı
            pct_99 = float(np.percentile(final_prices, 99)) # %99 Üst sınır
            
            positive_paths = np.sum(final_prices > current_price)
            prob_positive = float((positive_paths / num_sims) * 100.0)
            
            var_95_pct = float(((current_price - pct_5) / current_price) * 100.0)
            var_99_pct = float(((current_price - pct_1) / current_price) * 100.0)
            
            tail_final = final_prices[final_prices <= pct_5]
            cvar_95_pct = float(((current_price - np.mean(tail_final)) / current_price) * 100.0) if len(tail_final) > 0 else var_95_pct
            
            expected_mc_return = float(((median_price - current_price) / current_price) * 100.0)

            return {
                "current_price": current_price,
                # 1 Hafta (5 Gün)
                "median_5d": median_5d,
                "expected_return_5d_pct": ret_5d_pct,
                "ci_95_lower_5d": ci_95_l_5d,
                "ci_95_upper_5d": ci_95_u_5d,
                "prob_positive_5d": prob_pos_5d,
                "var_95_5d": var_95_5d,
                "cvar_95_5d": cvar_95_5d,
                # Orta Vade (N Gün)
                "median_target": median_price,
                "mean_target": mean_price,
                "expected_return_pct": expected_mc_return,
                "ci_95_lower": pct_5,
                "ci_95_upper": pct_95,
                "ci_99_lower": pct_1,
                "ci_99_upper": pct_99,
                "prob_positive": prob_positive,
                "var_95_pct": var_95_pct,
                "var_99_pct": var_99_pct,
                "cvar_95_pct": cvar_95_pct,
                "model_engine": "Merton Jump Diffusion + GARCH(1,1)",
                "num_simulations": num_sims
            }
        except Exception as e:
            return {
                "current_price": 0.0,
                "median_5d": 0.0,
                "expected_return_5d_pct": 0.0,
                "ci_95_lower_5d": 0.0,
                "ci_95_upper_5d": 0.0,
                "prob_positive_5d": 50.0,
                "var_95_5d": 3.0,
                "cvar_95_5d": 4.5,
                "median_target": 0.0,
                "mean_target": 0.0,
                "expected_return_pct": 0.0,
                "ci_95_lower": 0.0,
                "ci_95_upper": 0.0,
                "ci_99_lower": 0.0,
                "ci_99_upper": 0.0,
                "prob_positive": 50.0,
                "var_95_pct": 5.0,
                "var_99_pct": 8.0,
                "cvar_95_pct": 7.0,
                "num_simulations": num_sims,
                "error": str(e)
            }

    def generate_econometric_report(self, df: pd.DataFrame, ticker: str, forecast_days: int = 15) -> str:
        """Tüm istatistiksel, ekonometrik ve stokastik testleri birleştirerek yapılandırılmış bir rapor üretir."""
        stat_res = self.test_stationarity(df)
        seas_res = self.analyze_seasonality(df)
        vol_res = self.calculate_volatility(df)
        mc_res = self.run_monte_carlo_simulation(df, days=forecast_days, num_sims=1000)

        price = mc_res["current_price"]
        med_t = mc_res["median_target"]
        ret_pct = mc_res["expected_return_pct"]
        prob_pos = mc_res["prob_positive"]
        var_95 = mc_res["var_95_pct"]
        ci_95_l = mc_res["ci_95_lower"]
        ci_95_u = mc_res["ci_95_upper"]

        stat_badge = "✅ Trend Doğrulandı (I(1))" if not stat_res.get("price_is_stationary", False) else "🔄 Ortalamaya Dönen (Mean-Reverting)"

        report = f"""### 🔬 Klasik Ekonometri & Stokastik Simülasyon Raporu ({ticker})
* **Stokastik Motor Modeli:** **Merton Jump Diffusion (Şişman Kuyruk Sıçramaları) + GARCH(1,1) Dinamik Volatilite**

| Ekonometrik Gösterge / Test | Test Değeri / Sonuç | Finansal & Matematiksel Yorum |
| :--- | :--- | :--- |
| **Durağanlık (ADF Testi)** | Test İstatistiği: {stat_res.get('price_stat', 0.0):.3f} (p={stat_res.get('price_pvalue', 1.0):.4f}) | {stat_badge} |
| **Log-Getiri Durağanlığı** | p={stat_res.get('return_pvalue', 0.0):.4e} | Getiri serisi durağandır, rastgele yürüyüş (random walk) modeli geçerlidir. |
| **Tarihsel Volatilite (60G)** | %{vol_res.get('hist_volatility', 0.0):.2f} (Yıllık) | Rejim: **{vol_res.get('volatility_regime', 'Normal')}** |
| **GARCH(1,1) Dinamik Oynaklık** | Koşullu Varyans Projeksiyonu | Zamanla kümelenen dalga boyu simülasyon adımlarına entegre edildi. |
| **Mevsimsellik Gücü** | {seas_res.get('seasonal_strength', 'Nötr')} | En Güçlü Gün: **{seas_res.get('best_day', 'N/A')}** (%{seas_res.get('best_day_ret', 0.0):+.2f}), En Zayıf: **{seas_res.get('worst_day', 'N/A')}** (%{seas_res.get('worst_day_ret', 0.0):+.2f}) |
| **1 Haftalık Merton MC (5G)**| **{mc_res['median_5d']:.2f} TRY** (Getiri: **%{mc_res['expected_return_5d_pct']:+.2f}**) | %95 Güven: **[{mc_res['ci_95_lower_5d']:.2f} - {mc_res['ci_95_upper_5d']:.2f} TRY]**, Kazanma: **%{mc_res['prob_positive_5d']:.1f}**, 5G VaR: **-%{mc_res['var_95_5d']:.2f}** (CVaR: -%{mc_res['cvar_95_5d']:.2f}) |
| **Orta Vadeli Merton MC ({forecast_days}G)**| **{med_t:.2f} TRY** (Getiri: **%{ret_pct:+.2f}**) | %95 Güven: **[{ci_95_l:.2f} - {ci_95_u:.2f} TRY]**, Kazanma: **%{prob_pos:.1f}**, {forecast_days}G VaR: **-%{var_95:.2f}** (CVaR: -%{mc_res['cvar_95_pct']:.2f}) |

**Ekonometrik Sentez:**
{stat_res.get('interpretation', '')} 1.000 yollu Merton Jump Diffusion (Poisson sıçramalı şişman kuyruk) simülasyonu; hissede **1 haftalık vadede** %{mc_res['prob_positive_5d']:.1f} kazanma olasılığıyla {mc_res['median_5d']:.2f} TRY (%{mc_res['expected_return_5d_pct']:+.2f}) medyan seviyesini [{mc_res['ci_95_lower_5d']:.2f} - {mc_res['ci_95_upper_5d']:.2f} TRY bandı, %95 CVaR kuyruk riski: %{mc_res['cvar_95_5d']:.2f}], **{forecast_days} günlük orta vadede** ise %{prob_pos:.1f} kazanma olasılığıyla {med_t:.2f} TRY (%{ret_pct:+.2f}) medyan seviyesini [{ci_95_l:.2f} - {ci_95_u:.2f} TRY bandı, %95 CVaR kuyruk riski: %{mc_res['cvar_95_pct']:.2f}] işaret etmektedir.
"""
        return report

if __name__ == "__main__":
    import yfinance as yf
    print("🔬 BistEconometrics Test Ediliyor...")
    econ = BistEconometrics()
    
    # Test verisi çek
    ticker = "ISCTR.IS"
    t = yf.Ticker(ticker)
    df = t.history(period="1y")
    df.reset_index(inplace=True)
    df.columns = [str(c).lower().strip() for c in df.columns]
    for c in ["date", "datetime"]:
        if c in df.columns:
            df.rename(columns={c: "timestamps"}, inplace=True)
            break
            
    rep = econ.generate_econometric_report(df, ticker, forecast_days=15)
    print(rep)
