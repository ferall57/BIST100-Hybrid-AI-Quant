#!/usr/bin/env python3
"""
🏛️ BIST HİBRİT PORTFÖY OPTİMİZASYONU VE RİSK YÖNETİMİ MOTORU
Kurumsal Portföy Tahsis Modelleri:
1. Ledoit-Wolf Kovaryans Daraltması (Shrinkage Covariance Matrix)
2. Markowitz Ortalama-Varyans Optimizasyonu (MVO - Maksimum Sharpe & Minimum Varyans)
3. Hiyerarşik Risk Paritesi (Hierarchical Risk Parity - HRP by Marcos Lopez de Prado)
4. Black-Litterman Bayesyen Varlık Dağılım Modeli (Piyasa Dengesi + Yapay Zeka Görüşleri)
5. Multi-Asset Kelly Kriteri ve Fraksiyonel Kaldıraç Boyutlandırması
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from bist_quant.market_assumptions import risk_free_rate

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')


class BistPortfolioOptimizer:
    """
    BIST 100 hisseleri ve çoklu varlık portföyleri için Kantitatif Tahsis Motoru.
    """
    def __init__(self, risk_free_rate_annual: float = None):
        self.rf_annual = risk_free_rate() if risk_free_rate_annual is None else risk_free_rate_annual
        self.rf_daily = (1.0 + self.rf_annual) ** (1.0 / 252.0) - 1.0

    def compute_returns_and_covariance(self, price_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Fiyat matrisinden log-getirileri, ortalama yıllık getirileri ve Ledoit-Wolf daraltılmış kovaryans matrisini hesaplar.
        """
        log_ret = np.log(price_df / price_df.shift(1)).dropna()
        mean_ret_annual = log_ret.mean() * 252.0
        
        # Ledoit-Wolf Shrinkage (Gürültülü korelasyonları düzenleme)
        lw = LedoitWolf().fit(log_ret.values)
        cov_annual = pd.DataFrame(lw.covariance_ * 252.0, index=price_df.columns, columns=price_df.columns)
        corr = log_ret.corr()
        
        return mean_ret_annual, cov_annual, corr

    # =========================================================================
    # 1. MARKOWITZ ORTALAMA-VARYANS OPTİMİZASYONU (MVO)
    # =========================================================================
    def optimize_markowitz(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target: str = "max_sharpe", # 'max_sharpe' veya 'min_variance'
        max_weight_per_asset: float = 0.35
    ) -> dict:
        """
        Klasik Markowitz Etkin Sınır (Efficient Frontier) Çözücüsü (SLSQP Karesel Programlama).
        """
        n_assets = len(mean_returns)
        tickers = list(mean_returns.index)
        
        def portfolio_variance(weights):
            return float(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

        def portfolio_return(weights):
            return float(np.dot(weights, mean_returns.values))

        def negative_sharpe(weights):
            p_ret = portfolio_return(weights)
            p_vol = np.sqrt(portfolio_variance(weights))
            return -(p_ret - self.rf_annual) / (p_vol + 1e-6)

        init_weights = np.ones(n_assets) / n_assets
        bounds = tuple((0.0, max_weight_per_asset) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        if target == "max_sharpe":
            opt = minimize(negative_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        else: # min_variance
            opt = minimize(portfolio_variance, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        opt_weights = np.round(opt.x, 4) if opt.success else init_weights
        weights_dict = {tickers[i]: float(opt_weights[i]) for i in range(n_assets)}
        
        exp_ret = float(portfolio_return(opt_weights))
        exp_vol = float(np.sqrt(portfolio_variance(opt_weights)))
        sharpe = (exp_ret - self.rf_annual) / (exp_vol + 1e-6)

        return {
            "target": target,
            "weights": weights_dict,
            "expected_return_pct": round(exp_ret * 100.0, 2),
            "expected_volatility_pct": round(exp_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 3)
        }

    # =========================================================================
    # 2. HİYERARŞİK RİSK PARİTESİ (HRP - MARCOS LOPEZ DE PRADO)
    # =========================================================================
    def optimize_hrp(self, price_df: pd.DataFrame) -> dict:
        """
        Hierarchical Risk Parity (HRP) Algoritması:
        Kovaryans matrisi tersi (Sigma^-1) almadan, korelasyon ağacı kümelemesi ile risk tahsisi yapar.
        """
        log_ret = np.log(price_df / price_df.shift(1)).dropna()
        cov = log_ret.cov() * 252.0
        corr = log_ret.corr()
        tickers = list(price_df.columns)
        
        # 1. Aşama: Korelasyon Uzaklık Matrisi: d(i,j) = sqrt( 2 * (1 - rho(i,j)) )
        dist_matrix = np.sqrt(np.clip(2.0 * (1.0 - corr.values), 0.0, None))
        dist_condensed = squareform(dist_matrix, checks=False)
        
        # 2. Aşama: Hiyerarşik Ağaç Kümeleme (Single Linkage)
        link = linkage(dist_condensed, method='single')
        
        # 3. Aşama: Quasi-Diagonalization (Kümelerin sıralanması)
        def get_quasi_diag(link_mat):
            link_mat = link_mat.astype(int)
            sort_ix = pd.Series([link_mat[-1, 0], link_mat[-1, 1]])
            num_items = link_mat[-1, 3]
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df_temp = sort_ix[sort_ix >= num_items]
                i = df_temp.index
                j = df_temp.values - num_items
                sort_ix[i] = link_mat[j, 0]
                df_temp = pd.Series(link_mat[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df_temp]).sort_index()
                sort_ix.index = range(sort_ix.shape[0])
            return sort_ix.tolist()

        sorted_indices = get_quasi_diag(link)
        sorted_tickers = [tickers[i] for i in sorted_indices]
        
        # 4. Aşama: Recursive Bisection ile Ağırlık Dağıtımı (Inverse Variance Weighting)
        weights = pd.Series(1.0, index=sorted_tickers)
        clusters = [sorted_tickers]
        
        while len(clusters) > 0:
            clusters = [c[start:end] for c in clusters for start, end in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
            for i in range(0, len(clusters), 2):
                c_left = clusters[i]
                c_right = clusters[i + 1]
                
                cov_left = cov.loc[c_left, c_left]
                cov_right = cov.loc[c_right, c_right]
                
                # Küme varyansları (Ters varyans ağırlıklı)
                w_left_inv = 1.0 / np.diag(cov_left)
                w_left_norm = w_left_inv / np.sum(w_left_inv)
                var_left = float(np.dot(w_left_norm.T, np.dot(cov_left.values, w_left_norm)))
                
                w_right_inv = 1.0 / np.diag(cov_right)
                w_right_norm = w_right_inv / np.sum(w_right_inv)
                var_right = float(np.dot(w_right_norm.T, np.dot(cov_right.values, w_right_norm)))
                
                alpha = 1.0 - (var_left / (var_left + var_right + 1e-6))
                
                weights[c_left] *= alpha
                weights[c_right] *= (1.0 - alpha)

        weights = weights.loc[tickers] # Orijinal sıraya getir
        weights = weights / weights.sum()
        
        exp_ret = float(np.dot(weights.values, (log_ret.mean() * 252.0).values))
        exp_vol = float(np.sqrt(np.dot(weights.values.T, np.dot(cov.values, weights.values))))
        sharpe = (exp_ret - self.rf_annual) / (exp_vol + 1e-6)

        return {
            "model": "Hierarchical Risk Parity (HRP)",
            "weights": {k: round(float(v), 4) for k, v in weights.to_dict().items()},
            "expected_return_pct": round(exp_ret * 100.0, 2),
            "expected_volatility_pct": round(exp_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 3)
        }

    # =========================================================================
    # 3. BLACK-LITTERMAN BAYESYEN PORTFÖY MODELİ
    # =========================================================================
    def optimize_black_litterman(
        self,
        price_df: pd.DataFrame,
        market_caps: dict[str, float] = None,
        investor_views: dict[str, float] = None, # Örn: {"ISCTR.IS": 0.35, "THYAO.IS": 0.50} (Yıllık getiri beklentileri)
        tau: float = 0.05
    ) -> dict:
        """
        Black-Litterman Modeli:
        1. Piyasa Denge Getirilerini (Pi) Tersine Çözer (Reverse Optimization).
        2. Yapay Zeka / Quant Görüşlerini (Views: P, Q, Omega) Bayes Teoremi ile entegre eder.
        3. Nihai Denge Getirilerini (mu_BL) ve Optimal Portföy Ağırlıklarını hesaplar.
        """
        tickers = list(price_df.columns)
        n = len(tickers)
        log_ret = np.log(price_df / price_df.shift(1)).dropna()
        
        # Kovaryans matrisi (Yıllık)
        lw = LedoitWolf().fit(log_ret.values)
        Sigma = pd.DataFrame(lw.covariance_ * 252.0, index=tickers, columns=tickers)
        
        # 1. Piyasa Değeri Ağırlıkları (Eşit veya Piyasa Değeri Ağırlıklı)
        if market_caps is None or len(market_caps) != n:
            w_mkt = np.ones(n) / n
        else:
            caps = np.array([market_caps.get(t, 1.0) for t in tickers])
            w_mkt = caps / np.sum(caps)

        # Risk Aversiyon Katsayısı (delta)
        market_var = float(np.dot(w_mkt.T, np.dot(Sigma.values, w_mkt)))
        delta = max(1.0, (0.60 - self.rf_annual) / (market_var + 1e-6)) # %60 BIST 100 beklenen getiri varsayımı

        # İma Edilen Denge Getirileri (Pi = delta * Sigma * w_mkt)
        Pi = delta * np.dot(Sigma.values, w_mkt)

        # 2. Yatırımcı Görüşleri Matrisi (P, Q, Omega)
        if investor_views and len(investor_views) > 0:
            view_tickers = [t for t in investor_views if t in tickers]
            k = len(view_tickers)
            P = np.zeros((k, n))
            Q = np.zeros(k)
            
            for row_idx, vt in enumerate(view_tickers):
                col_idx = tickers.index(vt)
                P[row_idx, col_idx] = 1.0
                Q[row_idx] = investor_views[vt]

            # Ridge Regülarizasyonu ile Tekil Matris (Singular Matrix) Çökmesini Önleme
            ridge_eye_n = np.eye(n) * 1e-5
            ridge_eye_k = np.eye(k) * 1e-5
            
            tau_Sigma_reg = tau_Sigma + ridge_eye_n
            Omega_reg = Omega + ridge_eye_k

            # Bayesyen Formülü: E(R)_BL = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1 * [(tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
            inv_tau_Sigma = np.linalg.inv(tau_Sigma_reg)
            inv_Omega = np.linalg.inv(Omega_reg)

            M_inv = np.linalg.inv(inv_tau_Sigma + np.dot(P.T, np.dot(inv_Omega, P)) + ridge_eye_n)
            mu_BL = np.dot(M_inv, (np.dot(inv_tau_Sigma, Pi) + np.dot(P.T, np.dot(inv_Omega, Q))))
            Sigma_BL = Sigma.values + M_inv
        else:
            mu_BL = Pi
            Sigma_BL = Sigma.values

        # Optimal Ağırlıklar: w_BL = (delta * Sigma_BL)^-1 * mu_BL
        try:
            inv_Sigma_BL = np.linalg.inv((delta * Sigma_BL) + np.eye(n) * 1e-5)
            raw_weights = np.dot(inv_Sigma_BL, mu_BL)
            # Long-only ve bütçe kısıtı normalizasyonu
            w_bl = np.clip(raw_weights, 0.0, None)
            sum_w = np.sum(w_bl)
            w_bl = (w_bl / sum_w) if sum_w > 0 else w_mkt
        except Exception:
            w_bl = w_mkt

        exp_ret = float(np.dot(w_bl, mu_BL))
        exp_vol = float(np.sqrt(np.dot(w_bl.T, np.dot(Sigma_BL, w_bl))))
        sharpe = (exp_ret - self.rf_annual) / (exp_vol + 1e-6)

        return {
            "model": "Black-Litterman Bayesian Allocation",
            "weights": {tickers[i]: round(float(w_bl[i]), 4) for i in range(n)},
            "expected_return_pct": round(exp_ret * 100.0, 2),
            "expected_volatility_pct": round(exp_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 3),
            "implied_equilibrium_returns": {tickers[i]: round(float(Pi[i]) * 100.0, 2) for i in range(n)}
        }

    # =========================================================================
    # 4. KELLY CRITERION & FRAKSİYONEL KALDIRAÇ MOTORU
    # =========================================================================
    def calculate_kelly_portfolio(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        fraction: float = 0.5 # Half-Kelly (Güvenli Korumacı Büyüme)
    ) -> dict:
        """
        Multi-Asset Kelly Kriteri (Sermaye Büyüme Oranını Maksimize Eden Kaldıraç):
        f* = fraction * Sigma^-1 * (mu - r_f)
        """
        tickers = list(mean_returns.index)
        n = len(tickers)
        excess_returns = (mean_returns - self.rf_annual).values
        
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            raw_kelly = np.dot(inv_cov, excess_returns) * fraction
            
            # Negatif ağırlıkları ele (Long-only) ve toplam kaldıracı hesapla
            kelly_clipped = np.clip(raw_kelly, 0.0, 1.0)
            total_leverage = float(np.sum(kelly_clipped))
            
            normalized_weights = kelly_clipped / (np.sum(kelly_clipped) + 1e-6)
            
            return {
                "model": f"Kelly Criterion ({fraction:.1f}x Fractional)",
                "suggested_total_leverage": round(total_leverage, 2),
                "unconstrained_kelly_weights": {tickers[i]: round(float(kelly_clipped[i]), 4) for i in range(n)},
                "normalized_portfolio_weights": {tickers[i]: round(float(normalized_weights[i]), 4) for i in range(n)}
            }
        except Exception as e:
            return {"error": str(e), "suggested_total_leverage": 1.0}


if __name__ == "__main__":
    import yfinance as yf
    print("⚡ BistPortfolioOptimizer Kurumsal Test Ediliyor...")
    
    tickers = ["THYAO.IS", "ISCTR.IS", "AKBNK.IS", "ASELS.IS", "BIMAS.IS"]
    df_prices = pd.DataFrame()
    for t in tickers:
        data = yf.download(t, period="1y", progress=False)
        if not data.empty and "Close" in data.columns:
            df_prices[t] = data["Close"]

    optimizer = BistPortfolioOptimizer()
    mean_ret, cov, corr = optimizer.compute_returns_and_covariance(df_prices)
    
    # 1. Markowitz Max Sharpe
    mvo_res = optimizer.optimize_markowitz(mean_ret, cov, target="max_sharpe")
    print("\n" + "="*80)
    print("📊 1. MARKOWITZ MAKSİMUM SHARPE PORTFÖYÜ")
    print("="*80)
    for t, w in mvo_res["weights"].items():
        print(f"  • {t:10}: %{w*100.0:5.1f}")
    print(f"  🎯 Beklenen Getiri: %{mvo_res['expected_return_pct']} | Volatilite: %{mvo_res['expected_volatility_pct']} | Sharpe: {mvo_res['sharpe_ratio']}")

    # 2. HRP (Hierarchical Risk Parity)
    hrp_res = optimizer.optimize_hrp(df_prices)
    print("\n" + "="*80)
    print("🛡️ 2. HİYERARŞİK RİSK PARİTESİ (HRP - LOPEZ DE PRADO)")
    print("="*80)
    for t, w in hrp_res["weights"].items():
        print(f"  • {t:10}: %{w*100.0:5.1f}")
    print(f"  🎯 Beklenen Getiri: %{hrp_res['expected_return_pct']} | Volatilite: %{hrp_res['expected_volatility_pct']} | Sharpe: {hrp_res['sharpe_ratio']}")

    # 3. Black-Litterman
    bl_views = {"ISCTR.IS": 0.55, "THYAO.IS": 0.65}
    bl_res = optimizer.optimize_black_litterman(df_prices, investor_views=bl_views)
    print("\n" + "="*80)
    print("🧠 3. BLACK-LITTERMAN BAYESYEN PORTFÖYÜ (AI Views Entegre)")
    print("="*80)
    for t, w in bl_res["weights"].items():
        print(f"  • {t:10}: %{w*100.0:5.1f}")
    print(f"  🎯 Beklenen Getiri: %{bl_res['expected_return_pct']} | Volatilite: %{bl_res['expected_volatility_pct']} | Sharpe: {bl_res['sharpe_ratio']}")
    print("="*80 + "\n")
