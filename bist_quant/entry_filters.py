"""
Backtest giriş filtreleri: komite ve taramada kullanılan bileşenleri (ICT kurulumu, hacim tabanlı para akışı,
XU100 endeks rejimi) backtest'te tek tek açıp kapatabilmek için. Her filtre yalnızca giriş gününden önceki
mumları görür.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.bist_akd_flow import BistAkdFlowEngine
from bist_quant.bist_index_gatekeeper import compute_market_regime
from bist_quant.bist_price_action import BistPriceActionEngine, ict_setup_score

INDEX_TICKER = "XU100.IS"
FILTER_ICT = "ict"
FILTER_MONEY_FLOW = "money_flow"
FILTER_INDEX_GATE = "index_gate"
ALL_FILTERS = (FILTER_ICT, FILTER_MONEY_FLOW, FILTER_INDEX_GATE)


class EntryFilters:
    """Seçilen filtrelerin tamamı izin verirse LONG girişe onay verir."""

    def __init__(self, names=(), index_history: pd.DataFrame = None):
        self.names = tuple(names)
        unknown = [name for name in self.names if name not in ALL_FILTERS]
        if unknown:
            raise ValueError(f"bilinmeyen giriş filtresi: {unknown}; geçerli değerler: {ALL_FILTERS}")

        self._index_history = None
        self._index_days = None
        if FILTER_INDEX_GATE in self.names:
            if index_history is None or index_history.empty:
                raise ValueError(f"Endeks kapısı filtresi için {INDEX_TICKER} (XU100) geçmişi gerekir.")
            ordered = index_history.assign(_day=index_history["timestamps"].astype(str).str[:10])
            ordered = ordered.sort_values("_day").reset_index(drop=True)
            self._index_history = ordered
            self._index_days = ordered["_day"].to_numpy()

        self._price_action = BistPriceActionEngine()
        self._money_flow = BistAkdFlowEngine()

    def _ict_allows(self, hist_df: pd.DataFrame, as_of_day: str) -> bool:
        score = ict_setup_score(
            self._price_action.detect_liquidity_sweeps(hist_df),
            self._price_action.detect_fair_value_gaps(hist_df),
            self._price_action.detect_break_and_retest(hist_df),
        )
        return score > 0

    def _money_flow_allows(self, hist_df: pd.DataFrame, as_of_day: str) -> bool:
        return self._money_flow.analyze_akd_profile("", hist_df)["whale_score"] > 0

    def _index_gate_allows(self, hist_df: pd.DataFrame, as_of_day: str) -> bool:
        bars_before_entry = int(np.searchsorted(self._index_days, as_of_day, side="left"))
        return bool(compute_market_regime(self._index_history.iloc[:bars_before_entry])["is_long_allowed"])

    _CHECKS = {
        FILTER_ICT: _ict_allows,
        FILTER_MONEY_FLOW: _money_flow_allows,
        FILTER_INDEX_GATE: _index_gate_allows,
    }

    def blocking_filter(self, hist_df: pd.DataFrame, as_of_day: str) -> str | None:
        """Girişi engelleyen ilk filtrenin adı; hepsi izin veriyorsa None."""
        for name in self.names:
            if not self._CHECKS[name](self, hist_df, as_of_day):
                return name
        return None
