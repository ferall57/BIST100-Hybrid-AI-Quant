"""
Hisse bazında pencereleyen Kronos ince ayar veri kümesi.

Kronos'un kendi CustomKlineDataset sınıfı CSV'deki tüm satırları tarihe göre sıralayıp tek seri gibi
pencereler; çok hisseli bir dosyada her pencere aynı günün farklı hisselerinden oluşur. Bu sınıf aynı
arayüzü korur ama her pencereyi tek bir hissenin ardışık günlerinden üretir ve eğitim/doğrulama
ayrımını tüm hisseler için ortak bir tarihten yapar.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.backtest_engine import compute_training_cutoff

SYMBOL_COLUMN = "symbol"
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
TIME_FEATURE_COLUMNS = ["minute", "hour", "weekday", "day", "month"]
NORMALIZATION_EPSILON = 1e-5
# Kronos'un eğitim indeksini pencerelere dağıtmak için kullandığı asal adımlar (aynı örnekleme düzeni korunur).
INDEX_STRIDE = 9973
EPOCH_STRIDE = 104729
# Bir epoch'ta kullanılacak pencere sayısı sınırları (tanımsızsa tüm pencereler kullanılır).
TRAIN_WINDOWS_ENV = "KRONOS_TRAIN_WINDOWS_PER_EPOCH"
VAL_WINDOWS_ENV = "KRONOS_VAL_WINDOWS"


def _read_window_cap(env_name: str) -> int | None:
    raw = os.getenv(env_name)
    if raw is None or not raw.strip():
        return None
    try:
        cap = int(raw.strip())
    except ValueError:
        raise ValueError(f"{env_name} pozitif bir tam sayı olmalı; gelen değer: {raw!r}") from None
    if cap < 1:
        raise ValueError(f"{env_name} pozitif bir tam sayı olmalı; gelen değer: {raw!r}")
    return cap


@dataclass(frozen=True)
class SymbolSegment:
    """Tek bir hissenin, tek bir veri bölümündeki (train/val/test) kronolojik mumları."""
    symbol: str
    features: np.ndarray
    time_features: np.ndarray
    days: np.ndarray


class PerTickerKlineDataset(Dataset):
    """Kronos CustomKlineDataset ile aynı kurucu imzasına sahip, hisse bazlı pencereleyen veri kümesi."""

    def __init__(self, data_path, data_type='train', lookback_window=90, predict_window=10,
                 clip=5.0, seed=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.data_path = data_path
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.current_epoch = 0

        self.segments = self._build_segments(pd.read_csv(data_path))
        window_counts = [len(seg.features) - self.window + 1 for seg in self.segments]
        self._offsets = np.concatenate([[0], np.cumsum(window_counts)]).astype(np.int64)
        self.n_samples = int(self._offsets[-1])
        if self.n_samples == 0:
            raise ValueError(
                f"[{data_type.upper()}] Hiçbir hissede {self.window} mumluk tam bir pencere yok. "
                "Veri aralığını veya lookback/predict pencere uzunluklarını kontrol edin."
            )

        cap = _read_window_cap(TRAIN_WINDOWS_ENV if data_type == 'train' else VAL_WINDOWS_ENV)
        self.epoch_length = self.n_samples if cap is None else min(cap, self.n_samples)

        print(f"[{data_type.upper()}] Hisse sayısı: {len(self.segments)}, pencere sayısı: {self.n_samples} "
              f"(epoch başına kullanılan: {self.epoch_length}), "
              f"tarih aralığı: {min(seg.days[0] for seg in self.segments)} -> {max(seg.days[-1] for seg in self.segments)}")

    def _split_mask(self, days: pd.Series, timestamps: pd.Series) -> pd.Series:
        """Tüm hisseler için ortak tarih sınırlarıyla bu bölüme ait satırları seçer."""
        train_cutoff = compute_training_cutoff(timestamps, self.train_ratio)
        val_cutoff = compute_training_cutoff(timestamps, min(1.0, self.train_ratio + self.val_ratio))
        if self.data_type == 'train':
            return days <= train_cutoff
        if self.data_type == 'val':
            return (days > train_cutoff) & (days <= val_cutoff)
        return days > val_cutoff

    def _build_segments(self, df: pd.DataFrame) -> list[SymbolSegment]:
        missing = [c for c in [SYMBOL_COLUMN, "timestamps"] + FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Eğitim verisinde eksik sütun: {missing}. Hisse bazlı pencereleme için '{SYMBOL_COLUMN}' sütunu "
                "gerekir; veriyi 'python main.py --download-all' ile yeniden üretin."
            )

        df = df.dropna(subset=FEATURE_COLUMNS).copy()
        days = df["timestamps"].astype(str).str[:10]
        df = df.loc[self._split_mask(days, df["timestamps"])]

        segments = []
        for symbol, group in df.groupby(SYMBOL_COLUMN, sort=True):
            if len(group) < self.window:
                continue
            stamps = pd.to_datetime(group["timestamps"])
            order = np.argsort(stamps.to_numpy(), kind="stable")
            stamps = stamps.iloc[order]
            time_features = np.column_stack([
                stamps.dt.minute, stamps.dt.hour, stamps.dt.weekday, stamps.dt.day, stamps.dt.month
            ]).astype(np.float32)
            segments.append(SymbolSegment(
                symbol=str(symbol),
                features=group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[order],
                time_features=time_features,
                days=stamps.dt.strftime("%Y-%m-%d").to_numpy(),
            ))
        return segments

    def set_epoch_seed(self, epoch):
        self.current_epoch = epoch

    def __len__(self):
        return self.epoch_length

    def flat_index(self, idx: int) -> int:
        """
        Epoch içi indeksi tüm pencereler üzerindeki düz indekse çevirir.
        Eğitimde her epoch farklı pencerelere kayar; doğrulamada sınırlı örnek tüm hisselere eşit aralıkla yayılır.
        """
        if self.data_type == 'train':
            return (idx * INDEX_STRIDE + (self.current_epoch + 1) * EPOCH_STRIDE) % self.n_samples
        return (idx * (self.n_samples // self.epoch_length)) % self.n_samples

    def raw_window(self, flat_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Düz pencere indeksine karşılık gelen (özellikler, zaman özellikleri) dilimi; tek hisseden gelir."""
        segment_index = int(np.searchsorted(self._offsets, flat_index, side="right") - 1)
        start = int(flat_index - self._offsets[segment_index])
        segment = self.segments[segment_index]
        return segment.features[start:start + self.window], segment.time_features[start:start + self.window]

    def __getitem__(self, idx):
        x, x_stamp = self.raw_window(self.flat_index(idx))
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + NORMALIZATION_EPSILON)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(x_stamp.copy())
