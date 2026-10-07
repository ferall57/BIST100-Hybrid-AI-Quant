"""
Kronos sıralı ince ayarını hisse bazlı veri kümesiyle başlatır.
train_sequential.py yerine çalıştırılır; Kronos kaynak kodu değiştirilmez, yalnızca veri kümesi sınıfı
PerTickerKlineDataset ile değiştirilir. Komut satırı argümanları train_sequential.py ile aynıdır.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KRONOS_DIR = os.path.join(ROOT_DIR, "repos", "Kronos")
FINETUNE_CSV_DIR = os.path.join(KRONOS_DIR, "finetune_csv")


def install_per_ticker_dataset():
    """Kronos eğitim modüllerindeki veri kümesi sınıfını hisse bazlı sürümle değiştirir."""
    for path in (ROOT_DIR, KRONOS_DIR, FINETUNE_CSV_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)

    import finetune_base_model
    import finetune_tokenizer
    from bist_quant.bist_kline_dataset import PerTickerKlineDataset

    finetune_base_model.CustomKlineDataset = PerTickerKlineDataset
    finetune_tokenizer.CustomKlineDataset = PerTickerKlineDataset
    return finetune_base_model, finetune_tokenizer


def main():
    install_per_ticker_dataset()
    import train_sequential
    train_sequential.main()


if __name__ == "__main__":
    main()
