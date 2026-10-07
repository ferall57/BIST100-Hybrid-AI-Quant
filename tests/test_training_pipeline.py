import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bist_quant import bist_trainer
from bist_quant.backtest_engine import compute_training_cutoff
from bist_quant.bist_finetune_runner import install_per_ticker_dataset
from bist_quant.bist_kline_dataset import TRAIN_WINDOWS_ENV, VAL_WINDOWS_ENV, PerTickerKlineDataset
from bist_quant.bist_preprocess import preprocess_bist_for_kronos

LOOKBACK, PREDICT = 8, 3
WINDOW = LOOKBACK + PREDICT + 1
CLOSE = 3  # özellik sırası: open, high, low, close, volume, amount
PRICE_GAP = 1000.0


def _symbol_rows(symbol, base_price, n, start="2024-01-01"):
    """Kapanışı her gün tam 1 artan hisse: ardışıklık ve hisse kimliği kapanıştan okunabilir."""
    close = base_price + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamps": pd.date_range(start, periods=n, freq="B").strftime("%Y-%m-%d %H:%M:%S"),
            "open": close, "high": close + 1, "low": close - 1, "close": close,
            "volume": 1000.0, "amount": close * 1000.0,
        }
    )


@pytest.fixture
def unified_csv(tmp_path):
    df = pd.concat([_symbol_rows("AAA.IS", 10.0, 60), _symbol_rows("BBB.IS", 5000.0, 60)])
    path = tmp_path / "unified.csv"
    df.sample(frac=1.0, random_state=1).to_csv(path, index=False)  # satırlar karışık gelse de sonuç değişmemeli
    return str(path)


def _dataset(path, data_type="train", train_ratio=0.8):
    return PerTickerKlineDataset(path, data_type=data_type, lookback_window=LOOKBACK, predict_window=PREDICT,
                                 train_ratio=train_ratio, val_ratio=1.0 - train_ratio, test_ratio=0.0)


def _all_raw_closes(dataset):
    return [dataset.raw_window(i)[0][:, CLOSE] for i in range(dataset.n_samples)]


def test_no_window_mixes_two_symbols(unified_csv):
    for closes in _all_raw_closes(_dataset(unified_csv)):
        assert closes.max() - closes.min() < PRICE_GAP


def test_every_window_is_consecutive_days_of_one_symbol(unified_csv):
    for closes in _all_raw_closes(_dataset(unified_csv)):
        assert np.allclose(np.diff(closes), 1.0)


def test_sample_count_is_the_sum_of_per_symbol_windows(unified_csv):
    assert _dataset(unified_csv, "train").n_samples == 2 * (48 - WINDOW + 1)
    assert _dataset(unified_csv, "val").n_samples == 2 * (12 - WINDOW + 1)


def test_train_and_validation_are_separated_by_date(unified_csv):
    train, val = _dataset(unified_csv, "train"), _dataset(unified_csv, "val")
    cutoff = compute_training_cutoff(pd.read_csv(unified_csv)["timestamps"], 0.8)

    assert max(seg.days[-1] for seg in train.segments) == cutoff
    assert min(seg.days[0] for seg in val.segments) > cutoff


def test_items_have_the_shape_and_range_the_kronos_trainer_expects(unified_csv):
    dataset = _dataset(unified_csv)

    for index in range(len(dataset)):
        x, stamp = dataset[index]
        assert tuple(x.shape) == (WINDOW, 6)
        assert tuple(stamp.shape) == (WINDOW, 5)
        assert float(x.abs().max()) <= dataset.clip


def test_epoch_changes_which_window_a_training_index_maps_to(unified_csv):
    dataset = _dataset(unified_csv)
    first = dataset[0][1].clone()

    dataset.set_epoch_seed(1)

    assert not np.array_equal(first.numpy(), dataset[0][1].numpy())


def test_missing_symbol_column_fails_with_a_clear_message(tmp_path):
    path = tmp_path / "legacy.csv"
    _symbol_rows("AAA.IS", 10.0, 60).drop(columns=["symbol"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="symbol"):
        _dataset(str(path))


def test_symbols_shorter_than_one_window_are_skipped(tmp_path):
    path = tmp_path / "mixed.csv"
    pd.concat([_symbol_rows("AAA.IS", 10.0, 60), _symbol_rows("KISA.IS", 5000.0, 6)]).to_csv(path, index=False)

    assert {seg.symbol for seg in _dataset(str(path)).segments} == {"AAA.IS"}


def test_dataset_without_any_full_window_is_rejected(tmp_path):
    path = tmp_path / "tiny.csv"
    _symbol_rows("AAA.IS", 10.0, 6).to_csv(path, index=False)

    with pytest.raises(ValueError, match="pencere"):
        _dataset(str(path))


# --- ön işleme ---------------------------------------------------------------

def _write_raw(directory, name, timestamps):
    frame = _symbol_rows("x", 10.0, len(timestamps)).drop(columns=["symbol"]).assign(timestamps=timestamps)
    frame.to_csv(directory / name, index=False)


def test_preprocess_tags_rows_with_symbol_and_skips_index_and_intraday_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    days = pd.date_range("2015-10-20", periods=10, freq="B").strftime("%Y-%m-%d")
    mixed_offsets = [f"{d} 00:00:00+0{2 if i < 5 else 3}:00" for i, d in enumerate(days)]  # yaz/kış saati geçişi
    _write_raw(raw, "AAA.IS_1d.csv", mixed_offsets)
    _write_raw(raw, "BBB.IS_1d.csv", mixed_offsets)
    _write_raw(raw, "XU100.IS_1d.csv", mixed_offsets)
    _write_raw(raw, "AAA.IS_1h.csv", mixed_offsets)
    out = tmp_path / "unified.csv"

    preprocess_bist_for_kronos(raw_dir=str(raw), output_file=str(out), min_length=5)
    unified = pd.read_csv(out)

    assert set(unified["symbol"]) == {"AAA.IS", "BBB.IS"}
    assert len(unified) == 20
    assert unified["timestamps"].iloc[0] == "2015-10-20 00:00:00"


# --- Kronos eğitim koduna bağlanma ------------------------------------------

def test_kronos_trainer_modules_use_the_per_ticker_dataset(unified_csv):
    base_module, tokenizer_module = install_per_ticker_dataset()

    assert base_module.CustomKlineDataset is PerTickerKlineDataset
    assert tokenizer_module.CustomKlineDataset is PerTickerKlineDataset

    config = SimpleNamespace(data_path=unified_csv, lookback_window=LOOKBACK, predict_window=PREDICT, clip=5.0, seed=1,
                             train_ratio=0.8, val_ratio=0.2, test_ratio=0.0, batch_size=2, num_workers=0)
    train_loader = base_module.create_dataloaders(config)[0]
    x, stamp = next(iter(train_loader))

    assert tuple(x.shape) == (2, WINDOW, 6)
    assert tuple(stamp.shape) == (2, WINDOW, 5)


# --- epoch uzunluğu sınırı ---------------------------------------------------

def test_training_epoch_can_be_capped_without_mixing_symbols(unified_csv, monkeypatch):
    monkeypatch.setenv(TRAIN_WINDOWS_ENV, "10")
    dataset = _dataset(unified_csv)

    assert len(dataset) == 10
    assert dataset.n_samples == 2 * (48 - WINDOW + 1)
    for index in range(len(dataset)):
        assert tuple(dataset[index][0].shape) == (WINDOW, 6)


def test_cap_larger_than_the_data_is_ignored(unified_csv, monkeypatch):
    monkeypatch.setenv(TRAIN_WINDOWS_ENV, "100000")

    assert len(_dataset(unified_csv)) == 2 * (48 - WINDOW + 1)


def test_capped_validation_is_spread_across_symbols(unified_csv, monkeypatch):
    monkeypatch.setenv(VAL_WINDOWS_ENV, "4")
    dataset = _dataset(unified_csv, "val", train_ratio=0.5)

    flat_indices = [dataset.flat_index(i) for i in range(len(dataset))]
    symbols = {dataset.segments[int(np.searchsorted(dataset._offsets, f, side="right") - 1)].symbol for f in flat_indices}

    assert len(dataset) == 4
    assert symbols == {"AAA.IS", "BBB.IS"}


@pytest.mark.parametrize("bad", ["0", "-3", "abc"])
def test_invalid_cap_fails_fast(unified_csv, monkeypatch, bad):
    monkeypatch.setenv(TRAIN_WINDOWS_ENV, bad)

    with pytest.raises(ValueError, match=TRAIN_WINDOWS_ENV):
        _dataset(unified_csv)


# --- eğitim yöneticisi -------------------------------------------------------

class _FakeProcess:
    """Gerçek eğitimi başlatmayan alt süreç; isteğe bağlı olarak 'model kaydedildi' etkisini taklit eder."""
    launched = []
    on_run = None

    def __init__(self, cmd, env=None, cwd=None):
        _FakeProcess.launched.append({"cmd": cmd, "env": env})
        self.returncode = 0

    def wait(self):
        if _FakeProcess.on_run:
            _FakeProcess.on_run()


@pytest.fixture
def trainer_sandbox(tmp_path, monkeypatch, unified_csv):
    models_dir = tmp_path / "models"
    monkeypatch.setattr(bist_trainer, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(bist_trainer, "TRAINING_CUTOFF_PATH", str(models_dir / "training_cutoff.json"))
    monkeypatch.setattr(bist_trainer, "PROCESSED_DATA_PATH", unified_csv)
    monkeypatch.setattr(bist_trainer, "CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setattr(bist_trainer, "ensure_pretrained_tokenizer", lambda dest: os.makedirs(dest, exist_ok=True))
    monkeypatch.setattr(bist_trainer.subprocess, "Popen", _FakeProcess)
    _FakeProcess.launched = []
    _FakeProcess.on_run = None
    return SimpleNamespace(
        models_dir=models_dir,
        experiment_dir=models_dir / bist_trainer.EXPERIMENT_NAME,
        cutoff_path=models_dir / "training_cutoff.json",
        config_path=str(tmp_path / "config.yaml"),
    )


def _save_checkpoint(sandbox, completed_epochs):
    best_model = sandbox.experiment_dir / "basemodel" / "best_model"
    best_model.mkdir(parents=True, exist_ok=True)
    (best_model / "model.safetensors").write_bytes(b"agirlik")
    (best_model / "last_epoch.txt").write_text(str(completed_epochs))
    return best_model


def test_finished_checkpoint_is_reported_instead_of_silently_training_nothing(trainer_sandbox, capsys):
    _save_checkpoint(trainer_sandbox, completed_epochs=1)
    bist_trainer.generate_bist_config(epochs_predictor=1)

    result = bist_trainer.run_training(config_file=trainer_sandbox.config_path, skip_tokenizer=True)

    assert result is False
    assert _FakeProcess.launched == []
    assert "--fresh-train" in capsys.readouterr().out
    assert not trainer_sandbox.cutoff_path.exists()


def test_cutoff_is_not_recorded_when_the_run_saved_no_model(trainer_sandbox):
    bist_trainer.generate_bist_config(epochs_predictor=1)

    result = bist_trainer.run_training(config_file=trainer_sandbox.config_path, skip_tokenizer=True)

    assert len(_FakeProcess.launched) == 1
    assert result is False
    assert not trainer_sandbox.cutoff_path.exists()


def test_cutoff_is_recorded_when_the_run_saved_a_model(trainer_sandbox):
    bist_trainer.generate_bist_config(epochs_predictor=1)
    _FakeProcess.on_run = lambda: _save_checkpoint(trainer_sandbox, completed_epochs=1)

    result = bist_trainer.run_training(config_file=trainer_sandbox.config_path, skip_tokenizer=True)

    assert result is True
    assert trainer_sandbox.cutoff_path.exists()


def test_fresh_training_archives_the_old_model_and_starts_from_pretrained(trainer_sandbox):
    _save_checkpoint(trainer_sandbox, completed_epochs=1)
    trainer_sandbox.cutoff_path.write_text('{"train_end_date": "2020-01-01"}')
    bist_trainer.generate_bist_config(epochs_predictor=1)

    bist_trainer.run_training(config_file=trainer_sandbox.config_path, skip_tokenizer=True, fresh=True)

    archives = [p for p in trainer_sandbox.models_dir.iterdir() if "_legacy_" in p.name]
    assert len(archives) == 1
    assert (archives[0] / "basemodel" / "best_model" / "model.safetensors").exists()
    assert (archives[0] / "training_cutoff.json").exists()
    assert not (trainer_sandbox.experiment_dir / "basemodel").exists()
    assert len(_FakeProcess.launched) == 1


def test_step_limits_are_passed_to_the_training_process_as_window_counts(trainer_sandbox):
    bist_trainer.generate_bist_config(epochs_predictor=1, batch_size=2)

    bist_trainer.run_training(config_file=trainer_sandbox.config_path, skip_tokenizer=True, train_steps=500, val_steps=50)

    env = _FakeProcess.launched[0]["env"]
    assert env[TRAIN_WINDOWS_ENV] == "1000"
    assert env[VAL_WINDOWS_ENV] == "100"


def test_trainer_launches_the_per_ticker_entrypoint():
    assert os.path.basename(bist_trainer.TRAIN_ENTRYPOINT) == "bist_finetune_runner.py"
    assert os.path.exists(bist_trainer.TRAIN_ENTRYPOINT)
