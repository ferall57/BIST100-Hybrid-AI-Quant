import os
import sys
import json
import yaml
import subprocess
import argparse
from datetime import datetime

import pandas as pd

# Windows konsollarında Emoji ve UTF-8 karakter hatalarını önlemek için:
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

# Proje ana dizini ve Kronos repolarını PATH'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KRONOS_DIR = os.path.join(ROOT_DIR, "repos", "Kronos")
FINETUNE_CSV_DIR = os.path.join(KRONOS_DIR, "finetune_csv")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "bist_data", "processed", "bist100_unified_kline.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "bist_kronos")
CONFIG_PATH = os.path.join(ROOT_DIR, "bist_quant", "bist_train_config.yaml")
TRAINING_CUTOFF_PATH = os.path.join(MODELS_DIR, "training_cutoff.json")
EXPERIMENT_NAME = "bist100_kronos_base"
PRETRAINED_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
PRETRAINED_PREDICTOR = "NeoQuasar/Kronos-base"
TRAIN_RATIO = 0.9
PIPELINE_NAME = "per_ticker"
MODEL_WEIGHTS_FILE = "model.safetensors"
# Kronos'un train_sequential.py betiğini hisse bazlı veri kümesiyle çalıştıran giriş noktası
TRAIN_ENTRYPOINT = os.path.join(ROOT_DIR, "bist_quant", "bist_finetune_runner.py")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bist_quant.backtest_engine import compute_training_cutoff
from bist_quant.bist_kline_dataset import TRAIN_WINDOWS_ENV, VAL_WINDOWS_ENV

def _experiment_dir() -> str:
    return os.path.join(MODELS_DIR, EXPERIMENT_NAME)

def _best_model_dir(component: str) -> str:
    return os.path.join(_experiment_dir(), component, "best_model")

def _weights_mtime(component: str) -> float | None:
    path = os.path.join(_best_model_dir(component), MODEL_WEIGHTS_FILE)
    return os.path.getmtime(path) if os.path.exists(path) else None

def completed_epochs(component: str = "basemodel") -> int:
    """Kayıtlı checkpoint'in tamamladığı epoch sayısı; checkpoint yoksa 0."""
    epoch_file = os.path.join(_best_model_dir(component), "last_epoch.txt")
    if not os.path.exists(epoch_file):
        return 0
    try:
        with open(epoch_file, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except ValueError:
        return 0

def archive_existing_experiment() -> str | None:
    """
    Mevcut ince ayarlı modeli silmeden yeniden adlandırarak arşivler; böylece yeni eğitim
    eski checkpoint'ten değil önceden eğitilmiş Kronos ağırlıklarından başlar.
    """
    source = _experiment_dir()
    if not os.path.exists(source):
        return None
    archive = f"{source}_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.rename(source, archive)
    if os.path.exists(TRAINING_CUTOFF_PATH):
        os.rename(TRAINING_CUTOFF_PATH, os.path.join(archive, os.path.basename(TRAINING_CUTOFF_PATH)))
    print(f"📦 Eski model arşivlendi -> {archive}")
    return archive

def ensure_pretrained_tokenizer(dest: str) -> None:
    """Tokenizer eğitimi atlanırken yerelde tokenizer yoksa önceden eğitilmiş Kronos tokenizer'ını oraya kaydeder."""
    if os.path.exists(os.path.join(dest, MODEL_WEIGHTS_FILE)):
        return
    if KRONOS_DIR not in sys.path:
        sys.path.insert(0, KRONOS_DIR)
    from model import KronosTokenizer
    os.makedirs(dest, exist_ok=True)
    KronosTokenizer.from_pretrained(PRETRAINED_TOKENIZER).save_pretrained(dest)
    print(f"📥 Önceden eğitilmiş tokenizer hazırlandı ({PRETRAINED_TOKENIZER}) -> {dest}")

def write_training_cutoff(processed_csv: str, train_ratio: float, out_path: str) -> str:
    """
    Modelin eğitimde gördüğü son tarihi kaydeder. Backtest motoru bu dosyayı okuyarak
    test döneminin eğitim verisiyle çakışıp çakışmadığını (veri sızıntısı) denetler.
    """
    timestamps = pd.read_csv(processed_csv, usecols=["timestamps"])["timestamps"]
    cutoff = compute_training_cutoff(timestamps, train_ratio)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"train_end_date": cutoff, "train_ratio": train_ratio, "pipeline": PIPELINE_NAME,
                   "source": os.path.basename(processed_csv)}, f, indent=2)
    print(f"🗓️ Eğitim kesim tarihi kaydedildi: {cutoff} -> {out_path}")
    return cutoff

def generate_bist_config(epochs_tokenizer=30, epochs_predictor=20, batch_size=2, accum_steps=16, lookback=256, lr_predictor=1e-6, train_tokenizer=True, train_basemodel=True):
    """
    NVIDIA GTX 1650 (4 GB VRAM) donanımı için optimize edilmiş (düşük batch_size + yüksek gradient accumulation)
    BIST 100 özel ince ayar konfigürasyon dosyası üretir.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    config = {
        "data": {
            "data_path": PROCESSED_DATA_PATH.replace("\\", "/"),
            "lookback_window": lookback,
            "predict_window": 30,
            "max_context": 512,
            "clip": 5.0,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": round(1.0 - TRAIN_RATIO, 4),
            "test_ratio": 0.0
        },
        "training": {
            "tokenizer_epochs": epochs_tokenizer,
            "basemodel_epochs": epochs_predictor,
            "batch_size": batch_size,
            "log_interval": 20,
            "num_workers": 0,  # Windows PyTorch multiprocessing çökmelerini önlemek için 0
            "seed": 42,
            "tokenizer_learning_rate": 0.0002,
            "predictor_learning_rate": lr_predictor,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": accum_steps
        },
        "model_paths": {
            "pretrained_tokenizer": PRETRAINED_TOKENIZER,
            "pretrained_predictor": PRETRAINED_PREDICTOR,
            "exp_name": EXPERIMENT_NAME,
            "base_path": MODELS_DIR.replace("\\", "/") + "/",
            "base_save_path": "",
            "finetuned_tokenizer": "",
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel"
        },
        "experiment": {
            "name": "bist100_custom_finetune",
            "description": "BIST 100 Finansal Mum Formasyonları için Özelleştirilmiş Kronos-base Eğitimi",
            "use_comet": False,
            "train_tokenizer": train_tokenizer,
            "train_basemodel": train_basemodel,
            "skip_existing": False
        },
        "device": {
            "use_cuda": True,
            "device_id": 0
        }
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"⚙️ 4GB VRAM Optimize Eğitim Konfigürasyonu Üretildi -> {CONFIG_PATH}")
    return CONFIG_PATH

def _window_limits(train_steps: int | None, val_steps: int | None, batch_size: int) -> dict:
    """Adım sınırlarını veri kümesinin okuduğu pencere sayısı ortam değişkenlerine çevirir."""
    limits = {}
    if train_steps is not None:
        limits[TRAIN_WINDOWS_ENV] = str(train_steps * batch_size)
    if val_steps is not None:
        limits[VAL_WINDOWS_ENV] = str(val_steps * batch_size)
    return limits

def run_training(config_file=None, skip_tokenizer=False, skip_basemodel=False, fresh=False, train_steps=None, val_steps=None):
    """
    Kronos sıralı ince ayarını hisse bazlı veri kümesiyle başlatır.
    fresh=True: mevcut modeli arşivleyip önceden eğitilmiş Kronos ağırlıklarından başlar.
    train_steps / val_steps: epoch başına eğitim ve doğrulama adımı sınırı (tanımsızsa tüm pencereler).
    """
    config_file = config_file or CONFIG_PATH
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ [HATA] İşlenmiş veri seti bulunamadı: {PROCESSED_DATA_PATH}")
        print("Önce 'python main.py --download-all' veya 'python -m bist_quant.bist_preprocess' çalıştırmalısınız.")
        return False

    with open(config_file, "r", encoding="utf-8") as f:
        training_config = yaml.safe_load(f)["training"]
    target_epochs = int(training_config["basemodel_epochs"])
    batch_size = int(training_config["batch_size"])

    if fresh:
        archive_existing_experiment()

    # Kronos eğitim kodu kayıtlı checkpoint'ten ve onun epoch sayacından devam eder
    done_epochs = completed_epochs("basemodel")
    if not skip_basemodel and done_epochs >= target_epochs:
        print(f"⏹️ Eğitim başlatılmadı: kayıtlı model zaten {done_epochs} epoch tamamlamış, hedef {target_epochs} epoch.")
        print("   • Aynı modelin üzerine devam etmek için --pred-epochs değerini büyütün.")
        print("   • Yeni veri hattıyla sıfırdan eğitmek için --fresh-train ekleyin (eski model silinmez, arşivlenir).")
        return False
    if done_epochs > 0 and not os.path.exists(TRAINING_CUTOFF_PATH):
        print("⚠️ Kayıtlı model eski veri hattıyla (hisseleri karıştıran pencerelerle) eğitilmiş; eğitim onun üzerine devam edecek.")
        print("   Temiz bir başlangıç için --fresh-train kullanın.")

    if skip_tokenizer:
        ensure_pretrained_tokenizer(_best_model_dir("tokenizer"))

    print(f"🔥 BIST 100 Kronos-base Derin Eğitimi (Fine-Tuning) Başlatılıyor...")
    print(f"🧠 Hedef Model: {PRETRAINED_PREDICTOR} (102.3M Parametre)")
    print(f"🛡️ Bellek Koruma: Batch Size = {batch_size}, Gradient Accumulation = {training_config.get('accumulation_steps', 1)}")

    # PYTHONPATH ve UTF-8 ayarla
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{KRONOS_DIR};{FINETUNE_CSV_DIR};" + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.update(_window_limits(train_steps, val_steps, batch_size))

    cmd = [sys.executable, TRAIN_ENTRYPOINT, "--config", config_file]
    if skip_tokenizer:
        cmd.append("--skip-tokenizer")
    if skip_basemodel:
        cmd.append("--skip-basemodel")
    print(f"💻 Çalıştırılan Komut: {' '.join(cmd)}")

    weights_before = _weights_mtime("basemodel")
    try:
        # Gerçek zamanlı terminal log akışı ile eğitimi başlat
        process = subprocess.Popen(cmd, env=env, cwd=FINETUNE_CSV_DIR)
        process.wait()
    except KeyboardInterrupt:
        print("\n⏹️ Kullanıcı eğitimi durdurdu. Checkpoint'ler korunuyor.")
        return False

    if process.returncode != 0:
        print(f"\n❌ [UYARI] Eğitim komutu hata ile kapandı. Çıkış kodu: {process.returncode}")
        return False

    if not skip_basemodel and _weights_mtime("basemodel") == weights_before:
        print("\n⚠️ Eğitim süreci hatasız kapandı ama yeni bir model kaydedilmedi; eğitim kesim tarihi yazılmadı.")
        return False

    print(f"\n🎉 BIST 100 İnce Ayar Eğitimi Başarıyla Tamamlandı!")
    if not skip_basemodel:
        write_training_cutoff(PROCESSED_DATA_PATH, TRAIN_RATIO, TRAINING_CUTOFF_PATH)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 100 Kronos-Base Eğitim ve Fine-Tuning Yöneticisi")
    parser.add_argument("--tok-epochs", type=int, default=10, help="Tokenizer eğitim epok sayısı")
    parser.add_argument("--pred-epochs", type=int, default=15, help="Predictor (Base Model) eğitim epok sayısı")
    parser.add_argument("--batch-size", type=int, default=2, help="4GB VRAM için önerilen batch size (2 veya 4)")
    parser.add_argument("--accum", type=int, default=16, help="Gradient accumulation adımı (2 * 16 = 32)")
    parser.add_argument("--fresh-train", action="store_true", help="Mevcut modeli arşivleyip önceden eğitilmiş ağırlıklardan başla")
    parser.add_argument("--train-steps", type=int, default=None, help="Epoch başına eğitim adımı sınırı")
    parser.add_argument("--val-steps", type=int, default=None, help="Epoch başına doğrulama adımı sınırı")

    args = parser.parse_args()
    generate_bist_config(epochs_tokenizer=args.tok_epochs, epochs_predictor=args.pred_epochs, batch_size=args.batch_size, accum_steps=args.accum)
    run_training(fresh=args.fresh_train, train_steps=args.train_steps, val_steps=args.val_steps)
