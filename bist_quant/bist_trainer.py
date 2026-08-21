import os
import sys
import yaml
import subprocess
import argparse

# Proje ana dizini ve Kronos repolarını PATH'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KRONOS_DIR = os.path.join(ROOT_DIR, "repos", "Kronos")
FINETUNE_CSV_DIR = os.path.join(KRONOS_DIR, "finetune_csv")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "bist_data", "processed", "bist100_unified_kline.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "bist_kronos")
CONFIG_PATH = os.path.join(ROOT_DIR, "bist_quant", "bist_train_config.yaml")

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
            "train_ratio": 0.9,
            "val_ratio": 0.1,
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
            "pretrained_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
            "pretrained_predictor": "NeoQuasar/Kronos-base",
            "exp_name": "bist100_kronos_base",
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

def run_training(config_file=CONFIG_PATH, skip_tokenizer=False, skip_basemodel=False):
    """
    Kronos 'train_sequential.py' betiğini, PYTHONPATH değişkenini düzenleyerek başlatır.
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ [HATA] İşlenmiş veri seti bulunamadı: {PROCESSED_DATA_PATH}")
        print("Önce 'python main.py --download-data' veya 'python -m bist_quant.bist_preprocess' çalıştırmalısınız.")
        return False
        
    print(f"🔥 BIST 100 Kronos-base Derin Eğitimi (Fine-Tuning) Başlatılıyor...")
    print(f"🧠 Hedef Model: NeoQuasar/Kronos-base (102.3M Parametre)")
    print(f"🛡️ Bellek Koruma: Batch Size = 2, Gradient Accumulation = 16 (Effective Batch = 32)")
    
    # PYTHONPATH ve UTF-8 ayarla
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{KRONOS_DIR};{FINETUNE_CSV_DIR};" + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    train_script = os.path.join(FINETUNE_CSV_DIR, "train_sequential.py")
    
    cmd = [sys.executable, train_script, "--config", config_file]
    if skip_tokenizer:
        cmd.append("--skip-tokenizer")
    if skip_basemodel:
        cmd.append("--skip-basemodel")
    print(f"💻 Çalıştırılan Komut: {' '.join(cmd)}")
    
    try:
        # Gerçek zamanlı terminal log akışı ile eğitimi başlat
        process = subprocess.Popen(cmd, env=env, cwd=FINETUNE_CSV_DIR)
        process.wait()
        if process.returncode == 0:
            print(f"\n🎉 BIST 100 İnce Ayar Eğitimi Başarıyla Tamamlandı!")
            return True
        else:
            print(f"\n❌ [UYARI] Eğitim komutu hata ile kapandı. Çıkış kodu: {process.returncode}")
            return False
    except KeyboardInterrupt:
        print("\n⏹️ Kullanıcı eğitimi durdurdu. Checkpoint'ler korunuyor.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 100 Kronos-Base Eğitim ve Fine-Tuning Yöneticisi")
    parser.add_argument("--tok-epochs", type=int, default=10, help="Tokenizer eğitim epok sayısı")
    parser.add_argument("--pred-epochs", type=int, default=15, help="Predictor (Base Model) eğitim epok sayısı")
    parser.add_argument("--batch-size", type=int, default=2, help="4GB VRAM için önerilen batch size (2 veya 4)")
    parser.add_argument("--accum", type=int, default=16, help="Gradient accumulation adımı (2 * 16 = 32)")
    
    args = parser.parse_args()
    generate_bist_config(epochs_tokenizer=args.tok_epochs, epochs_predictor=args.pred_epochs, batch_size=args.batch_size, accum_steps=args.accum)
    run_training()
