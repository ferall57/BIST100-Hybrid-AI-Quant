import os
import glob
import pandas as pd
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT_DIR, "bist_data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "bist_data", "processed")

DAILY_FILE_SUFFIX = "_1d.csv"
SYMBOL_COLUMN = "symbol"
# Endeks serileri hisse mumu değildir (hacim/tutar anlamı farklı); eğitim verisine girmez.
EXCLUDED_SYMBOLS = frozenset({"XU100.IS", "XU030.IS"})
REQUIRED_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]

def preprocess_bist_for_kronos(raw_dir: str = RAW_DIR, output_file: str = None, min_length: int = 100):
    """
    İndirilen günlük BIST (.IS) CSV dosyalarını okur, temizler ve hisse bazlı ince ayar veri kümesinin
    tüketeceği (symbol, timestamps, open, high, low, close, volume, amount) formatında tek bir
    Birleşik BIST Veri Seti olarak kaydeder. Her satır ait olduğu hisseyle etiketlenir; eğitim
    pencereleri bu etikete göre hisse hisse kurulur.
    """
    if output_file is None:
        output_file = os.path.join(PROCESSED_DIR, "bist100_unified_kline.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(raw_dir, f"*{DAILY_FILE_SUFFIX}")))
    if not csv_files:
        print(f"❌ [HATA] '{raw_dir}' klasöründe hiç günlük CSV dosyası bulunamadı! Önce veri indirmelisiniz.")
        return None

    print(f"🔄 BIST 100 Veri Setleri Kronos formatına birleştiriliyor ({len(csv_files)} dosya okundu)...")

    combined_frames = []
    total_candles = 0

    for file_path in tqdm(csv_files, desc="BIST Ön İşleme"):
        symbol = os.path.basename(file_path)[:-len(DAILY_FILE_SUFFIX)]
        if symbol in EXCLUDED_SYMBOLS:
            continue
        try:
            df = pd.read_csv(file_path)
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                tqdm.write(f"[Atlandı] {symbol}: eksik sütun {missing}")
                continue

            df = df[REQUIRED_COLUMNS].dropna()
            df = df[df["close"] > 0]
            if len(df) < min_length:
                tqdm.write(f"[Atlandı] {symbol}: {len(df)} mum (en az {min_length} gerekli)")
                continue

            # Günlük mumda yalnızca takvim günü anlamlıdır; saat dilimi ekleri (yaz/kış saati) atılır
            df = df.assign(timestamps=df["timestamps"].astype(str).str[:10] + " 00:00:00")
            df = df.drop_duplicates(subset="timestamps", keep="last").sort_values("timestamps")
            df.insert(0, SYMBOL_COLUMN, symbol)

            combined_frames.append(df)
            total_candles += len(df)
        except Exception as e:
            tqdm.write(f"[Atlandı] {file_path}: {e}")

    if not combined_frames:
        print("❌ Hiçbir geçerli hisse verisi işlenemedi.")
        return None

    unified_df = pd.concat(combined_frames, ignore_index=True)

    print(f"💾 Birleşik veri seti yazılıyor -> {output_file}")
    unified_df.to_csv(output_file, index=False)

    print(f"\n✅ BIST Ön İşleme Başarıyla Tamamlandı!")
    print(f"📊 İşlenen Sembol Sayısı : {len(combined_frames)}")
    print(f"🕯️ Toplam Mum Sayısı      : {total_candles:,} adet k-line")
    print(f"🎯 Hedef Çıktı Dosyası    : {os.path.abspath(output_file)}")

    return output_file

if __name__ == "__main__":
    preprocess_bist_for_kronos()
