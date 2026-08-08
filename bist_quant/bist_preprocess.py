import os
import glob
import pandas as pd
from tqdm import tqdm

RAW_DIR = os.path.join("bist_data", "raw")
PROCESSED_DIR = os.path.join("bist_data", "processed")

def preprocess_bist_for_kronos(raw_dir: str = RAW_DIR, output_file: str = None, min_length: int = 100):
    """
    Tüm indirilen BIST (.IS) CSV dosyalarını okur, temizler ve Kronos modelinin 
    'finetune_csv' boru hattının tüketebileceği standart (timestamps, open, high, low, close, volume, amount) 
    formatında tek bir dev Birleşik BIST Veri Seti olarak birleştirip kaydeder.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if output_file is None:
        output_file = os.path.join(PROCESSED_DIR, "bist100_unified_kline.csv")
        
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        print(f"❌ [HATA] '{raw_dir}' klasöründe hiç CSV dosyası bulunamadı! Önce veri indirmelisiniz.")
        return None
        
    print(f"🔄 BIST 100 Veri Setleri Kronos formatına birleştiriliyor ({len(csv_files)} dosya okundu)...")
    
    combined_frames = []
    total_candles = 0
    valid_symbols = 0
    
    required_cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    
    for file_path in tqdm(csv_files, desc="BIST Ön İşleme"):
        try:
            df = pd.read_csv(file_path)
            # Sütunları denetle
            missing = [c for c in required_cols if c not in df.columns]
            if missing or len(df) < min_length:
                continue
                
            # Sadece gerekli sütunları sıralı tutalım
            df = df[required_cols].copy()
            df = df.dropna()
            df = df[df["close"] > 0]
            
            # Zaman damgasını string veya dt tutabiliriz
            df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            
            combined_frames.append(df)
            total_candles += len(df)
            valid_symbols += 1
        except Exception as e:
            tqdm.write(f"[Atlandı] {file_path}: {e}")
            
    if not combined_frames:
        print("❌ Hiçbir geçerli hisse verisi işlenemedi.")
        return None
        
    # Tüm veriyi birleştir
    unified_df = pd.concat(combined_frames, ignore_index=True)
    
    # Dosyaya kaydet
    print(f"💾 Birleşik veri seti yazılıyor -> {output_file}")
    unified_df.to_csv(output_file, index=False)
    
    print(f"\n✅ BIST Ön İşleme Başarıyla Tamamlandı!")
    print(f"📊 İşlenen Sembol Sayısı : {valid_symbols}")
    print(f"🕯️ Toplam Mum Sayısı      : {total_candles:,} adet k-line")
    print(f"🎯 Hedef Çıktı Dosyası    : {os.path.abspath(output_file)}")
    
    return output_file

if __name__ == "__main__":
    preprocess_bist_for_kronos()
