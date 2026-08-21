import os
import sys
import time
import logging
from typing import List, Any, Optional
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

# LangChain Google GenAI desteği
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("[UYARI] 'langchain_google_genai' bulunamadı. Lütfen 'pip install langchain-google-genai' çalıştırın.")

logger = logging.getLogger("GeminiRotator")

class GeminiRotator:
    """
    Kullanıcının elindeki 3 veya daha fazla Gemini API anahtarını yöneten akıllı rotasyon motoru.
    Herhangi bir anahtarda kota sorunu (HTTP 429, ResourceExhausted, Rate Limit) meydana gelirse
    hiçbir kesinti yaşatmadan bir sonraki yedek anahtara geçiş yapar (Failover & Rotation).
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.2, max_retries: int = 5):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env_file = os.path.join(root_dir, ".env")
        load_dotenv(dotenv_path=env_file, override=True)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.api_keys = self._load_api_keys()
        
        if not self.api_keys:
            raise ValueError(
                "❌ [HATA] .env dosyasında hiçbir Gemini API anahtarı bulunamadı!\n"
                "Lütfen .env dosyanıza GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3 değerlerini girin."
            )
            
        self.current_index = 0
        self.active_client = self._create_client(self.api_keys[self.current_index])
        print(f"🔑 Gemini Akıllı Rotasyon Motoru Devrede! (Toplan {len(self.api_keys)} API Anahtarı Yüklendi | İlk Aktif: #{self.current_index+1})")

    def _load_api_keys(self) -> List[str]:
        keys = []
        # Öncelik: GOOGLE_API_KEY_1, _2, _3 vb.
        for i in range(1, 10):
            k = os.getenv(f"GOOGLE_API_KEY_{i}") or os.getenv(f"GEMINI_API_KEY_{i}")
            if k and k.strip() and not k.strip().startswith("BURAYA_API_KEY") and k.strip() not in keys:
                keys.append(k.strip())
        # Eğer numaralı yoksa tekli GOOGLE_API_KEY ya da GEMINI_API_KEY bak
        if not keys:
            single = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if single and single.strip() and not single.strip().startswith("BURAYA_API_KEY"):
                keys.append(single.strip())
        return keys

    def _create_client(self, api_key: str):
        if not LANGCHAIN_AVAILABLE:
            return None
        # Ortam değişkenini de güncelle
        os.environ["GOOGLE_API_KEY"] = api_key
        # Gemini Modeli Yeniden Başlat
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=api_key,
            max_retries=1 # Kendi rotasyonumuz yönecek
        )

    def rotate_key(self, error_msg: str = ""):
        """Bir sonraki API anahtarına kesintisiz atlama gerçekleştirir."""
        old_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_index]
        self.active_client = self._create_client(new_key)
        print(f"\n⚡ [API ROTASYON] Anahtar #{old_index+1}'de sınır uyarısı algılandı -> Anahtar #{self.current_index+1} aktif edildi!")
        if error_msg:
            logger.debug(f"Rotasyon Gerekçesi: {error_msg}")

    def invoke(self, messages: Any, **kwargs) -> Any:
        """
        LangChain standardı invoke arayüzünün dayanıklı (fault-tolerant) versiyonu.
        429 veya ResourceExhaustions hatalarında diğer API key ile sorguyu otomatik tekrarlar.
        """
        if not self.active_client:
            raise RuntimeError("LangChain ChatGoogleGenerativeAI istemcisi oluşturulamadı.")
            
        attempts = 0
        while attempts < (len(self.api_keys) * 2):
            try:
                response = self.active_client.invoke(messages, **kwargs)
                return response
            except Exception as e:
                err_str = str(e).lower()
                attempts += 1
                # Kota, Rate Limit, ResourceExhausted, 429 veya 503 Sunucu Yoğunluğu kontrolü
                is_transient = any(x in err_str for x in ["429", "503", "quota", "exhausted", "rate limit", "permission", "limit", "unavailable", "high demand", "overloaded"])
                if is_transient:
                    print(f"⚠️ API Kota / Sunucu Yoğunluk Uyarısı ({e}). Başka API anahtarına deneniyor... (Deneme: {attempts}/{len(self.api_keys)*2})")
                    self.rotate_key(str(e))
                    time.sleep(2)  # Sunucunun toparlanması için kısa bekleme
                else:
                    if attempts >= (len(self.api_keys) * 2):
                        raise e
                    print(f"⚠️ Geçici Bağlantı Hatası: {e} -> Diğer API anahtarında tekrar deneniyor...")
                    self.rotate_key(str(e))
                    time.sleep(1)
        
        raise RuntimeError("❌ Tüm Gemini API anahtarları denendi fakat kota veya bağlantı sınırı aşılamadı.")

if __name__ == "__main__":
    # Kısa Doğrulama Testi
    try:
        rotator = GeminiRotator(model_name="gemini-2.5-flash", temperature=0.1)
        res = rotator.invoke("Merhaba Gemini, 3'lü API rotasyon altyapımızın çalıştığını test ediyorum. Bana 1 cümlelik yanıt ver.")
        print("🤖 Gemini Cevabı:", res.content)
    except Exception as ex:
        print("Test Sonucu:", ex)
