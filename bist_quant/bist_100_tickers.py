# BIST 100 (Borsa İstanbul 100) Güncel ve Lokomotif Hisse Listesi
# Yahoo Finance üzerinden veri çekerken '.IS' uzatısı gerekmektedir.

BIST_100_TICKERS = [
    "ACSEL.IS", "ADEL.IS", "ADESE.IS", "AEFES.IS", "AGHOL.IS", "AGROT.IS", "AHGAZ.IS", "AKBNK.IS",
    "AKCNS.IS", "AKFYE.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALBRK.IS", "ALFAS.IS", "ANSGR.IS",
    "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "ASUZU.IS", "BIMAS.IS", "BIOEN.IS", "BOBET.IS", "BRSAN.IS",
    "BRYAT.IS", "BUCIM.IS", "CANTE.IS", "CCOLA.IS", "CIMSA.IS", "CWENE.IS", "DOAS.IS", "DOHOL.IS",
    "ECILC.IS", "ECZYT.IS", "EGEEN.IS", "EKGYO.IS", "ENJSA.IS", "ENKAI.IS", "ERCB.IS", "EREGL.IS",
    "EUREN.IS", "EUPWR.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS", "GLYHO.IS", "GUBRF.IS",
    "HALKB.IS", "HEKTS.IS", "IMASM.IS", "IPEKE.IS", "ISCTR.IS", "ISDMR.IS", "ISFIN.IS", "ISGYO.IS",
    "ISMEN.IS", "IZMDC.IS", "KARSN.IS", "KAYSE.IS", "KCHOL.IS", "KMPUR.IS", "KNYAS.IS",
    "KONTR.IS", "KONYA.IS", "KORDS.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "KZBGY.IS", "MAVI.IS",
    "MGROS.IS", "MIATK.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PENGD.IS", "PENTA.IS", "PETKM.IS",
    "PGSUS.IS", "PRKME.IS", "QUAGR.IS", "REEDR.IS", "SAHOL.IS", "SASA.IS", "SAYAS.IS", "SISE.IS",
    "SKBNK.IS", "SNGYO.IS", "SOKM.IS", "TABGD.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS",
    "TMSN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", "TURSG.IS",
    "ULKER.IS", "VAKBN.IS", "VAKKO.IS", "VESBE.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "YYLGD.IS",
    "ZOREN.IS"
]

# Temizlenmiş tam liste (hatalı veya tekrarlı kayıtlar hariç)
BIST_100_TICKERS = sorted(list(set([t for t in BIST_100_TICKERS if t.endswith(".IS")])))

# En yüksek hacimli BIST 30 Hisseleri (Hızlı doğrulama ve denemeler için)
BIST_30_TICKERS = sorted(list(set([
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "DOAS.IS", "DOHOL.IS",
    "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HALKB.IS",
    "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS",
    "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS",
    "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS"
])))

def get_tickers(mode="bist100"):
    if mode.lower() == "bist30":
        return BIST_30_TICKERS
    return BIST_100_TICKERS

if __name__ == "__main__":
    print(f"BIST 100 Hisse Sayısı: {len(get_tickers('bist100'))}")
    print(f"BIST 30 Hisse Sayısı: {len(get_tickers('bist30'))}")
