import sys
sys.stdout.reconfigure(encoding="utf-8")

"""
Aşama 1: Ham verinin filtrelenmesi, temizlenmesi ve birleştirilmesi
Girdi : hucreler_hava_durumu_2023_2025.csv, soil_2023_2025.csv, firms_raw_2023_2025.csv
Çıktı : cleaned_master_table.csv, quality_report.txt
Not   : Imputation ve ölçekleme BİLEREK yapılmaz (train/test ayrımından sonra, sızıntıyı önlemek için).
"""
import pandas as pd
import numpy as np

# =============== AYARLAR (sadece burası) ===============
DATA_DIR     = "."   # CSV'lerin olduğu klasör
WEATHER_FILE = f"{DATA_DIR}/hucreler_hava_durumu_2023_2025.csv"
SOIL_FILE    = f"{DATA_DIR}/soil_2023_2025.csv"
FIRMS_FILE   = f"{DATA_DIR}/firms_raw_2023_2025.csv"
OUT_TABLE    = f"{DATA_DIR}/cleaned_master_table.csv"
OUT_REPORT   = f"{DATA_DIR}/quality_report.txt"
GRID         = 0.25
START, END   = "2023-01-01", "2025-12-31"
EXPECTED_DAYS = (pd.Timestamp(END) - pd.Timestamp(START)).days + 1  # 1096

report = []
def log(msg=""):
    print(msg)
    report.append(str(msg))

# =============== 1) HAVA DURUMU ===============
weather = pd.read_csv(WEATHER_FILE, parse_dates=["tarih"]).rename(columns={
    "tarih": "date", "sicaklik_maks": "temp_max", "sicaklik_ort": "temp_mean",
    "bagil_nem_ort": "rel_humidity_mean", "ciy_noktasi_ort": "dew_point_mean",
    "yagis_toplam": "precip_sum", "ruzgar_hizi_maks": "wind_speed_max",
    "ruzgar_hamlesi_maks": "wind_gust_max", "buharlasma_et0": "et0",
    "yuzey_toprak_nemi": "soil_moisture_surface",
    "kok_bolgesi_nem": "soil_moisture_root", "kuruluk_stresi_vpd": "vpd",
    "hucre_id": "cell_id", "enlem": "lat", "boylam": "lon",
    "il": "province", "blok_id": "block_id"})
log(f"[HAVA] ham satır: {len(weather)}")

weather = weather[weather["date"].between(START, END)]
n = len(weather)
weather = weather.drop_duplicates(subset=["cell_id", "date"])
log(f"[HAVA] mükerrer (cell_id, date) silindi: {n - len(weather)}")
weather[["lat", "lon"]] = weather[["lat", "lon"]].round(2)

# hücre merkezleri 0.25'in katı mı?
cells = weather[["cell_id", "lat", "lon"]].drop_duplicates()
off_grid = cells[((cells["lat"] / GRID).round() * GRID - cells["lat"]).abs().round(4).gt(0)
                 | ((cells["lon"] / GRID).round() * GRID - cells["lon"]).abs().round(4).gt(0)]
log(f"[HAVA] hücre sayısı: {cells['cell_id'].nunique()} | 0.25° ızgarasına uymayan: {len(off_grid)}")

# =============== 2) SOIL ===============
soil = (pd.read_csv(SOIL_FILE, parse_dates=["date"])
          .rename(columns={"point_id": "cell_id"}))
log(f"[SOIL] ham satır: {len(soil)}")
soil = soil[soil["date"].between(START, END)]
n = len(soil)
soil = soil.drop_duplicates(subset=["cell_id", "date"])
log(f"[SOIL] mükerrer (cell_id, date) silindi: {n - len(soil)}")
# Soil dosyasından yalnızca hava dosyasında olmayan sütunlar alınır.
# (et0, nem, vpd iki dosyada farklı katman/toplulaştırma ile var -> hava dosyası kullanılır)
soil[["soil_temperature", "elevation"]] = soil[["soil_temperature", "elevation"]].round(4)
soil = soil[["cell_id", "date", "soil_temperature", "elevation"]]

# =============== 3) FIRMS (filtre) ===============
fire = pd.read_csv(FIRMS_FILE, parse_dates=["acq_date"])
log(f"[FIRMS] ham satır: {len(fire)}")

fire["type"] = pd.to_numeric(fire["type"], errors="coerce")
fire["confidence"] = fire["confidence"].astype(str).str.strip().str.lower()

# Filtreleri sırayla uygula ve her adımda kaç satır gittiğini yaz
before = len(fire)
fire = fire[fire["type"] == 0]
log(f"[FIRMS] type != 0 atıldı: {before - len(fire)}")

before = len(fire)
fire = fire[fire["confidence"].isin(["n", "h"])]
log(f"[FIRMS] confidence 'l' (veya bilinmeyen) atıldı: {before - len(fire)}")

before = len(fire)
fire = fire[fire["acq_date"].between(START, END)]
log(f"[FIRMS] tarih aralığı dışı atıldı: {before - len(fire)}")

before = len(fire)
fire = fire.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
log(f"[FIRMS] mükerrer atıldı: {before - len(fire)}")
log(f"[FIRMS] kalan satır: {len(fire)}")

# En yakın 0.25° hücre merkezine ata
fire["lat"] = ((fire["latitude"] / GRID).round() * GRID).round(2)
fire["lon"] = ((fire["longitude"] / GRID).round() * GRID).round(2)
fire = fire.rename(columns={"acq_date": "date"})

# (hücre, gün) seviyesine indir
fire_cells = (fire.groupby(["lat", "lon", "date"])
                  .agg(fire_count=("frp", "size"), max_frp=("frp", "max"))
                  .reset_index())
fire_cells["fire_occurrence"] = 1
log(f"[FIRMS] yangın olan benzersiz (hücre, gün): {len(fire_cells)}")

# =============== 4) BİRLEŞTİR ===============
df = weather.merge(soil, on=["cell_id", "date"], how="inner", validate="one_to_one")
log(f"[MERGE] hava + soil sonrası satır: {len(df)} (hava: {len(weather)})")

df = df.merge(fire_cells[["lat", "lon", "date", "fire_occurrence", "fire_count", "max_frp"]],
              on=["lat", "lon", "date"], how="left", validate="many_to_one")
df["fire_occurrence"] = df["fire_occurrence"].fillna(0).astype(int)
df["fire_count"] = df["fire_count"].fillna(0).astype(int)

matched = fire_cells.merge(df[["lat", "lon", "date"]].drop_duplicates(),
                           on=["lat", "lon", "date"], how="inner")
log(f"[MERGE] yangın hücre-günü: {len(fire_cells)} | çalışma alanıyla eşleşen: {len(matched)} "
    f"| alan dışı kalan: {len(fire_cells) - len(matched)}")

# =============== 5) FİZİKSEL KONTROL ===============
# Kesin hatalar: silinir
hard = {
    "nem 0-100 dışı":       ~df["rel_humidity_mean"].between(0, 100) & df["rel_humidity_mean"].notna(),
    "yağış negatif":        df["precip_sum"] < 0,
    "rüzgar negatif":       (df["wind_speed_max"] < 0) | (df["wind_gust_max"] < 0),
    "et0 negatif":          df["et0"] < 0,
    "toprak nemi 0-1 dışı": (~df["soil_moisture_surface"].between(0, 1) & df["soil_moisture_surface"].notna())
                            | (~df["soil_moisture_root"].between(0, 1) & df["soil_moisture_root"].notna()),
    "vpd negatif":          df["vpd"] < 0,
}
drop_mask = pd.Series(False, index=df.index)
log("\n[KONTROL] Kesin hatalar (silinir):")
for name, mask in hard.items():
    log(f"  {name}: {int(mask.sum())}")
    drop_mask |= mask.fillna(False)
df = df[~drop_mask]
log(f"  toplam silinen satır: {int(drop_mask.sum())}")

# Şüpheli: silinmez, sadece raporlanır (inceleme gerekir)
soft = {
    "temp_max < temp_mean": df["temp_max"] < df["temp_mean"],
    "gust < wind":          df["wind_gust_max"] < df["wind_speed_max"],
    "dew_point > temp_mean": df["dew_point_mean"] > df["temp_mean"],
}
log("\n[KONTROL] Şüpheli (silinmez, incelenmeli):")
for name, mask in soft.items():
    log(f"  {name}: {int(mask.sum())}")

# =============== 6) EKSİK GÜN / EKSİK DEĞER RAPORU ===============
counts = df.groupby("cell_id")["date"].nunique()
log(f"\n[EKSİK GÜN] {EXPECTED_DAYS} günden eksik hücre sayısı: {int((counts < EXPECTED_DAYS).sum())} "
    f"/ {len(counts)}")
if (counts < EXPECTED_DAYS).any():
    log("  en eksik 5 hücre:\n" + counts.sort_values().head(5).to_string())

log("\n[NaN ORANI] (doldurma yapılmadı)")
log(df.isna().mean().round(4).sort_values(ascending=False).head(12).to_string())

log("\n[EKSTREM DEĞERLER] (silinmez; gerçek mi hata mı bak)")
num_cols = ["temp_max", "rel_humidity_mean", "precip_sum", "wind_speed_max",
            "wind_gust_max", "et0", "vpd", "soil_temperature", "elevation"]
log(df[num_cols].describe().loc[["min", "50%", "max"]].round(2).to_string())

# =============== 7) ÖZET + KAYDET ===============
log(f"\n[ÖZET] son satır: {len(df)} | hücre: {df['cell_id'].nunique()} | "
    f"fire=1: {int(df['fire_occurrence'].sum())} | fire oranı: {df['fire_occurrence'].mean():.5f}")

df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)
df.to_csv(OUT_TABLE, index=False)
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\nKaydedildi: {OUT_TABLE} ve {OUT_REPORT}")