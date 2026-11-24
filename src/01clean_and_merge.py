import sys
sys.stdout.reconfigure(encoding="utf-8")

"""
Stage 1: Filtering, cleaning, and merging raw data
Input : hucreler_hava_durumu_2023_2025.csv, soil_2023_2025.csv, firms_raw_2023_2025.csv
Output: cleaned_master_table.csv, quality_report.txt
Note  : Imputation and scaling are INTENTIONALLY omitted (to be done after train/test split to prevent data leakage).
"""
import pandas as pd
import numpy as np

# =============== SETTINGS (only edit here) ===============
DATA_DIR     = "."   # Directory containing the CSV files
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

# =============== 1) WEATHER ===============
weather = pd.read_csv(WEATHER_FILE, parse_dates=["tarih"]).rename(columns={
    "tarih": "date", "sicaklik_maks": "temp_max", "sicaklik_ort": "temp_mean",
    "bagil_nem_ort": "rel_humidity_mean", "ciy_noktasi_ort": "dew_point_mean",
    "yagis_toplam": "precip_sum", "ruzgar_hizi_maks": "wind_speed_max",
    "ruzgar_hamlesi_maks": "wind_gust_max", "buharlasma_et0": "et0",
    "yuzey_toprak_nemi": "soil_moisture_surface",
    "kok_bolgesi_nem": "soil_moisture_root", "kuruluk_stresi_vpd": "vpd",
    "hucre_id": "cell_id", "enlem": "lat", "boylam": "lon",
    "il": "province", "blok_id": "block_id"})
log(f"[WEATHER] raw rows: {len(weather)}")

weather = weather[weather["date"].between(START, END)]
n = len(weather)
weather = weather.drop_duplicates(subset=["cell_id", "date"])
log(f"[WEATHER] duplicate (cell_id, date) dropped: {n - len(weather)}")
weather[["lat", "lon"]] = weather[["lat", "lon"]].round(2)

# are cell centers multiples of 0.25?
cells = weather[["cell_id", "lat", "lon"]].drop_duplicates()
off_grid = cells[((cells["lat"] / GRID).round() * GRID - cells["lat"]).abs().round(4).gt(0)
                 | ((cells["lon"] / GRID).round() * GRID - cells["lon"]).abs().round(4).gt(0)]
log(f"[WEATHER] number of cells: {cells['cell_id'].nunique()} | off 0.25° grid: {len(off_grid)}")

# =============== 2) SOIL ===============
soil = (pd.read_csv(SOIL_FILE, parse_dates=["date"])
          .rename(columns={"point_id": "cell_id"}))
log(f"[SOIL] raw rows: {len(soil)}")
soil = soil[soil["date"].between(START, END)]
n = len(soil)
soil = soil.drop_duplicates(subset=["cell_id", "date"])
log(f"[SOIL] duplicate (cell_id, date) dropped: {n - len(soil)}")
# Only columns not present in the weather file are taken from the soil file.
# (et0, humidity, vpd exist in both files with different layers/aggregations -> weather file is used)
soil[["soil_temperature", "elevation"]] = soil[["soil_temperature", "elevation"]].round(4)
soil = soil[["cell_id", "date", "soil_temperature", "elevation"]]

# =============== 3) FIRMS (filter) ===============
fire = pd.read_csv(FIRMS_FILE, parse_dates=["acq_date"])
log(f"[FIRMS] raw rows: {len(fire)}")

fire["type"] = pd.to_numeric(fire["type"], errors="coerce")
fire["confidence"] = fire["confidence"].astype(str).str.strip().str.lower()

# Apply filters sequentially and log how many rows are dropped at each step
before = len(fire)
fire = fire[fire["type"] == 0]
log(f"[FIRMS] type != 0 dropped: {before - len(fire)}")

before = len(fire)
fire = fire[fire["confidence"].isin(["n", "h"])]
log(f"[FIRMS] confidence 'l' (or unknown) dropped: {before - len(fire)}")

before = len(fire)
fire = fire[fire["acq_date"].between(START, END)]
log(f"[FIRMS] out of date range dropped: {before - len(fire)}")

before = len(fire)
fire = fire.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
log(f"[FIRMS] duplicates dropped: {before - len(fire)}")
log(f"[FIRMS] remaining rows: {len(fire)}")

# Assign to nearest 0.25° cell center
fire["lat"] = ((fire["latitude"] / GRID).round() * GRID).round(2)
fire["lon"] = ((fire["longitude"] / GRID).round() * GRID).round(2)
fire = fire.rename(columns={"acq_date": "date"})

# Aggregate to (cell, day) level
fire_cells = (fire.groupby(["lat", "lon", "date"])
                  .agg(fire_count=("frp", "size"), max_frp=("frp", "max"))
                  .reset_index())
fire_cells["fire_occurrence"] = 1
log(f"[FIRMS] unique (cell, day) with fire: {len(fire_cells)}")

# =============== 4) MERGE ===============
df = weather.merge(soil, on=["cell_id", "date"], how="inner", validate="one_to_one")
log(f"[MERGE] rows after weather + soil: {len(df)} (weather: {len(weather)})")

df = df.merge(fire_cells[["lat", "lon", "date", "fire_occurrence", "fire_count", "max_frp"]],
              on=["lat", "lon", "date"], how="left", validate="many_to_one")
df["fire_occurrence"] = df["fire_occurrence"].fillna(0).astype(int)
df["fire_count"] = df["fire_count"].fillna(0).astype(int)

matched = fire_cells.merge(df[["lat", "lon", "date"]].drop_duplicates(),
                           on=["lat", "lon", "date"], how="inner")
log(f"[MERGE] fire cell-days: {len(fire_cells)} | matched with study area: {len(matched)} "
    f"| out of study area: {len(fire_cells) - len(matched)}")

# =============== 5) PHYSICAL CHECKS ===============
# Hard errors: will be dropped
hard = {
    "humidity out of 0-100":      ~df["rel_humidity_mean"].between(0, 100) & df["rel_humidity_mean"].notna(),
    "negative precipitation":     df["precip_sum"] < 0,
    "negative wind":              (df["wind_speed_max"] < 0) | (df["wind_gust_max"] < 0),
    "negative et0":               df["et0"] < 0,
    "soil moisture out of 0-1":   (~df["soil_moisture_surface"].between(0, 1) & df["soil_moisture_surface"].notna())
                                  | (~df["soil_moisture_root"].between(0, 1) & df["soil_moisture_root"].notna()),
    "negative vpd":               df["vpd"] < 0,
}
drop_mask = pd.Series(False, index=df.index)
log("\n[CHECK] Hard errors (will be dropped):")
for name, mask in hard.items():
    log(f"  {name}: {int(mask.sum())}")
    drop_mask |= mask.fillna(False)
df = df[~drop_mask]
log(f"  total dropped rows: {int(drop_mask.sum())}")

SEA_CELLS = ["BURAYA_HUCRE_ID"]   # cell_id obtained from the check script
n = len(df)
df = df[~df["cell_id"].isin(SEA_CELLS)]
log(f"\n[SEA] elevation=0 cell removed: {n - len(df)} rows")

# Suspicious: not dropped, only reported (requires inspection)
soft = {
    "temp_max < temp_mean": df["temp_max"] < df["temp_mean"],
    "gust < wind":          df["wind_gust_max"] < df["wind_speed_max"],
    "dew_point > temp_mean": df["dew_point_mean"] > df["temp_mean"],
}
log("\n[CHECK] Suspicious (not dropped, needs inspection):")
for name, mask in soft.items():
    log(f"  {name}: {int(mask.sum())}")

# =============== 6) MISSING DAY / MISSING VALUE REPORT ===============
counts = df.groupby("cell_id")["date"].nunique()
log(f"\n[MISSING DAY] cells with fewer than {EXPECTED_DAYS} days: {int((counts < EXPECTED_DAYS).sum())} "
    f"/ {len(counts)}")
if (counts < EXPECTED_DAYS).any():
    log("  top 5 cells with most missing days:\n" + counts.sort_values().head(5).to_string())

log("\n[NaN RATE] (no imputation applied)")
log(df.isna().mean().round(4).sort_values(ascending=False).head(12).to_string())

log("\n[EXTREME VALUES] (not dropped; check if real or error)")
num_cols = ["temp_max", "rel_humidity_mean", "precip_sum", "wind_speed_max",
            "wind_gust_max", "et0", "vpd", "soil_temperature", "elevation"]
log(df[num_cols].describe().loc[["min", "50%", "max"]].round(2).to_string())

# =============== 7) SUMMARY + SAVE ===============
log(f"\n[SUMMARY] final rows: {len(df)} | cells: {df['cell_id'].nunique()} | "
    f"fire=1: {int(df['fire_occurrence'].sum())} | fire rate: {df['fire_occurrence'].mean():.5f}")

df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)
df.to_csv(OUT_TABLE, index=False)
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\nSaved: {OUT_TABLE} and {OUT_REPORT}")