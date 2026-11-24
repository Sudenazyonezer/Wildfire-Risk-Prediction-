import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("cleaned_master_table.csv", parse_dates=["date"])

# 1) Add missing day (r05_c014, 2023-03-11)
cell = "r05_c014"
eksik_tarih = pd.Timestamp("2023-03-11")

sabit_kolonlar = ["cell_id", "lat", "lon", "province", "block_id", "elevation"]
yeni = df.loc[df["cell_id"] == cell, sabit_kolonlar].iloc[0].to_dict()
yeni["date"] = eksik_tarih
yeni["fire_occurrence"] = 0
yeni["fire_count"] = 0
# max_frp remains NaN (no fire)

df = pd.concat([df, pd.DataFrame([yeni])], ignore_index=True)
df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)

# 2) Linearly interpolate weather/soil columns only for that cell
interp_kolonlar = [
    "temp_max", "temp_mean", "rel_humidity_mean", "dew_point_mean",
    "precip_sum", "wind_speed_max", "wind_gust_max", "et0",
    "soil_moisture_surface", "soil_moisture_root", "vpd", "soil_temperature",
]
mask = df["cell_id"] == cell
df.loc[mask, interp_kolonlar] = (
    df.loc[mask, interp_kolonlar].astype(float).interpolate(method="linear")
)
df.loc[mask, "rel_humidity_mean"] = df.loc[mask, "rel_humidity_mean"].round()

# 3) r13_c020 elevation: 0 -> NaN
df.loc[df["cell_id"] == "r13_c020", "elevation"] = np.nan

# 4) Checks
counts = df.groupby("cell_id").size()
print("Rows per cell min/max:", counts.min(), counts.max())
print("Total rows:", len(df))
print("Positive count:", df["fire_occurrence"].sum())
print("Duplicate (cell_id, date):", df.duplicated(["cell_id", "date"]).sum())
print("\nAdded row and its neighbors:")
print(df.loc[mask & df["date"].between("2023-03-10", "2023-03-12"),
             ["date", "temp_max", "precip_sum", "vpd", "soil_moisture_root", "fire_occurrence"]])
print("\nMissing values:")
print(df.isna().sum()[df.isna().sum() > 0])

df.to_csv("cleaned_master_table_v2.csv", index=False)
print("\nSaved: cleaned_master_table_v2.csv")