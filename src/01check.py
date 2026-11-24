import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")



df = pd.read_csv("cleaned_master_table.csv", parse_dates=["date"])

# 1) En yüksek rüzgâr hamleleri
print(df.nlargest(5, "wind_gust_max")[
    ["cell_id", "date", "province", "elevation", "wind_speed_max", "wind_gust_max"]])

# 2) Deniz seviyesi (elevation=0) hücreleri
sea = df[df["elevation"] == 0]
print("elevation=0 hücre:", sea["cell_id"].nunique(),
      "| satır:", len(sea),
      "| fire=1:", sea["fire_occurrence"].sum())

# 3) Deniz hücresi detayı
sea = df[df["elevation"] == 0]
print(sea[["cell_id", "lat", "lon", "province"]].drop_duplicates())
print(sea[sea["fire_occurrence"] == 1][["date", "fire_count", "max_frp"]])