import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("cleaned_master_table_v2.csv", parse_dates=["date"])
df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)

g = df.groupby("cell_id", sort=False)
yeni_kolonlar = []

# 1) Rolling features (including today, last N days)
rolling_tanim = {
    "temp_mean": ["mean"],
    "precip_sum": ["sum"],
    "rel_humidity_mean": ["mean"],
    "vpd": ["mean"],
    "soil_moisture_root": ["mean"],
}
for kol, fonksiyonlar in rolling_tanim.items():
    for n in [3, 7, 14, 30]:
        for f in fonksiyonlar:
            ad = f"{kol}_roll{n}_{f}"
            df[ad] = g[kol].transform(lambda s: s.rolling(n, min_periods=n).agg(f))
            yeni_kolonlar.append(ad)

# 2) Lag features
lag_kolonlar = ["vpd", "temp_max", "rel_humidity_mean", "precip_sum", "soil_moisture_root"]
for kol in lag_kolonlar:
    for n in [1, 3, 7]:
        ad = f"{kol}_lag{n}"
        df[ad] = g[kol].shift(n)
        yeni_kolonlar.append(ad)

# 3) Days since last rain (precipitation threshold: >= 1 mm)
def kurak_gun_sayisi(s, esik=1.0):
    sayac, cikti = 0, []
    for v in s.values:
        sayac = 0 if v >= esik else sayac + 1
        cikti.append(sayac)
    return pd.Series(cikti, index=s.index)

df["days_since_rain"] = g["precip_sum"].transform(kurak_gun_sayisi)
yeni_kolonlar.append("days_since_rain")

# 4) Time: month and day of year sin/cos
df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)
yeni_kolonlar += ["month_sin", "month_cos", "doy_sin", "doy_cos"]

# Checks
print("Shape:", df.shape)
print("Number of new columns:", len(yeni_kolonlar))
print("Positive count:", df["fire_occurrence"].sum())
print("\nNaN count in new columns (expected for initial days):")
print(df[yeni_kolonlar].isna().sum()[df[yeni_kolonlar].isna().sum() > 0].to_string())
print("\ndays_since_rain summary:")
print(df["days_since_rain"].describe().round(1))

# Leakage/ordering verification: lag must be NaN on the first row of a cell
ilk = df.groupby("cell_id").head(1)
print("\nNaN rate of vpd_lag1 on the first day of each cell:", ilk["vpd_lag1"].isna().mean())

df.to_csv("features_table.csv", index=False)

with open("outputs/new_features.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(yeni_kolonlar))
print("\nSaved: features_table.csv and outputs/new_features.txt")