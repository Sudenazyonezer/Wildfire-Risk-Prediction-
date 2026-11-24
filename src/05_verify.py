import pandas as pd
import os

yol = "features_table.csv"
print("Dosya boyutu (MB):", round(os.path.getsize(yol) / 1e6, 1))

df = pd.read_csv(yol, parse_dates=["date"])
print("Şekil:", df.shape)
print("Pozitif sayısı:", df["fire_occurrence"].sum())
print("Son 3 sütun:", df.columns[-3:].tolist())
print(df[["cell_id", "date", "vpd", "vpd_lag1", "vpd_roll7_mean", "days_since_rain"]].head(10))