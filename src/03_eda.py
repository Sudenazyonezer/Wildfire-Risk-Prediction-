import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding="utf-8")

os.makedirs("outputs/eda", exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")

df = pd.read_csv("cleaned_master_table_v2.csv", parse_dates=["date"])
df["ay"] = df["date"].dt.month
df["yil"] = df["date"].dt.year

# 1) Class imbalance
fig, ax = plt.subplots(figsize=(6, 5))
sayilar = df["fire_occurrence"].value_counts().sort_index()
ax.bar(["No fire (0)", "Fire (1)"], sayilar.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(sayilar.values):
    ax.text(i, v, f"{v:,}\n({100*v/len(df):.2f}%)", ha="center", va="bottom", fontsize=12)
ax.set_ylabel("Number of rows")
ax.set_title("Class imbalance")
ax.set_ylim(0, sayilar.max() * 1.15)
plt.tight_layout()
plt.savefig("outputs/eda/01_class_imbalance.png", dpi=200)
plt.close()

# 2) Number of fires by month (grouped by year)
aylik = df.groupby(["yil", "ay"])["fire_occurrence"].sum().unstack(0)
fig, ax = plt.subplots(figsize=(10, 5))
aylik.plot(kind="bar", ax=ax, width=0.8)
ax.set_xlabel("Month")
ax.set_ylabel("Number of fire days (cell-days)")
ax.set_title("Number of fires by month")
ax.set_xticklabels(range(1, 13), rotation=0)
ax.legend(title="Year")
plt.tight_layout()
plt.savefig("outputs/eda/02_monthly_fires.png", dpi=200)
plt.close()

# 3) Top provinces with most fires (top 15)
il = df.groupby("province")["fire_occurrence"].sum().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 7))
il[::-1].plot(kind="barh", ax=ax, color="#DD8452")
ax.set_xlabel("Number of fire days (cell-days)")
ax.set_ylabel("")
ax.set_title("Top 15 provinces with most fires")
plt.tight_layout()
plt.savefig("outputs/eda/03_province_distribution.png", dpi=200)
plt.close()

# 4) Fire vs no fire: box plots
degiskenler = {
    "temp_max": "Max temperature (°C)",
    "rel_humidity_mean": "Mean relative humidity (%)",
    "vpd": "VPD (kPa)",
    "wind_speed_max": "Max wind speed",
}
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, (kol, baslik) in zip(axes, degiskenler.items()):
    sns.boxplot(data=df, x="fire_occurrence", y=kol, ax=ax,
                palette=["#4C72B0", "#C44E52"], showfliers=False)
    ax.set_xticklabels(["No", "Yes"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(baslik, fontsize=13)
plt.suptitle("Comparison of fire and non-fire days", y=1.02)
plt.tight_layout()
plt.savefig("outputs/eda/04_box_plots.png", dpi=200, bbox_inches="tight")
plt.close()

# 5) Correlation matrix (excluding leaky and ID columns)
sayisal = [
    "temp_max", "temp_mean", "rel_humidity_mean", "dew_point_mean", "precip_sum",
    "wind_speed_max", "wind_gust_max", "et0", "soil_moisture_surface",
    "soil_moisture_root", "vpd", "soil_temperature", "elevation", "fire_occurrence",
]
corr = df[sayisal].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 8}, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature correlation matrix")
plt.tight_layout()
plt.savefig("outputs/eda/05_correlation.png", dpi=200)
plt.close()

# 6) Fire rate: month x year heatmap (seasonality summary)
oran = df.groupby(["yil", "ay"])["fire_occurrence"].mean().unstack() * 100
fig, ax = plt.subplots(figsize=(10, 3.5))
sns.heatmap(oran, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "Fire rate (%)"})
ax.set_xlabel("Month")
ax.set_ylabel("Year")
ax.set_title("Monthly fire rate (%)")
plt.tight_layout()
plt.savefig("outputs/eda/06_month_year_rate.png", dpi=200)
plt.close()

# Numerical summaries (for report text)
print("Fire / no-fire means:")
print(df.groupby("fire_occurrence")[list(degiskenler.keys())].mean().round(2))
print("\nFire rate by year (%):")
print((df.groupby("yil")["fire_occurrence"].mean() * 100).round(2))
print("\nNumber of zero or negative rows in soil_moisture_root:",
      (df["soil_moisture_root"] <= 0).sum())
print("soil_moisture_root min/max:",
      df["soil_moisture_root"].min(), df["soil_moisture_root"].max())
print("\nCorrelation with fire_occurrence (top 6):")
print(corr["fire_occurrence"].drop("fire_occurrence").abs().sort_values(ascending=False).head(6).round(3))