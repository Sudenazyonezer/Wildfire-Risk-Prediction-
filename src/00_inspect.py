import pandas as pd
import sys
sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("cleaned_master_table.csv", parse_dates=["date"])

print("Shape:", df.shape)
print("\nColumns and types:")
print(df.dtypes)

print("\nNumber of cells:", df["cell_id"].nunique())
print("Date range:", df["date"].min(), "->", df["date"].max())
print("Number of days:", df["date"].nunique())

print("\nPositive count:", df["fire_occurrence"].sum())
print("Positive rate: %.2f%%" % (100 * df["fire_occurrence"].mean()))

print("\nMissing values:")
print(df.isna().sum()[df.isna().sum() > 0])

print("\nRows per cell (min/max):")
counts = df.groupby("cell_id").size()
print(counts.min(), counts.max())

print("\nDuplicate (cell_id, date):", df.duplicated(["cell_id", "date"]).sum())

print("\nr13_c020 elevation:")
print(df.loc[df["cell_id"] == "r13_c020", "elevation"].describe())

print("\nProvince distribution:")
print(df.groupby("province")["fire_occurrence"].agg(["size", "sum"]))