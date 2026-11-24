import pandas as pd
import config as C
import sys
sys.stdout.reconfigure(encoding="utf-8")

cols = pd.read_csv(C.FEATURES_FILE, nrows=5).columns.tolist()

for ad in ["FEATURES_NO_LATLON", "FEATURES_WITH_LATLON", "FEATURES_LOGREG"]:
    liste = getattr(C, ad)
    eksik = [c for c in liste if c not in cols]
    sizinti = [c for c in liste if c in C.DROP_COLS or c == C.TARGET]
    tekrar = len(liste) - len(set(liste))
    print(f"{ad}: {len(liste)} features | missing from table: {eksik} | forbidden: {sizinti} | duplicates: {tekrar}")

with open("outputs/new_features.txt", encoding="utf-8") as f:
    dosyadaki = set(f.read().split())
print("\nconfig.ENGINEERED == new_features.txt:", set(C.ENGINEERED) == dosyadaki)
print("Number of new features:", len(C.ENGINEERED))