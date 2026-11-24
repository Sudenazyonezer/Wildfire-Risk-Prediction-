"""
Aşama 3 - Zaman bazlı ayrım (Kişi B)
Girdi : data/processed/features_table.csv
Çıktı : data/processed/train.csv (2023), val.csv (2024), test.csv (2025)

Çalıştır (proje kökünden):  python src/06_split_preprocess.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TARGET = "fire_occurrence"

# Ortak karar: ayarlama için 2023 train / 2024 val, final için 2023-24 train / 2025 test
SPLIT_YEARS = {
    "train": [2023],
    "val": [2024],
    "test": [2025],
}


def load():
    df = pd.read_csv(PROCESSED / "features_table.csv", parse_dates=["date"])
    print(f"Yüklendi: {df.shape}")
    return df


def split_by_year(df):
    years = df["date"].dt.year
    return {name: df[years.isin(ys)].copy() for name, ys in SPLIT_YEARS.items()}


def check_splits(splits, full_len):
    # 1) Hiçbir satır kaybolmadı mı?
    total = sum(len(s) for s in splits.values())
    assert total == full_len, f"Satır kaybı: {total} != {full_len}"

    # 2) Zaman sırası doğru mu? (train < val < test, çakışma yok)
    assert splits["train"]["date"].max() < splits["val"]["date"].min()
    assert splits["val"]["date"].max() < splits["test"]["date"].min()

    # 3) Her sette aynı hücreler var mı? (zaman bazlı ayrımda olmalı)
    cells = {n: set(s["cell_id"]) for n, s in splits.items()}
    assert cells["train"] == cells["val"] == cells["test"], "Hücre kümeleri farklı!"
    print("[OK] Satır kaybı yok, tarihler çakışmıyor, hücre kümeleri aynı.")


def report(splits):
    rows = []
    for name, s in splits.items():
        rows.append({
            "set": name,
            "satir": len(s),
            "pozitif": int(s[TARGET].sum()),
            "pozitif_oran_%": round(100 * s[TARGET].mean(), 2),
            "baslangic": s["date"].min().date(),
            "bitis": s["date"].max().date(),
        })
    print("\n" + pd.DataFrame(rows).to_string(index=False))


def main():
    df = load()
    splits = split_by_year(df)
    check_splits(splits, len(df))
    report(splits)

    for name, s in splits.items():
        s.to_csv(PROCESSED / f"{name}.csv", index=False)
        print(f"[OK] {name}.csv kaydedildi")


if __name__ == "__main__":
    main()