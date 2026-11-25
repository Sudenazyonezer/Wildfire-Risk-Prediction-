"""
Stage 3 - Time-based split (Person B)
Input  : data/processed/features_table.csv
Output : data/processed/train.csv (2023), val.csv (2024), test.csv (2025)

Run (from project root): python src/06_split_preprocess.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TARGET = "fire_occurrence"

# Joint decision: 2023 train / 2024 val for tuning, 2023-24 train / 2025 test for final
SPLIT_YEARS = {
    "train": [2023],
    "val": [2024],
    "test": [2025],
}


def load():
    df = pd.read_csv(PROCESSED / "features_table.csv", parse_dates=["date"])
    print(f"Loaded: {df.shape}")
    return df


def split_by_year(df):
    years = df["date"].dt.year
    return {name: df[years.isin(ys)].copy() for name, ys in SPLIT_YEARS.items()}


def check_splits(splits, full_len):
    # 1) Are any rows missing?
    total = sum(len(s) for s in splits.values())
    assert total == full_len, f"Row loss: {total} != {full_len}"

    # 2) Is the chronological order correct? (train < val < test, no overlap)
    assert splits["train"]["date"].max() < splits["val"]["date"].min()
    assert splits["val"]["date"].max() < splits["test"]["date"].min()

    # 3) Do all sets contain the same cells? (required for time-based split)
    cells = {n: set(s["cell_id"]) for n, s in splits.items()}
    assert cells["train"] == cells["val"] == cells["test"], "Cell sets do not match!"
    print("[OK] No row loss, dates do not overlap, cell sets are identical.")


def report(splits):
    rows = []
    for name, s in splits.items():
        rows.append({
            "set": name,
            "rows": len(s),
            "positive": int(s[TARGET].sum()),
            "positive_rate_%": round(100 * s[TARGET].mean(), 2),
            "start_date": s["date"].min().date(),
            "end_date": s["date"].max().date(),
        })
    print("\n" + pd.DataFrame(rows).to_string(index=False))


def main():
    df = load()
    splits = split_by_year(df)
    check_splits(splits, len(df))
    report(splits)

    for name, s in splits.items():
        s.to_csv(PROCESSED / f"{name}.csv", index=False)
        print(f"[OK] {name}.csv saved")


if __name__ == "__main__":
    main()