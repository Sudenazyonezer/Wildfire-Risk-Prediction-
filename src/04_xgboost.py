"""XGBoost skeleton .
Data: data/processed/features_table.csv (produced by A, derived from cleaned_master_table_v2).
Feature list comes from config.py. Split: 2023 train, 2024 val (time-based).
Once B's train.csv / val.csv are ready, update load_data(); the rest stays the same.
The test set is NEVER used here (tuning and threshold selection are done on val).
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config as C
from evaluate import evaluate, find_best_threshold

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent  # project root (one level above src/)
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
MODELS = OUT / "models"      # saved models + config
RESULTS = OUT / "results"    # predictions, metric tables, plots
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
USE_LATLON = True  # run twice with True/False and compare

# Columns to exclude from the model (per plan rules)
DROP_COLS = ["cell_id", "block_id", "province", "date", "fire_count", "max_frp"]
TARGET = "fire_occurrence"


def load_data():
    # Table produced by A (derived from v2, including elevation fix)
    df = pd.read_csv(DATA / "features_table.csv", parse_dates=["date"])
    train = df[df["date"].dt.year == 2023]
    val = df[df["date"].dt.year == 2024]
    return train, val


def split_xy(df):
    # Feature list from A's config.py: leakage columns are already excluded
    cols = C.FEATURES_WITH_LATLON if USE_LATLON else C.FEATURES_NO_LATLON
    return df[cols], df[TARGET].astype(int)


def main():
    train, val = load_data()
    X_tr, y_tr = split_xy(train)
    X_va, y_va = split_xy(val)

    print(f"train: {len(X_tr)} rows, fire rate {y_tr.mean():.4f}")
    print(f"val:   {len(X_va)} rows, fire rate {y_va.mean():.4f}")
    print(f"num features: {X_tr.shape[1]}  (lat/lon: {USE_LATLON})")

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()  # scale_pos_weight
    print(f"scale_pos_weight: {spw:.1f}")

    # Small grid (for initial run); expand depending on runtime
    # 8 combinations: depth 7 was consistently poor in the first grid, subsample difference was noise
    grid = {
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 5],
        "n_estimators": [300, 600],
        "subsample": [0.8],
    }
    keys = list(grid)
    results = []
    best = None

    for values in itertools.product(*grid.values()):
        params = dict(zip(keys, values))
        model = XGBClassifier(
            **params,
            scale_pos_weight=spw,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        res = evaluate(y_va, p_va, threshold=0.5)  # model selection is threshold-independent: PR-AUC
        results.append({**params, "val_pr_auc": res["pr_auc"]})
        print(params, f"val PR-AUC={res['pr_auc']:.4f}")
        if best is None or res["pr_auc"] > best["pr_auc"]:
            best = {"pr_auc": res["pr_auc"], "params": params, "model": model, "p_va": p_va}

    pd.DataFrame(results).sort_values("val_pr_auc", ascending=False).to_csv(
        RESULTS / "xgb_tuning.csv", index=False
    )

    # Select threshold on val
    thr, f1 = find_best_threshold(y_va, best["p_va"], mode="f1")
    final_val = evaluate(y_va, best["p_va"], threshold=thr, name="XGBoost (val)")
    print("\nBest parameters:", best["params"])
    print(f"Selected threshold (val, F1): {thr:.4f}")
    print({k: round(v, 4) for k, v in final_val.items()
           if k in ("pr_auc", "precision", "recall", "f1_macro", "f1_weighted")})
    print(final_val["confusion_matrix"])

    # Save for next steps
    best["model"].save_model(str(MODELS / "xgb_best.json"))
    with open(MODELS / "xgb_best_config.json", "w") as f:
        json.dump({"params": best["params"], "threshold": thr,
                   "scale_pos_weight": spw, "use_latlon": USE_LATLON}, f, indent=2)


if __name__ == "__main__":
    main()
