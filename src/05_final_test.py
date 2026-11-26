"""XGBoost final test phase.

04_xgboost.py must be run first (generates xgb_best_config.json).
Plan: the final model is trained on 2023-2024 and EVALUATED on the 2025 test set.
The threshold was selected on val (xgb_best_config.json); it is not re-selected on test.

THE TEST SET IS ONLY EXAMINED IN THIS FILE, AT THE VERY FINAL STEP. Do not change parameters after seeing the results.

Two variants are run for verification:
  A) Model trained in 04 (2023 only) + val threshold        -> threshold is consistent with the model
  B) Model retrained on 2023-2024 + val threshold          -> final model required by the plan
In B, probability scale may shift due to increased data; close results between A and B are a good sign.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config as C
from evaluate import evaluate, results_table

RANDOM_STATE = 42
TARGET = "fire_occurrence"
DROP_COLS = ["cell_id", "block_id", "province", "date", "fire_count", "max_frp"]
ROOT = Path(__file__).resolve().parent.parent  # project root (one level above src/)
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
MODELS = OUT / "models"      # saved models + config
RESULTS = OUT / "results"    # predictions, metric tables, plots
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def load_all():
    # Update to read train/val/test.csv once B's split is ready
    return pd.read_csv(DATA / "features_table.csv", parse_dates=["date"])


def split_xy(df, use_latlon):
    cols = C.FEATURES_WITH_LATLON if use_latlon else C.FEATURES_NO_LATLON
    return df[cols], df[TARGET].astype(int)


def make_model(params, spw):
    return XGBClassifier(
        **params,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def save_preds(name, y, p, thr):
    out = pd.DataFrame({"y_true": y.values, "y_prob": p, "threshold": thr})
    out.to_csv(RESULTS / f"test_pred_{name}.csv", index=False)


def main():
    cfg = json.load(open(MODELS / "xgb_best_config.json"))
    params, thr, use_ll = cfg["params"], cfg["threshold"], cfg["use_latlon"]
    print("Parameters:", params, "| threshold (val):", round(thr, 4), "| lat/lon:", use_ll)

    df = load_all()
    year = df["date"].dt.year
    tr23, tr2324, test = df[year == 2023], df[year.isin([2023, 2024])], df[year == 2025]

    X_te, y_te = split_xy(test, use_ll)
    print(f"test: {len(X_te)} rows, fire rate {y_te.mean():.4f}")

    results = []

    # A) model trained on 2023 only
    X_a, y_a = split_xy(tr23, use_ll)
    spw_a = (y_a == 0).sum() / (y_a == 1).sum()
    m_a = make_model(params, spw_a).fit(X_a, y_a)
    p_a = m_a.predict_proba(X_te)[:, 1]
    results.append(evaluate(y_te, p_a, threshold=thr, name="XGBoost (train 2023)"))

    # B) final model retrained on 2023-2024
    X_b, y_b = split_xy(tr2324, use_ll)
    spw_b = (y_b == 0).sum() / (y_b == 1).sum()
    m_b = make_model(params, spw_b).fit(X_b, y_b)
    p_b = m_b.predict_proba(X_te)[:, 1]
    results.append(evaluate(y_te, p_b, threshold=thr, name="XGBoost (final, 2023-24)"))
    m_b.save_model(str(MODELS / "xgb_final.json"))  # use this for A's SHAP analysis

    print(results_table(results).to_string(index=False))
    print("\nConfusion matrix (final):\n", results[1]["confusion_matrix"])

    # Save final model predictions for the comparison file
    save_preds("xgboost", y_te, p_b, thr)


if __name__ == "__main__":
    main()
