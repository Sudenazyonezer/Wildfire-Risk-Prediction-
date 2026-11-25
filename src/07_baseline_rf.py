"""
Stage 3 - Baseline and Random Forest (Person B)
Input  : data/processed/train.csv (2023), val.csv (2024)
Output : outputs/results/baseline_rf_val.csv

Run (from project root): python src/07_baseline_rf.py

NOTE: The test set is NOT USED in this script. Model selection is performed solely on val.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs" / "results"

# Small hyperparameter grid (RF)
RF_GRID = {
    "max_depth": [8, 16],
    "min_samples_leaf": [5, 20],
}
RF_TREES = 200


# ---------- Helpers ----------
def load(name):
    return pd.read_csv(PROCESSED / f"{name}.csv")


def get_xy(df, features):
    # Safety check: has any forbidden column (id, date, leakage) entered the feature list?
    assert not set(features) & set(C.DROP_COLS), "Forbidden column in feature list!"
    return df[features], df[C.TARGET]


def make_pipeline(model, scale=False):
    """Imputer (median) -> [Scaler] -> model. All fitted ONLY on train."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def compute_metrics(y_true, proba):
    """
    TEMPORARY: Replace this function once C's evaluate.py is ready.
    Threshold is the value that maximizes F1 (positive class) on val.
    """
    pr_auc = average_precision_score(y_true, proba)
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    threshold = t[np.argmax(f1[:-1])]
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": round(pr_auc, 4),
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, pred, average="macro"), 4),
    }


def fit_and_score(name, features, model, scale, train, val):
    X_tr, y_tr = get_xy(train, features)
    X_va, y_va = get_xy(val, features)
    pipe = make_pipeline(model, scale=scale)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_va)[:, 1]
    row = {"model": name, "n_features": len(features)}
    row.update(compute_metrics(y_va, proba))
    print(f"[OK] {name}: PR-AUC={row['pr_auc']}")
    return row


def new_rf(depth, leaf):
    return RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=depth, min_samples_leaf=leaf,
        class_weight="balanced", n_jobs=-1, random_state=C.RANDOM_STATE,
    )


# ---------- Main flow ----------
def main():
    train, val = load("train"), load("val")
    print(f"train={train.shape}, val={val.shape}")
    random_pr_auc = val[C.TARGET].mean()
    print(f"Random guess PR-AUC (val) ~ {random_pr_auc:.4f}\n")

    rows = []

    # 1) Logistic regression (simple baseline)
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000,
                                random_state=C.RANDOM_STATE)
    rows.append(fit_and_score("LogReg (13 features)", C.FEATURES_LOGREG,
                              logreg, True, train, val))

    # 2) Random Forest: grid search, selection based on val PR-AUC
    best = None
    for depth, leaf in itertools.product(RF_GRID["max_depth"], RF_GRID["min_samples_leaf"]):
        name = f"RF depth={depth} leaf={leaf} (no latlon)"
        row = fit_and_score(name, C.FEATURES_NO_LATLON, new_rf(depth, leaf),
                            False, train, val)
        row["_depth"], row["_leaf"] = depth, leaf
        rows.append(row)
        if best is None or row["pr_auc"] > best["pr_auc"]:
            best = row

    # 3) Try best RF with lat/lon (location memorization check)
    name = f"RF depth={best['_depth']} leaf={best['_leaf']} (WITH latlon)"
    rows.append(fit_and_score(name, C.FEATURES_WITH_LATLON,
                              new_rf(best["_depth"], best["_leaf"]),
                              False, train, val))

    result = pd.DataFrame(rows).drop(columns=["_depth", "_leaf"], errors="ignore")
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS / "baseline_rf_val.csv", index=False)
    print("\n" + result.to_string(index=False))
    print(f"\nRandom guess PR-AUC (val) ~ {random_pr_auc:.4f}")
    print("[OK] outputs/results/baseline_rf_val.csv saved")


if __name__ == "__main__":
    main()