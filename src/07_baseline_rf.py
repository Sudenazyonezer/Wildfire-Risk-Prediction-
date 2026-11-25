"""
Stage 3 - Baseline and Random Forest (Person B)
Input : data/processed/train.csv (2023), val.csv (2024)
Output: outputs/results/baseline_rf_val.csv

Run from project root:  python src/07_baseline_rf.py

NOTE: The test set is NOT used in this script. Model selection is done on validation only.
"""
import itertools
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from evaluate import evaluate, find_best_threshold, results_table

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
    # Safety check: identifier / date / leakage columns must never be features
    assert not set(features) & set(C.DROP_COLS), "Forbidden column in feature list!"
    return df[features], df[C.TARGET]


def make_pipeline(model, scale=False):
    """Imputer (median) -> [Scaler] -> model. All statistics are learned on TRAIN only."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def fit_and_score(name, features, model, scale, train, val):
    X_tr, y_tr = get_xy(train, features)
    X_va, y_va = get_xy(val, features)
    pipe = make_pipeline(model, scale=scale)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_va)[:, 1]

    # Threshold is selected on validation (shared function from evaluate.py)
    thr, _ = find_best_threshold(y_va, proba, mode="f1")
    res = evaluate(y_va, proba, threshold=thr, name=name)
    res["n_features"] = len(features)
    print(f"[OK] {name}: PR-AUC={res['pr_auc']:.4f}")
    return res


def new_rf(depth, leaf):
    return RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=depth, min_samples_leaf=leaf,
        class_weight="balanced", n_jobs=-1, random_state=C.RANDOM_STATE,
    )


# ---------- Main ----------
def main():
    train, val = load("train"), load("val")
    print(f"train={train.shape}, val={val.shape}")
    random_pr_auc = val[C.TARGET].mean()
    print(f"Random-guess PR-AUC (val) ~ {random_pr_auc:.4f}\n")

    rows = []

    # 1) Logistic regression (simple reference)
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000,
                                random_state=C.RANDOM_STATE)
    rows.append(fit_and_score("LogReg (13 features)", C.FEATURES_LOGREG,
                              logreg, True, train, val))

    # 2) Random Forest: grid search, selection by validation PR-AUC
    best = None
    for depth, leaf in itertools.product(RF_GRID["max_depth"], RF_GRID["min_samples_leaf"]):
        name = f"RF depth={depth} leaf={leaf} (no latlon)"
        res = fit_and_score(name, C.FEATURES_NO_LATLON, new_rf(depth, leaf),
                            False, train, val)
        res["_depth"], res["_leaf"] = depth, leaf
        rows.append(res)
        if best is None or res["pr_auc"] > best["pr_auc"]:
            best = res

    # 3) Best RF with lat/lon (location memorisation check)
    name = f"RF depth={best['_depth']} leaf={best['_leaf']} (WITH latlon)"
    rows.append(fit_and_score(name, C.FEATURES_WITH_LATLON,
                              new_rf(best["_depth"], best["_leaf"]),
                              False, train, val))

    table = results_table(rows)
    table.insert(1, "n_features", [r["n_features"] for r in rows])
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "baseline_rf_val.csv", index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nRandom-guess PR-AUC (val) ~ {random_pr_auc:.4f}")
    print("[OK] outputs/results/baseline_rf_val.csv saved")


if __name__ == "__main__":
    main()