"""
Stage 3 - Class imbalance strategy comparison (Person B)
Input : data/processed/train.csv (2023), val.csv (2024)
Output: outputs/results/imbalance_comparison_val.csv

Run from project root:  python src/07c_imbalance.py

Undersampling is applied to TRAIN only. Validation is always the full, untouched set.
The test set is NOT used here.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import config as C
from evaluate import evaluate, find_best_threshold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs" / "results"

RF_DEPTH, RF_LEAF, RF_TREES = 8, 20, 200   # best setting from 07 on validation
SEEDS = [42, 43, 44]                        # undersampling is random -> average over seeds


def make_pipeline(class_weight, seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=RF_TREES, max_depth=RF_DEPTH, min_samples_leaf=RF_LEAF,
            class_weight=class_weight, n_jobs=-1, random_state=seed)),
    ])


def undersample(train, neg_per_pos, seed):
    """Keep all positives, sample neg_per_pos negatives per positive."""
    pos = train[train[C.TARGET] == 1]
    neg = train[train[C.TARGET] == 0]
    n_neg = min(len(neg), neg_per_pos * len(pos))
    neg_sampled = neg.sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg_sampled]).sample(frac=1, random_state=seed)


def run_strategy(name, train, val, features, neg_per_pos, class_weight, seeds):
    X_va, y_va = val[features], val[C.TARGET]
    runs = []
    for seed in seeds:
        tr = train if neg_per_pos is None else undersample(train, neg_per_pos, seed)
        pipe = make_pipeline(class_weight, seed)
        pipe.fit(tr[features], tr[C.TARGET])
        proba = pipe.predict_proba(X_va)[:, 1]
        thr, _ = find_best_threshold(y_va, proba, mode="f1")   # shared function
        res = evaluate(y_va, proba, threshold=thr, name=name)
        res.pop("confusion_matrix")                            # keep numeric columns only
        runs.append(res)
    df = pd.DataFrame(runs)
    row = {
        "strategy": name,
        "train_rows": len(tr),
        "train_pos_rate_%": round(100 * tr[C.TARGET].mean(), 2),
        "pr_auc_mean": df["pr_auc"].mean(),
        "pr_auc_std": df["pr_auc"].std(ddof=0),
        "precision": df["precision"].mean(),
        "recall": df["recall"].mean(),
        "f1_macro": df["f1_macro"].mean(),
        "f1_weighted": df["f1_weighted"].mean(),
    }
    print(f"[OK] {name}: PR-AUC={row['pr_auc_mean']:.4f} (+/- {row['pr_auc_std']:.4f})")
    return row


def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    features = C.FEATURES_NO_LATLON
    assert not set(features) & set(C.DROP_COLS), "Forbidden column in feature list!"
    print(f"train={train.shape}, val={val.shape}")
    print(f"Random-guess PR-AUC (val) ~ {val[C.TARGET].mean():.4f}\n")

    rows = [
        # class_weight on full data: one run is enough (std not defined -> reported as 0)
        run_strategy("class_weight=balanced (full data)", train, val, features,
                     None, "balanced", [C.RANDOM_STATE]),
        run_strategy("undersample 1:5 (no class weight)", train, val, features,
                     5, None, SEEDS),
        run_strategy("undersample 1:10 (no class weight)", train, val, features,
                     10, None, SEEDS),
    ]

    result = pd.DataFrame(rows).round(4)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS / "imbalance_comparison_val.csv", index=False)
    print("\n" + result.to_string(index=False))
    print("\n[OK] outputs/results/imbalance_comparison_val.csv saved")


if __name__ == "__main__":
    main()