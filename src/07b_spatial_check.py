"""
Stage 3 - Spatial block check (Person B)
Question: Is the model memorizing location?
Input   : data/processed/train.csv (2023), val.csv (2024)
Output  : outputs/results/spatial_check_val.csv

Run (from project root): python src/07b_spatial_check.py

NOTE: The test set is not used. RF setting is the best one selected on val in script 07.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

import config as C

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs" / "results"

N_FOLDS = 5
RF_DEPTH, RF_LEAF, RF_TREES = 8, 20, 200   # Best setting selected on val in 07


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=RF_TREES, max_depth=RF_DEPTH, min_samples_leaf=RF_LEAF,
            class_weight="balanced", n_jobs=-1, random_state=C.RANDOM_STATE)),
    ])


def make_block_folds(blocks):
    """Shuffle blocks with a fixed seed and split into N_FOLDS groups."""
    blocks = np.array(sorted(blocks))
    rng = np.random.default_rng(C.RANDOM_STATE)
    rng.shuffle(blocks)
    return np.array_split(blocks, N_FOLDS)


def run(name, features, train, val, folds):
    assert not set(features) & set(C.DROP_COLS), "Forbidden column in feature list!"

    # "Seen": single model trained on all blocks
    full = make_pipeline().fit(train[features], train[C.TARGET])
    proba_all = full.predict_proba(val[features])[:, 1]

    rows = []
    for i, fold_blocks in enumerate(folds, start=1):
        mask = val["block_id"].isin(fold_blocks).values
        y = val.loc[mask, C.TARGET]
        if y.sum() == 0:
            print(f"  fold {i}: no positives, skipped")
            continue

        seen = average_precision_score(y, proba_all[mask])

        # "Unseen": exclude these blocks from training
        tr_f = train[~train["block_id"].isin(fold_blocks)]
        m = make_pipeline().fit(tr_f[features], tr_f[C.TARGET])
        unseen = average_precision_score(y, m.predict_proba(val.loc[mask, features])[:, 1])

        print(f"  fold {i}: {len(fold_blocks)} blocks, {int(y.sum())} positives | "
              f"seen={seen:.4f}  unseen={unseen:.4f}")
        rows.append({"features": name, "fold": i, "n_blocks": len(fold_blocks),
                     "n_pos": int(y.sum()), "pr_auc_seen": seen, "pr_auc_unseen": unseen})
    return pd.DataFrame(rows)


def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    assert set(train["block_id"]) == set(val["block_id"]), "Block sets do not match!"

    folds = make_block_folds(train["block_id"].unique())
    print(f"{train['block_id'].nunique()} blocks, split into {N_FOLDS} groups\n")

    results = []
    for name, feats in [("NO latlon", C.FEATURES_NO_LATLON),
                        ("WITH latlon", C.FEATURES_WITH_LATLON)]:
        print(f"[{name}]")
        results.append(run(name, feats, train, val, folds))

    res = pd.concat(results, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "spatial_check_val.csv", index=False)

    summary = res.groupby("features")[["pr_auc_seen", "pr_auc_unseen"]].mean()
    summary["drop"] = summary["pr_auc_seen"] - summary["pr_auc_unseen"]
    summary["drop_%"] = 100 * summary["drop"] / summary["pr_auc_seen"]
    print("\nMean (across folds):")
    print(summary.round(4).to_string())
    print("\n[OK] outputs/results/spatial_check_val.csv saved")


if __name__ == "__main__":
    main()