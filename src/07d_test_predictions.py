"""
Stage 3 - Test-set predictions 
Input : data/processed/train.csv (2023), val.csv (2024), test.csv (2025)
Output: outputs/test_pred_baseline.csv     (LogReg, 13 features)
        outputs/test_pred_rf.csv           (RF depth=8 leaf=20, WITH lat/lon: best on validation)
        outputs/test_pred_rf_nolatlon.csv  (same RF, without lat/lon, for comparison)
        columns: y_true, y_prob, threshold

Run from project root:  python src/07d_test_predictions.py

Procedure (the test set is only PREDICTED here, never used for any decision):
  1. Fit on 2023, choose the decision threshold on 2024 (max F1, shared evaluate.py).
  2. Refit on 2023 + 2024 with the same settings.
  3. Predict 2025 and save probabilities with the validation-selected threshold.

Rows keep the same order as data/processed/test.csv.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from evaluate import find_best_threshold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "results"

RF_DEPTH, RF_LEAF, RF_TREES = 8, 20, 200   # best setting from 07 on validation


def new_logreg():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                     random_state=C.RANDOM_STATE)),
    ])


def new_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=RF_TREES, max_depth=RF_DEPTH, min_samples_leaf=RF_LEAF,
            class_weight="balanced", n_jobs=-1, random_state=C.RANDOM_STATE)),
    ])


def predict_test(file_name, make_model, features, train, val, test):
    assert not set(features) & set(C.DROP_COLS), "Forbidden column in feature list!"

    # Step 1: threshold chosen on validation, from a model fitted on 2023 only
    tuning = make_model().fit(train[features], train[C.TARGET])
    thr, _ = find_best_threshold(val[C.TARGET],
                                 tuning.predict_proba(val[features])[:, 1], mode="f1")

    # Step 2: final model on 2023 + 2024
    final_train = pd.concat([train, val], ignore_index=True)
    final = make_model().fit(final_train[features], final_train[C.TARGET])

    # Step 3: predict test, save in the format Person C asked for
    out = pd.DataFrame({
        "y_true": test[C.TARGET].values,
        "y_prob": final.predict_proba(test[features])[:, 1],
        "threshold": thr,
    })
    out.to_csv(OUT / file_name, index=False)
    print(f"[OK] {file_name}: rows={len(out)}, positives={int(out['y_true'].sum())}, "
          f"threshold={thr:.4f}")


def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    print(f"train={train.shape}, val={val.shape}, test={test.shape}\n")
    OUT.mkdir(parents=True, exist_ok=True)

    predict_test("test_pred_baseline.csv", new_logreg, C.FEATURES_LOGREG, train, val, test)
    predict_test("test_pred_rf.csv", new_rf, C.FEATURES_WITH_LATLON, train, val, test)
    predict_test("test_pred_rf_nolatlon.csv", new_rf, C.FEATURES_NO_LATLON, train, val, test)


if __name__ == "__main__":
    main()