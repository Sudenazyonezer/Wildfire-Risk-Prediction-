"""
Stage 3 - Save final baseline and Random Forest models (Person B)
Input : data/processed/train.csv (2023), val.csv (2024)
Output: outputs/models/logreg_final.joblib
        outputs/models/rf_final.joblib
        outputs/models/rf_latlon_final.joblib
        outputs/models/model_info.json

Run from project root:  python src/07e_save_models.py

These are the SAME final models used in 07d_test_predictions.py
(fitted on 2023 + 2024, same settings, same random_state), so SHAP explains the
exact models that were evaluated on the test set. The test set is not used here.
"""
import json
from pathlib import Path

import joblib
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
MODELS = ROOT / "outputs" / "models"

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


def save_model(name, make_model, features, train, val, final_train):
    assert not set(features) & set(C.DROP_COLS), "Forbidden column in feature list!"

    # Threshold: model fitted on 2023, chosen on 2024 (same as 07d)
    tuning = make_model().fit(train[features], train[C.TARGET])
    thr, _ = find_best_threshold(val[C.TARGET],
                                 tuning.predict_proba(val[features])[:, 1], mode="f1")

    # Final model: 2023 + 2024
    final = make_model().fit(final_train[features], final_train[C.TARGET])
    path = MODELS / f"{name}_final.joblib"
    joblib.dump(final, path)
    size_mb = path.stat().st_size / 1e6
    print(f"[OK] {path.name}: {size_mb:.1f} MB, threshold={thr:.4f}, features={len(features)}")
    return {"file": path.name, "threshold": round(thr, 4), "features": features}


def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    final_train = pd.concat([train, val], ignore_index=True)
    print(f"train={train.shape}, val={val.shape}, final_train={final_train.shape}\n")

    MODELS.mkdir(parents=True, exist_ok=True)
    info = {
        "logreg": save_model("logreg", new_logreg, C.FEATURES_LOGREG, train, val, final_train),
        "rf": save_model("rf", new_rf, C.FEATURES_NO_LATLON, train, val, final_train),
        "rf_latlon": save_model("rf_latlon", new_rf, C.FEATURES_WITH_LATLON, train, val, final_train),
    }
    with open(MODELS / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("[OK] model_info.json saved")


if __name__ == "__main__":
    main()