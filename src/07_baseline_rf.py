"""
Aşama 3 - Baseline ve Random Forest (Kişi B)
Girdi : data/processed/train.csv (2023), val.csv (2024)
Çıktı : outputs/results/baseline_rf_val.csv

Çalıştır (proje kökünden):  python src/07_baseline_rf.py

NOT: Test seti bu scriptte KULLANILMAZ. Model seçimi sadece val üzerinde yapılır.
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

# Küçük hiperparametre ızgarası (RF)
RF_GRID = {
    "max_depth": [8, 16],
    "min_samples_leaf": [5, 20],
}
RF_TREES = 200


# ---------- Yardımcılar ----------
def load(name):
    return pd.read_csv(PROCESSED / f"{name}.csv")


def get_xy(df, features):
    # Güvenlik: yasaklı sütun (kimlik, tarih, sızıntı) özellik listesine girmiş mi?
    assert not set(features) & set(C.DROP_COLS), "Yasaklı sütun özellik listesinde!"
    return df[features], df[C.TARGET]


def make_pipeline(model, scale=False):
    """Imputer (medyan) -> [Scaler] -> model. Hepsi SADECE train'den öğrenilir."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def compute_metrics(y_true, proba):
    """
    GEÇİCİ: C'nin evaluate.py'si gelince bu fonksiyonu onunla değiştir.
    Eşik, val üzerinde F1'i (pozitif sınıf) maksimize eden değerdir.
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


# ---------- Ana akış ----------
def main():
    train, val = load("train"), load("val")
    print(f"train={train.shape}, val={val.shape}")
    random_pr_auc = val[C.TARGET].mean()
    print(f"Rastgele tahmin PR-AUC (val) ~ {random_pr_auc:.4f}\n")

    rows = []

    # 1) Lojistik regresyon (basit referans)
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000,
                                random_state=C.RANDOM_STATE)
    rows.append(fit_and_score("LogReg (13 ozellik)", C.FEATURES_LOGREG,
                              logreg, True, train, val))

    # 2) Random Forest: ızgara, seçim val PR-AUC'ye göre
    best = None
    for depth, leaf in itertools.product(RF_GRID["max_depth"], RF_GRID["min_samples_leaf"]):
        name = f"RF depth={depth} leaf={leaf} (latlon yok)"
        row = fit_and_score(name, C.FEATURES_NO_LATLON, new_rf(depth, leaf),
                            False, train, val)
        row["_depth"], row["_leaf"] = depth, leaf
        rows.append(row)
        if best is None or row["pr_auc"] > best["pr_auc"]:
            best = row

    # 3) En iyi RF'yi lat/lon ile dene (konum ezberleme kontrolü)
    name = f"RF depth={best['_depth']} leaf={best['_leaf']} (latlon VAR)"
    rows.append(fit_and_score(name, C.FEATURES_WITH_LATLON,
                              new_rf(best["_depth"], best["_leaf"]),
                              False, train, val))

    result = pd.DataFrame(rows).drop(columns=["_depth", "_leaf"], errors="ignore")
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS / "baseline_rf_val.csv", index=False)
    print("\n" + result.to_string(index=False))
    print(f"\nRastgele tahmin PR-AUC (val) ~ {random_pr_auc:.4f}")
    print("[OK] outputs/results/baseline_rf_val.csv kaydedildi")


if __name__ == "__main__":
    main()