"""XGBoost final test aşaması (Kişi C).

Önce 04_xgboost.py çalışmış olmalı (xgb_best_config.json üretir).
Plan: son model 2023-2024 ile eğitilir, 2025 test üzerinde DEĞERLENDİRİLİR.
Eşik val'de seçilmişti (xgb_best_config.json), test'te yeniden seçilmez.

TEST SETİNE SADECE BU DOSYADA, EN SON ADIMDA BAKILIR. Sonuca bakıp parametre değiştirme.

Kontrol için iki varyant çalışır:
  A) 04'te eğitilen model (sadece 2023) + val eşiği      -> eşik ile model tutarlı
  B) 2023-2024 ile yeniden eğitilen model + val eşiği    -> planın istediği final model
B'de veri arttığı için olasılık ölçeği kayabilir; A ile B'nin yakın çıkması iyi işarettir.
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
ROOT = Path(__file__).resolve().parent.parent  # proje kökü (src/ bir üstü)
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_all():
    # B'nin ayrımı hazır olunca train/val/test.csv okuyacak şekilde değiştir
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
    out.to_csv(OUT / f"test_pred_{name}.csv", index=False)


def main():
    cfg = json.load(open(OUT / "xgb_best_config.json"))
    params, thr, use_ll = cfg["params"], cfg["threshold"], cfg["use_latlon"]
    print("Parametreler:", params, "| eşik (val):", round(thr, 4), "| lat/lon:", use_ll)

    df = load_all()
    year = df["date"].dt.year
    tr23, tr2324, test = df[year == 2023], df[year.isin([2023, 2024])], df[year == 2025]

    X_te, y_te = split_xy(test, use_ll)
    print(f"test: {len(X_te)} satır, yangın oranı {y_te.mean():.4f}")

    results = []

    # A) sadece 2023 ile eğitilmiş model
    X_a, y_a = split_xy(tr23, use_ll)
    spw_a = (y_a == 0).sum() / (y_a == 1).sum()
    m_a = make_model(params, spw_a).fit(X_a, y_a)
    p_a = m_a.predict_proba(X_te)[:, 1]
    results.append(evaluate(y_te, p_a, threshold=thr, name="XGBoost (train 2023)"))

    # B) 2023-2024 ile yeniden eğitilmiş final model
    X_b, y_b = split_xy(tr2324, use_ll)
    spw_b = (y_b == 0).sum() / (y_b == 1).sum()
    m_b = make_model(params, spw_b).fit(X_b, y_b)
    p_b = m_b.predict_proba(X_te)[:, 1]
    results.append(evaluate(y_te, p_b, threshold=thr, name="XGBoost (final, 2023-24)"))
    m_b.save_model(str(OUT / "xgb_final.json"))  # A'nın SHAP analizi için bunu kullan

    print(results_table(results).to_string(index=False))
    print("\nKarışıklık matrisi (final):\n", results[1]["confusion_matrix"])

    # Karşılaştırma dosyası için final modelin tahminlerini kaydet
    save_preds("xgboost", y_te, p_b, thr)


if __name__ == "__main__":
    main()