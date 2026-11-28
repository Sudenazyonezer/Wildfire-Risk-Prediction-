"""XGBoost lat/lon ablation (Person C).

Goal: quantify the contribution of lat/lon on the test set (for the report).
Same hyperparameters (xgb_best_config.json), same procedure as the main model:
  1) Train on 2023 -> choose the threshold on 2024 validation (max F1)
  2) Retrain on 2023-2024 -> apply the threshold to 2025 test
The only difference: feature set is C.FEATURES_NO_LATLON.

This is an ABLATION, not model selection: the final model stays with lat/lon.
No hyperparameter tuning here. Existing files are not overwritten.
Outputs:
  outputs/results/test_pred_xgboost_nolatlon.csv   (picked up by 06_compare.py)
  outputs/results/latlon_ablation_test.csv
  outputs/models/xgb_final_nolatlon.json
"""
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

import config as C
from evaluate import evaluate, find_best_threshold, results_table

RANDOM_STATE = 42
TARGET = "fire_occurrence"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"
RESULTS = ROOT / "outputs" / "results"


def make_model(params, spw):
    return XGBClassifier(
        **params,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def fit(train, cols, params):
    y = train[TARGET].astype(int)
    spw = (y == 0).sum() / (y == 1).sum()
    return make_model(params, spw).fit(train[cols], y)


def main():
    params = json.load(open(MODELS / "xgb_best_config.json"))["params"]
    print("Parameters (taken from the lat/lon model, unchanged):", params)

    df = pd.read_csv(DATA / "features_table.csv", parse_dates=["date"])
    year = df["date"].dt.year
    tr23, val = df[year == 2023], df[year == 2024]
    tr2324, test = df[year.isin([2023, 2024])], df[year == 2025]
    cols = C.FEATURES_NO_LATLON
    print(f"number of features: {len(cols)} (no lat/lon)")

    # 1) Threshold selection on validation (model trained on 2023 only)
    m_val = fit(tr23, cols, params)
    y_val = val[TARGET].astype(int)
    p_val = m_val.predict_proba(val[cols])[:, 1]
    thr, _ = find_best_threshold(y_val, p_val, mode="f1")
    val_res = evaluate(y_val, p_val, threshold=thr, name="XGBoost no lat/lon (val)")
    print(f"val PR-AUC: {val_res['pr_auc']:.4f} | selected threshold: {thr:.4f}")

    # 2) Final: train on 2023-24, apply the val threshold to test
    m_fin = fit(tr2324, cols, params)
    y_te = test[TARGET].astype(int)
    p_te = m_fin.predict_proba(test[cols])[:, 1]
    res_no = evaluate(y_te, p_te, threshold=thr, name="XGBoost no lat/lon")
    m_fin.save_model(str(MODELS / "xgb_final_nolatlon.json"))
    pd.DataFrame({"y_true": y_te.values, "y_prob": p_te, "threshold": thr}).to_csv(
        RESULTS / "test_pred_xgboost_nolatlon.csv", index=False
    )

    # 3) Side by side with the lat/lon final model (from saved predictions)
    ll = pd.read_csv(RESULTS / "test_pred_xgboost.csv")
    assert (ll["y_true"].values == y_te.values).all(), "test row order differs!"
    res_ll = evaluate(ll["y_true"], ll["y_prob"], threshold=ll["threshold"].iloc[0],
                      name="XGBoost with lat/lon")

    table = results_table([res_ll, res_no])
    base = y_te.mean()
    table.insert(3, "lift_vs_random", (table["pr_auc"] / base).round(2))
    table["f1_fire"] = (2 * table["precision"] * table["recall"]
                        / (table["precision"] + table["recall"])).round(4)
    print(f"\nTest (random PR-AUC = {base:.4f}):")
    print(table.to_string(index=False))
    table.to_csv(RESULTS / "latlon_ablation_test.csv", index=False)

    d = res_ll["pr_auc"] - res_no["pr_auc"]
    print(f"\nLat/lon contribution (test PR-AUC difference): {d:+.4f} "
          f"({d / res_no['pr_auc']:+.1%} relative to the no-lat/lon model)")


if __name__ == "__main__":
    main()