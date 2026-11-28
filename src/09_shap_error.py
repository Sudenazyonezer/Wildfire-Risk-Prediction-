import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "outputs" / "models" / "xgb_final.json"
PRED_PATH = ROOT / "outputs" / "results" / "test_pred_xgboost.csv"

# features_table is too large for Git, so it may be stored as .csv or .zip
candidates = [
    ROOT / "data" / "processed" / "features_table.csv",
    ROOT / "data" / "processed" / "features_table.zip",
    ROOT.parent / "data" / "processed" / "features_table.csv",
    ROOT.parent / "features_table.csv",
]
FEATURES_PATH = next((p for p in candidates if p.exists()), None)
if FEATURES_PATH is None:
    print("features_table not found. Looked in:")
    for p in candidates:
        print("  ", p)
    sys.exit("STOP: put features_table.csv or .zip into data/processed/ and run again.")
print("Using features file:", FEATURES_PATH)

OUT = ROOT / "outputs" / "shap"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- settings
SEED = 42
N_SAMPLE = 20000
TOP_N = 20

# ---------------------------------------------------------------- load model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
feature_names = model.get_booster().feature_names
print("Model features:", len(feature_names))

# ---------------------------------------------------------------- load 2025 test rows
df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
test = df[df["date"] >= "2025-01-01"].sort_values(["cell_id", "date"]).reset_index(drop=True)
del df

preds = pd.read_csv(PRED_PATH)
threshold = float(preds["threshold"].iloc[0])
print("Test rows (features_table):", len(test), "| prediction rows:", len(preds))
print("Threshold:", round(threshold, 4))

# ---------------------------------------------------------------- alignment checks
if len(test) != len(preds):
    sys.exit("STOP: row counts differ, cannot align predictions with features.")
if not np.array_equal(test["fire_occurrence"].values, preds["y_true"].values):
    sys.exit("STOP: y_true does not match fire_occurrence in this row order.")

X = test[feature_names]
recomputed = model.predict_proba(X)[:, 1]
close = np.isclose(recomputed, preds["y_prob"].values, atol=1e-4)
print(f"Alignment OK. Recomputed probabilities match saved ones for {close.mean():.2%} of rows.")
if close.mean() < 0.99:
    print("WARNING: less than 99% match; the final model may have used different preprocessing.")

test["y_true"] = preds["y_true"].values
test["y_prob"] = preds["y_prob"].values
test["y_pred"] = (test["y_prob"] >= threshold).astype(int)
test["month"] = test["date"].dt.month


def outcome(row_true, row_pred):
    if row_true == 1 and row_pred == 1:
        return "TP"
    if row_true == 1 and row_pred == 0:
        return "FN"
    if row_true == 0 and row_pred == 1:
        return "FP"
    return "TN"


test["outcome"] = [outcome(a, b) for a, b in zip(test["y_true"], test["y_pred"])]
print("\nOutcome counts:\n", test["outcome"].value_counts().to_string())

# ================================================================ SHAP
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(X), size=min(N_SAMPLE, len(X)), replace=False)
X_s = X.iloc[sample_idx]

explainer = shap.TreeExplainer(model)
sv_s = explainer.shap_values(X_s)  # log-odds contributions

# 1) beeswarm (global, random sample)
plt.figure()
shap.summary_plot(sv_s, X_s, max_display=TOP_N, show=False)
plt.title("SHAP summary (2025 test sample, 20k rows)")
plt.tight_layout()
plt.savefig(OUT / "shap_beeswarm_global.png", dpi=200, bbox_inches="tight")
plt.close()

# 2) mean |SHAP| bar chart + CSV for all features
imp = pd.Series(np.abs(sv_s).mean(axis=0), index=feature_names).sort_values(ascending=False)
imp.to_csv(OUT / "shap_importance_all_features.csv", header=["mean_abs_shap"])
top = imp.head(TOP_N)[::-1]
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(top.index, top.values, color="#DD8452")
ax.set_xlabel("Mean |SHAP value| (log-odds)")
ax.set_title(f"Top {TOP_N} features by mean |SHAP|")
plt.tight_layout()
plt.savefig(OUT / "shap_importance_bar.png", dpi=200)
plt.close()
print("\nTop 10 features by mean |SHAP|:\n", imp.head(10).round(4).to_string())


# 3) grouped importance (correlated features share importance, so group them)
def group_of(name):
    if name in ("lat", "lon", "elevation"):
        return "Location & terrain"
    if name.startswith(("month_", "doy_")):
        return "Seasonality"
    if name == "days_since_rain" or name.startswith("precip_sum"):
        return "Precipitation / dryness"
    if name.startswith("soil_moisture"):
        return "Soil moisture"
    if name.startswith("wind"):
        return "Wind"
    if name.startswith(("vpd", "rel_humidity", "dew_point", "et0")):
        return "Humidity / VPD / ET0"
    if name.startswith("temp_") or name == "soil_temperature":
        return "Temperature"
    return "Other"


groups = {}
for i, name in enumerate(feature_names):
    groups.setdefault(group_of(name), []).append(i)

group_imp = pd.Series(
    {g: np.abs(sv_s[:, idx].sum(axis=1)).mean() for g, idx in groups.items()}
).sort_values()
group_imp.sort_values(ascending=False).to_csv(
    OUT / "shap_importance_by_group.csv", header=["mean_abs_group_shap"]
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(group_imp.index, group_imp.values, color="#4C72B0")
ax.set_xlabel("Mean |summed SHAP| per feature group (log-odds)")
ax.set_title("Grouped SHAP importance")
plt.tight_layout()
plt.savefig(OUT / "shap_importance_by_group.png", dpi=200)
plt.close()
print("\nGroup importance:\n", group_imp.sort_values(ascending=False).round(4).to_string())
print("Group sizes:", {g: len(i) for g, i in groups.items()})

# 4) SHAP on actual fire days only (all 2025 positives)
X_pos = X[test["y_true"].values == 1]
sv_pos = explainer.shap_values(X_pos)
plt.figure()
shap.summary_plot(sv_pos, X_pos, max_display=TOP_N, show=False)
plt.title(f"SHAP summary, fire days only (n={len(X_pos)})")
plt.tight_layout()
plt.savefig(OUT / "shap_beeswarm_fire_days.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================ error analysis
def summarize(g):
    tp = (g["outcome"] == "TP").sum()
    fn = (g["outcome"] == "FN").sum()
    fp = (g["outcome"] == "FP").sum()
    return pd.Series({
        "positives": tp + fn,
        "TP": tp, "FN": fn, "FP": fp,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
    })


# 1) by month
by_month = test.groupby("month").apply(summarize, include_groups=False)
by_month.to_csv(OUT / "error_by_month.csv")
print("\nErrors by month:\n", by_month.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(by_month.index, by_month["TP"], label="Caught (TP)", color="#55A868")
axes[0].bar(by_month.index, by_month["FN"], bottom=by_month["TP"], label="Missed (FN)", color="#C44E52")
axes[0].set_title("Fire days: caught vs missed, by month")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Fire cell-days")
axes[0].set_xticks(range(1, 13))
axes[0].legend()
axes[1].bar(by_month.index, by_month["FP"], color="#DD8452")
axes[1].set_title("False positives by month")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Non-fire cell-days flagged")
axes[1].set_xticks(range(1, 13))
plt.tight_layout()
plt.savefig(OUT / "error_by_month.png", dpi=200)
plt.close()

# 2) by province (top 15 by number of positives)
by_prov = test.groupby("province").apply(summarize, include_groups=False)
by_prov.to_csv(OUT / "error_by_province.csv")
top_prov = by_prov.sort_values("positives", ascending=False).head(15)
print("\nTop 15 provinces by fire count:\n", top_prov.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
y = np.arange(len(top_prov))[::-1]
axes[0].barh(y, top_prov["TP"], color="#55A868", label="Caught (TP)")
axes[0].barh(y, top_prov["FN"], left=top_prov["TP"], color="#C44E52", label="Missed (FN)")
axes[0].set_yticks(y)
axes[0].set_yticklabels(top_prov.index)
axes[0].set_title("Fire days: caught vs missed")
axes[0].legend()
axes[1].barh(y, top_prov["FP"], color="#DD8452")
axes[1].set_title("False positives")
plt.tight_layout()
plt.savefig(OUT / "error_by_province.png", dpi=200)
plt.close()

# 3) by fire size (max_frp: analysis only, never a model input)
pos = test[test["y_true"] == 1].copy()
pos = pos[pos["max_frp"].notna()]
pos["frp_bin"] = pd.qcut(pos["max_frp"], q=4, duplicates="drop")
by_frp = pos.groupby("frp_bin", observed=True).apply(summarize, include_groups=False)
by_frp.to_csv(OUT / "error_by_fire_size.csv")
print("\nRecall by fire size (max_frp quartiles):\n",
      by_frp[["positives", "TP", "FN", "recall"]].round(3).to_string())
print("\nMedian max_frp, caught vs missed:")
print(pos.groupby("outcome")["max_frp"].median().round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
labels = [str(i) for i in by_frp.index]
ax.bar(labels, by_frp["recall"], color="#8172B2")
for i, (r, n) in enumerate(zip(by_frp["recall"], by_frp["positives"])):
    ax.text(i, r, f"{r:.2f}\n(n={int(n)})", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, max(by_frp["recall"].max() * 1.3, 0.1))
ax.set_ylabel("Recall")
ax.set_xlabel("Fire size bin (max FRP quartiles)")
ax.set_title("Recall by fire size")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT / "error_by_fire_size.png", dpi=200)
plt.close()

print("\nDone. Files written to:", OUT)