"""Final comparison: all models, same test set, single table + single PR plot.

Each model saves test predictions in this format:
    outputs/test_pred_<model_name>.csv   columns: y_true, y_prob, threshold
    e.g.: test_pred_baseline.csv, test_pred_rf.csv, test_pred_xgboost.csv

B produces this file for baseline and RF, C for XGBoost (save_preds function is in 05).
Each model's threshold must have been selected on its OWN val set; no threshold is selected on test.
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from evaluate import evaluate, results_table

files = sorted(glob.glob("outputs/test_pred_*.csv"))
if not files:
    raise SystemExit("outputs/test_pred_*.csv not found")

results, curves = [], {}
for f in files:
    name = os.path.basename(f)[len("test_pred_"):-len(".csv")]
    d = pd.read_csv(f)
    thr = float(d["threshold"].iloc[0])
    results.append(evaluate(d["y_true"], d["y_prob"], threshold=thr, name=name))
    curves[name] = (d["y_true"], d["y_prob"])

table = results_table(results)
print(table.to_string(index=False))
table.to_csv("outputs/final_results_table.csv", index=False)

# PR curves on a single plot
fig, ax = plt.subplots(figsize=(7, 5))
for r in results:
    y, p = curves[r["model"]]
    prec, rec, _ = precision_recall_curve(y, p)
    ax.plot(rec, prec, label=f"{r['model']} (PR-AUC={r['pr_auc']:.3f})")
base = next(iter(curves.values()))[0].mean()
ax.axhline(base, color="gray", linestyle="--", label=f"Random (={base:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Test set: Precision-Recall curves")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/pr_curves.png", dpi=200)
print("Saved: outputs/final_results_table.csv, outputs/pr_curves.png")