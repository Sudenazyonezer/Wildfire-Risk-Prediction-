"""Shared evaluation functions (used by both B and C).

Usage:
    from evaluate import evaluate, find_best_threshold, results_table

    thr, _ = find_best_threshold(y_val, p_val, mode="f1")   # select threshold on val
    res = evaluate(y_test, p_test, threshold=thr)           # apply only on test
    
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_prob, threshold=0.5, name=None):
    """Compute PR-AUC, precision, recall, F1 (macro/weighted), and confusion matrix.

    Accuracy is intentionally omitted because the data is heavily imbalanced (2.37% positive).
    Precision and recall refer to the positive class (fire present).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "model": name,
        "threshold": float(threshold),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
    }


def find_best_threshold(y_true, y_prob, mode="f1", target_recall=0.80):
    """Select the threshold on the VAL set. Do not call on the test set!

    mode="f1":     threshold that maximizes positive-class F1
    mode="recall": highest-precision threshold among those with recall >= target_recall
    Returns: (threshold, metric_value_at_threshold)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has length len(precision) - 1; drop the last point
    precision, recall = precision[:-1], recall[:-1]

    if mode == "f1":
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        i = int(np.argmax(f1))
        return float(thresholds[i]), float(f1[i])

    if mode == "recall":
        ok = recall >= target_recall
        if not ok.any():
            raise ValueError(f"No threshold satisfies recall >= {target_recall}")
        idx = np.where(ok)[0]
        i = idx[np.argmax(precision[idx])]
        return float(thresholds[i]), float(precision[i])

    raise ValueError("mode must be 'f1' or 'recall'")


def results_table(results):
    """Build a comparison table from a list of evaluate() outputs (for the final table)."""
    cols = ["model", "threshold", "pr_auc", "precision", "recall",
            "f1_macro", "f1_weighted", "tn", "fp", "fn", "tp"]
    return pd.DataFrame(results)[cols].round(4)


if __name__ == "__main__":
    # Quick check: synthetic imbalanced data
    rng = np.random.default_rng(42)
    y = (rng.random(20000) < 0.0237).astype(int)
    p = np.clip(0.02 + 0.5 * y * rng.random(20000) + 0.1 * rng.random(20000), 0, 1)
    thr, f1 = find_best_threshold(y, p, mode="f1")
    r = evaluate(y, p, threshold=thr, name="demo")
    print(f"threshold={thr:.3f}  F1={f1:.3f}")
    print(results_table([r]).to_string(index=False))
    print(r["confusion_matrix"])