"""
Ensemble pipeline — combine SVM and VARX predictions.

Strategies:
  1. Simple average: (SVM + VARX) / 2
  2. Weighted average: optimize weights on validation set
  3. VARX-only (best single model)

Run:  python3 run_ensemble.py
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from src.evaluation import create_submission

# ── Load individual submissions ──────────────────────────────────────────────
svm_sub = pd.read_csv("submissions/svm_submission.csv")
varx_sub = pd.read_csv("submissions/varx_submission.csv")

assert len(svm_sub) == 13098, f"SVM submission has {len(svm_sub)} rows"
assert len(varx_sub) == 13098, f"VARX submission has {len(varx_sub)} rows"
assert list(svm_sub["id"]) == list(varx_sub["id"]), "IDs don't match"

print("SVM predictions:  mean={:.2f}, std={:.2f}, range=[{:.2f}, {:.2f}]".format(
    svm_sub["target"].mean(), svm_sub["target"].std(),
    svm_sub["target"].min(), svm_sub["target"].max()))
print("VARX predictions: mean={:.2f}, std={:.2f}, range=[{:.2f}, {:.2f}]".format(
    varx_sub["target"].mean(), varx_sub["target"].std(),
    varx_sub["target"].min(), varx_sub["target"].max()))

# ── Strategy 1: Simple average ──────────────────────────────────────────────
avg_target = (svm_sub["target"] + varx_sub["target"]) / 2
avg_sub = pd.DataFrame({"id": svm_sub["id"], "target": avg_target})
avg_sub.to_csv("submissions/ensemble_avg_submission.csv", index=False)
print(f"\n✅  Simple average: mean={avg_target.mean():.2f}, range=[{avg_target.min():.2f}, {avg_target.max():.2f}]")
print(f"   Saved → submissions/ensemble_avg_submission.csv")

# ── Strategy 2: Weighted average (favor VARX since it scored much better) ───
# VARX: MAE=13.24, R²=0.690
# SVM:  MAE=17.03, R²=0.061
# Weight inversely by MAE
w_varx = 17.03 / (13.24 + 17.03)
w_svm = 13.24 / (13.24 + 17.03)
print(f"\n  Weights (inverse MAE): SVM={w_svm:.3f}, VARX={w_varx:.3f}")

weighted_target = w_svm * svm_sub["target"] + w_varx * varx_sub["target"]
weighted_sub = pd.DataFrame({"id": svm_sub["id"], "target": weighted_target})
weighted_sub.to_csv("submissions/ensemble_weighted_submission.csv", index=False)
print(f"✅  Weighted avg: mean={weighted_target.mean():.2f}, range=[{weighted_target.min():.2f}, {weighted_target.max():.2f}]")
print(f"   Saved → submissions/ensemble_weighted_submission.csv")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUBMISSION FILES:")
print("=" * 70)
print("  1. submissions/svm_submission.csv              — SGD SVR (MAE=17.03 on val)")
print("  2. submissions/varx_submission.csv             — Per-market ARX (MAE=13.24 on val)")
print("  3. submissions/ensemble_avg_submission.csv     — Simple average")
print("  4. submissions/ensemble_weighted_submission.csv — Weighted average (VARX-heavy)")
print()
print("  RECOMMENDED: submissions/varx_submission.csv")
print("    (VARX outperformed SVM by ~4 MAE points and R² 0.69 vs 0.06)")
print("    Ensemble may slightly smooth predictions but unlikely to beat")
print("    the strong VARX model given SVM's weak performance.")
print("=" * 70)
