"""
Main training pipeline — end-to-end workflow:
  1. Load & impute data
  2. Feature engineering
  3. Feature selection (RF importance)
  4. Temporal train/val split
  5. Scale features
  6. Train SVM (SGDRegressor for speed, then SVR if time allows)
  7. Evaluate on validation set
  8. Generate test predictions & submission CSV

Run:  python3 run_pipeline.py

User preferences:
  - Single model (market as feature, not per-market)
  - No subsampling (full dataset)
"""
from __future__ import annotations
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import (
    load_data, impute_missing, temporal_split, encode_market, scale_features,
    NUMERIC_FEATURES, TARGET, MARKET, DELIVERY_START, DELIVERY_END,
)
from src.feature_engineering import engineer_features
from src.feature_selection import rf_importance, correlation_filter
from src.svm_model import train_sgd_svr, tune_sgd_svr, train_svr, train_and_evaluate
from src.evaluation import evaluate, evaluate_by_market, create_submission


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & IMPUTE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading & imputing data")
print("=" * 70)
t0 = time.time()

train_raw, test_raw = load_data()
print(f"  Raw train: {train_raw.shape}  |  Raw test: {test_raw.shape}")

train_raw = impute_missing(train_raw, strategy="ffill")
test_raw = impute_missing(test_raw, strategy="ffill")
print(f"  Nulls after impute — train: {train_raw[NUMERIC_FEATURES].isnull().sum().sum()}"
      f"  test: {test_raw[NUMERIC_FEATURES].isnull().sum().sum()}")

# Drop rows without target (can't train on them)
train_raw = train_raw.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"  Train after dropping null target: {train_raw.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Feature engineering")
print("=" * 70)

train_eng = engineer_features(train_raw, include_lags=True, include_rolling=True)
test_eng = engineer_features(test_raw, include_lags=False, include_rolling=False)
print(f"  Train columns: {len(train_eng.columns)}  |  Test columns: {len(test_eng.columns)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MARKET ENCODING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Market encoding (one-hot)")
print("=" * 70)

train_eng = encode_market(train_eng, method="onehot")
test_eng = encode_market(test_eng, method="onehot")
print(f"  Train columns after encoding: {len(train_eng.columns)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Temporal train/validation split (last 3 months → val)")
print("=" * 70)

tr, val = temporal_split(train_eng, val_months=3)
print(f"  Train: {tr.shape}  |  Val: {val.shape}")
print(f"  Train dates: {tr[DELIVERY_START].min()} → {tr[DELIVERY_START].max()}")
print(f"  Val dates:   {val[DELIVERY_START].min()} → {val[DELIVERY_START].max()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DEFINE FEATURE SET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Feature set selection")
print("=" * 70)

# Exclude non-feature columns
EXCLUDE_COLS = {"id", TARGET, MARKET, DELIVERY_START, DELIVERY_END}
all_feature_cols = [c for c in tr.columns if c not in EXCLUDE_COLS]

# Also exclude any columns that test doesn't have
# (lag/rolling features are only in train)
test_feature_cols = [c for c in test_eng.columns if c not in EXCLUDE_COLS]
feature_cols = [c for c in all_feature_cols if c in test_feature_cols]

# Add lag/rolling features that ARE available in train (for SVM val evaluation)
# but we'll need to handle test prediction separately
svm_train_features = feature_cols.copy()

# For the SVM model we'll use features available in BOTH train and test
print(f"  Features available in both train & test: {len(feature_cols)}")
print(f"  Feature names: {feature_cols}")

# Drop rows with NaN in feature columns (from lag/rolling features)
tr_clean = tr.dropna(subset=feature_cols).reset_index(drop=True)
val_clean = val.dropna(subset=feature_cols).reset_index(drop=True)
print(f"  Train after NaN drop: {tr_clean.shape}  |  Val: {val_clean.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  FEATURE IMPORTANCE (quick RF check)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Random Forest feature importance (quick ranking)")
print("=" * 70)

importances = rf_importance(tr_clean, feature_cols, top_k=None)
print("\nTop 20 features:")
print(importances.head(20).to_string())

# Use top features (importance > 0.01)
important_features = importances[importances > 0.005].index.tolist()
print(f"\n  Features with importance > 0.005: {len(important_features)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SCALE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Scaling features")
print("=" * 70)

scaler = StandardScaler()
X_train = scaler.fit_transform(tr_clean[important_features])
X_val = scaler.transform(val_clean[important_features])
y_train = tr_clean[TARGET].values
y_val = val_clean[TARGET].values

# Also prepare test set
test_clean = test_eng.copy()
test_clean = encode_market(test_clean, method="onehot") if MARKET in test_clean.columns and "market_Market A" not in test_clean.columns else test_clean
# Handle any missing values in test features
for col in important_features:
    if col not in test_clean.columns:
        test_clean[col] = 0  # fill missing engineered features with 0
    elif test_clean[col].isnull().any():
        test_clean[col] = test_clean[col].fillna(0)

X_test = scaler.transform(test_clean[important_features])

print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  TRAIN SGD SVR (linear, fast)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8a: Training SGD SVR (linear, full dataset)")
print("=" * 70)

t_start = time.time()
sgd_model = train_sgd_svr(X_train, y_train, alpha=1e-4, epsilon=0.1, max_iter=3000)
t_sgd = time.time() - t_start
print(f"  Training time: {t_sgd:.1f}s")

sgd_pred_train = sgd_model.predict(X_train)
sgd_pred_val = sgd_model.predict(X_val)

print("\n── SGD SVR Results ──")
sgd_train_metrics = evaluate(y_train, sgd_pred_train, label="Train")
sgd_val_metrics = evaluate(y_val, sgd_pred_val, label="Validation")

# Per-market evaluation
val_eval_df = val_clean.copy()
val_eval_df["pred"] = sgd_pred_val
print("\n── Per-Market Validation Results (SGD SVR) ──")
market_metrics = evaluate_by_market(val_eval_df)
print(market_metrics.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 8b. TUNE SGD SVR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8b: Hyperparameter tuning (SGD SVR via GridSearchCV)")
print("=" * 70)

best_sgd, gs_sgd = tune_sgd_svr(
    X_train, y_train,
    param_grid={
        "alpha": [1e-5, 1e-4, 1e-3],
        "epsilon": [0.01, 0.1, 0.5],
        "learning_rate": ["invscaling", "adaptive"],
        "eta0": [0.001, 0.01],
    },
    n_splits=3,
    verbose=0,
)

tuned_pred_val = best_sgd.predict(X_val)
print("\n── Tuned SGD SVR Validation ──")
tuned_val_metrics = evaluate(y_val, tuned_pred_val, label="Tuned Val")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  TRAIN SVR (RBF, may be slow)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 9: Training SVR (RBF kernel, full dataset — this may take a while)")
print("=" * 70)

t_start = time.time()
svr_model = train_svr(X_train, y_train, kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
t_svr = time.time() - t_start
print(f"  Training time: {t_svr:.1f}s")

svr_pred_val = svr_model.predict(X_val)
print("\n── SVR (RBF) Validation ──")
svr_val_metrics = evaluate(y_val, svr_pred_val, label="Validation")

val_eval_df["pred"] = svr_pred_val
print("\n── Per-Market Validation Results (SVR RBF) ──")
svr_market_metrics = evaluate_by_market(val_eval_df)
print(svr_market_metrics.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 10. COMPARE & SELECT BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 10: Model comparison")
print("=" * 70)

results = pd.DataFrame({
    "SGD SVR (default)": sgd_val_metrics,
    "SGD SVR (tuned)": tuned_val_metrics,
    "SVR (RBF)": svr_val_metrics,
}).T
print(results.to_string())

# Pick best model by MAE
best_name = results["MAE"].idxmin()
print(f"\n  Best model: {best_name}  (MAE = {results.loc[best_name, 'MAE']:.4f})")

# Select best model for prediction
if best_name == "SVR (RBF)":
    best_model = svr_model
elif best_name == "SGD SVR (tuned)":
    best_model = best_sgd
else:
    best_model = sgd_model


# ═══════════════════════════════════════════════════════════════════════════════
# 11. GENERATE PREDICTIONS & SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 11: Generating test predictions & submission")
print("=" * 70)

test_predictions = best_model.predict(X_test)

# Sanity checks
print(f"  Predictions range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
print(f"  Predictions mean:  {test_predictions.mean():.2f}")
print(f"  Predictions std:   {test_predictions.std():.2f}")

# Note: electricity prices CAN be negative, so we don't clip the lower bound
submission = create_submission(test_raw, test_predictions,
                               path="submissions/svm_submission.csv")

total_time = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Pipeline complete in {total_time:.1f}s")
print(f"{'=' * 70}")
