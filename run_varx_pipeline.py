"""
VARX pipeline — per-market VARX/AR model for electricity price forecasting.

For the competition we use a simpler AR(X) formulation via statsmodels OLS
since full VARMAX on 20k+ rows per market is prohibitively slow.
This uses autoregressive lags + exogenous regressors (a "reduced-form VARX").

Run:  python3 run_varx_pipeline.py
"""
from __future__ import annotations
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error

from src.data_preprocessing import (
    load_data, impute_missing, temporal_split, encode_market,
    NUMERIC_FEATURES, TARGET, MARKET, DELIVERY_START, DELIVERY_END,
)
from src.feature_engineering import (
    add_temporal_features, add_lag_features, add_rolling_features,
    add_squared_features, add_interaction_features,
)
from src.varx_model import check_stationarity_by_market
from src.evaluation import evaluate, evaluate_by_market, create_submission


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & PREPARE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading & preparing data")
print("=" * 70)
t0 = time.time()

train_raw, test_raw = load_data()
train_raw = impute_missing(train_raw, strategy="ffill")
test_raw = impute_missing(test_raw, strategy="ffill")
train_raw = train_raw.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"  Train: {train_raw.shape}  |  Test: {test_raw.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  STATIONARITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Stationarity checks")
print("=" * 70)

stat_results = check_stationarity_by_market(train_raw)
print(stat_results)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING (with lags & rolling stats)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Feature engineering (temporal + lags + rolling)")
print("=" * 70)

train = train_raw.copy()
train = add_temporal_features(train)
train = add_lag_features(train, lags=[1, 2, 3, 6, 12, 24, 48, 168])
train = add_rolling_features(train, windows=[6, 24, 168])
train = add_squared_features(train)
train = add_interaction_features(train)
print(f"  Train columns after engineering: {len(train.columns)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DEFINE FEATURES FOR ARX
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Defining features for AR(X) model")
print("=" * 70)

# Exogenous features (available in test)
EXOG_FEATURES = [
    "solar_forecast", "wind_forecast", "load_forecast",
    "air_temperature_2m", "wind_speed_10m", "wind_speed_80m",
    "cloud_cover_total", "surface_pressure",
    "hour", "day_of_week", "month", "is_weekend",
    "wind_speed_10m_sq", "wind_speed_80m_sq", "solar_forecast_sq",
    "solar_forecast_x_cloud_cover_total",
    "wind_forecast_x_wind_speed_10m",
    "load_forecast_x_air_temperature_2m",
]

# Autoregressive features (only available in train where target exists)
AR_FEATURES = [
    "target_lag_1h", "target_lag_2h", "target_lag_3h",
    "target_lag_6h", "target_lag_12h", "target_lag_24h",
    "target_lag_48h", "target_lag_168h",
    "target_roll_mean_6h", "target_roll_mean_24h", "target_roll_mean_168h",
    "target_roll_std_6h", "target_roll_std_24h", "target_roll_std_168h",
]

ALL_ARX_FEATURES = AR_FEATURES + EXOG_FEATURES
print(f"  AR features: {len(AR_FEATURES)}  |  Exog features: {len(EXOG_FEATURES)}")
print(f"  Total ARX features: {len(ALL_ARX_FEATURES)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAIN/VAL SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Temporal train/val split")
print("=" * 70)

tr, val = temporal_split(train, val_months=3)
tr_clean = tr.dropna(subset=ALL_ARX_FEATURES).reset_index(drop=True)
val_clean = val.dropna(subset=ALL_ARX_FEATURES).reset_index(drop=True)
print(f"  Train: {tr_clean.shape}  |  Val: {val_clean.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TRAIN ARX MODEL (per-market Ridge regression with AR lags)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Training per-market ARX models (Ridge regression)")
print("=" * 70)

models = {}
val_predictions = []

for market in sorted(tr_clean[MARKET].unique()):
    tr_m = tr_clean[tr_clean[MARKET] == market]
    val_m = val_clean[val_clean[MARKET] == market]

    X_tr = tr_m[ALL_ARX_FEATURES].values
    y_tr = tr_m[TARGET].values
    X_val = val_m[ALL_ARX_FEATURES].values
    y_val = val_m[TARGET].values

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    models[market] = model

    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    print(f"  {market}: MAE={mae:.4f}  (train={len(tr_m)}, val={len(val_m)})")

    val_m_copy = val_m.copy()
    val_m_copy["pred"] = pred
    val_predictions.append(val_m_copy)

# Combine validation results
val_all = pd.concat(val_predictions, ignore_index=True)
print("\n── Overall ARX Validation ──")
arx_metrics = evaluate(val_all[TARGET], val_all["pred"], label="ARX (Ridge)")

print("\n── Per-Market ARX Validation ──")
arx_market = evaluate_by_market(val_all)
print(arx_market.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ALSO TRAIN SINGLE-MARKET OLS (with all data, for comparison)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Single model ARX (market as feature) for comparison")
print("=" * 70)

# Encode market for single model
tr_enc = encode_market(tr_clean, method="onehot")
val_enc = encode_market(val_clean, method="onehot")

market_dummies = [c for c in tr_enc.columns if c.startswith("market_")]
single_features = ALL_ARX_FEATURES + market_dummies

X_tr_single = tr_enc[single_features].values
y_tr_single = tr_enc[TARGET].values
X_val_single = val_enc[single_features].values
y_val_single = val_enc[TARGET].values

single_model = Ridge(alpha=1.0)
single_model.fit(X_tr_single, y_tr_single)
single_pred = single_model.predict(X_val_single)

print("\n── Single Model ARX Validation ──")
single_metrics = evaluate(y_val_single, single_pred, label="Single ARX (Ridge)")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  GENERATE TEST PREDICTIONS (using exog-only model, no AR lags for test)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: Generating test predictions")
print("=" * 70)
print("  Note: Test set has no target → no AR lag features available.")
print("  Training exog-only models for test prediction.")

# Prepare test features
test = test_raw.copy()
test = add_temporal_features(test)
test = add_squared_features(test)
test = add_interaction_features(test)

# Fill any missing exog features in test
for col in EXOG_FEATURES:
    if col not in test.columns:
        test[col] = 0
    elif test[col].isnull().any():
        test[col] = test[col].fillna(0)

# Train exog-only models (no AR features) per market, using full training data
print("\n  Training exog-only models per market …")
train_full = train.copy()
train_full = add_temporal_features(train_raw) if "hour" not in train_full.columns else train_full
train_full_clean = train_full.dropna(subset=EXOG_FEATURES)

exog_models = {}
test_predictions = []

for market in sorted(train_full_clean[MARKET].unique()):
    tr_m = train_full_clean[train_full_clean[MARKET] == market]
    test_m = test[test[MARKET] == market]

    X_tr = tr_m[EXOG_FEATURES].values
    y_tr = tr_m[TARGET].values
    X_test = test_m[EXOG_FEATURES].values

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    exog_models[market] = model

    pred = model.predict(X_test)
    print(f"    {market}: predicted {len(pred)} rows, range=[{pred.min():.1f}, {pred.max():.1f}]")

    test_m_copy = test_m.copy()
    test_m_copy["target_pred"] = pred
    test_predictions.append(test_m_copy)

# Combine test predictions
test_all = pd.concat(test_predictions, ignore_index=True)
test_all = test_all.sort_values("id").reset_index(drop=True)

print(f"\n  Total test predictions: {len(test_all)}")
print(f"  Predictions range: [{test_all['target_pred'].min():.2f}, {test_all['target_pred'].max():.2f}]")
print(f"  Predictions mean:  {test_all['target_pred'].mean():.2f}")

# Save submission
create_submission(test_raw, test_all["target_pred"].values,
                  path="submissions/varx_submission.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
total_time = time.time() - t0
print(f"\n{'=' * 70}")
print("RESULTS SUMMARY")
print(f"{'=' * 70}")
print(f"  Per-market ARX (with AR lags) — Val MAE: {arx_metrics['MAE']:.4f}, R²: {arx_metrics['R²']:.4f}")
print(f"  Single ARX (with AR lags)     — Val MAE: {single_metrics['MAE']:.4f}, R²: {single_metrics['R²']:.4f}")
print(f"  (For comparison: SVM SGD      — Val MAE: 17.03, R²: 0.061)")
print(f"\n  VARX submission saved → submissions/varx_submission.csv")
print(f"  Pipeline complete in {total_time:.1f}s")
print(f"{'=' * 70}")
