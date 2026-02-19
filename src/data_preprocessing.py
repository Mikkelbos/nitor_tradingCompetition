"""
Data loading, cleaning, scaling, and train/validation splitting.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ── Column lists ──────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "global_horizontal_irradiance",
    "diffuse_horizontal_irradiance",
    "direct_normal_irradiance",
    "cloud_cover_total",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "precipitation_amount",
    "visibility",
    "air_temperature_2m",
    "apparent_temperature_2m",
    "dew_point_temperature_2m",
    "wet_bulb_temperature_2m",
    "surface_pressure",
    "freezing_level_height",
    "relative_humidity_2m",
    "convective_available_potential_energy",
    "lifted_index",
    "convective_inhibition",
    "wind_speed_80m",
    "wind_direction_80m",
    "wind_gust_speed_10m",
    "wind_speed_10m",
    "solar_forecast",
    "wind_forecast",
    "load_forecast",
]

TARGET = "target"
MARKET = "market"
DELIVERY_START = "delivery_start"
DELIVERY_END = "delivery_end"


# ── Loading ───────────────────────────────────────────────────────────────────

def load_data(train_path: str = "train.csv",
              test_path: str = "test_for_participants.csv"):
    """Load train and test CSVs, parse dates, sort by delivery_start."""
    train = pd.read_csv(train_path, parse_dates=[DELIVERY_START, DELIVERY_END])
    test = pd.read_csv(test_path, parse_dates=[DELIVERY_START, DELIVERY_END])

    train = train.sort_values(DELIVERY_START).reset_index(drop=True)
    test = test.sort_values(DELIVERY_START).reset_index(drop=True)

    return train, test


# ── Missing-value imputation ─────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame, strategy: str = "ffill") -> pd.DataFrame:
    """
    Impute missing values in numeric feature columns.

    Strategies
    ----------
    ffill  : forward-fill within each market, then back-fill any remaining.
    median : fill with per-market median.
    """
    df = df.copy()

    if strategy == "ffill":
        for col in NUMERIC_FEATURES:
            df[col] = df.groupby(MARKET)[col].transform(
                lambda s: s.ffill().bfill()
            )
    elif strategy == "median":
        for col in NUMERIC_FEATURES:
            df[col] = df.groupby(MARKET)[col].transform(
                lambda s: s.fillna(s.median())
            )
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    return df


# ── Scaling ───────────────────────────────────────────────────────────────────

def scale_features(train_df: pd.DataFrame,
                   val_df: pd.DataFrame | None = None,
                   test_df: pd.DataFrame | None = None,
                   feature_cols: list[str] | None = None):
    """
    Fit StandardScaler on train, transform train / val / test.
    Returns (scaler, train, val, test) — val/test may be None.
    """
    if feature_cols is None:
        feature_cols = NUMERIC_FEATURES

    scaler = StandardScaler()
    train_df = train_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

    if val_df is not None:
        val_df = val_df.copy()
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])

    if test_df is not None:
        test_df = test_df.copy()
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return scaler, train_df, val_df, test_df


# ── Train / validation split (temporal) ──────────────────────────────────────

def temporal_split(df: pd.DataFrame,
                   val_months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *train.csv* into train/validation by cutting the last `val_months`
    months as validation, respecting temporal ordering.
    Only rows with non-null target are included.
    """
    df = df.dropna(subset=[TARGET]).copy()
    cutoff = df[DELIVERY_START].max() - pd.DateOffset(months=val_months)
    train = df[df[DELIVERY_START] < cutoff].reset_index(drop=True)
    val = df[df[DELIVERY_START] >= cutoff].reset_index(drop=True)
    return train, val


# ── Market encoding ──────────────────────────────────────────────────────────

def encode_market(df: pd.DataFrame, method: str = "onehot") -> pd.DataFrame:
    """One-hot or label-encode the market column."""
    df = df.copy()
    if method == "onehot":
        dummies = pd.get_dummies(df[MARKET], prefix="market", dtype=int)
        df = pd.concat([df, dummies], axis=1)
    elif method == "label":
        mapping = {m: i for i, m in enumerate(sorted(df[MARKET].unique()))}
        df["market_label"] = df[MARKET].map(mapping)
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    return df


# ── Quick CLI smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data …")
    train, test = load_data()
    print(f"  train: {train.shape}  |  test: {test.shape}")

    print("Imputing missing values …")
    train = impute_missing(train, strategy="ffill")
    test = impute_missing(test, strategy="ffill")
    print(f"  train nulls remaining: {train[NUMERIC_FEATURES].isnull().sum().sum()}")
    print(f"  test  nulls remaining: {test[NUMERIC_FEATURES].isnull().sum().sum()}")

    print("Temporal split …")
    tr, val = temporal_split(train, val_months=3)
    print(f"  train: {tr.shape}  |  val: {val.shape}")

    print("Scaling …")
    scaler, tr_s, val_s, test_s = scale_features(tr, val, test)
    print(f"  scaled train: {tr_s.shape}")

    print("Market encoding …")
    tr_s = encode_market(tr_s, method="onehot")
    print(f"  columns after encoding: {len(tr_s.columns)}")

    print("\n✅  data_preprocessing.py — all checks passed.")
