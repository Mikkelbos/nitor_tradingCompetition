"""
Feature engineering: temporal, lag, squared, and interaction features.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from src.data_preprocessing import (
    DELIVERY_START, DELIVERY_END, MARKET, TARGET, NUMERIC_FEATURES,
)


# ── Temporal features ────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day-of-week, month, weekend flag from delivery_start."""
    df = df.copy()
    ds = df[DELIVERY_START]
    df["hour"] = ds.dt.hour
    df["day_of_week"] = ds.dt.dayofweek          # 0=Mon … 6=Sun
    df["month"] = ds.dt.month
    df["is_weekend"] = (ds.dt.dayofweek >= 5).astype(int)
    return df


# ── Lag features (target) ────────────────────────────────────────────────────

LAG_HOURS = [1, 2, 6, 12, 24, 48, 168]  # up to 1 week

def add_lag_features(df: pd.DataFrame,
                     lags: list[int] | None = None) -> pd.DataFrame:
    """
    Add lagged target columns *within each market*, sorted by delivery_start.
    Only applicable to the train set (target available).
    """
    if lags is None:
        lags = LAG_HOURS
    df = df.copy().sort_values([MARKET, DELIVERY_START])
    for lag in lags:
        df[f"target_lag_{lag}h"] = df.groupby(MARKET)[TARGET].shift(lag)
    return df


# ── Rolling features ─────────────────────────────────────────────────────────

ROLLING_WINDOWS = [6, 24, 168]  # 6h, 1 day, 1 week

def add_rolling_features(df: pd.DataFrame,
                         windows: list[int] | None = None) -> pd.DataFrame:
    """Rolling mean and std of target within each market."""
    if windows is None:
        windows = ROLLING_WINDOWS
    df = df.copy().sort_values([MARKET, DELIVERY_START])
    for w in windows:
        df[f"target_roll_mean_{w}h"] = (
            df.groupby(MARKET)[TARGET]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"target_roll_std_{w}h"] = (
            df.groupby(MARKET)[TARGET]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).std())
        )
    return df


# ── Squared features ─────────────────────────────────────────────────────────

SQUARED_COLS = [
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_gust_speed_10m",
    "solar_forecast",
    "wind_forecast",
]

def add_squared_features(df: pd.DataFrame,
                         cols: list[str] | None = None) -> pd.DataFrame:
    """Add squared terms to capture diminishing marginal effects."""
    if cols is None:
        cols = SQUARED_COLS
    df = df.copy()
    for c in cols:
        df[f"{c}_sq"] = df[c] ** 2
    return df


# ── Interaction features ─────────────────────────────────────────────────────

INTERACTIONS = [
    ("solar_forecast", "cloud_cover_total"),
    ("wind_forecast", "wind_speed_10m"),
    ("load_forecast", "air_temperature_2m"),
    ("solar_forecast", "global_horizontal_irradiance"),
    ("wind_forecast", "wind_speed_80m"),
]

def add_interaction_features(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Multiply pairs of features to capture interactions."""
    if pairs is None:
        pairs = INTERACTIONS
    df = df.copy()
    for a, b in pairs:
        df[f"{a}_x_{b}"] = df[a] * df[b]
    return df


# ── Convenience: apply all engineering steps ─────────────────────────────────

def engineer_features(df: pd.DataFrame,
                      include_lags: bool = True,
                      include_rolling: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Parameters
    ----------
    include_lags : bool
        Whether to add lag features (requires target column).
    include_rolling : bool
        Whether to add rolling features (requires target column).
    """
    df = add_temporal_features(df)
    df = add_squared_features(df)
    df = add_interaction_features(df)

    if include_lags and TARGET in df.columns:
        df = add_lag_features(df)
    if include_rolling and TARGET in df.columns:
        df = add_rolling_features(df)

    return df


# ── Quick CLI smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_preprocessing import load_data, impute_missing

    print("Loading data …")
    train, test = load_data()
    train = impute_missing(train)

    print("Engineering features (train) …")
    train_eng = engineer_features(train, include_lags=True, include_rolling=True)
    print(f"  columns before: {len(train.columns)}  →  after: {len(train_eng.columns)}")
    new_cols = set(train_eng.columns) - set(train.columns)
    print(f"  new features ({len(new_cols)}): {sorted(new_cols)}")

    print("Engineering features (test – no lags/rolling) …")
    test_eng = engineer_features(test, include_lags=False, include_rolling=False)
    print(f"  columns before: {len(test.columns)}  →  after: {len(test_eng.columns)}")

    print("\n✅  feature_engineering.py — all checks passed.")
