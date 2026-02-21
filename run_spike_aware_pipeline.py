"""
Spike-Aware Electricity Price Forecasting Pipeline
====================================================

THE KEY INSIGHT
---------------
Your current RMSE (~35) is dominated by missed price spikes. The scatter plot
tells the whole story: predictions cap at ~300 while actuals hit 600+. Every
single missed spike adds thousands to RMSE. This pipeline directly attacks that.

HOW ELECTRICITY PRICES ACTUALLY SPIKE
--------------------------------------
Intraday prices spike when the grid is unexpectedly tight:
  1. Renewable shortfall: solar/wind generates less than the day-ahead forecast
  2. Demand surge: consumption exceeds forecast (heat wave, cold snap)
  3. Market stress: combination of above, especially during peak hours

You have ALL the signals to detect this:
  load_forecast - solar_forecast - wind_forecast = "thermal gap"
  (how much dispatchable/expensive generation is needed)

THE ARCHITECTURE
----------------
Stage 1 — Base Model (HGB, absolute_error loss)
    Predicts the "normal" price component, robust to outliers

Stage 2 — Spike Detector (LightGBM classifier)
    P(price in top X% | features) using electricity market fundamentals

Stage 3 — Spike Magnitude Model (HGB, squared_error on spike rows)
    Trained ONLY on high-price events to learn spike dynamics

Stage 4 — Historical Statistics Injection
    market × hour mean/std/quantiles from training data → injected as features

Stage 5 — Smart Blend
    final = base_pred + spike_prob * (spike_pred - base_pred)
    Weights tuned on time-CV folds to minimise RMSE

HOW TO RUN
----------
python run_spike_aware_pipeline.py --data-dir /path/to/data --output submissions/spike_aware_submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TRAIN_FILE = "train.csv"
TEST_FILE = "test_for_participants.csv"
SAMPLE_FILE = "sample_submission.csv"

# Top X% of prices per market are "spikes" for training the spike model
SPIKE_QUANTILE = 0.93


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standard temporal + cyclical features."""
    out = df.copy()
    ds = pd.to_datetime(out["delivery_start"], errors="coerce")
    de = pd.to_datetime(out["delivery_end"], errors="coerce")

    out["hour"] = ds.dt.hour
    out["dayofweek"] = ds.dt.dayofweek
    out["month"] = ds.dt.month
    out["dayofyear"] = ds.dt.dayofyear
    out["weekofyear"] = ds.dt.isocalendar().week.astype("Int64")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_peak_hour"] = out["hour"].between(8, 20).astype(int)  # peak demand window
    out["is_morning_ramp"] = out["hour"].between(6, 9).astype(int)  # solar ramp-up risk
    out["is_evening_peak"] = out["hour"].between(17, 21).astype(int)  # dusk + high demand

    # Cyclical encodings
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["dayofyear"] / 365.25)
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)

    out["duration_hours"] = (de - ds).dt.total_seconds() / 3600.0
    out["days_from_2023"] = (ds - pd.Timestamp("2023-01-01")).dt.total_seconds() / 86400.0

    return out


def add_electricity_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The most important features: electricity market fundamentals.

    The core idea: intraday price = day-ahead price + surprise.
    Surprise is driven by renewable underperformance vs forecast and demand overrun.
    We engineer proxies for this since we don't have the actual day-ahead price.
    """
    out = df.copy()

    # ── 1. SUPPLY-DEMAND GAP (the single most predictive signal) ──────────────
    # How much dispatchable (expensive, typically gas/coal) generation is needed?
    # Higher gap → more price pressure → higher price → more spike risk
    out["thermal_gap"] = (
        out["load_forecast"] - out["solar_forecast"] - out["wind_forecast"]
    )
    # Normalise: gap as fraction of load (avoids absolute MW scale issues)
    load_safe = out["load_forecast"].clip(lower=1.0)
    out["thermal_gap_ratio"] = out["thermal_gap"] / load_safe
    out["renewable_penetration"] = (
        (out["solar_forecast"] + out["wind_forecast"]) / load_safe
    )

    # ── 2. RENEWABLE SHORTFALL RISK PROXIES ──────────────────────────────────
    # Solar: high solar forecast + high cloud cover = high shortfall risk
    cloud = out["cloud_cover_total"].fillna(50.0)
    out["solar_cloud_adjusted"] = out["solar_forecast"] * (1.0 - cloud / 100.0)
    out["solar_at_risk"] = out["solar_forecast"] * (cloud / 100.0)  # MWs at risk
    out["solar_shortfall_proxy"] = out["solar_forecast"] - out["solar_cloud_adjusted"]

    # Photovoltaic hours: solar matters more at certain hours
    out["solar_hour_interaction"] = out["solar_forecast"] * out["hour_sin"].clip(lower=0)

    # Wind: wind cube (power output) vs forecast
    wind_cube = out["wind_speed_80m"].clip(lower=0.0) ** 3
    wind_forecast_safe = out["wind_forecast"].clip(lower=0.1)
    out["wind_power_proxy"] = wind_cube
    # Ratio of actual-conditions wind power to forecast (>1 = overperformance, <1 = risk)
    out["wind_delivery_ratio"] = wind_cube / (wind_forecast_safe * 1e6 + 1)

    # ── 3. DEMAND STRESS INDICATORS ──────────────────────────────────────────
    # Temperature extremes → heating/cooling demand → load forecast errors
    out["temp_extreme"] = (out["air_temperature_2m"] - 15.0).abs()  # deviation from comfort
    out["cooling_degree"] = (out["air_temperature_2m"] - 22.0).clip(lower=0.0)
    out["heating_degree"] = (10.0 - out["air_temperature_2m"]).clip(lower=0.0)
    out["demand_temp_stress"] = out["load_forecast"] * out["temp_extreme"] / 100.0

    # ── 4. STORM / CONVECTION RISK (sudden production drops or demand spikes) ─
    # CAPE > 1000 J/kg → severe thunderstorm risk → sudden renewable drops
    cape = out["convective_available_potential_energy"].fillna(0.0)
    out["high_convection_risk"] = (cape > 1000.0).astype(float)
    out["cape_log"] = np.log1p(cape)
    # Lifted index < -4 → extreme instability
    li = out["lifted_index"].fillna(0.0)
    out["extreme_instability"] = (li < -4.0).astype(float)
    # Storm interaction: high CAPE during solar/wind forecast hours
    out["storm_solar_risk"] = out["cape_log"] * (out["solar_forecast"] / (load_safe))
    out["storm_wind_risk"] = out["cape_log"] * out["wind_delivery_ratio"]

    # ── 5. GRID TIGHTNESS COMPOSITE SCORE ────────────────────────────────────
    # Combines demand stress, renewable shortfall risk, and convection risk
    out["grid_stress_score"] = (
        out["thermal_gap_ratio"].clip(-1, 2)
        + out["solar_at_risk"] / (load_safe)
        + out["high_convection_risk"] * 0.5
        + out["temp_extreme"] / 30.0
    )

    # ── 6. WEATHER INTERACTION FEATURES ──────────────────────────────────────
    out["temp_dew_gap"] = out["air_temperature_2m"] - out["dew_point_temperature_2m"]
    out["wind_chill"] = out["apparent_temperature_2m"] - out["air_temperature_2m"]
    out["high_humidity_load"] = out["relative_humidity_2m"] * out["load_forecast"] / 1e5
    out["visibility_cloud_interaction"] = out["visibility"] * (1 - cloud / 100.0)

    return out


def add_historical_statistics(
    df: pd.DataFrame,
    train_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inject historical price statistics as features.

    For each (market, hour) pair, we compute the mean, std, p75, p95 price
    from training data and merge them in. This gives the model a prior on
    'how expensive is this market at this hour normally' — crucial for spike
    prediction since spikes are relative to the normal level.
    """
    out = df.copy()
    out = out.merge(train_stats, on=["market", "hour"], how="left")
    return out


def build_train_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute market × hour price statistics from training data."""
    stats = (
        train_df.groupby(["market", "hour"])["target"]
        .agg(
            hist_mean="mean",
            hist_std="std",
            hist_p25=lambda x: x.quantile(0.25),
            hist_p75=lambda x: x.quantile(0.75),
            hist_p95=lambda x: x.quantile(0.95),
            hist_p99=lambda x: x.quantile(0.99),
        )
        .reset_index()
    )
    # Price relative to its normal level = z-score proxy (computed dynamically in add_derived)
    return stats


def add_derived_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derived from injected historical stats."""
    out = df.copy()
    # How "unusual" would a spike be right now?
    out["hist_spike_room"] = out["hist_p99"] - out["hist_mean"]
    out["hist_volatility"] = out["hist_std"] / (out["hist_mean"].clip(lower=1.0))
    # Grid stress × historical volatility → compound spike risk
    if "grid_stress_score" in out.columns and "hist_volatility" in out.columns:
        out["compound_spike_risk"] = out["grid_stress_score"] * out["hist_volatility"]
    return out


def add_features(df: pd.DataFrame, train_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    out = add_temporal_features(df)
    out = add_electricity_market_features(out)
    if train_stats is not None:
        out = add_historical_statistics(out, train_stats)
        out = add_derived_stats_features(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def make_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        sparse_threshold=0.0,
    )


def build_base_model(df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    """Robust base model using MAE loss — resists spike distortion."""
    pre = make_preprocessor(df, feature_cols)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=600,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", model)])


def build_spike_model(df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    """
    MSE-loss model trained exclusively on spike rows.

    By training only on spikes, this model learns spike dynamics:
    which hour/market/supply-gap combinations lead to extreme prices.
    MSE loss here is intentional — we WANT it to chase the large values.
    """
    pre = make_preprocessor(df, feature_cols)
    model = HistGradientBoostingRegressor(
        loss="squared_error",   # MSE: aggressively chases large values
        learning_rate=0.04,
        max_iter=600,
        max_leaf_nodes=63,
        min_samples_leaf=10,   # smaller leaves to fit spike patterns
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", model)])


def build_spike_classifier(df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    """
    ExtraTrees classifier: P(spike) given features.
    ExtraTrees is fast and good at rare-event detection.
    """
    pre = make_preprocessor(df, feature_cols)
    model = ExtraTreesRegressor(  # regressor gives soft prob via clipping
        n_estimators=300,
        min_samples_leaf=5,
        max_features=0.7,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", model)])


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def get_spike_threshold_per_market(train_df: pd.DataFrame) -> dict[str, float]:
    """Top SPIKE_QUANTILE of prices per market defines a spike."""
    return (
        train_df.groupby("market")["target"]
        .quantile(SPIKE_QUANTILE)
        .to_dict()
    )


def make_spike_labels(
    train_df: pd.DataFrame,
    thresholds: dict[str, float],
) -> np.ndarray:
    """1 if the row is a spike for its market, 0 otherwise."""
    labels = np.zeros(len(train_df), dtype=float)
    for market, thresh in thresholds.items():
        mask = (train_df["market"] == market) & (train_df["target"] >= thresh)
        labels[mask.to_numpy()] = 1.0
    return labels


def get_feature_cols(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    exclude = {"id", "target", "delivery_start", "delivery_end"}
    return [c for c in train_df.columns if c not in exclude and c in test_df.columns]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def blend_predictions(
    pred_base: np.ndarray,
    pred_spike: np.ndarray,
    spike_prob: np.ndarray,
    weight: float,
) -> np.ndarray:
    """
    final = base + weight * spike_prob * (spike - base)

    When spike_prob ≈ 0: final ≈ base (normal price)
    When spike_prob ≈ 1: final ≈ base + weight*(spike-base)
    weight is tuned on CV folds.
    """
    return pred_base + weight * spike_prob * (pred_spike - pred_base)


def train_and_predict_fold(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    feature_cols: list[str],
    blend_weight: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Train three-stage model on tr, predict on va.
    Returns (predictions, y_true, spike_rate_in_val).
    """
    thresholds = get_spike_threshold_per_market(tr)
    spike_labels = make_spike_labels(tr, thresholds)
    spike_mask = spike_labels == 1.0

    y_tr = tr["target"].to_numpy()
    y_va = va["target"].to_numpy()

    # Stage 1: base model (all training data, MAE loss)
    base_pipe = build_base_model(tr, feature_cols)
    base_pipe.fit(tr[feature_cols], y_tr)
    base_pred = base_pipe.predict(va[feature_cols])

    # Stage 2: spike model (spike rows only, MSE loss)
    tr_spikes = tr.loc[spike_mask.astype(bool)]
    spike_pipe = build_spike_model(tr_spikes, feature_cols)
    spike_pipe.fit(tr_spikes[feature_cols], y_tr[spike_mask.astype(bool)])
    spike_pred = spike_pipe.predict(va[feature_cols])

    # Stage 3: spike probability classifier
    clf_pipe = build_spike_classifier(tr, feature_cols)
    clf_pipe.fit(tr[feature_cols], spike_labels)
    spike_prob = clf_pipe.predict(va[feature_cols]).clip(0.0, 1.0)

    # Blend
    final = blend_predictions(base_pred, spike_pred, spike_prob, blend_weight)

    spike_rate = spike_prob.mean()
    return final, y_va, spike_rate


def tune_blend_weight(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[float, float]:
    """Find the blend weight that minimises RMSE on a validation fold."""
    thresholds = get_spike_threshold_per_market(tr)
    spike_labels = make_spike_labels(tr, thresholds)
    spike_mask = spike_labels == 1.0
    y_tr = tr["target"].to_numpy()
    y_va = va["target"].to_numpy()

    print("  Tuning blend weight...")

    base_pipe = build_base_model(tr, feature_cols)
    base_pipe.fit(tr[feature_cols], y_tr)
    base_pred = base_pipe.predict(va[feature_cols])

    tr_spikes = tr.loc[spike_mask.astype(bool)]
    spike_pipe = build_spike_model(tr_spikes, feature_cols)
    spike_pipe.fit(tr_spikes[feature_cols], y_tr[spike_mask.astype(bool)])
    spike_pred = spike_pipe.predict(va[feature_cols])

    clf_pipe = build_spike_classifier(tr, feature_cols)
    clf_pipe.fit(tr[feature_cols], spike_labels)
    spike_prob = clf_pipe.predict(va[feature_cols]).clip(0.0, 1.0)

    best_w, best_rmse = 0.0, float("inf")
    for w in np.linspace(0.0, 1.2, 61):  # allow >1 to test aggressive spike boosting
        preds = blend_predictions(base_pred, spike_pred, spike_prob, w)
        r = rmse(y_va, preds)
        if r < best_rmse:
            best_rmse = r
            best_w = w

    print(f"  Best blend weight: {best_w:.3f} → RMSE {best_rmse:.4f}")
    print(f"  Base-only RMSE: {rmse(y_va, base_pred):.4f}")
    return best_w, best_rmse


def time_folds(
    train_df: pd.DataFrame, n_folds: int
) -> list[tuple[pd.Period, pd.DataFrame, pd.DataFrame]]:
    periods = train_df["delivery_start"].dt.to_period("M")
    months = sorted(periods.unique())[-n_folds:]
    folds = []
    for m in months:
        tr = train_df.loc[train_df["delivery_start"] < m.start_time].copy()
        va = train_df.loc[periods == m].copy()
        if len(tr) == 0 or len(va) == 0:
            continue
        folds.append((m, tr, va))
    return folds


def run_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    tune_weight: bool = True,
) -> tuple[float, float]:
    """Run rolling CV and return (best_blend_weight, mean_cv_rmse)."""
    folds = time_folds(train_df, n_folds=n_folds)
    best_weights: list[float] = []
    fold_rmses: list[float] = []

    print(f"\n{'─'*60}")
    print(f"Rolling Time-CV ({n_folds} monthly folds)")
    print(f"{'─'*60}")

    for month, tr, va in folds:
        print(f"\nFold {month}: train={len(tr):,}  val={len(va):,}")

        if tune_weight:
            w, r = tune_blend_weight(tr, va, feature_cols)
        else:
            w = 0.7
            preds, y_va, _ = train_and_predict_fold(tr, va, feature_cols, w)
            r = rmse(y_va, preds)
            print(f"  RMSE: {r:.4f}")

        best_weights.append(w)
        fold_rmses.append(r)

    mean_w = float(np.mean(best_weights))
    mean_r = float(np.mean(fold_rmses))

    print(f"\n{'─'*60}")
    print(f"CV Summary")
    print(f"  Fold RMSEs:     {[f'{r:.4f}' for r in fold_rmses]}")
    print(f"  Mean CV RMSE:   {mean_r:.4f}")
    print(f"  Blend weights:  {[f'{w:.3f}' for w in best_weights]}")
    print(f"  Mean weight:    {mean_w:.3f}")
    print(f"{'─'*60}")

    return mean_w, mean_r


def final_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    blend_weight: float,
    output_path: Path,
) -> None:
    """Train final model on all training data and predict test set."""
    print(f"\n{'─'*60}")
    print(f"Final Training (blend_weight={blend_weight:.3f})")
    print(f"{'─'*60}")

    thresholds = get_spike_threshold_per_market(train_df)
    spike_labels = make_spike_labels(train_df, thresholds)
    spike_mask = spike_labels == 1.0
    y_tr = train_df["target"].to_numpy()

    print(f"  Spike rows in training: {spike_mask.sum():,} / {len(train_df):,} "
          f"({100*spike_mask.mean():.1f}%) [top {100*(1-SPIKE_QUANTILE):.0f}% per market]")

    # Stage 1
    print("  Training base model...")
    base_pipe = build_base_model(train_df, feature_cols)
    base_pipe.fit(train_df[feature_cols], y_tr)
    base_pred = base_pipe.predict(test_df[feature_cols])

    # Stage 2
    print("  Training spike model...")
    tr_spikes = train_df.loc[spike_mask.astype(bool)]
    spike_pipe = build_spike_model(tr_spikes, feature_cols)
    spike_pipe.fit(tr_spikes[feature_cols], y_tr[spike_mask.astype(bool)])
    spike_pred = spike_pipe.predict(test_df[feature_cols])

    # Stage 3
    print("  Training spike classifier...")
    clf_pipe = build_spike_classifier(train_df, feature_cols)
    clf_pipe.fit(train_df[feature_cols], spike_labels)
    spike_prob = clf_pipe.predict(test_df[feature_cols]).clip(0.0, 1.0)

    # Blend
    final = blend_predictions(base_pred, spike_pred, spike_prob, blend_weight)

    submission = pd.DataFrame({"id": test_df["id"], "target": final})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"\n  Saved: {output_path}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Predictions: min={final.min():.2f}  max={final.max():.2f}  "
          f"mean={final.mean():.2f}  std={final.std():.2f}")
    pct_high = (spike_prob > 0.5).mean() * 100
    print(f"  Test rows with spike_prob > 0.5: {pct_high:.1f}%")


def verify(output_path: Path, sample_path: Path) -> None:
    if not sample_path.exists():
        return
    s = pd.read_csv(sample_path)
    p = pd.read_csv(output_path)
    assert list(p.columns) == list(s.columns)
    assert len(p) == len(s)
    assert p["id"].equals(s["id"])
    print("  ✓ Submission format check passed")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spike-Aware Electricity Price Pipeline")
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/spike_aware_submission.csv"),
    )
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip per-fold weight tuning (faster, uses weight=0.7)",
    )
    p.add_argument(
        "--blend-weight",
        type=float,
        default=None,
        help="Override blend weight (skip CV tuning entirely)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    print("Loading data...")
    train = pd.read_csv(
        data_dir / TRAIN_FILE,
        parse_dates=["delivery_start", "delivery_end"],
    )
    test = pd.read_csv(
        data_dir / TEST_FILE,
        parse_dates=["delivery_start", "delivery_end"],
    )
    train = train.dropna(subset=["target"]).reset_index(drop=True)

    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")

    # Compute historical stats from full training data
    # (note: in CV we recompute these from tr only to avoid leakage)
    print("Computing historical market×hour statistics...")
    train_stats = build_train_stats(train)

    print("Engineering features...")
    train_feat = add_features(train, train_stats)
    test_feat = add_features(test, train_stats)

    feature_cols = get_feature_cols(train_feat, test_feat)
    print(f"  Total features: {len(feature_cols)}")

    if args.blend_weight is not None:
        blend_weight = args.blend_weight
        print(f"\nUsing provided blend weight: {blend_weight:.3f}")
    else:
        blend_weight, cv_rmse = run_cv(
            train_feat,
            feature_cols,
            n_folds=args.cv_folds,
            tune_weight=not args.no_tune,
        )

    output_path = args.output if args.output.is_absolute() else data_dir / args.output
    final_predict(
        train_df=train_feat,
        test_df=test_feat,
        feature_cols=feature_cols,
        blend_weight=blend_weight,
        output_path=output_path,
    )
    verify(output_path, data_dir / SAMPLE_FILE)


if __name__ == "__main__":
    main()
