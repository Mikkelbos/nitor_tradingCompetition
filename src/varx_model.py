"""
VARX model: stationarity checks, lag selection, training, prediction.

Single-model approach — trains one VARX per market (necessary for VAR structure)
but orchestrated from a single entry point.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

from src.data_preprocessing import TARGET, MARKET, DELIVERY_START
from src.evaluation import evaluate


# ── Stationarity checks ─────────────────────────────────────────────────────

def adf_test(series: pd.Series, name: str = "", significance: float = 0.05):
    """Run Augmented Dickey-Fuller test and print result."""
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    stationary = p_value < significance
    if name:
        status = "✅ stationary" if stationary else "❌ non-stationary"
        print(f"  {name}: ADF p={p_value:.4f} → {status}")
    return stationary, p_value


def check_stationarity_by_market(df: pd.DataFrame):
    """Run ADF test on target for each market."""
    results = {}
    for market, grp in df.groupby(MARKET):
        is_stat, pval = adf_test(grp[TARGET], name=market)
        results[market] = {"stationary": is_stat, "p_value": pval}
    return pd.DataFrame(results).T


# ── Differencing ─────────────────────────────────────────────────────────────

def difference_if_needed(series: pd.Series, significance: float = 0.05):
    """Apply first-order differencing if series is non-stationary."""
    is_stat, _ = adf_test(series)
    if is_stat:
        return series, False
    return series.diff().dropna(), True


# ── Lag order selection ──────────────────────────────────────────────────────

def select_lag_order(endog: pd.DataFrame, maxlags: int = 24):
    """Use VAR to determine optimal lag order via AIC/BIC."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VAR(endog)
        results = model.select_order(maxlags=maxlags)
    print(results.summary())
    return results


# ── VARX training (per market) ───────────────────────────────────────────────

def train_varx_market(df_market: pd.DataFrame,
                      endog_cols: list[str],
                      exog_cols: list[str],
                      order: int = 24,
                      trend: str = "c"):
    """
    Fit a VARMAX model on a single market's data.

    Parameters
    ----------
    df_market : DataFrame sorted by delivery_start for one market.
    endog_cols : endogenous variable columns (e.g. [target]).
    exog_cols : exogenous variable columns (forecast features, weather, etc.)
    order : AR lag order.
    """
    endog = df_market[endog_cols].astype(float)
    exog = df_market[exog_cols].astype(float) if exog_cols else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VARMAX(endog, exog=exog, order=(order, 0), trend=trend)
        result = model.fit(disp=False, maxiter=200)

    print(f"  AIC: {result.aic:.2f}  |  BIC: {result.bic:.2f}")
    return result


# ── Full pipeline: train all markets ─────────────────────────────────────────

def train_varx_all_markets(
    df: pd.DataFrame,
    endog_cols: list[str] | None = None,
    exog_cols: list[str] | None = None,
    order: int = 24,
):
    """
    Train separate VARX models per market (VAR requires this structure).
    Returns dict of {market: fitted_result}.
    """
    if endog_cols is None:
        endog_cols = [TARGET]
    if exog_cols is None:
        exog_cols = ["solar_forecast", "wind_forecast", "load_forecast"]

    models = {}
    for market, grp in df.groupby(MARKET):
        print(f"\nTraining VARX for {market} …")
        grp = grp.sort_values(DELIVERY_START).reset_index(drop=True)
        try:
            result = train_varx_market(grp, endog_cols, exog_cols, order=order)
            models[market] = result
        except Exception as e:
            print(f"  ⚠️  Failed for {market}: {e}")
            models[market] = None

    return models


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_varx(result, steps: int, exog_future=None):
    """Generate forecasts from a fitted VARMAX result."""
    forecast = result.forecast(steps=steps, exog=exog_future)
    return forecast


# ── Quick CLI smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_preprocessing import load_data, impute_missing

    print("Loading data …")
    train, _ = load_data()
    train = impute_missing(train)
    train = train.dropna(subset=[TARGET])

    print("\nStationarity checks …")
    stat_results = check_stationarity_by_market(train)
    print(stat_results)

    # Quick test on one market, small order
    market_a = train[train[MARKET] == "Market A"].sort_values(DELIVERY_START).head(500)
    print(f"\nSmoke test: VARX on Market A (first 500 rows, order=2) …")
    try:
        result = train_varx_market(
            market_a,
            endog_cols=[TARGET],
            exog_cols=["solar_forecast", "wind_forecast", "load_forecast"],
            order=2,
        )
        print("  ✅  fit succeeded")
    except Exception as e:
        print(f"  ⚠️  {e}")

    print("\n✅  varx_model.py — smoke test done.")
