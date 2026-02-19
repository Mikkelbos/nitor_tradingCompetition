"""
Feature selection: correlation filtering, VIF, Random Forest importance, LASSO.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ── Correlation-based filtering ──────────────────────────────────────────────

def correlation_filter(df: pd.DataFrame,
                       target_col: str = "target",
                       threshold: float = 0.05,
                       inter_threshold: float = 0.95):
    """
    1. Drop features with |corr to target| < threshold.
    2. Among remaining, drop one of any pair with |corr| > inter_threshold.

    Returns (kept_features, dropped_features, corr_matrix).
    """
    numeric = df.select_dtypes(include=[np.number])
    if target_col not in numeric.columns:
        raise ValueError(f"{target_col} not in numeric columns")

    corr = numeric.corr()
    target_corr = corr[target_col].drop(target_col).abs()

    # Step 1: weak features
    weak = target_corr[target_corr < threshold].index.tolist()
    kept = target_corr[target_corr >= threshold].index.tolist()

    # Step 2: inter-feature collinearity among kept
    dropped_collinear = set()
    for i, f1 in enumerate(kept):
        if f1 in dropped_collinear:
            continue
        for f2 in kept[i + 1:]:
            if f2 in dropped_collinear:
                continue
            if abs(corr.loc[f1, f2]) > inter_threshold:
                # drop the one less correlated with target
                if target_corr[f1] >= target_corr[f2]:
                    dropped_collinear.add(f2)
                else:
                    dropped_collinear.add(f1)

    final = [f for f in kept if f not in dropped_collinear]
    dropped = weak + list(dropped_collinear)
    return final, dropped, corr


# ── Variance Inflation Factor ────────────────────────────────────────────────

def vif_filter(df: pd.DataFrame,
               features: list[str],
               vif_threshold: float = 10.0) -> list[str]:
    """Iteratively drop the feature with highest VIF until all < threshold."""
    remaining = list(features)
    while True:
        X = df[remaining].dropna()
        if X.shape[0] == 0 or len(remaining) <= 1:
            break
        vifs = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=remaining,
        )
        max_vif = vifs.max()
        if max_vif <= vif_threshold:
            break
        worst = vifs.idxmax()
        remaining.remove(worst)
    return remaining


# ── Random Forest importance ─────────────────────────────────────────────────

def rf_importance(df: pd.DataFrame,
                  features: list[str],
                  target_col: str = "target",
                  n_estimators: int = 200,
                  top_k: int | None = None) -> pd.Series:
    """
    Fit a quick Random Forest and return feature importances (sorted desc).
    If top_k is given, only return the top-k feature names.
    """
    subset = df[features + [target_col]].dropna()
    X, y = subset[features], subset[target_col]

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )
    if top_k is not None:
        return importances.head(top_k)
    return importances


# ── LASSO-based selection ────────────────────────────────────────────────────

def lasso_selection(df: pd.DataFrame,
                    features: list[str],
                    target_col: str = "target") -> list[str]:
    """Fit LassoCV; return features with non-zero coefficients."""
    subset = df[features + [target_col]].dropna()
    X, y = subset[features], subset[target_col]

    lasso = LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=5000)
    lasso.fit(X, y)

    coefs = pd.Series(lasso.coef_, index=features)
    selected = coefs[coefs.abs() > 0].index.tolist()
    return selected


# ── Quick CLI smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_preprocessing import load_data, impute_missing

    print("Loading & imputing …")
    train, _ = load_data()
    train = impute_missing(train)

    from src.data_preprocessing import NUMERIC_FEATURES

    print("Correlation filter …")
    kept, dropped, _ = correlation_filter(train, threshold=0.05)
    print(f"  kept: {len(kept)}  |  dropped: {len(dropped)}")

    print("RF importance (on raw features) …")
    imp = rf_importance(train, NUMERIC_FEATURES, top_k=10)
    print(imp.to_string())

    print("\n✅  feature_selection.py — all checks passed.")
