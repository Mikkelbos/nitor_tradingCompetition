"""
SVM model: SVR training, hyperparameter tuning, prediction.

Single-model approach — market is encoded as a feature (not separate models).
Uses full dataset (no subsampling) per user preference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate


# ── Baseline SVR ─────────────────────────────────────────────────────────────

def train_svr(X_train, y_train,
              kernel: str = "rbf",
              C: float = 10.0,
              epsilon: float = 0.1,
              gamma: str = "scale"):
    """Train a single SVR model."""
    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma,
                cache_size=1000, max_iter=50_000)
    model.fit(X_train, y_train)
    return model


# ── SGD-based linear SVR (fast alternative for large data) ───────────────────

def train_sgd_svr(X_train, y_train,
                  alpha: float = 1e-4,
                  epsilon: float = 0.1,
                  max_iter: int = 2000):
    """Train a linear SVR via SGD — scales to large datasets."""
    model = SGDRegressor(
        loss="epsilon_insensitive",
        alpha=alpha,
        epsilon=epsilon,
        max_iter=max_iter,
        random_state=42,
        learning_rate="invscaling",
        eta0=0.01,
    )
    model.fit(X_train, y_train)
    return model


# ── Hyperparameter tuning ────────────────────────────────────────────────────

def tune_svr(X_train, y_train,
             param_grid: dict | None = None,
             n_splits: int = 3,
             scoring: str = "neg_mean_absolute_error",
             verbose: int = 1):
    """
    GridSearchCV with TimeSeriesSplit for SVR.

    Default param_grid explores RBF and linear kernels.
    """
    if param_grid is None:
        param_grid = {
            "kernel": ["rbf", "linear"],
            "C": [1, 10, 100],
            "epsilon": [0.01, 0.1, 0.5],
            "gamma": ["scale", "auto"],
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        SVR(cache_size=1000, max_iter=50_000),
        param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV score ({scoring}): {gs.best_score_:.4f}")
    return gs.best_estimator_, gs


# ── Tuning for SGD-based SVR ────────────────────────────────────────────────

def tune_sgd_svr(X_train, y_train,
                 param_grid: dict | None = None,
                 n_splits: int = 3,
                 scoring: str = "neg_mean_absolute_error",
                 verbose: int = 1):
    """GridSearchCV with TimeSeriesSplit for SGDRegressor (linear SVR)."""
    if param_grid is None:
        param_grid = {
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "epsilon": [0.01, 0.05, 0.1, 0.5],
            "learning_rate": ["invscaling", "adaptive"],
            "eta0": [0.001, 0.01, 0.1],
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        SGDRegressor(loss="epsilon_insensitive", max_iter=2000, random_state=42),
        param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV score ({scoring}): {gs.best_score_:.4f}")
    return gs.best_estimator_, gs


# ── Train + evaluate convenience ─────────────────────────────────────────────

def train_and_evaluate(X_train, y_train, X_val, y_val,
                       method: str = "sgd", **kwargs):
    """
    Train SVM model and evaluate on validation set.
    method: 'svr' for full SVR, 'sgd' for SGDRegressor.
    """
    if method == "svr":
        model = train_svr(X_train, y_train, **kwargs)
    elif method == "sgd":
        model = train_sgd_svr(X_train, y_train, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    print("\n── Training set ──")
    evaluate(y_train, pred_train, label="Train")
    print("\n── Validation set ──")
    metrics = evaluate(y_val, pred_val, label="Validation")

    return model, pred_val, metrics
