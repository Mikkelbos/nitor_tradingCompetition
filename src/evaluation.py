"""
Evaluation utilities: MAE, RMSE, MAPE, R², and comparison helpers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred, epsilon=1e-8):
    """Mean Absolute Percentage Error (avoids division by zero)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def evaluate(y_true, y_pred, label: str = "") -> dict:
    """Compute all metrics and print a summary."""
    metrics = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R²": r2(y_true, y_pred),
    }
    if label:
        print(f"\n── {label} ──")
    for k, v in metrics.items():
        print(f"  {k:>6s}: {v:.4f}")
    return metrics


def evaluate_by_market(df: pd.DataFrame,
                       y_true_col: str = "target",
                       y_pred_col: str = "pred",
                       market_col: str = "market") -> pd.DataFrame:
    """Per-market evaluation table."""
    rows = []
    for market, grp in df.groupby(market_col):
        m = evaluate(grp[y_true_col], grp[y_pred_col], label=str(market))
        m["market"] = market
        rows.append(m)
    return pd.DataFrame(rows).set_index("market")


# ── Submission helpers ────────────────────────────────────────────────────────

def create_submission(test_df: pd.DataFrame,
                      predictions: np.ndarray,
                      path: str = "submissions/submission.csv"):
    """Create and validate a submission CSV based on test_for_participants.csv IDs."""
    
    actual_rows = len(predictions)
    expected_rows = 13098
    
    # Let the user know if there is a discrepancy in rows versus the accepted number of rows
    if expected_rows != actual_rows:
        print(f"❌ ERROR: Shape mismatch! Your prediction data has {actual_rows} rows vs the expected {expected_rows} rows.")
    
    # Use IDs directly from the test dataframe
    sub = pd.DataFrame({"id": test_df["id"], "target": predictions})

    # Step 4: Validate Your Submission format explicitly
    assert list(sub.columns) == ['id', 'target'], "Wrong columns!"
    assert len(sub) == 13098, f"Wrong row count: {len(sub)}"
    assert sub['id'].min() == 133627, "IDs must start at 133627"
    assert sub['id'].max() == 146778, "IDs must end at 146778"
    assert sub['target'].notna().all(), "No NaN values allowed!"
    assert np.isfinite(sub['target']).all(), "No infinite values allowed!"

    print("✅ Validation passed!")

    # Step 5: Save Submission
    sub.to_csv(path, index=False)
    print(f"✅  Submission saved → {path}  ({len(sub)} rows)")
    return sub
