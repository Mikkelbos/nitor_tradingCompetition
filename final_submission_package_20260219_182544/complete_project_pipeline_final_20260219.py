from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TRAIN_FILE = "train.csv"
TEST_FILE = "test_for_participants.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ds = pd.to_datetime(out["delivery_start"], errors="coerce")
    de = pd.to_datetime(out["delivery_end"], errors="coerce")

    out["hour"] = ds.dt.hour
    out["dayofweek"] = ds.dt.dayofweek
    out["month"] = ds.dt.month
    out["dayofyear"] = ds.dt.dayofyear
    out["weekofyear"] = ds.dt.isocalendar().week.astype("Int64")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["dayofyear"] / 365.25)

    out["duration_hours"] = (de - ds).dt.total_seconds() / 3600.0
    out["days_from_2023"] = (ds - pd.Timestamp("2023-01-01")).dt.total_seconds() / 86400.0

    cloud = out["cloud_cover_total"].fillna(0.0)
    out["solar_cloud_adjusted"] = out["solar_forecast"] * (1.0 - cloud / 100.0)
    out["wind_power_proxy"] = out["wind_speed_80m"].clip(lower=0.0) ** 3
    out["temp_dew_gap"] = out["air_temperature_2m"] - out["dew_point_temperature_2m"]
    out["wind_chill_proxy"] = out["apparent_temperature_2m"] - out["air_temperature_2m"]

    return out


def build_feature_lists(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[list[str], list[str], list[str]]:
    exclude = {"id", "target", "delivery_start", "delivery_end"}
    feature_cols = [c for c in train_df.columns if c not in exclude and c in test_df.columns]

    cat_cols: list[str] = []
    num_cols: list[str] = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    return feature_cols, num_cols, cat_cols


def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        sparse_threshold=0.0,
    )

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def time_folds(
    train_df: pd.DataFrame, n_folds: int
) -> list[tuple[pd.Period, pd.Series, pd.Series]]:
    periods = train_df["delivery_start"].dt.to_period("M")
    unique_periods = sorted(periods.unique())
    val_periods = unique_periods[-n_folds:]

    folds: list[tuple[pd.Period, pd.Series, pd.Series]] = []
    for p in val_periods:
        val_mask = periods == p
        train_mask = train_df["delivery_start"] < p.start_time
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        folds.append((p, train_mask, val_mask))
    return folds


def evaluate_with_time_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    n_folds: int,
) -> None:
    folds = time_folds(train_df, n_folds=n_folds)
    if not folds:
        raise RuntimeError("No valid time folds were created for cross-validation.")

    maes: list[float] = []
    rmses: list[float] = []
    y = train_df["target"].to_numpy()
    X = train_df[feature_cols]

    print("\nRolling time validation (month-based)")
    for month, tr_mask, va_mask in folds:
        pipe = build_pipeline(num_cols=num_cols, cat_cols=cat_cols)
        pipe.fit(X.loc[tr_mask], y[tr_mask.to_numpy()])
        pred = pipe.predict(X.loc[va_mask])
        y_true = y[va_mask.to_numpy()]

        mae = mean_absolute_error(y_true, pred)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        maes.append(mae)
        rmses.append(rmse)
        print(
            f"  Fold {month}: train={int(tr_mask.sum())}, val={int(va_mask.sum())}, "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}"
        )

    print(
        f"  CV mean: MAE={np.mean(maes):.4f}, RMSE={np.mean(rmses):.4f} "
        f"(over {len(folds)} folds)"
    )


def fit_predict_and_save(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    output_path: Path,
) -> None:
    pipe = build_pipeline(num_cols=num_cols, cat_cols=cat_cols)
    pipe.fit(train_df[feature_cols], train_df["target"].to_numpy())

    preds = pipe.predict(test_df[feature_cols])
    submission = pd.DataFrame({"id": test_df["id"], "target": preds})
    submission.to_csv(output_path, index=False)

    print("\nFinal training complete")
    print(f"  Saved submission: {output_path}")
    print(f"  Rows: {len(submission)}")
    print(
        f"  Prediction stats: min={submission['target'].min():.3f}, "
        f"max={submission['target'].max():.3f}, "
        f"mean={submission['target'].mean():.3f}, "
        f"std={submission['target'].std():.3f}"
    )


def verify_submission(output_path: Path, sample_path: Path) -> None:
    if not sample_path.exists():
        print("\nSample submission not found, skipping schema check.")
        return

    sample = pd.read_csv(sample_path)
    pred = pd.read_csv(output_path)

    assert list(pred.columns) == list(sample.columns), "Submission columns do not match sample."
    assert len(pred) == len(sample), "Submission row count does not match sample."
    assert pred["id"].equals(sample["id"]), "Submission ids are not in the required order."
    print("  Submission format check passed against sample_submission.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end training and submission pipeline for the trading competition."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing train/test/sample CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_complete_hgb.csv"),
        help="Output path for submission CSV.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of last monthly folds to use for rolling validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    train_path = data_dir / TRAIN_FILE
    test_path = data_dir / TEST_FILE
    sample_path = data_dir / SAMPLE_SUBMISSION_FILE

    train = pd.read_csv(train_path, parse_dates=["delivery_start", "delivery_end"])
    test = pd.read_csv(test_path, parse_dates=["delivery_start", "delivery_end"])

    train = train.dropna(subset=["target"]).reset_index(drop=True)
    train = add_features(train)
    test = add_features(test)

    feature_cols, num_cols, cat_cols = build_feature_lists(train, test)
    print("Data loaded")
    print(f"  Train shape: {train.shape}")
    print(f"  Test shape:  {test.shape}")
    print(f"  Features used: {len(feature_cols)} (numeric={len(num_cols)}, categorical={len(cat_cols)})")

    evaluate_with_time_cv(
        train_df=train,
        feature_cols=feature_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        n_folds=args.cv_folds,
    )

    output_path = args.output if args.output.is_absolute() else data_dir / args.output
    fit_predict_and_save(
        train_df=train,
        test_df=test,
        feature_cols=feature_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        output_path=output_path,
    )
    verify_submission(output_path=output_path, sample_path=sample_path)


if __name__ == "__main__":
    main()
