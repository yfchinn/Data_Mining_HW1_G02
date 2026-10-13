"""Shared data preparation for station-level HSR demand models.

Each target is a station's monthly entry or exit count. The first feature set
contains only calendar information available before the month being predicted.
OLS and Lasso should use the same split returned by ``load_model_data`` and
the same training-only validation folds from ``make_time_series_cv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from preprocessing import STATIONS as SOURCE_STATIONS


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "merged_monthly.csv"
STATIONS = tuple(station for station in SOURCE_STATIONS if station != "total")
TARGET_COLUMNS = tuple(
    f"hsr_{direction}_{station}"
    for direction in ("entry", "exit")
    for station in STATIONS
)
FEATURE_COLUMNS = (
    "month_index",
    "month_sin",
    "month_cos",
    "days_in_month",
    "long_holiday_count",
)
HOLIDAY_FEATURE_COLUMNS = FEATURE_COLUMNS + (
    "long_holiday_days_started",
    "long_holiday_days_ended",
)


@dataclass
class ModelData:
    """Date-indexed training/development and held-out test data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


def load_model_data(
    csv_path: str | Path = DEFAULT_DATA_PATH, test_months: int = 24,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> ModelData:
    """Read monthly data and reserve the last ``test_months`` for final testing.

    No scaling or parameter fitting happens here. Fit those steps inside each
    model's training folds to avoid using validation or test information.
    """
    if isinstance(test_months, bool) or not isinstance(test_months, int) or test_months < 1:
        raise ValueError("test_months must be a positive integer")

    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if not feature_columns or feature_columns[0] != "month_index" or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must start with month_index and contain unique columns")
    required = {"date", *TARGET_COLUMNS, *feature_columns[1:]}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Dates must be present and unique")
    frame = frame.sort_values("date").set_index("date")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="MS", name="date")
    if not frame.index.equals(expected):
        raise ValueError("Data must contain one row for every consecutive month")
    if len(frame) <= test_months:
        raise ValueError("test_months must leave at least one training month")

    first = frame.index[0]
    month_index = (frame.index.year - first.year) * 12 + frame.index.month - first.month
    X = frame.loc[:, feature_columns[1:]].apply(pd.to_numeric, errors="raise")
    X.insert(0, "month_index", month_index)
    y = frame.loc[:, TARGET_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Selected features and targets must have no missing or infinite values")

    split_at = len(frame) - test_months
    return ModelData(
        X_train=X.iloc[:split_at].copy(),
        X_test=X.iloc[split_at:].copy(),
        y_train=y.iloc[:split_at].copy(),
        y_test=y.iloc[split_at:].copy(),
    )


def add_target_lag(data: ModelData, target: str, months: int = 24) -> ModelData:
    """Add one target's known value from exactly `months` earlier.

    Only supports test windows no longer than the lag, so every test feature
    is known at the start of the held-out forecast period. Early training rows
    without lag history are removed; validation folds remain chronological.
    """
    if target not in data.y_train or months < len(data.X_test):
        raise ValueError("Target must exist and lag must cover the test horizon")
    values = pd.concat([data.y_train[target], data.y_test[target]])
    lag = values.shift(months).rename(f"{target}_lag_{months}")
    train_lag = lag.loc[data.X_train.index]
    valid_train = train_lag.notna()
    X_train = data.X_train.loc[valid_train].copy()
    X_train[lag.name] = train_lag.loc[valid_train]
    X_test = data.X_test.copy()
    X_test[lag.name] = lag.loc[X_test.index]
    if X_test[lag.name].isna().any():
        raise ValueError("Missing lag history for test period")
    return ModelData(X_train=X_train, X_test=X_test,
                     y_train=data.y_train.loc[valid_train].copy(),
                     y_test=data.y_test.copy())


def make_time_series_cv(
    data: ModelData, n_splits: int = 4, validation_months: int = 24
) -> TimeSeriesSplit:
    """Create expanding training folds with fixed-length validation windows.

    Pass this splitter to model selection with ``data.X_train`` only. The
    held-out ``X_test``/``y_test`` are never part of parameter selection.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if (isinstance(validation_months, bool) or not isinstance(validation_months, int)
            or validation_months < 1):
        raise ValueError("validation_months must be a positive integer")
    if len(data.X_train) <= n_splits * validation_months:
        raise ValueError("Not enough training months for the requested validation folds")
    return TimeSeriesSplit(n_splits=n_splits, test_size=validation_months)


if __name__ == "__main__":
    data = load_model_data()
    print(f"Train: {data.X_train.index.min():%Y-%m} to {data.X_train.index.max():%Y-%m}, "
          f"{len(data.X_train)} months")
    print(f"Test:  {data.X_test.index.min():%Y-%m} to {data.X_test.index.max():%Y-%m}, "
          f"{len(data.X_test)} months")
    print(f"Features: {', '.join(data.X_train.columns)}")
    print(f"Targets: {len(data.y_train.columns)} station entry/exit series")
    for fold, (train_idx, val_idx) in enumerate(make_time_series_cv(data).split(data.X_train), 1):
        train_dates = data.X_train.index[train_idx]
        val_dates = data.X_train.index[val_idx]
        print(f"Fold {fold}: train {train_dates.min():%Y-%m} to {train_dates.max():%Y-%m}; "
              f"validate {val_dates.min():%Y-%m} to {val_dates.max():%Y-%m}")
