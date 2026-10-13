"""Operating-history rules shared by the v2 OLS and Lasso experiments."""
import pandas as pd

from model_data import TARGET_COLUMNS
from preprocessing import ANALYSIS_START, OPENING_MONTHS

MIN_TRAIN_MONTHS = 24


def operating_target(y):
    """Keep monthly alignment and genuine zeros; mask only pre-opening rows."""
    if y.name not in TARGET_COLUMNS:
        raise ValueError(f"Unknown station target: {y.name}")
    station = y.name.split("_", 2)[2]
    opening = pd.Timestamp(OPENING_MONTHS.get(station, ANALYSIS_START))
    return y.where(y.index >= opening)


def features_for_target(calendar, target_series):
    y = operating_target(target_series)
    X = calendar.copy()
    for lag in (1, 3, 12):
        X[f"lag_{lag}"] = y.shift(lag)
    X["rolling_mean_3"] = y.shift(1).rolling(3).mean()
    X["rolling_mean_12"] = y.shift(1).rolling(12).mean()
    # Eligibility of the target month uses only the known opening date.
    station = y.name.split("_", 2)[2]
    X.loc[X.index < pd.Timestamp(OPENING_MONTHS.get(station, ANALYSIS_START))] = float("nan")
    return X


def fold_info(target, fold, train, validation):
    reason = ("insufficient_training_history" if len(train) < MIN_TRAIN_MONTHS
              else "incomplete_validation_features" if validation.isna().any().any()
              else "")
    return {"target": target, "fold": fold,
            "status": "skipped" if reason else "used", "skip_reason": reason,
            "n_train": len(train), "n_validation": len(validation),
            "train_start": train.index[0].strftime("%Y-%m") if len(train) else "",
            "train_end": train.index[-1].strftime("%Y-%m") if len(train) else "",
            "validation_start": validation.index[0].strftime("%Y-%m"),
            "validation_end": validation.index[-1].strftime("%Y-%m")}


def common_windows(cache, dates, windows):
    """Require exact window lengths for every station and validation month."""
    rows = []
    for window in windows:
        for target, X in cache.items():
            available = min(len(X.loc[X.index < date].dropna()) for date in dates)
            complete = not X.loc[dates].isna().any().any()
            eligible = available >= max(window, MIN_TRAIN_MONTHS) and complete
            rows.append({"target": target, "window_months": window,
                         "min_available_train": available, "eligible": bool(eligible),
                         "reason": "" if eligible else "insufficient_history_or_features"})
    audit = pd.DataFrame(rows)
    eligible = audit.groupby("window_months").eligible.all()
    selected = tuple(int(w) for w in eligible.index[eligible])
    audit["used_for_all_targets"] = audit.window_months.isin(selected)
    if not selected:
        raise ValueError("No common rolling window has enough operating history")
    return selected, audit
