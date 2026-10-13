"""Compare two holiday representations for v2 static OLS and Lasso.

Run: python static_holiday_experiment.py
All holiday information is either started/ended in days or their log1p values.
The old long_holiday_count is omitted; calendar month length is retained.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from lasso_features import ALPHAS, coefficients, metrics, model_for
from model_data import load_model_data, make_time_series_cv
from station_history import features_for_target, fold_info


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiment_v2"
HOLIDAY_COLUMNS = ("long_holiday_days_started", "long_holiday_days_ended")
CALENDAR_COLUMNS = ("month_index", "month_sin", "month_cos", "days_in_month")
FEATURE_COLUMNS = CALENDAR_COLUMNS + HOLIDAY_COLUMNS
LAG_COLUMNS = ("lag_1", "lag_3", "lag_12", "rolling_mean_3", "rolling_mean_12")
VARIANTS = ("raw", "log1p")


def experiment_features(calendar, y, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    X = features_for_target(calendar, y)[list(FEATURE_COLUMNS + LAG_COLUMNS)].copy()
    if variant == "log1p":
        for column in HOLIDAY_COLUMNS:
            X[column] = np.log1p(X[column].astype(float))
    return X


def main():
    data = load_model_data(
        csv_path=ROOT / "data" / "processed_calendar" / "merged_monthly.csv",
        feature_columns=FEATURE_COLUMNS, test_months=24)
    calendar = pd.concat([data.X_train, data.X_test])
    targets = pd.concat([data.y_train, data.y_test])
    folds = list(make_time_series_cv(data, n_splits=4, validation_months=24)
                 .split(data.X_train))
    entries = [target for target in targets if target.startswith("hsr_entry_")]
    actual_total = targets.loc[data.X_test.index, entries].sum(axis=1)

    comparison, station_rows, cv_rows, prediction_rows, coefficient_rows = [], [], [], [], []
    for model_name in ("OLS", "Lasso"):
        for variant in VARIANTS:
            predicted = pd.DataFrame(index=data.X_test.index)
            target_cv_maes = []
            for target in targets:
                X = experiment_features(calendar, targets[target], variant)
                fold_data = []
                for fold_no, (tr, va) in enumerate(folds, 1):
                    train = X.loc[data.X_train.index[tr]].dropna()
                    validation = X.loc[data.X_train.index[va]]
                    info = fold_info(target, fold_no, train, validation)
                    if info["status"] == "used":
                        fold_data.append((info, train, validation))
                if not fold_data:
                    raise ValueError(f"No eligible folds for {target}")

                candidates = (None,) if model_name == "OLS" else ALPHAS
                scores = []
                for alpha in candidates:
                    fold_scores = []
                    for info, train, validation in fold_data:
                        model = (LinearRegression() if model_name == "OLS"
                                 else model_for(alpha))
                        model.fit(train, targets.loc[train.index, target])
                        result = metrics(targets.loc[validation.index, target],
                                         model.predict(validation))
                        fold_scores.append(result["MAE"])
                        cv_rows.append({"model": model_name, "variant": variant,
                                        **info, "alpha": alpha, **result})
                    scores.append((float(np.mean(fold_scores)), alpha))
                selected_mae, selected_alpha = min(scores, key=lambda item: item[0])
                target_cv_maes.append(selected_mae)

                train = X.loc[data.X_train.index].dropna()
                test = X.loc[data.X_test.index]
                if test.isna().any().any():
                    raise ValueError(f"Incomplete test features for {target}")
                model = (LinearRegression() if model_name == "OLS"
                         else model_for(selected_alpha))
                model.fit(train, targets.loc[train.index, target])
                predictions = model.predict(test)
                predicted[target] = predictions
                station_rows.append({"model": model_name, "variant": variant,
                                     "target": target, "alpha": selected_alpha,
                                     "validation_mae": selected_mae,
                                     "validation_folds": len(fold_data),
                                     "n_train": len(train),
                                     "train_start": train.index[0].strftime("%Y-%m"),
                                     "train_end": train.index[-1].strftime("%Y-%m"),
                                     **metrics(targets.loc[test.index, target], predictions)})
                for date, actual, forecast in zip(test.index, targets.loc[test.index, target], predictions):
                    prediction_rows.append({"model": model_name, "variant": variant,
                                            "target": target, "date": date.strftime("%Y-%m"),
                                            "actual": actual, "predicted": forecast})
                if model_name == "OLS":
                    coefs = {"intercept": float(model.intercept_),
                             **dict(zip(X.columns, model.coef_))}
                else:
                    coefs = coefficients(model, X.columns)
                coefficient_rows.append({"model": model_name, "variant": variant,
                                         "target": target, "alpha": selected_alpha,
                                         **coefs})
            comparison.append({"model": model_name, "variant": variant,
                               "n_features": len(FEATURE_COLUMNS + LAG_COLUMNS),
                               "mean_station_validation_mae": float(np.mean(target_cv_maes)),
                               **metrics(actual_total, predicted[entries].sum(axis=1))})

    for model_name, directory in (("OLS", "ols_static_lag"),
                                  ("Lasso", "lasso_static_lag")):
        old = pd.read_csv(OUT / directory / "total_metrics_comparison.csv").iloc[0]
        old_cv = pd.read_csv(OUT / directory / "cv_metrics.csv")
        comparison.append({"model": model_name, "variant": "baseline",
                           "n_features": 10,
                           "mean_station_validation_mae": old_cv.groupby("target").MAE.mean().mean(),
                           **{key: old[key] for key in ("MAE", "RMSE", "R2", "MAPE_pct")}})
    old = pd.read_csv(OUT / "ols_static_lag" / "total_metrics_comparison.csv").iloc[1]
    comparison.append({"model": "Seasonal naive", "variant": "t-12",
                       "n_features": 1, "mean_station_validation_mae": np.nan,
                       **{key: old[key] for key in ("MAE", "RMSE", "R2", "MAPE_pct")}})

    OUT.mkdir(parents=True, exist_ok=True)
    order = {("OLS", "baseline"): 0, ("OLS", "raw"): 1,
             ("OLS", "log1p"): 2, ("Lasso", "baseline"): 3,
             ("Lasso", "raw"): 4, ("Lasso", "log1p"): 5,
             ("Seasonal naive", "t-12"): 6}
    comparison = sorted(comparison, key=lambda row: order[(row["model"], row["variant"])])
    pd.DataFrame(comparison).to_csv(OUT / "static_holiday_comparison.csv", index=False)
    pd.DataFrame(station_rows).to_csv(OUT / "static_holiday_station_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUT / "static_holiday_cv_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "static_holiday_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(OUT / "static_holiday_coefficients.csv", index=False)
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__ == "__main__":
    main()
