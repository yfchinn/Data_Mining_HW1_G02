"""Compare raw and raw-plus-log holiday features for v2 static/rolling models.

Run: python static_holiday_experiment.py
Holiday inputs are started/ended, optionally supplemented by their log1p values.
The old long_holiday_count is omitted; calendar month length is retained.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from lasso_features import ALPHAS, coefficients, metrics, model_for
from lasso_rolling import walk_forward as lasso_walk_forward
from OLS_rolling import walk_forward as ols_walk_forward
from model_data import load_model_data, make_time_series_cv
from station_history import common_windows, features_for_target, fold_info


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiment_v2"
HOLIDAY_COLUMNS = ("long_holiday_days_started", "long_holiday_days_ended")
CALENDAR_COLUMNS = ("month_index", "month_sin", "month_cos", "days_in_month")
FEATURE_COLUMNS = CALENDAR_COLUMNS + HOLIDAY_COLUMNS
LAG_COLUMNS = ("lag_1", "lag_3", "lag_12", "rolling_mean_3", "rolling_mean_12")
VARIANTS = ("raw", "raw_log1p")


def experiment_features(calendar, y, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    X = features_for_target(calendar, y)[list(FEATURE_COLUMNS + LAG_COLUMNS)].copy()
    if variant == "raw_log1p":
        for column in HOLIDAY_COLUMNS:
            X[f"{column}_log1p"] = np.log1p(X[column].astype(float))
    return X


def rolling_predictions(X, y, dates, window, model_name, alpha,
                        keep_coefficients=False):
    if model_name == "Lasso":
        return lasso_walk_forward(X, y, dates, window, alpha,
                                  keep_coefficients=keep_coefficients)
    predictions, coefs = ols_walk_forward(X, y, dates, window)
    records = coefs.reset_index().to_dict("records") if keep_coefficients else []
    for row in records:
        row["date"] = row["date"].strftime("%Y-%m")
    return predictions.predicted, records


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
            print(f"Static {model_name}: {variant}", flush=True)
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
                        cv_rows.append({"method": "static", "model": model_name, "variant": variant,
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
                station_rows.append({"method": "static", "model": model_name, "variant": variant,
                                     "target": target, "alpha": selected_alpha,
                                     "validation_mae": selected_mae,
                                     "validation_folds": len(fold_data),
                                     "n_train": len(train),
                                     "train_start": train.index[0].strftime("%Y-%m"),
                                     "train_end": train.index[-1].strftime("%Y-%m"),
                                     **metrics(targets.loc[test.index, target], predictions)})
                for date, actual, forecast in zip(test.index, targets.loc[test.index, target], predictions):
                    prediction_rows.append({"method": "static", "model": model_name, "variant": variant,
                                            "target": target, "date": date.strftime("%Y-%m"),
                                            "actual": actual, "predicted": forecast})
                if model_name == "OLS":
                    coefs = {"intercept": float(model.intercept_),
                             **dict(zip(X.columns, model.coef_))}
                else:
                    coefs = coefficients(model, X.columns)
                coefficient_rows.append({"method": "static", "model": model_name, "variant": variant,
                                         "target": target, "alpha": selected_alpha,
                                         **coefs})
            comparison.append({"method": "static", "model": model_name, "variant": variant,
                               "window_months": np.nan, "n_features": X.shape[1],
                               "mean_station_validation_mae": float(np.mean(target_cv_maes)),
                               **metrics(actual_total, predicted[entries].sum(axis=1))})

    validation_dates = data.X_train.index[-24:]
    for model_name in ("OLS", "Lasso"):
        for variant in VARIANTS:
            cache = {target: experiment_features(calendar, targets[target], variant)
                     for target in targets}
            windows, _ = common_windows(cache, validation_dates, (36, 60, 120))
            rows = []
            for window in windows:
                print(f"Rolling {model_name}: {variant}, validating window={window}", flush=True)
                for target in targets:
                    for alpha in ((None,) if model_name == "OLS" else ALPHAS):
                        values, _ = rolling_predictions(
                            cache[target], targets[target], validation_dates, window,
                            model_name, alpha)
                        rows.append({"method": "rolling", "model": model_name,
                                     "variant": variant, "target": target,
                                     "window_months": window, "alpha": alpha,
                                     "n_train": window,
                                     "validation_start": validation_dates[0].strftime("%Y-%m"),
                                     "validation_end": validation_dates[-1].strftime("%Y-%m"),
                                     **metrics(targets.loc[validation_dates, target], values)})
            validation = pd.DataFrame(rows)
            selected = validation.loc[validation.groupby(["window_months", "target"]).MAE.idxmin()]
            summary = selected.groupby("window_months").MAE.mean()
            window = int(summary.idxmin())
            cv_rows.extend(rows)
            selection = selected[selected.window_months.eq(window)].set_index("target")
            print(f"Rolling {model_name}: {variant}, selected window={window}", flush=True)
            predicted = pd.DataFrame(index=data.X_test.index)
            for target in targets:
                alpha = int(selection.loc[target, "alpha"]) if model_name == "Lasso" else None
                values, coefs = rolling_predictions(
                    cache[target], targets[target], data.X_test.index, window,
                    model_name, alpha, keep_coefficients=True)
                predicted[target] = values
                station_rows.append({"method": "rolling", "model": model_name,
                                     "variant": variant, "target": target,
                                     "alpha": alpha, "window_months": window,
                                     "validation_mae": selection.loc[target, "MAE"],
                                     "n_train": window,
                                     **metrics(targets.loc[data.X_test.index, target], values)})
                for date, forecast in values.items():
                    prediction_rows.append({"method": "rolling", "model": model_name,
                                            "variant": variant, "target": target,
                                            "date": date.strftime("%Y-%m"),
                                            "actual": targets.loc[date, target],
                                            "predicted": forecast})
                coefficient_rows.extend({"method": "rolling", "model": model_name,
                                         "variant": variant, "target": target,
                                         "alpha": alpha, "window_months": window, **row}
                                        for row in coefs)
            comparison.append({"method": "rolling", "model": model_name, "variant": variant,
                               "window_months": window,
                               "n_features": cache[targets.columns[0]].shape[1],
                               "mean_station_validation_mae": summary.loc[window],
                               **metrics(actual_total, predicted[entries].sum(axis=1))})

    for method, model_name, directory in (("static", "OLS", "ols_static_lag"),
                                         ("static", "Lasso", "lasso_static_lag"),
                                         ("rolling", "OLS", "ols_rolling"),
                                         ("rolling", "Lasso", "lasso_rolling")):
        old = pd.read_csv(OUT / directory / "total_metrics_comparison.csv").iloc[0]
        if method == "static":
            old_cv = pd.read_csv(OUT / directory / "cv_metrics.csv")
            validation_mae = old_cv.groupby("target").MAE.mean().mean()
            window = np.nan
        else:
            old_cv = pd.read_csv(OUT / directory / "window_validation_summary.csv")
            best = old_cv.loc[old_cv.MAE.idxmin()]
            validation_mae, window = best.MAE, int(best.window_months)
        comparison.append({"method": method, "model": model_name, "variant": "baseline",
                           "n_features": 10, "window_months": window,
                           "mean_station_validation_mae": validation_mae,
                           **{key: old[key] for key in ("MAE", "RMSE", "R2", "MAPE_pct")}})
    old = pd.read_csv(OUT / "ols_static_lag" / "total_metrics_comparison.csv").iloc[1]
    comparison.append({"method": "reference", "model": "Seasonal naive", "variant": "t-12",
                       "n_features": 1, "mean_station_validation_mae": np.nan,
                       **{key: old[key] for key in ("MAE", "RMSE", "R2", "MAPE_pct")}})

    OUT.mkdir(parents=True, exist_ok=True)
    baseline_maes = {(row["method"], row["model"]): row["MAE"]
                     for row in comparison if row["variant"] == "baseline"}
    for row in comparison:
        key = (row["method"], row["model"])
        if key in baseline_maes:
            row["mae_improvement_vs_baseline_pct"] = (1 - row["MAE"] / baseline_maes[key]) * 100
        row["mae_improvement_vs_seasonal_pct"] = (1 - row["MAE"] / old["MAE"]) * 100
    comparison.sort(key=lambda row: (
        {"static": 0, "rolling": 1, "reference": 2}[row["method"]],
        {"OLS": 0, "Lasso": 1, "Seasonal naive": 2}[row["model"]],
        {"baseline": 0, "raw": 1, "raw_log1p": 2, "t-12": 3}[row["variant"]]))
    pd.DataFrame(comparison).to_csv(OUT / "static_holiday_comparison.csv", index=False)
    pd.DataFrame(station_rows).to_csv(OUT / "static_holiday_station_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUT / "static_holiday_cv_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "static_holiday_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(OUT / "static_holiday_coefficients.csv", index=False)
    print(pd.DataFrame(comparison).to_string(index=False))


if __name__ == "__main__":
    main()
