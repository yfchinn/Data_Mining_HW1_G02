"""Shared features and reporting aligned with feature/experiment-v1 OLS.

Forecasts are one month ahead: prior test months' observed counts are available.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from station_history import features_for_target
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model_data import load_model_data

ALPHAS = (1, 10, 100, 1000, 10000, 100000, 1000000)
ROOT = Path(__file__).resolve().parent


def load_experiment():
    data = load_model_data(test_months=24)
    calendar = pd.concat([data.X_train, data.X_test])
    targets = pd.concat([data.y_train, data.y_test])
    return data, targets, {t: features_for_target(calendar, targets[t]) for t in targets}


def model_for(alpha):
    # Scaling is fitted only on each model's training rows; targets stay in persons.
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=100000))


def metrics(actual, predicted):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    nonzero = a != 0
    return {"MAE": float(mean_absolute_error(a, p)),
            "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "R2": float(r2_score(a, p)),
            "MAPE_pct": float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100)
            if nonzero.any() else np.nan}


def coefficients(model, columns):
    scaler, estimator = model.named_steps["standardscaler"], model.named_steps["lasso"]
    coef = estimator.coef_ / scaler.scale_
    return {"intercept": float(estimator.intercept_ - np.dot(coef, scaler.mean_)),
            **dict(zip(columns, coef))}


def save_results(out, targets, predicted, settings, coefficient_rows):
    out.mkdir(parents=True, exist_ok=True)
    rows, prediction_rows = [], []
    for target in predicted:
        actual = targets.loc[predicted.index, target]
        rows.append({"target": target, **settings[target], **metrics(actual, predicted[target])})
        for date in predicted.index:
            prediction_rows.append({"date": date.strftime("%Y-%m"), "target": target,
                                    "actual": actual.loc[date], "predicted": predicted.loc[date, target],
                                    "error": predicted.loc[date, target] - actual.loc[date]})
    pd.DataFrame(rows).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(out / "coefficients.csv", index=False)
    print(pd.DataFrame(rows)[["MAE", "RMSE", "R2", "MAPE_pct"]].mean().to_string(), flush=True)
    print(f"Saved: {out}", flush=True)
