"""Lasso with a 12-month target lag and recursive 24-month forecasts.

For months 13–24 of each validation/test window, lag_12 comes from the
model's earlier predictions in that window, never from held-out actuals.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_data import load_model_data, make_time_series_cv


ALPHAS = (1, 10, 100, 1000, 10000, 100000, 1000000)
LAG_MONTHS = 12
ROOT = Path(__file__).resolve().parent


def fit_model(features: pd.DataFrame, targets: pd.Series, alpha: int) -> Pipeline:
    lagged = targets.shift(LAG_MONTHS).rename("lag_12")
    available = lagged.notna()
    training_features = features.loc[available].copy()
    training_features["lag_12"] = lagged.loc[available]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha, max_iter=100000)),
    ])
    model.fit(training_features, targets.loc[available])
    return model


def recursive_predict(model: Pipeline, features: pd.DataFrame,
                      known_targets: pd.Series) -> pd.Series:
    """Forecast contiguous future months using only known or predicted lags."""
    history = known_targets.to_dict()
    predictions = {}
    for month, row in features.iterrows():
        source = month - pd.DateOffset(months=LAG_MONTHS)
        if source not in history:
            raise ValueError(f"Missing known or predicted lag source: {source:%Y-%m}")
        inputs = row.to_frame().T
        inputs["lag_12"] = history[source]
        value = float(model.predict(inputs)[0])
        predictions[month] = value
        history[month] = value
    return pd.Series(predictions, name=known_targets.name)


def main() -> None:
    data = load_model_data()
    cv = make_time_series_cv(data, n_splits=3, validation_months=24)
    folds = list(cv.split(data.X_train))
    if (data.y_test <= 0).any().any():
        raise ValueError("Test targets must be positive to calculate MAPE")

    results = []
    predictions = {}
    for target in data.y_train.columns:
        y = data.y_train[target]
        if any((y.iloc[validation_idx] <= 0).any()
               for _, validation_idx in folds):
            raise ValueError(f"Validation target {target} must be positive to calculate MAPE")

        scores = []
        for alpha in ALPHAS:
            fold_mae, fold_mape = [], []
            for train_idx, validation_idx in folds:
                train_X = data.X_train.iloc[train_idx]
                train_y = y.iloc[train_idx]
                validation_X = data.X_train.iloc[validation_idx]
                actual = y.iloc[validation_idx]
                model = fit_model(train_X, train_y, alpha)
                predicted = recursive_predict(model, validation_X, train_y)
                fold_mae.append(mean_absolute_error(actual, predicted))
                fold_mape.append(mean_absolute_percentage_error(actual, predicted))
            scores.append((sum(fold_mae) / len(fold_mae),
                           sum(fold_mape) / len(fold_mape), alpha))

        validation_mae, validation_mape, best_alpha = min(scores, key=lambda score: score[0])
        final_model = fit_model(data.X_train, y, best_alpha)
        predicted = recursive_predict(final_model, data.X_test, y)
        actual = data.y_test[target]
        predictions[target] = predicted
        results.append({
            "target": target,
            "alpha": best_alpha,
            "validation_mae": validation_mae,
            "validation_mape_pct": validation_mape * 100,
            "test_mae": mean_absolute_error(actual, predicted),
            "test_rmse": root_mean_squared_error(actual, predicted),
            "test_mape_pct": mean_absolute_percentage_error(actual, predicted) * 100,
        })

    actual_test = data.y_test.copy()
    predicted_test = pd.DataFrame(predictions, index=data.X_test.index)
    for direction in ("entry", "exit"):
        columns = [column for column in data.y_test.columns
                   if column.startswith(f"hsr_{direction}_")]
        total = f"hsr_{direction}_total"
        actual_test[total] = actual_test[columns].sum(axis=1)
        predicted_test[total] = predicted_test[columns].sum(axis=1)
        actual, predicted = actual_test[total], predicted_test[total]
        results.append({
            "target": total,
            "alpha": None,
            "validation_mae": None,
            "validation_mape_pct": None,
            "test_mae": mean_absolute_error(actual, predicted),
            "test_rmse": root_mean_squared_error(actual, predicted),
            "test_mape_pct": mean_absolute_percentage_error(actual, predicted) * 100,
        })

    output_dir = ROOT / "reports" / "lasso_lag12_recursive"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / "metrics.csv", index=False)
    prediction_rows = [pd.DataFrame({
        "date": data.X_test.index,
        "target": target,
        "actual": actual_test[target].to_numpy(),
        "predicted": predicted_test[target].to_numpy(),
    }) for target in actual_test.columns]
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False, date_format="%Y-%m-%d")
    print(pd.DataFrame(results).to_string(index=False))
    print("測試期間：", data.X_test.index.min().strftime("%Y-%m"), "至",
          data.X_test.index.max().strftime("%Y-%m"))
    print("結果已儲存至：", output_dir)


if __name__ == "__main__":
    main()
