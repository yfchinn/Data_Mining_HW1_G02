from pathlib import Path

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_data import load_model_data, make_time_series_cv

def main():
    data = load_model_data()
    cv = make_time_series_cv(data, n_splits=3)
    if any((data.y_train.iloc[val_idx] <= 0).any().any()
           for _, val_idx in cv.split(data.X_train)):
        raise ValueError("Validation targets must be positive to calculate MAPE")
    if (data.y_test <= 0).any().any():
        raise ValueError("Test targets must be positive to calculate MAPE")

    results = []
    predictions = {}

    for target in data.y_train.columns:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(max_iter=100000)),
        ])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid={"lasso__alpha": [1, 10, 100, 1000, 10000, 100000, 1000000]},
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "mape": "neg_mean_absolute_percentage_error",
            },
            refit="mae",
        )
        search.fit(data.X_train, data.y_train[target])
        predicted = search.predict(data.X_test)
        actual = data.y_test[target]
        predictions[target] = predicted

        results.append({
            "target": target,
            "alpha": search.best_params_["lasso__alpha"],
            "validation_mae": -search.best_score_,
            "validation_mape_pct": -search.cv_results_["mean_test_mape"][search.best_index_] * 100,
            "test_mae": mean_absolute_error(actual, predicted),
            "test_rmse": root_mean_squared_error(actual, predicted),
            "test_mape_pct": mean_absolute_percentage_error(actual, predicted) * 100,
        })

    actual_test = data.y_test.copy()
    predicted_test = pd.DataFrame(predictions, index=data.X_test.index)
    for direction in ("entry", "exit"):
        station_columns = [column for column in data.y_test.columns
                           if column.startswith(f"hsr_{direction}_")]
        total_target = f"hsr_{direction}_total"
        actual_test[total_target] = actual_test[station_columns].sum(axis=1)
        predicted_test[total_target] = predicted_test[station_columns].sum(axis=1)
        actual = actual_test[total_target]
        predicted = predicted_test[total_target]
        results.append({
            "target": total_target,
            "alpha": None,
            "validation_mae": None,
            "validation_mape_pct": None,
            "test_mae": mean_absolute_error(actual, predicted),
            "test_rmse": root_mean_squared_error(actual, predicted),
            "test_mape_pct": mean_absolute_percentage_error(actual, predicted) * 100,
        })

    summary = pd.DataFrame(results)
    prediction_rows = [
        pd.DataFrame({
            "date": data.X_test.index,
            "target": target,
            "actual": actual_test[target].to_numpy(),
            "predicted": predicted_test[target].to_numpy(),
        })
        for target in actual_test.columns
    ]
    output_dir = Path(__file__).resolve().parent / "reports" / "lasso"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False, date_format="%Y-%m-%d"
    )

    print(summary.to_string(index=False))
    print("車站目標數：", len(predictions))
    print("測試期間：", data.X_test.index.min().strftime("%Y-%m"), "至",
          data.X_test.index.max().strftime("%Y-%m"))
    print("結果已儲存至：", output_dir)


if __name__ == "__main__":
    main()
