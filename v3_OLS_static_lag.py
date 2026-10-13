"""Static OLS + lag features for monthly HSR station demand.

Run from the repository root: python OLS_static_lag.py
Uses the same features as OLS_rolling.py, but fits coefficients only once.
Evaluation is one-step-ahead: at each month, prior months' observed counts
are assumed available. No target-month or future actual is used as a feature.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_data import load_model_data, make_time_series_cv

OUT = Path(__file__).resolve().parent / "results" / "ols_static_lag"
LAGS = (1, 3, 12)


def metrics(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    nonzero = a != 0
    return {
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "R2": float(r2_score(a, p)),
        "MAPE_pct": float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100)
        if nonzero.any() else np.nan,
    }


def features_for_target(calendar, target_series):
    X = calendar.copy()
    for lag in LAGS:
        X[f"lag_{lag}"] = target_series.shift(lag)
    X["rolling_mean_3"] = target_series.shift(1).rolling(3).mean()
    X["rolling_mean_12"] = target_series.shift(1).rolling(12).mean()
    return X


def main():
    data = load_model_data(test_months=24)
    OUT.mkdir(parents=True, exist_ok=True)
    calendar = pd.concat([data.X_train, data.X_test]).sort_index()
    targets = pd.concat([data.y_train, data.y_test]).sort_index()
    train_dates = data.X_train.index
    test_dates = data.X_test.index
    folds = list(make_time_series_cv(data, n_splits=4, validation_months=24).split(data.X_train))

    cv_rows, test_rows, prediction_rows, coefficient_rows = [], [], [], []
    predictions_by_target = pd.DataFrame(index=test_dates)

    for target in targets.columns:
        y = targets[target]
        X = features_for_target(calendar, y)
        valid_train = X.loc[train_dates].notna().all(axis=1)
        X_train = X.loc[train_dates][valid_train]
        y_train = y.loc[X_train.index]
        X_test = X.loc[test_dates]
        if X_test.isna().any().any():
            raise ValueError(f"測試特徵存在缺失值：{target}")

        # Fold boundaries match model_data.py; within each fold, remove only
        # the first 12 unavailable lag rows (normally only in the first fold).
        for fold_no, (tr_idx, val_idx) in enumerate(folds, 1):
            fold_train_dates = train_dates[tr_idx]
            fold_val_dates = train_dates[val_idx]
            fold_train = X.loc[fold_train_dates].dropna()
            fold_val = X.loc[fold_val_dates]
            if fold_val.isna().any().any():
                raise ValueError("驗證月份的 Lag Features 不完整")
            model = LinearRegression().fit(fold_train, y.loc[fold_train.index])
            pred = model.predict(fold_val)
            cv_rows.append({
                "target": target, "fold": fold_no,
                "train_end": fold_train.index[-1].strftime("%Y-%m"),
                "validation_start": fold_val.index[0].strftime("%Y-%m"),
                "validation_end": fold_val.index[-1].strftime("%Y-%m"),
                **metrics(y.loc[fold_val.index], pred),
            })

        # Static fit: train once through 2024-06; no refitting during test.
        model = LinearRegression().fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions_by_target[target] = predicted
        test_rows.append({"target": target, **metrics(y.loc[test_dates], predicted)})
        coefficient_rows.append({
            "target": target, "intercept": float(model.intercept_),
            **dict(zip(X_train.columns, model.coef_)),
        })
        for date, actual, pred in zip(test_dates, y.loc[test_dates], predicted):
            prediction_rows.append({
                "date": date.strftime("%Y-%m"), "target": target,
                "actual": float(actual), "predicted": float(pred),
                "error": float(pred - actual),
            })

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    cv_df.to_csv(OUT / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT / "test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficient_rows).to_csv(OUT / "coefficients.csv", index=False, encoding="utf-8-sig")

    entry_cols = [c for c in targets if c.startswith("hsr_entry_")]
    actual_total = targets.loc[test_dates, entry_cols].sum(axis=1)
    static_total = predictions_by_target[entry_cols].sum(axis=1)
    seasonal_total = targets[entry_cols].sum(axis=1).shift(12).loc[test_dates]
    if seasonal_total.isna().any():
        raise ValueError("季節性基準缺少去年同月資料")
    total_predictions = pd.DataFrame({
        "actual_total_entries": actual_total,
        "static_OLS_lag_predicted": static_total,
        "seasonal_naive_predicted": seasonal_total,
    })
    total_predictions.index.name = "date"
    total_predictions.to_csv(OUT / "total_predictions.csv", encoding="utf-8-sig")
    comparison = pd.DataFrame([
        {"model": "Static OLS + Lag", **metrics(actual_total, static_total)},
        {"model": "Seasonal Naive (t-12)", **metrics(actual_total, seasonal_total)},
    ])
    comparison.to_csv(OUT / "total_metrics_comparison.csv", index=False, encoding="utf-8-sig")

    print("【實驗設定】")
    print(f"訓練期間：{train_dates.min():%Y-%m} ～ {train_dates.max():%Y-%m}（前 12 個月因 Lag 缺值不納入訓練）")
    print(f"最終測試：{test_dates.min():%Y-%m} ～ {test_dates.max():%Y-%m}")
    print(f"特徵數量：{len(calendar.columns) + len(LAGS) + 2}；預測目標：{len(targets.columns)}")
    print("訓練方式：只訓練一次；測試期間不更新係數")
    print("預測方式：逐月單步預測，使用已觀測的歷史真實運量建立 Lag Features")
    print("\n【各站進出站：交叉驗證平均指標】")
    print(cv_df[["MAE", "RMSE", "R2", "MAPE_pct"]].mean().round(3).to_string())
    print("\n【各站進出站：測試集平均指標】")
    print(test_df[["MAE", "RMSE", "R2", "MAPE_pct"]].mean().round(3).to_string())
    print("\n【全高鐵每月總進站人次：測試集比較】")
    print(comparison.to_string(index=False, formatters={
        "MAE": "{:,.0f}".format, "RMSE": "{:,.0f}".format,
        "R2": "{:.3f}".format, "MAPE_pct": "{:.2f}%".format,
    }))
    print(f"\n結果已儲存至：{OUT}")


if __name__ == "__main__":
    main()
