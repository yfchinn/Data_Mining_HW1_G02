"""動態 OLS：歷史運量特徵 + Rolling Window + 逐月回測。

執行：python OLS_rolling.py
資料：model_data.py 使用的 merged_monthly.csv
測試集：最後 24 個月（與 OLS.py 相同）
視窗選擇：僅在測試期之前的最後 24 個月驗證 36/60/120 月視窗。
每次只用預測月份之前已知的真實運量，避免未來資料洩漏。
"""
from pathlib import Path
import numpy as np
import pandas as pd
from station_history import features_for_target, common_windows
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_data import load_model_data

OUT = Path(__file__).resolve().parent / "results" / "experiment_v2" / "ols_rolling"
WINDOWS = (36, 60, 120)
VALIDATION_MONTHS = 24
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


def walk_forward(X, y, dates, window):
    """每月重新 fit；訓練僅使用當月以前、最近 window 筆可用樣本。"""
    rows, coef_rows = [], []
    for date in dates:
        history = X.index[X.index < date]
        valid = X.loc[history].notna().all(axis=1) & y.loc[history].notna()
        history = history[valid][-window:]
        if len(history) < window:
            raise ValueError(f"{date:%Y-%m} 可用訓練月份不足 {window} 個")
        if X.loc[[date]].isna().any().any():
            raise ValueError(f"{date:%Y-%m} 特徵含缺失值")
        model = LinearRegression().fit(X.loc[history], y.loc[history])
        prediction = float(model.predict(X.loc[[date]])[0])
        rows.append({"date": date, "actual": float(y.loc[date]), "predicted": prediction})
        coef_rows.append({
            "date": date, "n_train": len(history),
            "train_start": history[0].strftime("%Y-%m"),
            "train_end": history[-1].strftime("%Y-%m"),
            "intercept": float(model.intercept_),
            **{name: float(value) for name, value in zip(X.columns, model.coef_)},
        })
    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(coef_rows).set_index("date")


def main():
    data = load_model_data(test_months=24)
    OUT.mkdir(parents=True, exist_ok=True)
    calendar = pd.concat([data.X_train, data.X_test]).sort_index()
    targets = pd.concat([data.y_train, data.y_test]).sort_index()
    train_dates = data.X_train.index
    test_dates = data.X_test.index
    validation_dates = train_dates[-VALIDATION_MONTHS:]
    feature_cache = {target: features_for_target(calendar, targets[target])
                     for target in targets.columns}

    # 對每個視窗，使用訓練期末端的 walk-forward 驗證，
    # 並以 24 個進出站目標的平均 MAE 選擇視窗。
    windows, audit = common_windows(feature_cache, validation_dates, WINDOWS)
    audit.to_csv(OUT / "window_eligibility.csv", index=False)
    validation_rows = []
    for window in windows:
        for target in targets.columns:
            predictions, _ = walk_forward(
                feature_cache[target], targets[target], validation_dates, window
            )
            validation_rows.append({
                "window_months": window, "target": target,
                **metrics(predictions.actual, predictions.predicted)
            })
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(OUT / "window_validation_metrics.csv", index=False)
    summary = validation_df.groupby("window_months")[["MAE", "RMSE", "R2", "MAPE_pct"]].mean()
    summary.to_csv(OUT / "window_validation_summary.csv")
    best_window = int(summary["MAE"].idxmin())

    print("【實驗設定】")
    print(f"訓練期間：{train_dates.min():%Y-%m} ～ {train_dates.max():%Y-%m}")
    print(f"視窗驗證：{validation_dates.min():%Y-%m} ～ {validation_dates.max():%Y-%m}")
    print(f"最終測試：{test_dates.min():%Y-%m} ～ {test_dates.max():%Y-%m}")
    print(f"所有車站皆可使用的候選視窗：{windows} 個月；根據驗證集各站平均 MAE 選擇")
    print("\n【視窗驗證平均結果】")
    print(summary.round(3).to_string())
    print(f"\n選定視窗：{best_window} 個月")

    station_rows, prediction_rows, coefficient_rows = [], [], []
    for target in targets.columns:
        predictions, coefs = walk_forward(
            feature_cache[target], targets[target], test_dates, best_window
        )
        station_rows.append({"target": target, "window_months": best_window, **metrics(predictions.actual, predictions.predicted)})
        for date, row in predictions.iterrows():
            prediction_rows.append({
                "date": date.strftime("%Y-%m"), "target": target,
                "actual": row.actual, "predicted": row.predicted,
                "error": row.predicted - row.actual,
            })
        for date, row in coefs.iterrows():
            coefficient_rows.append({
                "date": date.strftime("%Y-%m"), "target": target, **row.to_dict()
            })

    station_df = pd.DataFrame(station_rows)
    station_df.to_csv(OUT / "test_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(OUT / "monthly_coefficients.csv", index=False)

    print("\n【各站進出站：測試集平均指標】")
    print(station_df[["MAE", "RMSE", "R2", "MAPE_pct"]].mean().round(3).to_string())
    print("\n提醒：此實驗每個月會使用上一個月已觀測到的真實運量重新訓練。")
    print("比較 LASSO 前，請先確認目標、資料期間、特徵與評估方法一致。")
    print(f"結果儲存位置：{OUT}")


if __name__ == "__main__":
    main()
