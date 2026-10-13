"""Rolling Lasso matching OLS_rolling.py: ten features, monthly refits.

Select an eligible common window on the last 24 pre-test months, with
per-target alpha selected on that same validation period. Freeze both for test.
"""
import pandas as pd
from station_history import common_windows
from lasso_features import (ALPHAS, ROOT, coefficients, load_experiment,
                            metrics, model_for, save_results)

WINDOWS = (36, 60, 120)


def walk_forward(X, y, dates, window, alpha, keep_coefficients=False):
    predicted, coefficient_rows = {}, []
    for date in dates:
        history = X.loc[X.index < date].dropna().tail(window)
        if len(history) != window:
            raise ValueError(f"Insufficient history for {date}: {window} months")
        model = model_for(alpha).fit(history, y.loc[history.index])
        predicted[date] = float(model.predict(X.loc[[date]])[0])
        if keep_coefficients:
            coefficient_rows.append({"date": date.strftime("%Y-%m"),
                                     "n_train": len(history),
                                     "train_start": history.index[0].strftime("%Y-%m"),
                                     "train_end": history.index[-1].strftime("%Y-%m"),
                                     **coefficients(model, X.columns)})
    return pd.Series(predicted), coefficient_rows


def main():
    data, targets, cache = load_experiment()
    out = ROOT / "results" / "experiment_v2" / "lasso_rolling"
    out.mkdir(parents=True, exist_ok=True)
    validation_dates = data.X_train.index[-24:]
    windows, audit = common_windows(cache, validation_dates, WINDOWS)
    audit.to_csv(out / "window_eligibility.csv", index=False)
    print(f"Eligible common windows: {windows}", flush=True)
    rows = []
    for window in windows:
        print(f"Validating window={window}", flush=True)
        for target in targets:
            for alpha in ALPHAS:
                predicted, _ = walk_forward(cache[target], targets[target], validation_dates, window, alpha)
                rows.append({"window_months": window, "target": target, "alpha": alpha,
                             **metrics(targets.loc[validation_dates, target], predicted)})
    validation = pd.DataFrame(rows)
    validation.to_csv(out / "alpha_window_validation_metrics.csv", index=False)
    best_rows = validation.loc[validation.groupby(["window_months", "target"]).MAE.idxmin()]
    best_rows.to_csv(out / "window_validation_metrics.csv", index=False)
    summary = best_rows.groupby("window_months")[["MAE", "RMSE", "R2", "MAPE_pct"]].mean()
    summary.to_csv(out / "window_validation_summary.csv")
    window = int(summary.MAE.idxmin())
    print(f"Selected window={window}\n{summary}", flush=True)
    selection = best_rows[best_rows.window_months.eq(window)].set_index("target")
    settings, coefficient_rows = {}, []
    predicted = pd.DataFrame(index=data.X_test.index)
    for target in targets:
        alpha = int(selection.loc[target, "alpha"])
        settings[target] = {"window_months": window, "alpha": alpha,
                            "validation_mae": selection.loc[target, "MAE"]}
        values, coefs = walk_forward(cache[target], targets[target], data.X_test.index,
                                    window, alpha, keep_coefficients=True)
        predicted[target] = values
        coefficient_rows.extend({"target": target, "alpha": alpha, "window_months": window,
                                 **row} for row in coefs)
    save_results(out, "Rolling Lasso", targets, predicted, settings, coefficient_rows)


if __name__ == "__main__":
    main()
