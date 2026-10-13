"""Static Lasso matching OLS_static_lag.py with eligible operating-history folds.

Coefficients remain fixed during test, but prior months' observed usage updates
the features. Alpha is selected per target using training-only mean CV MAE.
"""
import pandas as pd
from station_history import fold_info
from model_data import make_time_series_cv
from lasso_features import (ALPHAS, ROOT, coefficients, load_experiment,
                            metrics, model_for, save_results)


def main():
    data, targets, cache = load_experiment()
    out = ROOT / "results" / "experiment_v2" / "lasso_static_lag"
    out.mkdir(parents=True, exist_ok=True)
    folds = list(make_time_series_cv(data, n_splits=4, validation_months=24).split(data.X_train))
    validation_rows, coefficient_rows, settings = [], [], {}
    fold_audit = []
    predicted = pd.DataFrame(index=data.X_test.index)
    for target in targets:
        X, y = cache[target], targets[target]
        target_rows = []
        for alpha in ALPHAS:
            for fold_no, (tr, va) in enumerate(folds, 1):
                train = X.loc[data.X_train.index[tr]].dropna()
                validation = X.loc[data.X_train.index[va]]
                info = fold_info(target, fold_no, train, validation)
                if alpha == ALPHAS[0]:
                    fold_audit.append(info)
                if info["status"] == "skipped":
                    continue
                model = model_for(alpha).fit(train, y.loc[train.index])
                target_rows.append({**info, "alpha": alpha,
                                    "train_end": train.index[-1].strftime("%Y-%m"),
                                    "validation_start": validation.index[0].strftime("%Y-%m"),
                                    "validation_end": validation.index[-1].strftime("%Y-%m"),
                                    **metrics(y.loc[validation.index], model.predict(validation))})
        if not target_rows:
            raise ValueError(f"No usable validation folds for {target}")
        scores = pd.DataFrame(target_rows).groupby("alpha").MAE.mean()
        alpha = int(scores.idxmin())
        validation_rows.extend(target_rows)
        settings[target] = {"alpha": alpha, "validation_mae": scores.loc[alpha]}
        train = X.loc[data.X_train.index].dropna()
        model = model_for(alpha).fit(train, y.loc[train.index])
        settings[target].update(n_train=len(train), train_start=train.index[0].strftime("%Y-%m"),
                                train_end=train.index[-1].strftime("%Y-%m"),
                                n_validation_folds=len(target_rows) // len(ALPHAS))
        predicted[target] = model.predict(X.loc[data.X_test.index])
        coefficient_rows.append({"target": target, "alpha": alpha, **coefficients(model, X.columns)})
    pd.DataFrame(fold_audit).to_csv(out / "fold_audit.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(out / "alpha_validation_metrics.csv", index=False)
    selected = validation.apply(lambda row: row.alpha == settings[row.target]["alpha"], axis=1)
    validation.loc[selected].to_csv(out / "cv_metrics.csv", index=False)
    save_results(out, "Static Lasso + Lag", targets, predicted, settings, coefficient_rows)


if __name__ == "__main__":
    main()
