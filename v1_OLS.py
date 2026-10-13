"""OLS station-level HSR forecasting and comparable total-entry evaluation.

Run: python OLS.py
Requires model_data.py, preprocessing.py and the processed monthly CSV.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_data import load_model_data, make_time_series_cv

OUTPUT_DIR = Path(__file__).resolve().parent / 'results' / 'ols'


def metrics(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    nonzero = a != 0
    return {
        'MAE': mean_absolute_error(a, p),
        'RMSE': np.sqrt(mean_squared_error(a, p)),
        'R2': r2_score(a, p),
        'MAPE_pct': float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100) if nonzero.any() else np.nan,
    }


def main():
    data = load_model_data(test_months=24)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = list(make_time_series_cv(data, n_splits=4, validation_months=24).split(data.X_train))
    cv_rows, test_rows, prediction_rows, coefficients = [], [], [], []
    all_predictions = pd.DataFrame(index=data.X_test.index)

    print(f'Train: {data.X_train.index.min():%Y-%m} ~ {data.X_train.index.max():%Y-%m}')
    print(f'Test:  {data.X_test.index.min():%Y-%m} ~ {data.X_test.index.max():%Y-%m}')
    print(f'Features: {list(data.X_train.columns)}')
    print(f'Targets: {len(data.y_train.columns)}')

    for target in data.y_train.columns:
        y_train, y_test = data.y_train[target], data.y_test[target]
        for fold_no, (tr, va) in enumerate(folds, start=1):
            model = LinearRegression().fit(data.X_train.iloc[tr], y_train.iloc[tr])
            pred = model.predict(data.X_train.iloc[va])
            cv_rows.append({'target': target, 'fold': fold_no,
                            'train_end': data.X_train.index[tr[-1]].strftime('%Y-%m'),
                            'validation_start': data.X_train.index[va[0]].strftime('%Y-%m'),
                            'validation_end': data.X_train.index[va[-1]].strftime('%Y-%m'),
                            **metrics(y_train.iloc[va], pred)})
        model = LinearRegression().fit(data.X_train, y_train)
        pred = model.predict(data.X_test)
        all_predictions[target] = pred
        test_rows.append({'target': target, **metrics(y_test, pred)})
        coefficients.append({'target': target, 'intercept': model.intercept_,
                             **dict(zip(data.X_train.columns, model.coef_))})
        prediction_rows.extend({'date': dt.strftime('%Y-%m'), 'target': target,
                                'actual': actual, 'predicted': forecast,
                                'error': forecast - actual}
                               for dt, actual, forecast in zip(data.X_test.index, y_test, pred))

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    cv_df.to_csv(OUTPUT_DIR / 'cv_metrics.csv', index=False)
    test_df.to_csv(OUTPUT_DIR / 'test_metrics.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUTPUT_DIR / 'coefficients.csv', index=False)

    # Each trip normally contributes one entry and one exit; use ENTRY only.
    entry_cols = [c for c in data.y_test.columns if c.startswith('hsr_entry_')]
    actual_total = data.y_test[entry_cols].sum(axis=1)
    ols_total = all_predictions[entry_cols].sum(axis=1)
    # Previous year's same month is available for every held-out month.
    all_actual_entries = pd.concat([data.y_train[entry_cols], data.y_test[entry_cols]])
    seasonal_total = all_actual_entries.sum(axis=1).shift(12).loc[data.X_test.index]
    if seasonal_total.isna().any():
        raise ValueError('Seasonal naive baseline requires actual data 12 months earlier.')

    total_predictions = pd.DataFrame({'actual_total_entries': actual_total,
                                      'OLS_predicted': ols_total,
                                      'seasonal_naive_predicted': seasonal_total})
    total_predictions.index.name = 'date'
    total_predictions.to_csv(OUTPUT_DIR / 'total_predictions.csv', encoding='utf-8-sig')
    comparison = pd.DataFrame([
        {'model': 'OLS', **metrics(actual_total, ols_total)},
        {'model': 'Seasonal Naive (t-12)', **metrics(actual_total, seasonal_total)},
    ])
    comparison.to_csv(OUTPUT_DIR / 'total_metrics_comparison.csv', index=False)

    print('\nMean validation metrics across station targets and folds:')
    print(cv_df[['MAE', 'RMSE', 'R2', 'MAPE_pct']].mean().round(3).to_string())
    print('\nMean test metrics across station targets:')
    print(test_df[['MAE', 'RMSE', 'R2', 'MAPE_pct']].mean().round(3).to_string())
    print('\nTotal monthly ENTRY passenger counts (not entry + exit):')
    print(comparison.to_string(index=False, formatters={
        'MAE': '{:,.0f}'.format, 'RMSE': '{:,.0f}'.format,
        'R2': '{:.3f}'.format, 'MAPE_pct': '{:.2f}%'.format,
    }))
    print('\nNOTE: Compare with LASSO only if target, time split and MAPE definition match.')
    print(f'Saved results to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
