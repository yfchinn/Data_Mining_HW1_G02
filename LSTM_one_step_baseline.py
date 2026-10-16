"""One-step LSTM baseline for monthly Taiwan HSR demand.

Forecast each month using the preceding observed months (walk-forward evaluation).
Weights remain fixed during the test period. This matches a static one-step
OLS with observed lag features, NOT a 24-month forecast from one origin.

Run: python LSTM_one_step_baseline.py --csv data/processed/merged_monthly.csv
Dependencies: numpy pandas scikit-learn tensorflow
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

STATIONS = ('nangang','taipei','banqiao','taoyuan','hsinchu','miaoli',
            'taichung','changhua','yunlin','chiayi','tainan','zuoying')
TARGETS = [f'hsr_{d}_{s}' for d in ('entry','exit') for s in STATIONS]


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = actual - predicted
    nonzero = actual != 0
    return dict(MAE=float(mean_absolute_error(actual.ravel(), predicted.ravel())),
                RMSE=float(np.sqrt(mean_squared_error(actual.ravel(), predicted.ravel()))),
                R2=float(r2_score(actual, predicted)),
                MAPE_pct=float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan,
                WAPE_pct=float(100 * np.abs(error).sum() / np.abs(actual).sum()) if np.abs(actual).sum() else np.nan)


def show_metrics(title, m):
    print(f'\n【{title}】')
    for k in ('MAE','RMSE','R2','MAPE_pct','WAPE_pct'):
        print(f'{k:<13}{m[k]:>14.3f}')


def make_sequences(scaled, lookback, last_index):
    """Train targets at lookback..last_index-1, never later."""
    xs = np.stack([scaled[t-lookback:t] for t in range(lookback, last_index)])
    ys = scaled[lookback:last_index]
    return xs.astype('float32'), ys.astype('float32')


def network(lookback, n_targets, args):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, n_targets)),
        tf.keras.layers.LSTM(args.units),
        tf.keras.layers.Dropout(args.dropout),
        tf.keras.layers.Dense(n_targets),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss='mse')
    return m


def fit_predict(arr, fit_end, predict_end, args, epochs=None):
    """Scaler/model fitted only on arr[:fit_end]. Later actuals enter X only.

    Predictions at t use arr[t-lookback:t], never arr[t]. This is walk-forward
    evaluation with fixed model weights, not recursive multi-step forecasting.
    """
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    scaler = StandardScaler().fit(arr[:fit_end])
    scaled = scaler.transform(arr[:predict_end])
    X, y = make_sequences(scaled, args.lookback, fit_end)
    model = network(args.lookback, arr.shape[1], args)
    model.fit(X, y, epochs=epochs or args.epochs, batch_size=args.batch_size,
              shuffle=False, verbose=args.verbose)
    future_X = np.stack([scaled[t-args.lookback:t] for t in range(fit_end, predict_end)]).astype('float32')
    predicted = scaler.inverse_transform(model.predict(future_X, verbose=0))
    return np.maximum(predicted, 0), model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='data/processed/merged_monthly.csv')
    p.add_argument('--out', default='results/lstm_one_step')
    p.add_argument('--lookback', type=int, default=12)
    p.add_argument('--units', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', type=int, default=0)
    p.add_argument('--cv-folds', type=int, default=4)
    p.add_argument('--validation-months', type=int, default=24)
    p.add_argument('--test-months', type=int, default=24)
    args = p.parse_args()
    if args.lookback < 1 or args.epochs < 1 or args.test_months < 1:
        p.error('lookback, epochs and test-months must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    df = pd.read_csv(args.csv, encoding='utf-8-sig')
    missing = set(['date', *TARGETS]) - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    if df.index.has_duplicates or not df.index.equals(pd.date_range(df.index.min(), df.index.max(), freq='MS', name='date')):
        raise ValueError('Dates must be unique consecutive month starts')
    arr = df[TARGETS].apply(pd.to_numeric, errors='raise').to_numpy(dtype='float64')
    if not np.isfinite(arr).all(): raise ValueError('Missing or infinite target values')
    split = len(arr) - args.test_months
    if split <= args.lookback + args.cv_folds * args.validation_months:
        raise ValueError('Insufficient training months for chosen CV configuration')
    print('【實驗設定】')
    print(f'訓練期間：{df.index[0]:%Y-%m} ～ {df.index[split-1]:%Y-%m}（最初 {args.lookback} 個月僅作歷史輸入）')
    print(f'最終測試：{df.index[split]:%Y-%m} ～ {df.index[-1]:%Y-%m}')
    print(f'特徵數量：{len(TARGETS)}（歷史運量）；預測目標：{len(TARGETS)}')
    print(f'LSTM 歷史視窗：{args.lookback} 個月；Epoch：{args.epochs}；Units：{args.units}')
    print('訓練方式：每折訓練一次；最終模型在測試前訓練一次，測試期間不更新權重')
    print('預測方式：逐月單步預測，使用截至前一個月的已觀測真實運量')
    print('注意：這是 walk-forward one-step 評估，非一次預測未來 24 個月')
    cv = TimeSeriesSplit(n_splits=args.cv_folds, test_size=args.validation_months)
    fold_results = []
    for i, (tr, va) in enumerate(cv.split(arr[:split]), 1):
        # TimeSeriesSplit gives contiguous prefix training and subsequent validation.
        pred, _ = fit_predict(arr, len(tr), va[-1] + 1, args)
        truth = arr[va]
        m = metrics(truth, pred)
        fold_results.append({'fold':i, 'train_end':str(df.index[tr[-1]].date()),
                             'validation_start':str(df.index[va[0]].date()),
                             'validation_end':str(df.index[va[-1]].date()), **m})
        print(f'Fold {i}: {df.index[va[0]]:%Y-%m} ～ {df.index[va[-1]]:%Y-%m} | MAE={m["MAE"]:,.2f} | RMSE={m["RMSE"]:,.2f} | MAPE={m["MAPE_pct"]:.2f}%')
    fold_df = pd.DataFrame(fold_results)
    cv_mean = {k:float(fold_df[k].mean()) for k in ('MAE','RMSE','R2','MAPE_pct','WAPE_pct')}
    show_metrics('各站進出站：交叉驗證平均指標', cv_mean)
    pred, model = fit_predict(arr, split, len(arr), args)
    actual = arr[split:]
    show_metrics('各站進出站：測試集平均指標', metrics(actual, pred))
    n = len(STATIONS)
    actual_total = actual[:,:n].sum(axis=1)
    lstm_total = pred[:,:n].sum(axis=1)
    naive_total = np.stack([arr[t-12,:n].sum() for t in range(split, len(arr))])
    print('\n【全高鐵每月總進站人次：測試集比較】')
    print(f'{"model":<27}{"MAE":>12}{"RMSE":>12}{"R2":>10}{"MAPE_pct":>13}')
    for name, prediction in [('LSTM (One-step)',lstm_total),('Seasonal Naive (t-12)',naive_total)]:
        m = metrics(actual_total, prediction)
        print(f'{name:<27}{m["MAE"]:>12,.0f}{m["RMSE"]:>12,.0f}{m["R2"]:>10.3f}{m["MAPE_pct"]:>12.2f}%')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out/'cv_metrics.csv', index=False)
    pd.DataFrame([{'split':'cv_mean',**cv_mean},{'split':'test',**metrics(actual,pred)}]).to_csv(out/'metrics.csv', index=False)
    rows = [{'date':str(date.date()),'target':name,'actual':float(actual[i,j]),
             'predicted':float(pred[i,j]),'abs_error':float(abs(actual[i,j]-pred[i,j]))}
            for i,date in enumerate(df.index[split:]) for j,name in enumerate(TARGETS)]
    pd.DataFrame(rows).to_csv(out/'test_predictions.csv', index=False)
    pd.DataFrame([{'target':name,**metrics(actual[:,j],pred[:,j])} for j,name in enumerate(TARGETS)]).to_csv(out/'test_metrics_by_target.csv',index=False)
    model.save(out/'lstm_model.keras')
    print(f'\n結果已儲存至：{out.resolve()}')

if __name__ == '__main__':
    main()
