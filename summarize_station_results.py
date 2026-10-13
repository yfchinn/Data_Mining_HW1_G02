from pathlib import Path
import pandas as pd
import numpy as np
from lasso_features import metrics as calculate_metrics

root = Path(__file__).resolve().parent
out = root / 'results' / 'experiment_v2'
metrics = ['MAE', 'RMSE', 'R2', 'MAPE_pct']
keys = ['method', 'model', 'variant']
frames = [pd.read_csv(out / 'static_holiday_station_metrics.csv')[keys + ['target'] + metrics]]
for method, model, folder in [('static','OLS','ols_static_lag'),('static','Lasso','lasso_static_lag'),('rolling','OLS','ols_rolling'),('rolling','Lasso','lasso_rolling')]:
    frame = pd.read_csv(out / folder / 'test_metrics.csv')
    frame['method'], frame['model'], frame['variant'] = method, model, 'baseline'
    frames.append(frame[keys + ['target'] + metrics])
source = pd.read_csv(root / 'data' / 'processed_calendar' / 'merged_monthly.csv',
                     index_col='date', parse_dates=True)
test_dates = pd.period_range('2024-07', '2026-06', freq='M').to_timestamp()
seasonal_rows = []
for target in frames[0].target.unique():
    actual = source.loc[test_dates, target]
    previous_year = source[target].shift(12).loc[test_dates]
    assert actual.notna().all() and previous_year.notna().all()
    seasonal_rows.append({'method':'seasonal', 'model':'Naive', 'variant':'t-12',
                          'target':target, **calculate_metrics(actual, previous_year)})
frames.append(pd.DataFrame(seasonal_rows))
df = pd.concat(frames, ignore_index=True)
assert len(df) == 312 and not df.duplicated(keys + ['target']).any()
assert df.groupby(keys).size().eq(24).all()
assert np.isfinite(df[metrics].to_numpy()).all()
base = df[df.variant.eq('baseline')][['method','model','target','MAE','MAPE_pct']].rename(columns={'MAE':'baseline_MAE','MAPE_pct':'baseline_MAPE_pct'})
df = df.merge(base, on=['method','model','target'], how='left', validate='many_to_one')
df['mae_improvement_vs_baseline_pct'] = (1 - df.MAE / df.baseline_MAE) * 100
df['mape_improvement_vs_baseline_pp'] = df.baseline_MAPE_pct - df.MAPE_pct
seasonal = df[df.method.eq('seasonal')][['target','MAE','MAPE_pct']].rename(columns={'MAE':'seasonal_MAE','MAPE_pct':'seasonal_MAPE_pct'})
df = df.merge(seasonal, on='target', validate='many_to_one')
df['mae_improvement_vs_seasonal_pct'] = (1 - df.MAE / df.seasonal_MAE) * 100
df['mape_improvement_vs_seasonal_pp'] = df.seasonal_MAPE_pct - df.MAPE_pct
df['direction'] = df.target.str.split('_').str[1]
df['station'] = df.target.str.replace(r'^hsr_(entry|exit)_', '', regex=True)
df.to_csv(out/'station_method_comparison.csv',index=False,encoding='utf-8-sig')
summary = df.groupby(keys)[metrics].mean().reset_index()
counts = df.groupby(keys).agg(targets_improved_MAPE=('mape_improvement_vs_baseline_pp', lambda s: int((s > 1e-9).sum())), targets_improved_MAE=('mae_improvement_vs_baseline_pct', lambda s: int((s > 1e-9).sum())))
summary = summary.merge(counts,on=keys)
summary['mae_improvement_vs_seasonal_pct'] = (1 - summary.MAE / summary.loc[summary.method.eq('seasonal'), 'MAE'].iloc[0]) * 100
summary['mape_improvement_vs_seasonal_pp'] = summary.loc[summary.method.eq('seasonal'), 'MAPE_pct'].iloc[0] - summary.MAPE_pct
seasonal_counts = df.groupby(keys).agg(targets_better_than_seasonal_MAE=('mae_improvement_vs_seasonal_pct', lambda s: int((s > 1e-9).sum())),targets_better_than_seasonal_MAPE=('mape_improvement_vs_seasonal_pp', lambda s: int((s > 1e-9).sum())))
summary = summary.merge(seasonal_counts,on=keys)
summary.to_csv(out/'station_method_summary.csv',index=False,encoding='utf-8-sig')
station_names = dict(zip(['nangang','taipei','banqiao','taoyuan','hsinchu','miaoli','taichung','changhua','yunlin','chiayi','tainan','zuoying'],['南港','台北','板橋','桃園','新竹','苗栗','台中','彰化','雲林','嘉義','台南','左營']))
variant_names = {'baseline':'B','raw':'H','raw_log1p':'HL','t-12':'t−12'}
def label(row):
    if row['method'] == 'seasonal':
        return '前一年同月 t−12'
    return f"{row['method'].title()} {row['model']} {variant_names[row['variant']]}"
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'] + ['| '+' | '.join(map(str,row))+' |' for row in rows])
lines = ['# 各站預測表現比較', '', '測試期間：2024-07～2026-06，共 24 個月。每站進站、出站分別評估，共 24 個目標。', '', '- 前一年同月 t−12：每站每月以前一年同月的實際人次預測。', '- B（baseline）：原本特徵，包含 long_holiday_count。', '- H：以 started、ended 取代 long_holiday_count。', '- HL：H 再加上 log1p(started)、log1p(ended)，保留原值。', '- Rolling OLS window = 60 個月；Rolling Lasso window = 36 個月。', '- 各方法沿用營運後資料、時間切分與逐月預測規則。', '- MAE、RMSE、MAPE 越小越好；R² 越大越好。MAPE 單位為 %；MAE、RMSE 單位為人次。', '', '## 各設定的逐站平均', '', '先各自計算每個目標的測試指標，再對 24 個目標等權平均。不是全線加總指標，也不是客流加權平均。改善目標數以同一方法、同一模型的 B 為比較對象；進出站分開計數。與 t−12 的差值為「t−12 平均 MAPE − 該設定平均 MAPE」，正值表示模型更好。', '']
rows=[]
for _,r in summary.iterrows():
    rows.append([label(r),f'{r.MAE:,.0f}',f'{r.RMSE:,.0f}',f'{r.R2:.3f}',f'{r.MAPE_pct:.2f}',f'{r.mape_improvement_vs_seasonal_pp:+.2f}' if r.method != 'seasonal' else '—',f'{int(r.targets_better_than_seasonal_MAPE)}/24' if r.method != 'seasonal' else '—'])
lines += [table(['設定','平均 MAE','平均 RMSE','平均 R²','平均 MAPE (%)','對 t−12 差值 (百分點)','優於 t−12 目標數'],rows),'']
lines += ['## 各站 t−12 MAPE (%)', '', '每個數值由該站 24 個測試月份計算；下方四張表的同站同方向 MAPE 可直接與此表比較。', '']
rows=[]
for station,name in station_names.items():
    vals=[df[(df.method=='seasonal')&(df.station==station)&(df.direction==direction)].MAPE_pct.iloc[0] for direction in ['entry','exit']]
    rows.append([name,f'{vals[0]:.2f}',f'{vals[1]:.2f}'])
lines += [table(['站名','進站 t−12','出站 t−12'],rows),'']
for method,model in [('static','OLS'),('static','Lasso'),('rolling','OLS'),('rolling','Lasso')]:
    lines += [f'## {method.title()} {model}：各站 MAPE (%)','','粗體為同站同方向在 B、H、HL 中最低值。','']
    rows=[]
    for station,name in station_names.items():
        vals=[]
        for direction in ['entry','exit']:
            ss=df[(df.method==method)&(df.model==model)&(df.station==station)&(df.direction==direction)].set_index('variant')
            best=ss.MAPE_pct.min()
            vals += [f'**{ss.loc[v,"MAPE_pct"]:.2f}**' if ss.loc[v,'MAPE_pct']==best else f'{ss.loc[v,"MAPE_pct"]:.2f}' for v in ['baseline','raw','raw_log1p']]
        rows.append([name]+vals)
    lines += [table(['站名','進站 B','進站 H','進站 HL','出站 B','出站 H','出站 HL'],rows),'']
lines += ['## 各站在 13 種設定中的最低測試 MAPE','','包含 t−12 與 12 種模型設定。這是測試集的事後比較，不能將每站最低值拼成已驗證的新模型。正式選擇模型仍須依 validation 結果。','']
winnerrows=[]
for station,name in station_names.items():
    vals=[]
    for direction in ['entry','exit']:
        ss=df[(df.station==station)&(df.direction==direction)]
        win=ss.loc[ss.MAPE_pct.idxmin()]
        vals += [label(win),f'{win.MAPE_pct:.2f}']
    winnerrows.append([name]+vals)
lines += [table(['站名','進站最低設定','進站 MAPE (%)','出站最低設定','出站 MAPE (%)'],winnerrows),'','## 完整數值','','- [逐站各方法完整指標](station_method_comparison.csv)：312 筆，含 t−12、MAE、RMSE、R²、MAPE、相對 B 與 t−12 的差值。','- [各設定逐站平均](station_method_summary.csv)：13 筆，指標為 24 個目標的等權平均。','']
(out/'station_method_report.md').write_text('\n'.join(lines),encoding='utf-8')
print(summary.to_string(index=False))
print(table(['站名','進站最低設定','進站 MAPE (%)','出站最低設定','出站 MAPE (%)'],winnerrows))
