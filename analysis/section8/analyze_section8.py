#!/usr/bin/env python3
"""Reproducible, immutable read-only Section 8 research diagnostics.

All model hyperparameters and test predictions come from the existing frozen
experiment_v2 outputs; the script never selects variants based on test values.
HAC inference is auxiliary full-feature OLS; it is NOT inference on Lasso betas.
Moving-block Lasso resampling is a sensitivity interval, not selective inference.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr

BASE=Path(__file__).resolve().parent
# Supports both the original self-contained research package and a clean
# repository checkout with the analysis in <repo>/analysis/section8/.
PACKAGED_REPO = BASE / 'repo' / 'Data_Mining_HW1_G02-main'
REPO = PACKAGED_REPO if PACKAGED_REPO.is_dir() else BASE.parents[1]
if not (REPO / 'model_data.py').is_file() or not (REPO / 'results' / 'experiment_v2' / 'static_holiday_predictions.csv').is_file():
    raise SystemExit(f'Cannot find the HW1 repository or experiment_v2 files at {REPO}. See analysis/section8/README.md.')
RESULTS=REPO/'results'/'experiment_v2'
OUT=BASE/'data'
FIG=BASE/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(REPO))
from model_data import load_model_data
from static_holiday_experiment import experiment_features, FEATURE_COLUMNS
from lasso_features import model_for

RNG=np.random.default_rng(20261014)
LBOOT=999
LENGTH=6

P=pd.read_csv(RESULTS/'static_holiday_predictions.csv')
C=pd.read_csv(RESULTS/'static_holiday_coefficients.csv')
B=pd.read_csv(RESULTS/'lasso_rolling/test_predictions.csv')
S=pd.read_csv(RESULTS/'station_method_summary.csv')
H=P.query('method=="rolling" and model=="Lasso" and variant=="raw"').copy()
assert H.target.nunique()==24 and H.date.nunique()==24 and len(H)==576
assert len(B)==576
H['signed_error']=H['actual']-H['predicted']
H['absolute_error']=H.signed_error.abs()
H['APE_pct']=100*H.absolute_error/H.actual
B['signed_error']=B.actual-B.predicted
B['absolute_error']=B.signed_error.abs()
MERGED=H.merge(B[['date','target','signed_error','absolute_error']],on=['date','target'],suffixes=('_H','_B'),validate='one_to_one')
assert len(MERGED)==576
MON=H.groupby('date').agg(mean_signed_error=('signed_error','mean'),mean_abs_error=('absolute_error','mean'),mean_APE=('APE_pct','mean'),positive=('signed_error',lambda x:(x>0).sum()),negative=('signed_error',lambda x:(x<0).sum())).reset_index()
MON['positive_fraction']=MON.positive/24
# Monthly common-error variance decomposition, purely descriptive on test outcomes.
H['signed_pct_error']=100*H.signed_error/H.actual
month_mean=H.groupby('date').signed_pct_error.mean()
month_R2=1-float(((H.signed_pct_error-H.date.map(month_mean))**2).sum()/((H.signed_pct_error-H.signed_pct_error.mean())**2).sum())
MON.to_csv(OUT/'monthly_residual_signs.csv',index=False)
H.sort_values('absolute_error',ascending=False).head(24).to_csv(OUT/'top_24_absolute_errors.csv',index=False)
H.sort_values('APE_pct',ascending=False).head(24).to_csv(OUT/'top_24_relative_errors.csv',index=False)
H.to_csv(OUT/'frozen_rolling_lasso_H_test_predictions_with_errors.csv',index=False)

# Paired test-set contrasts: resample entire months to retain within-month
# cross-station dependence, and circular blocks to acknowledge temporal drift.
per_month=MERGED.groupby('date').apply(lambda z: pd.Series({
    'mean_abs_improvement_BminusH':(z.absolute_error_B-z.absolute_error_H).mean(),
    'mean_abs_H':z.absolute_error_H.mean(),
    'mean_abs_B':z.absolute_error_B.mean(),
}),include_groups=False).reset_index()
per_month.to_csv(OUT/'paired_monthly_difference.csv',index=False)
vals=per_month.mean_abs_improvement_BminusH.to_numpy(); assert len(vals)==24

def circular_block_boot(values,block,nrep=20000,seed=417):
    rng=np.random.default_rng(seed); n=len(values);ans=np.empty(nrep)
    for k in range(nrep):
        indices=[]
        while len(indices)<n:
            start=int(rng.integers(0,n))
            indices.extend((start+q)%n for q in range(block))
        ans[k]=values[np.array(indices[:n])].mean()
    return ans
unc={}
for bl in [3,6,12]:
    ds=circular_block_boot(vals,bl)
    unc[str(bl)]={'observed_passengers':float(vals.mean()),'percentile_95_CI_passengers':np.quantile(ds,[.025,.975]).tolist(),'Pr_gap_positive_bootstrap':float(np.mean(ds>0))}
# One monthly aggregate is not 24 iid paired observations; outcome signs are paired.

# Prepare targets and X for rolling June 2026 (latest fitted model for every
# station-direction). 11 columns; training histories are the model-specific 36
# most recent complete months; trained without post-date rows.
data=load_model_data(csv_path=REPO/'data/processed_calendar/merged_monthly.csv',feature_columns=FEATURE_COLUMNS,test_months=24)
calendar=pd.concat([data.X_train,data.X_test]).sort_index()
Y=pd.concat([data.y_train,data.y_test]).sort_index()
FINAL_DATE=pd.Timestamp('2026-06-01')
C_2026=C.query('method=="rolling" and model=="Lasso" and variant=="raw" and date=="2026-06"').copy()
assert len(C_2026)==24

coef_rows=[]; audit_rows=[]; full_lasso=[]
for k,target in enumerate(sorted(Y.columns)):
    candidate=C_2026.loc[C_2026.target.eq(target)].iloc[0]
    allX=experiment_features(calendar,Y[target],'raw')
    history=allX.loc[allX.index<FINAL_DATE].dropna().tail(int(candidate.n_train))
    assert len(history)==36 and history.index.min().strftime('%Y-%m')==candidate.train_start and history.index.max().strftime('%Y-%m')==candidate.train_end
    columns=history.columns.tolist()
    train_y=Y[target].loc[history.index].to_numpy(dtype=float)
    X=history.to_numpy(dtype=float)
    alpha=float(candidate.alpha)
    trained=model_for(alpha).fit(X,train_y)
    estimator=trained.named_steps['lasso']; scaler=trained.named_steps['standardscaler']
    raw_coef=estimator.coef_/scaler.scale_
    recorded=candidate[columns].astype(float).to_numpy()
    co_gap=float(np.max(np.abs(raw_coef-recorded)))
    actual_ref=H.query('date=="2026-06" and target==@target').iloc[0]
    predicted=float(trained.predict(allX.loc[[FINAL_DATE]].to_numpy())[0])
    prediction_gap=abs(predicted-float(actual_ref.predicted))
    assert prediction_gap<.01,(target,prediction_gap)
    assert co_gap <.01,(target,co_gap)
    audit_rows.append({'target':target,'alpha':alpha,'window_months':36,'n_train':36,
        'start_date':candidate.train_start,'end_date':candidate.train_end,
        'max_abs_raw_beta_reproduction_difference':co_gap,'abs_test_prediction_reproduction_difference':prediction_gap})
    # Feature standardization holds the original LAST-FIT SD for each coefficient
    # across all bootstraps to make scales comparable.
    standard_betas=raw_coef*scaler.scale_
    yfit=trained.predict(X)
    residual=train_y-yfit
    residual=residual-residual.mean()
    rng=np.random.default_rng(40401+k)
    boot_betas=np.empty((LBOOT,len(columns)))
    for q in range(LBOOT):
        blocks=[]
        while len(blocks)<len(train_y):
            start=int(rng.integers(0,len(train_y)))
            blocks.extend((start+j)%len(train_y) for j in range(LENGTH))
        yboot=yfit+residual[np.asarray(blocks[:len(train_y)])]
        bootfit=model_for(alpha).fit(X,yboot)
        sb=bootfit.named_steps['lasso'].coef_ / bootfit.named_steps['standardscaler'].scale_ *scaler.scale_
        boot_betas[q]=sb
    # Auxiliary OLS HAC on exactly the same window and ALL 11 features.
    # Its p-values are NOT significance tests on penalized Lasso coefficients.
    Xz=(X-scaler.mean_)/scaler.scale_
    XX=sm.add_constant(Xz,has_constant='add')
    ols=sm.OLS(train_y,XX).fit(cov_type='HAC',cov_kwds={'maxlags':3},use_t=True)
    # Same OLS design matrix, variance-inflation calculation excluding intercept
    vif=np.array([variance_inflation_factor(XX,j+1) for j in range(len(columns))])
    bh=multipletests(ols.pvalues[1:],method='fdr_bh')[1]
    correlations=[float(np.corrcoef(X[:,j],train_y)[0,1]) for j in range(len(columns))]
    for j,feature in enumerate(columns):
        boot=boot_betas[:,j]
        # Report Lasso 95% resampling sensitivity intervals, but never claim
        # confidence validity after data-adaptive L1 selection.
        q025,q975=np.quantile(boot,[.025,.975]); cilo,cihi=ols.conf_int()[j+1]
        # Conditional model-selection stability = fraction nonzero.
        coef_rows.append({'target':target,'feature':feature,'alpha':alpha,'training_months':len(history),
            'forecast_month':'2026-06', 'lasso_beta_per_training_SD_passengers':standard_betas[j],
            'lasso_beta_raw_units':raw_coef[j],
            'lasso_boot_sensitivity_lower_95':q025,'lasso_boot_sensitivity_upper_95':q975,
            'lasso_boot_selection_rate':float(np.mean(abs(boot)>1e-8)),
            'ols_HAC_beta_per_training_SD_passengers':ols.params[j+1],
            'ols_HAC_lower_95':cilo,'ols_HAC_upper_95':cihi,
            'ols_HAC_p_nominal':ols.pvalues[j+1],
            'ols_HAC_p_BH11_exploratory':bh[j],
            'vif':vif[j],'pearson_r_training':correlations[j]})
    full_lasso.append(pd.DataFrame(boot_betas,columns=columns).assign(target=target))

AUDIT=pd.DataFrame(audit_rows); COEFF=pd.DataFrame(coef_rows)
AUDIT.to_csv(OUT/'model_reproduction_audit.csv',index=False)
COEFF.to_csv(OUT/'ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv',index=False)
assert len(COEFF)==264

# Comprehensive 24*24 coefficient export into per-training SD effect sizes;
# status and selection stability aggregated across real, frozen rolling months.
rolling=C.query('method=="rolling" and model=="Lasso" and variant=="raw"').copy()
assert len(rolling)==24*24
all_col=COEFF.feature.unique().tolist()
sel=(rolling[all_col].abs()>1e-12).mean().sort_values(ascending=False)
sel.to_csv(OUT/'rolling_lasso_feature_selection_frequencies.csv',header=['selection_frequency'])

# Per target sample dates. Large errors include both absolute and relative
# failures, without guessing causal attributions.
FAIL=H.sort_values('absolute_error',ascending=False).head(6).copy()
FAIL2=H.sort_values('APE_pct',ascending=False).head(3).copy()
CASES=pd.concat([FAIL,FAIL2]).drop_duplicates(['date','target'])
ctx=[]
for row in CASES.itertuples():
    target=row.target;date=pd.Timestamp(row.date)
    xx=experiment_features(calendar,Y[target],'raw').loc[date]
    ctx.append({'date':row.date,'target':row.target,'actual_passengers':row.actual,
        'predicted_passengers':row.predicted,'signed_error_actual_minus_pred':row.signed_error,
        'APE_pct':row.APE_pct,'lag_1_passengers':xx.lag_1,'lag_12_passengers':xx.lag_12,
        'holiday_days_started':xx.long_holiday_days_started,'holiday_days_ended':xx.long_holiday_days_ended,
        'days_in_month':xx.days_in_month})
CTX=pd.DataFrame(ctx)
CTX.to_csv(OUT/'worst_test_cases_with_available_inputs.csv',index=False)

# Systematic seasonal missed trends. Temporal residual correlation for individual
# series and average across series; descriptive due 24 months.
from scipy.stats import pearsonr
r1=[]
for tar,g in H.groupby('target'):
    g=g.sort_values('date');res=g.signed_error.to_numpy()
    ac=np.corrcoef(res[1:],res[:-1])[0,1]
    r1.append({'target':tar,'lag1_residual_autocorr':ac,
               'LjungBox_3_pvalue_exploratory':float(acorr_ljungbox(res,lags=[3],return_df=True)['lb_pvalue'].iloc[0])})
pd.DataFrame(r1).to_csv(OUT/'residual_autocorrelation_by_target.csv',index=False)

summary={
 'source':'Uploaded ZIP Data_Mining_HW1_G02-main, experiment_v2',
 'reference_model':'rolling Lasso, holiday raw (H), per-target alpha, 36 month window',
 'train_window_for_coefficients':'2023-06 through 2026-05, 36 observations per target',
 'test_period':'2024-07 through 2026-06, 24 months x 24 targets',
 'test_MAE_equal_target_weight':float(H.groupby('target').absolute_error.mean().mean()),
 'test_MAPE_equal_target_weight':float(H.groupby('target').APE_pct.mean().mean()),
 'test_MAE_B':float(B.groupby('target').absolute_error.mean().mean()),
 'mean_gap_BminusH':float(vals.mean()),
 'relative_MAE_gain_pct':float(100*vals.mean()/B.absolute_error.mean()),
 'paired_month_block_bootstrap':unc,
 'month_2025_10_mean_signed_error':float(MON.set_index('date').loc['2025-10','mean_signed_error']),
 'month_fixed_effect_R2_on_signed_percentage_errors_descriptive':month_R2,
 'month_2025_10_n_positive':int(MON.set_index('date').loc['2025-10','positive']),
 'month_2026_03_n_positive':int(MON.set_index('date').loc['2026-03','positive']),
 'month_2024_10_n_positive':int(MON.set_index('date').loc['2024-10','positive']),
 'feature_selection_over_all_month_target_models':sel.to_dict(),
 'lastfit_models_reproduced':int(len(AUDIT)),
 'max_reproduction_pred_error':float(AUDIT.abs_test_prediction_reproduction_difference.max()),
 'max_reproduction_coefficient_error':float(AUDIT.max_abs_raw_beta_reproduction_difference.max()),
 'n_targets':24,'n_months':24,'B_residual_block_Lasso_sensitivity':LBOOT,'bootstrap_residual_block_length':LENGTH,
 'Lasso_coefficients_are_not_OLS_inference':True,
 'no_test_time_hyperparameter_selection':True,
}
(BASE/'analysis_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

# Plots for ACL integration, no interactivity necessary.
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':230})
colordark='#132f48'; colormid='#166c8c';coloralt='#a64c3f'
rep=COEFF[COEFF.target.eq('hsr_entry_taipei')].copy().iloc[::-1]
short={'month_index':'Long-run month index','month_sin':'Month sine','month_cos':'Month cosine','days_in_month':'Days in month','long_holiday_days_started':'Holiday start-days','long_holiday_days_ended':'Holiday end-days','lag_1':'Ridership lag 1 mo.','lag_3':'Ridership lag 3 mo.','lag_12':'Ridership lag 12 mo.','rolling_mean_3':'Rolling mean 3 mo.','rolling_mean_12':'Rolling mean 12 mo.'}
fig,ax=plt.subplots(figsize=(6.75,3.05)); xx=rep.lasso_beta_per_training_SD_passengers.to_numpy()/1000;
lo=rep.lasso_boot_sensitivity_lower_95.to_numpy()/1000;hi=rep.lasso_boot_sensitivity_upper_95.to_numpy()/1000
ys=np.arange(len(rep))
ax.axvline(0,c='#666666',lw=.7)
ax.errorbar(xx,ys,xerr=np.vstack([xx-lo,hi-xx]),fmt='o',markersize=3.6,elinewidth=1.0,capsize=2,color=colormid,ecolor=colormid)
ax.set_yticks(ys,rep.feature.map(short))
ax.set_xlabel('Lasso coefficient (thousand passengers per training-feature SD)')
ax.grid(axis='x',alpha=.16)
fig.tight_layout();fig.savefig(FIG/'taipei_entry_Lasso_bootstrap_sensitivity.pdf',bbox_inches='tight');fig.savefig(FIG/'taipei_entry_Lasso_bootstrap_sensitivity.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.75,2.6));months=pd.to_datetime(MON.date); w=.65
bars=ax.bar(np.arange(24),MON.mean_signed_error/1000,color=[coloralt if v<0 else colormid for v in MON.mean_signed_error],alpha=.91)
ax.axhline(0,color=colordark,lw=.7)
ax.set_xticks(np.arange(0,24,2),[f'{d:%b %y}' for d in months.iloc[::2]],rotation=35,ha='right')
ax.set_ylabel('Mean actual − predicted\n(thousand passengers/target)')
ax.grid(axis='y',alpha=.16)
fig.tight_layout();fig.savefig(FIG/'monthly_common_residual_bias.pdf',bbox_inches='tight');fig.savefig(FIG/'monthly_common_residual_bias.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.75,2.35));sel2=sel.iloc[::-1];ax.barh([short[x] for x in sel2.index],sel2.values*100,color=colormid,alpha=.9)
ax.set_xlim(0,101);ax.set_xlabel('Nonzero Lasso weight (% of 576 target–month fits)');ax.grid(axis='x',alpha=.16)
fig.tight_layout();fig.savefig(FIG/'lasso_rolling_selection_frequency.pdf',bbox_inches='tight');fig.savefig(FIG/'lasso_rolling_selection_frequency.png',bbox_inches='tight');plt.close(fig)
print(json.dumps(summary,indent=2))
print('\nTOP_CONTEXT\n',CTX[['date','target','actual_passengers','predicted_passengers','signed_error_actual_minus_pred','APE_pct','lag_12_passengers','holiday_days_started','holiday_days_ended']].round(2).to_string(index=False))
print('\nREP_COEFF\n',COEFF.query('target=="hsr_entry_taipei"')[['feature','lasso_beta_per_training_SD_passengers','lasso_boot_sensitivity_lower_95','lasso_boot_sensitivity_upper_95','ols_HAC_beta_per_training_SD_passengers','ols_HAC_lower_95','ols_HAC_upper_95','ols_HAC_p_nominal','vif']].round(3).to_string(index=False))
