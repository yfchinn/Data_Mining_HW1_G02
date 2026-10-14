"""Exploratory, conditional Lasso zero-effect bootstrap tests for one station.
For H0: coefficient for feature j = 0 in linear conditional-mean model,
fit restricted full-sample OLS excluding j to obtain null mean, then resample
centered restricted residuals in circular six-month blocks; refit the fixed-
alpha Lasso test statistic. Hyperparameter CV selection is NOT bootstrapped.
"""
import sys, json
from pathlib import Path
import numpy as np,pandas as pd,statsmodels.api as sm
base=Path(__file__).resolve().parent
packaged_repo=base/'repo'/'Data_Mining_HW1_G02-main'
repo=packaged_repo if packaged_repo.is_dir() else base.parents[1]
if not (repo/'model_data.py').is_file():
    raise SystemExit(f'Cannot find HW1 repository at {repo}. See analysis/section8/README.md.')
sys.path.insert(0,str(repo))
from model_data import load_model_data
from static_holiday_experiment import experiment_features,FEATURE_COLUMNS
from lasso_features import model_for
CO=pd.read_csv(base/'data'/'ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv')
data=load_model_data(csv_path=repo/'data/processed_calendar/merged_monthly.csv',feature_columns=FEATURE_COLUMNS,test_months=24)
cal=pd.concat([data.X_train,data.X_test]); targets=pd.concat([data.y_train,data.y_test])
target='hsr_entry_taipei'; X=experiment_features(cal,targets[target],'raw').loc['2023-06':'2026-05']; yy=targets[target].loc[X.index].to_numpy(); a=10000.;cols=list(X.columns)
fit=model_for(a).fit(X,yy);scale=fit.named_steps['standardscaler'].scale_;observed=fit.named_steps['lasso'].coef_; B=1999; rng=np.random.default_rng(20261014); L=6; N=len(yy)
assert N==36
rows=[]
for j,feature in enumerate(cols):
    restricted=sm.OLS(yy,sm.add_constant(X.drop(columns=feature).to_numpy())).fit()
    mu=restricted.fittedvalues; err=np.asarray(restricted.resid)-np.mean(restricted.resid)
    nullstats=np.empty(B)
    for b in range(B):
        ids=[]
        while len(ids)<N:
            start=int(rng.integers(N))
            ids.extend((start+k)%N for k in range(L))
        ystar=mu+err[np.asarray(ids[:N])]
        model=model_for(a).fit(X,ystar)
        nullstats[b]=model.named_steps['lasso'].coef_[j]
    p=(1+int(np.sum(np.abs(nullstats)>=abs(observed[j])-1e-9)))/(B+1)
    rows.append({'target':target,'feature':feature,'observed_lasso_beta_per_feature_SD':observed[j],
        'approx_null_bootstrap_two_sided_p':p,'null_test_reps':B,'block_len':L})
Q=pd.DataFrame(rows);Q.to_csv(base/'data'/'taipei_lasso_conditional_null_bootstrap_pvalues.csv',index=False)
A=CO.query('target==@target').merge(Q[['feature','approx_null_bootstrap_two_sided_p']],on='feature',validate='one_to_one');A.to_csv(base/'data'/'taipei_entry_complete_beta_CI_p_VIF.csv',index=False)
print(A[['feature','lasso_beta_per_training_SD_passengers','lasso_boot_sensitivity_lower_95','lasso_boot_sensitivity_upper_95','approx_null_bootstrap_two_sided_p','vif','ols_HAC_p_nominal','ols_HAC_p_BH11_exploratory']].round(4).to_string(index=False))
