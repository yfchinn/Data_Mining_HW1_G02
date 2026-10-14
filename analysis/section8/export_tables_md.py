#!/usr/bin/env python3
"""Generate readable GitHub Markdown tables from existing Section 8 CSV/JSON outputs.

Execute after analyze_section8.py and lasso_null_tests.py:
    cd analysis/section8
    python export_tables_md.py

Reads existing audit files; does NOT refit forecasting models, recalculate
inferential statistics, or modify the original results/experiment_v2 directory.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = HERE / 'TABLES.md'
OUT_ALL = HERE / 'ALL_24_TARGETS_TABLES.md'
OUT_SUPP = HERE / 'SUPPLEMENTAL_TABLES.md'


def read_csv(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def number(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f'Invalid numerical value: {value}')
    return x


def signed(value, decimals=1, thousands=False):
    val = number(value) / (1000.0 if thousands else 1.0)
    if abs(val) < 0.5 * 10 ** -decimals:
        val = 0.0
    prefix = '+' if val > 0 else ''
    return f'{prefix}{val:,.{decimals}f}'


def integer(value):
    return f'{round(number(value)):,}'


def prob(value):
    v = number(value)
    if v < 0.001 and v > 0:
        return f'{v:.2e}'
    return f'{v:.4g}'


def interval(lo, hi, decimals=1, thousands=True):
    return f'[{signed(lo, decimals, thousands)}, {signed(hi, decimals, thousands)}]'


def name(feature):
    names = {
        'month_index': 'Month index',
        'month_sin': 'Month sine',
        'month_cos': 'Month cosine',
        'days_in_month': 'Days / month',
        'long_holiday_days_started': 'Holiday start days',
        'long_holiday_days_ended': 'Holiday end days',
        'lag_1': 'Lag 1 month',
        'lag_3': 'Lag 3 months',
        'lag_12': 'Lag 12 months',
        'rolling_mean_3': 'Mean, prev. 3 mo.',
        'rolling_mean_12': 'Mean, prev. 12 mo.',
    }
    return names.get(feature, feature)


def station(target):
    prefix, direction, location = target.split('_', 2)
    return f'{location.title()} {direction}'


def markdown_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(lines)


def main():
    a = json.loads((HERE / 'analysis_results.json').read_text(encoding='utf-8'))
    taipei = read_csv('taipei_entry_complete_beta_CI_p_VIF.csv')
    all24 = read_csv('ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv')
    worst = read_csv('worst_test_cases_with_available_inputs.csv')
    month = read_csv('monthly_residual_signs.csv')
    sel = read_csv('rolling_lasso_feature_selection_frequencies.csv')
    audit = read_csv('model_reproduction_audit.csv')
    if len(taipei) != 11 or len(all24) != 264 or len(month) != 24 or len(audit) != 24:
        raise AssertionError('Unexpected row count: re-run analyze_section8.py, then lasso_null_tests.py')
    if any(not row.get('approx_null_bootstrap_two_sided_p', '').strip() for row in taipei):
        raise AssertionError('Missing approximate Lasso null p-values: run lasso_null_tests.py first')
    lines = [
        '# Section 8 — Statistical and error analysis tables',
        '',
        '> **Provenance:** Generated from the CSV and JSON artifacts in this directory by `python export_tables_md.py`. '
        'The model outputs originate in `results/experiment_v2/`. This document is a readable Markdown rendering of '
        'those computed analyses, not an additional experiment. Values are rounded only for presentation.',
        '',
        '## 1. Model and test summary',
        '',
        markdown_table(['Quantity', 'Result'], [
            ['Reference forecast', 'Rolling Lasso, holiday-duration features (H)'],
            ['Final-fit coefficient window', str(a['train_window_for_coefficients'])],
            ['Held-out test period', str(a['test_period'])],
            ['Separate station–direction targets', str(a['n_targets'])],
            ['Held-out forecasts', str(a['n_targets'] * a['n_months'])],
            ['H test MAE, equal-target average (passengers)', integer(a['test_MAE_equal_target_weight'])],
            ['H test MAPE, equal-target average', f"{a['test_MAPE_equal_target_weight']:.2f}%"],
            ['Baseline rolling Lasso B test MAE (passengers)', integer(a['test_MAE_B'])],
            ['Observed B − H MAE difference (passengers)', signed(a['mean_gap_BminusH'], 1)],
            ['Observed relative MAE improvement', f"{a['relative_MAE_gain_pct']:.2f}%"],
        ]), '',
        '**Important:** H is an exploratory test-set comparison, not a retrospectively proven prospectively selected winner. '
        'The target outcomes share temporal and system-wide dependence.',
        '',
        '## 2. All 11 June 2026 Taipei-entry rolling Lasso predictors',
        '',
        '**Units:** β and bootstrap interval = **thousands of predicted passengers per one training-window standard deviation of the predictor**, '
        'not passengers per raw input unit. The 95% intervals are residual-block **sensitivity intervals**, not formal selective-inference confidence intervals. '
        'The `p₀` column is an exploratory restricted-null fixed-α block-bootstrap p-value (1,999 draws).',
        '',
        markdown_table(['Predictor', 'Lasso β (thousands)', '95% bootstrap interval', 'Approx. p₀', 'VIF'], [
            [name(r['feature']), signed(r['lasso_beta_per_training_SD_passengers'], 1, True),
             interval(r['lasso_boot_sensitivity_lower_95'], r['lasso_boot_sensitivity_upper_95']),
             prob(r['approx_null_bootstrap_two_sided_p']), f"{number(r['vif']):.1f}"]
            for r in taipei
        ]), '',
        'Predictors shrunk to exactly zero are **not** evidence of zero population association. '
        'VIF is a collinearity diagnostic of the training design, not a Lasso p-value.',
        '',
        '## 3. Separate auxiliary OLS / Newey–West HAC table',
        '',
        '**Units:** thousands of passengers per one training-feature SD, as above. '
        'This is an **unpenalized auxiliary OLS estimator**, fitted on the same 36-month design, '
        'with Newey–West HAC covariance (3 lags). These βs, 95% intervals, p-values and BH-adjusted q-values '
        '**do not refer to the fitted Lasso coefficients**. BH correction is exploratory within 11 predictors.',
        '',
        markdown_table(['Predictor', 'OLS β (thousands)', 'HAC 95% CI', 'HAC p', 'BH q'], [
            [name(r['feature']), signed(r['ols_HAC_beta_per_training_SD_passengers'], 1, True),
             interval(r['ols_HAC_lower_95'], r['ols_HAC_upper_95']),
             prob(r['ols_HAC_p_nominal']), prob(r['ols_HAC_p_BH11_exploratory'])]
            for r in taipei
        ]), '',
        '### Collinearity and generalization',
        '',
    ]
    n_hi = sum(number(r['vif']) > 10 for r in all24)
    n_zero = sum(abs(number(r['lasso_beta_per_training_SD_passengers'])) < 1e-9 for r in all24)
    max_vif = max(all24, key=lambda r: number(r['vif']))
    lines.extend([
        f'- **{n_hi}/{len(all24)}** final-window predictor–target pairs have VIF > 10.',
        f'- **{n_zero}/{len(all24)}** final-window Lasso coefficients are zero.',
        f"- Maximum observed VIF: **{number(max_vif['vif']):.1f}** ({station(max_vif['target'])}, {name(max_vif['feature'])}).",
        '- Full readable tables for all 24 targets are in [ALL_24_TARGETS_TABLES.md](ALL_24_TARGETS_TABLES.md).',
        '- Every one of the **24 × 11 = 264** final-window entries, including standardized Lasso coefficients, '
        'bootstrap intervals and auxiliary OLS/HAC p/q-values, is available in '
        '[the complete 24-target audit CSV](data/ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv).',
        '',
        '## 4. Feature selection across all 576 monthly Lasso fits',
        '',
        markdown_table(['Feature', 'Nonzero selection rate'], [
            [name(r.get('Unnamed: 0') or r.get('') or next(iter(r.values()))), f"{number(r['selection_frequency'])*100:.1f}%"]
            for r in sel
        ]), '',
        'Selection frequency describes how often a fitted coefficient is nonzero; it is not causal importance or statistical significance.',
        '',
        '## 5. Highest held-out errors, with available feature context',
        '',
        '**Signed error = actual − predicted**, in passengers. A negative value means overprediction. '
        'The first three rows are the largest absolute errors among the 576 target-month cases. '
        'The remaining examples expose other important failure patterns.',
        '',
        markdown_table(['Month', 'Station / flow', 'Actual', 'Predicted', 'Signed error', 'APE', 'Lag-1', 'Lag-12', 'Holiday start/end'], [
            [r['date'], station(r['target']), integer(r['actual_passengers']), integer(r['predicted_passengers']),
             signed(r['signed_error_actual_minus_pred'], 0), f"{number(r['APE_pct']):.2f}%",
             integer(r['lag_1_passengers']), integer(r['lag_12_passengers']),
             f"{integer(r['holiday_days_started'])} / {integer(r['holiday_days_ended'])}"]
            for r in worst
        ]), '',
        'For the full ranked lists, see [24 largest absolute errors](data/top_24_absolute_errors.csv) '
        'and [24 largest percentage errors](data/top_24_relative_errors.csv). '
        'An observed error and its calendar context do **not** establish its cause.',
        '',
        '## 6. Monthly shared error pattern',
        '',
        'All figures below summarize the 24 station–direction forecasts in each month. '
        'The counts of positive/negative errors are **not independent statistical replications**.',
        '',
        markdown_table(['Test month', 'Mean signed error', 'Mean absolute error', 'Mean APE', 'Underpredicted', 'Overpredicted'], [
            [r['date'], signed(r['mean_signed_error'], 0), integer(r['mean_abs_error']),
             f"{number(r['mean_APE']):.2f}%", r['positive'], r['negative']]
            for r in month
        ]), '',
        f"In October 2025, all {int(a['month_2025_10_n_positive'])} targets were underpredicted, "
        f"with mean signed error **{signed(a['month_2025_10_mean_signed_error'], 0)} passengers**. "
        f"Month group means account for **{100*a['month_fixed_effect_R2_on_signed_percentage_errors_descriptive']:.1f}%** "
        'of the variation in signed percentage errors as a *post-hoc descriptive decomposition*, '
        'not a pre-forecast predictive variable.',
        '',
        '## 7. Paired rolling Lasso B vs H test-error sensitivity',
        '',
        'The comparison uses the same 24 targets × 24 months. It pairs errors within target–month, '
        'averages across targets in each month, and resamples entire months in circular blocks. '
        'Positive B − H means H has lower MAE.',
        '',
        markdown_table(['Month block length', 'Observed MAE gain (passengers)', 'Bootstrap 95% percentile interval (passengers)',
                        'Proportion of bootstrap differences > 0'], [
            [f'{block} months', signed(v['observed_passengers'], 1),
             interval(v['percentile_95_CI_passengers'][0],v['percentile_95_CI_passengers'][1], decimals=0, thousands=False),
             f"{100*v['Pr_gap_positive_bootstrap']:.2f}%"]
            for block,v in sorted(a['paired_month_block_bootstrap'].items(),key=lambda x:int(x[0]))
        ]), '',
        'The conclusion is sensitive to the block length. With only 24 test months, '
        'this is **not decisive proof** of H superiority; feature variant comparison was exploratory.',
        '',
        '## 8. Final-window model reproduction audit',
        '',
        markdown_table(['Quantity', 'Verified result'], [
            ['Models refitted and checked', f"{a['lastfit_models_reproduced']} / 24"],
            ['Training window', a['train_window_for_coefficients']],
            ['Maximum absolute forecast discrepancy (passengers)', f"{a['max_reproduction_pred_error']:.3e}"],
            ['Maximum coefficient discrepancy (raw units)', f"{a['max_reproduction_coefficient_error']:.3e}"],
        ]), '',
        'Individual target-specific checks appear in [the reproduction audit CSV](data/model_reproduction_audit.csv).',
        '',
        '## 9. Source files and limitations',
        '',
        '- The original regression predictions, coefficients and comparative performance are in '
        '[`../../results/experiment_v2/`](../../results/experiment_v2/) (relative to the project root: '
        '`results/experiment_v2/`).',
        '- The full coefficient diagnostics, exact test values, month-level residuals, per-target autocorrelations '
        'and error rankings remain in the [data directory](data/).',
        '- Figures are in [figures/](figures/), generated by `analyze_section8.py`.',
        '- Do not interpret a predictive association as a causal effect. The Lasso uncertainty summaries '
        'and approximate null-bootstrap p-values are exploratory and are not formal post-selection inference. '
        'The auxiliary OLS/HAC table refers to a different estimator.',
        '- The complete 264-row CSV is deliberately linked rather than printed as an excessively long Markdown table.',
        '',
    ])
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    targets = sorted(set(x['target'] for x in all24))
    all_lines = [
        '# Section 8 — All 24 final-month model coefficient tables',
        '',
        'These 24 tables contain **all 264 predictor–target combinations** from the June 2026 rolling-Lasso final fits. '
        'Generated from [the full audited CSV](data/ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv) '
        'by `export_tables_md.py`. The coefficients and intervals are **thousands of passengers per original training-feature SD**.',
        '',
        '**Estimator distinction:** Lasso β and its block-resampled sensitivity interval describe the penalized forecast model; '
        'OLS β, HAC interval, HAC p and BH q come from an **unpenalized auxiliary OLS regression**, '
        'not tests of Lasso weights. HAC covariance uses three lags. '
        'There are no restricted-null Lasso p-values for 23 of these models; '
        'the representative Taipei-entry null tests appear in [TABLES.md](TABLES.md). '
        'Neither set of inferential summaries is causal or valid post-selection inference.',
        '',
        '## Index', '',
        *[f'- [{station(t)}](#{station(t).lower().replace(" ", "-")})' for t in targets],
        ''
    ]
    for target in targets:
        rs = [x for x in all24 if x['target'] == target]
        if len(rs) != 11:
            raise AssertionError(f'Expected 11 predictors for {target}, got {len(rs)}')
        all_lines.extend([
            f'## {station(target)}',
            '',
            f"**α = {number(rs[0]['alpha']):g}; training window = {rs[0]['training_months']} months; "
            f"forecast month = {rs[0]['forecast_month']}.**",
            '',
            markdown_table(['Predictor', 'Lasso β', 'Lasso 95% sensitivity interval', 'VIF',
                            'OLS β', 'OLS HAC 95% CI', 'HAC p', 'BH q'], [
                [name(r['feature']), signed(r['lasso_beta_per_training_SD_passengers'], 1, True),
                 interval(r['lasso_boot_sensitivity_lower_95'], r['lasso_boot_sensitivity_upper_95']),
                 f"{number(r['vif']):.1f}", signed(r['ols_HAC_beta_per_training_SD_passengers'], 1, True),
                 interval(r['ols_HAC_lower_95'], r['ols_HAC_upper_95']),
                 prob(r['ols_HAC_p_nominal']), prob(r['ols_HAC_p_BH11_exploratory'])]
                for r in rs
            ]),
            '',
        ])
    OUT_ALL.write_text('\n'.join(all_lines), encoding='utf-8')
    # Additional numerical diagnostics displayed as GitHub-readable tables.
    absolute = read_csv('top_24_absolute_errors.csv')
    relative = read_csv('top_24_relative_errors.csv')
    corrs = read_csv('residual_autocorrelation_by_target.csv')
    gap = read_csv('paired_monthly_difference.csv')
    if not all(len(rows) == 24 for rows in (absolute, relative, corrs, gap)):
        raise AssertionError('Expected exactly 24 rows in each supplemental diagnostic CSV.')
    suppl = [
        '# Section 8 — Supplemental diagnostic tables',
        '',
        'These tables are generated from existing `data/*.csv` results, without re-fitting or selecting models.',
        'Counts refer to test observations in the rolling Lasso H experiment unless otherwise noted.',
        'For concise core results see [TABLES.md](TABLES.md); for all 24 coefficient tables see '
        '[ALL_24_TARGETS_TABLES.md](ALL_24_TARGETS_TABLES.md).',
        '',
        '## 1. Largest 24 absolute forecast errors', '',
        '**Signed error = actual minus predicted** (passengers). Positive means underprediction.',
        '',
        markdown_table(['Rank', 'Month', 'Station / direction', 'Actual', 'Predicted', 'Signed error', 'Absolute error', 'APE'], [
            [i, r['date'], station(r['target']), integer(r['actual']), integer(r['predicted']),
             signed(r['signed_error'], 0), integer(r['absolute_error']), f"{number(r['APE_pct']):.2f}%"]
            for i, r in enumerate(absolute, 1)
        ]), '',
        '## 2. Largest 24 percentage forecast errors', '',
        markdown_table(['Rank', 'Month', 'Station / direction', 'Actual', 'Predicted', 'Signed error', 'APE'], [
            [i, r['date'], station(r['target']), integer(r['actual']), integer(r['predicted']),
             signed(r['signed_error'], 0), f"{number(r['APE_pct']):.2f}%"]
            for i, r in enumerate(relative, 1)
        ]), '',
        '## 3. Residual serial dependence for each target', '',
        'Exploratory three-lag Ljung–Box p-values are descriptive diagnostics based on only 24 test months; '
        'they are not independent hypothesis tests of 24 unrelated series.', '',
        markdown_table(['Target', 'Lag-1 signed residual correlation', 'Ljung–Box (3 lags), exploratory p'], [
            [station(r['target']), signed(r['lag1_residual_autocorr'], 3),
             prob(r['LjungBox_3_pvalue_exploratory'])] for r in corrs
        ]), '',
        '## 4. Paired baseline B versus holiday-duration H, by test month', '',
        'Positive B − H means H has smaller mean absolute error for that month (passengers); '
        'these are descriptive held-out comparisons.', '',
        markdown_table(['Month', 'B − H MAE improvement', 'H mean absolute error', 'B mean absolute error'], [
            [r['date'], signed(r['mean_abs_improvement_BminusH'], 0),
             integer(r['mean_abs_H']), integer(r['mean_abs_B'])] for r in gap
        ]), '',
        '## 5. Per-target final-month reproduction audit', '',
        'Numbers measure agreement between refitted frozen models and the original experiment outputs.', '',
        markdown_table(['Target', 'Lasso α', 'Train months', 'Maximum coefficient difference (raw units)',
                        'Forecast difference (passengers)'], [
            [station(r['target']), f"{number(r['alpha']):g}", r['n_train'],
             f"{number(r['max_abs_raw_beta_reproduction_difference']):.3e}",
             f"{number(r['abs_test_prediction_reproduction_difference']):.3e}"]
            for r in audit
        ]), '',
        '## 6. Full machine-readable result files', '',
        '- The [complete final model feature audit](data/ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv) '
        'has 264 feature-target entries.',
        '- The [complete held-out predictions](data/frozen_rolling_lasso_H_test_predictions_with_errors.csv) '
        'have 576 target-month observations.',
        '- The [11 Taipei Lasso restricted-null p-values](data/taipei_lasso_conditional_null_bootstrap_pvalues.csv) '
        'are exploratory fixed-penalty approximations, not formal selective inference.',
        '',
        '**Statistical caveat:** The bootstrap sensitivity intervals do not establish formal post-selection coverage, '
        'and auxiliary OLS/HAC coefficients and tests concern a separate estimator. '
        'Observed shared residuals do not identify causal failure mechanisms.',
        ''
    ]
    OUT_SUPP.write_text('\n'.join(suppl), encoding='utf-8')
    print(f'Wrote {OUT_SUPP} ({OUT_SUPP.stat().st_size:,} bytes)')
    print(f'Wrote {OUT} ({OUT.stat().st_size:,} bytes)')
    print(f'Wrote {OUT_ALL} ({OUT_ALL.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
