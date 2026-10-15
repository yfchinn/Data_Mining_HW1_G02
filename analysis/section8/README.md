# Section 8 — Statistical significance and error diagnostics (Python only)

This directory contains reproducible **additional analysis** of the existing
[`results/experiment_v2/`](../../results/experiment_v2/) experiments: final
rolling-Lasso coefficient audits, uncertainty and multicollinearity diagnostics,
feature selection, residual/failure cases, and model comparisons. It does **not**
retune the original models, overwrite their frozen forecasts, or modify the
original experiments. Work from a clone of the **full** repository, not this
folder in isolation.

## 1. Install dependencies

From the repository root on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r analysis/section8/requirements-analysis.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## 2. Reproduce the results and formatted tables

Run these commands **in order**, starting from the repository root:

```bash
cd analysis/section8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analyze_section8.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python lasso_null_tests.py
python export_tables_md.py
```

On Windows, run `python analyze_section8.py`, `python lasso_null_tests.py`,
and `python export_tables_md.py` from the `analysis/section8/` folder
without the Unix-style thread-variable prefixes.

| Script | Purpose | Produced / updated files |
| --- | --- | --- |
| [`analyze_section8.py`](analyze_section8.py) | Refit and verify 24 frozen June 2026 rolling Lasso forecasts; compute standardized Lasso coefficients, bootstrap sensitivity intervals, VIFs, auxiliary OLS/HAC associations, feature-selection stability, worst test cases, residual patterns, and paired model contrasts. | `analysis_results.json`, `data/*.csv`, `figures/*.png` |
| [`lasso_null_tests.py`](lasso_null_tests.py) | Additional **exploratory** restricted-null block-bootstrap p-values for all 11 predictors of the representative Taipei-entry model (1,999 draws per feature). **Run second.** | `data/taipei_lasso_conditional_null_bootstrap_pvalues.csv`, updates `data/taipei_entry_complete_beta_CI_p_VIF.csv` |
| [`export_tables_md.py`](export_tables_md.py) | **Format** the already-computed CSV/JSON results as human-readable Markdown. Performs no model fitting or inferential recalculation. **Run last.** | [`TABLES.md`](TABLES.md), [`ALL_24_TARGETS_TABLES.md`](ALL_24_TARGETS_TABLES.md), [`SUPPLEMENTAL_TABLES.md`](SUPPLEMENTAL_TABLES.md) |

**Tables-only refresh:** If you already have current CSV/JSON analysis files,
run **only** `python export_tables_md.py` in this folder (or
`python analysis/section8/export_tables_md.py` from the repository root).
The exporter requires the Taipei restricted-null p-value CSV; if it is missing,
run the first and second scripts before exporting. There is no need to rerun
the forecasting benchmark or the slower statistical bootstrap just to render
previously computed results.

The three Markdown files, CSVs, and PNGs are checked into the repository for
convenience. Rerunning this workflow overwrites **only** generated artifacts in
`analysis/section8/`, not the source files under `results/experiment_v2/`.

## 3. Where to read the formatted tables

| Markdown document | Contents |
| --- | --- |
| **[`TABLES.md`](TABLES.md)** | Main summary; June 2026 Taipei-entry Lasso β, exploratory 95% bootstrap intervals, approximate p-values and VIF; separate auxiliary OLS/HAC β, confidence intervals, p-values and BH q-values; test failure cases, monthly residual signs, model comparisons, and selection stability. |
| **[`ALL_24_TARGETS_TABLES.md`](ALL_24_TARGETS_TABLES.md)** | Full predictor-by-predictor tables for **all 24 station/direction models** (11 predictors each; 264 entries), including inference method distinctions. |
| **[`SUPPLEMENTAL_TABLES.md`](SUPPLEMENTAL_TABLES.md)** | Top 24 absolute errors, top 24 percentage errors, residual autocorrelation by target, monthly B-versus-H comparisons, and model reproduction checks. |

These are GitHub Markdown tables, **not** LaTeX or PDF exports.

## 4. Machine-readable statistical results

All paths below are relative to this `analysis/section8/` directory.

| Dataset | Contents |
| --- | --- |
| [`analysis_results.json`](analysis_results.json) | Summary statistics, comparison results, audit metadata, and configuration. |
| [`data/taipei_entry_complete_beta_CI_p_VIF.csv`](data/taipei_entry_complete_beta_CI_p_VIF.csv) | Representative Taipei-entry model: 11 standardized Lasso effects, bootstrap intervals, approximate null p-values, auxiliary OLS/HAC statistics, BH q-values, and VIF. |
| [`data/ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv`](data/ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv) | All 24 final-month models × 11 features: 264 rows of standardized coefficients, diagnostics, and auxiliary tests. |
| [`data/taipei_lasso_conditional_null_bootstrap_pvalues.csv`](data/taipei_lasso_conditional_null_bootstrap_pvalues.csv) | Exploratory restricted-null bootstrap tests for Taipei entry only. |
| [`data/frozen_rolling_lasso_H_test_predictions_with_errors.csv`](data/frozen_rolling_lasso_H_test_predictions_with_errors.csv) | All 576 held-out forecasts with actual, predicted, and signed/absolute/relative errors. |
| [`data/worst_test_cases_with_available_inputs.csv`](data/worst_test_cases_with_available_inputs.csv) | Highest-error cases with previous-ridership and holiday feature context. |
| [`data/top_24_absolute_errors.csv`](data/top_24_absolute_errors.csv) and [`data/top_24_relative_errors.csv`](data/top_24_relative_errors.csv) | Rankings by absolute and percentage prediction error. |
| [`data/monthly_residual_signs.csv`](data/monthly_residual_signs.csv) and [`data/residual_autocorrelation_by_target.csv`](data/residual_autocorrelation_by_target.csv) | Shared monthly error directions and serial dependence. |
| [`data/paired_monthly_difference.csv`](data/paired_monthly_difference.csv) | Paired month-wise error differences between rolling Lasso B and H. |
| [`data/rolling_lasso_feature_selection_frequencies.csv`](data/rolling_lasso_feature_selection_frequencies.csv) | Frequency of Lasso coefficient selection over rolling fits. |
| [`data/model_reproduction_audit.csv`](data/model_reproduction_audit.csv) | Agreement between independently refitted final models and frozen repository predictions. |

## 5. Figures

The first analysis script regenerates the PNGs in [`figures/`](figures/):

- [`monthly_common_residual_bias.png`](figures/monthly_common_residual_bias.png) — month-level signed-error pattern across targets.
- [`lasso_rolling_selection_frequency.png`](figures/lasso_rolling_selection_frequency.png) — Lasso feature-selection frequency.
- [`taipei_entry_Lasso_bootstrap_sensitivity.png`](figures/taipei_entry_Lasso_bootstrap_sensitivity.png) — resampling sensitivity of Taipei-entry coefficients.

## 6. Methods and interpretation boundaries

- The reference forecast is rolling Lasso with holiday-duration features (H),
  using frozen per-target tuning and a 36-month rolling window. Tests cover
  24 monthly forecasts (2024-07 to 2026-06) for 24 station/direction targets.
- The H-versus-original-holiday B advantage is **exploratory**; comparisons
  chosen or emphasized after inspecting the held-out test do not establish an
  independently selected winner.
- Lasso residual-block intervals measure **sensitivity** to data resampling;
  they are not formally post-selection-valid 95% confidence intervals.
  Restricted-null bootstrap p-values are likewise exploratory and conditional
  on the fitted penalty. The separate OLS/Newey–West HAC p-values and
  confidence intervals describe **auxiliary OLS coefficients**, not Lasso.
- Closely related station entry/exit and monthly outcomes are not independent
  replicates. Predictive coefficients and error associations do not establish
  causal holiday effects or causes of individual forecasting failures.

For the original datasets, model definitions, validation folds, and forecast
setup, see the [repository root README](../../README.md).
