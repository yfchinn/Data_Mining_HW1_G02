# Section 8 — Statistical and error analysis tables

> **Provenance:** Generated from the CSV and JSON artifacts in this directory by `python export_tables_md.py`. The model outputs originate in `results/experiment_v2/`. This document is a readable Markdown rendering of those computed analyses, not an additional experiment. Values are rounded only for presentation.

## 1. Model and test summary

| Quantity | Result |
| --- | ---: |
| Reference forecast | Rolling Lasso, holiday-duration features (H) |
| Final-fit coefficient window | 2023-06 through 2026-05, 36 observations per target |
| Held-out test period | 2024-07 through 2026-06, 24 months x 24 targets |
| Separate station–direction targets | 24 |
| Held-out forecasts | 576 |
| H test MAE, equal-target average (passengers) | 28,276 |
| H test MAPE, equal-target average | 5.42% |
| Baseline rolling Lasso B test MAE (passengers) | 28,455 |
| Observed B − H MAE difference (passengers) | +178.2 |
| Observed relative MAE improvement | 0.63% |

**Important:** H is an exploratory test-set comparison, not a retrospectively proven prospectively selected winner. The target outcomes share temporal and system-wide dependence.

## 2. All 11 June 2026 Taipei-entry rolling Lasso predictors

**Units:** β and bootstrap interval = **thousands of predicted passengers per one training-window standard deviation of the predictor**, not passengers per raw input unit. The 95% intervals are residual-block **sensitivity intervals**, not formal selective-inference confidence intervals. The `p₀` column is an exploratory restricted-null fixed-α block-bootstrap p-value (1,999 draws).

| Predictor | Lasso β (thousands) | 95% bootstrap interval | Approx. p₀ | VIF |
| --- | ---: | ---: | ---: | ---: |
| Month index | +19.0 | [0.0, +25.9] | 0.077 | 29.3 |
| Month sine | 0.0 | [-2.5, +2.8] | 1 | 1.1 |
| Month cosine | 0.0 | [-1.4, +3.4] | 1 | 3.0 |
| Days / month | +24.3 | [0.0, +28.8] | 0.0535 | 2.7 |
| Holiday start days | 0.0 | [-8.0, 0.0] | 1 | 2.9 |
| Holiday end days | 0.0 | [-10.1, 0.0] | 1 | 3.1 |
| Lag 1 month | 0.0 | [0.0, +11.7] | 1 | 3.4 |
| Lag 3 months | +0.8 | [0.0, +7.7] | 0.0925 | 3.5 |
| Lag 12 months | +25.4 | [0.0, +39.9] | 0.0935 | 23.9 |
| Mean, prev. 3 mo. | 0.0 | [0.0, +8.7] | 1 | 12.9 |
| Mean, prev. 12 mo. | 0.0 | [0.0, +31.3] | 1 | 82.9 |

Predictors shrunk to exactly zero are **not** evidence of zero population association. VIF is a collinearity diagnostic of the training design, not a Lasso p-value.

## 3. Separate auxiliary OLS / Newey–West HAC table

**Units:** thousands of passengers per one training-feature SD, as above. This is an **unpenalized auxiliary OLS estimator**, fitted on the same 36-month design, with Newey–West HAC covariance (3 lags). These βs, 95% intervals, p-values and BH-adjusted q-values **do not refer to the fitted Lasso coefficients**. BH correction is exploratory within 11 predictors.

| Predictor | OLS β (thousands) | HAC 95% CI | HAC p | BH q |
| --- | ---: | ---: | ---: | ---: |
| Month index | +150.8 | [+114.0, +187.5] | 1.12e-08 | 1.23e-07 |
| Month sine | -6.4 | [-15.2, +2.3] | 0.143 | 0.1747 |
| Month cosine | +10.5 | [-3.1, +24.1] | 0.1234 | 0.1697 |
| Days / month | +20.2 | [+6.3, +34.1] | 0.006238 | 0.01716 |
| Holiday start days | +18.5 | [+3.0, +33.9] | 0.02104 | 0.0463 |
| Holiday end days | -5.1 | [-20.0, +9.8] | 0.4842 | 0.5326 |
| Lag 1 month | +4.1 | [-14.5, +22.6] | 0.6544 | 0.6544 |
| Lag 3 months | +18.8 | [-0.6, +38.2] | 0.05692 | 0.08944 |
| Lag 12 months | +90.5 | [+55.1, +125.9] | 2.06e-05 | 1.14e-04 |
| Mean, prev. 3 mo. | -40.2 | [-81.4, +1.1] | 0.05585 | 0.08944 |
| Mean, prev. 12 mo. | -154.7 | [-221.3, -88.1] | 7.01e-05 | 2.57e-04 |

### Collinearity and generalization

- **87/264** final-window predictor–target pairs have VIF > 10.
- **181/264** final-window Lasso coefficients are zero.
- Maximum observed VIF: **307.5** (Nangang exit, Mean, prev. 12 mo.).
- Full readable tables for all 24 targets are in [ALL_24_TARGETS_TABLES.md](ALL_24_TARGETS_TABLES.md).
- Every one of the **24 × 11 = 264** final-window entries, including standardized Lasso coefficients, bootstrap intervals and auxiliary OLS/HAC p/q-values, is available in [the complete 24-target audit CSV](data/ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv).

## 4. Feature selection across all 576 monthly Lasso fits

| Feature | Nonzero selection rate |
| --- | ---: |
| Mean, prev. 12 mo. | 59.7% |
| Month index | 58.3% |
| Days / month | 54.7% |
| Lag 1 month | 54.3% |
| Lag 12 months | 50.7% |
| Month cosine | 46.5% |
| Holiday start days | 34.7% |
| Lag 3 months | 32.5% |
| Month sine | 29.2% |
| Holiday end days | 24.5% |
| Mean, prev. 3 mo. | 9.4% |

Selection frequency describes how often a fitted coefficient is nonzero; it is not causal importance or statistical significance.

## 5. Highest held-out errors, with available feature context

**Signed error = actual − predicted**, in passengers. A negative value means overprediction. The first three rows are the largest absolute errors among the 576 target-month cases. The remaining examples expose other important failure patterns.

| Month | Station / flow | Actual | Predicted | Signed error | APE | Lag-1 | Lag-12 | Holiday start/end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-01 | Taipei exit | 1,356,368 | 1,520,266 | -163,898 | 12.08% | 1,606,770 | 1,386,171 | 9 / 0 |
| 2024-10 | Taipei exit | 1,337,731 | 1,470,904 | -133,173 | 9.96% | 1,394,012 | 1,414,290 | 0 / 0 |
| 2024-10 | Taipei entry | 1,295,528 | 1,426,832 | -131,304 | 10.14% | 1,333,457 | 1,323,935 | 0 / 0 |
| 2025-10 | Zuoying entry | 1,002,974 | 872,204 | +130,770 | 13.04% | 805,629 | 785,666 | 9 / 9 |
| 2025-02 | Zuoying exit | 790,950 | 913,648 | -122,698 | 15.51% | 965,790 | 943,743 | 3 / 9 |
| 2025-08 | Taipei exit | 1,569,094 | 1,447,520 | +121,574 | 7.75% | 1,456,118 | 1,468,032 | 0 / 0 |
| 2026-03 | Chiayi entry | 378,728 | 273,358 | +105,370 | 27.82% | 312,359 | 290,465 | 0 / 3 |
| 2026-03 | Chiayi exit | 372,328 | 271,547 | +100,781 | 27.07% | 315,203 | 281,022 | 0 / 3 |
| 2025-02 | Changhua exit | 67,370 | 82,931 | -15,561 | 23.10% | 92,383 | 85,038 | 3 / 9 |

For the full ranked lists, see [24 largest absolute errors](data/top_24_absolute_errors.csv) and [24 largest percentage errors](data/top_24_relative_errors.csv). An observed error and its calendar context do **not** establish its cause.

## 6. Monthly shared error pattern

All figures below summarize the 24 station–direction forecasts in each month. The counts of positive/negative errors are **not independent statistical replications**.

| Test month | Mean signed error | Mean absolute error | Mean APE | Underpredicted | Overpredicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024-07 | -26,973 | 26,973 | 5.31% | 0 | 24 |
| 2024-08 | +22,106 | 22,106 | 4.45% | 24 | 0 |
| 2024-09 | -17,052 | 20,667 | 3.35% | 4 | 20 |
| 2024-10 | -44,931 | 45,674 | 7.62% | 2 | 22 |
| 2024-11 | +19,273 | 20,637 | 3.81% | 21 | 3 |
| 2024-12 | +30,727 | 31,079 | 5.12% | 22 | 2 |
| 2025-01 | -18,285 | 33,264 | 5.43% | 10 | 14 |
| 2025-02 | -17,247 | 35,028 | 9.17% | 6 | 18 |
| 2025-03 | +27,424 | 32,277 | 6.08% | 22 | 2 |
| 2025-04 | +4,943 | 9,519 | 2.36% | 17 | 7 |
| 2025-05 | +15,911 | 17,594 | 4.47% | 21 | 3 |
| 2025-06 | -16,603 | 17,713 | 4.21% | 3 | 21 |
| 2025-07 | +8,911 | 11,462 | 2.05% | 14 | 10 |
| 2025-08 | +48,247 | 48,247 | 8.23% | 24 | 0 |
| 2025-09 | -13,252 | 18,291 | 4.12% | 4 | 20 |
| 2025-10 | +56,553 | 56,553 | 10.09% | 24 | 0 |
| 2025-11 | +26,208 | 26,208 | 4.32% | 24 | 0 |
| 2025-12 | +44,592 | 44,592 | 7.42% | 24 | 0 |
| 2026-01 | -19,143 | 19,143 | 3.64% | 0 | 24 |
| 2026-02 | +238 | 26,822 | 5.47% | 13 | 11 |
| 2026-03 | +43,683 | 43,683 | 8.52% | 24 | 0 |
| 2026-04 | +27,172 | 27,172 | 6.12% | 24 | 0 |
| 2026-05 | +21,575 | 21,696 | 4.05% | 22 | 2 |
| 2026-06 | -20,615 | 22,232 | 4.65% | 3 | 21 |

In October 2025, all 24 targets were underpredicted, with mean signed error **+56,553 passengers**. Month group means account for **60.7%** of the variation in signed percentage errors as a *post-hoc descriptive decomposition*, not a pre-forecast predictive variable.

## 7. Paired rolling Lasso B vs H test-error sensitivity

The comparison uses the same 24 targets × 24 months. It pairs errors within target–month, averages across targets in each month, and resamples entire months in circular blocks. Positive B − H means H has lower MAE.

| Month block length | Observed MAE gain (passengers) | Bootstrap 95% percentile interval (passengers) | Proportion of bootstrap differences > 0 |
| --- | ---: | ---: | ---: |
| 3 months | +178.2 | [-42, +422] | 93.80% |
| 6 months | +178.2 | [+4, +355] | 97.85% |
| 12 months | +178.2 | [+82, +275] | 100.00% |

The conclusion is sensitive to the block length. With only 24 test months, this is **not decisive proof** of H superiority; feature variant comparison was exploratory.

## 8. Final-window model reproduction audit

| Quantity | Verified result |
| --- | ---: |
| Models refitted and checked | 24 / 24 |
| Training window | 2023-06 through 2026-05, 36 observations per target |
| Maximum absolute forecast discrepancy (passengers) | 1.164e-10 |
| Maximum coefficient discrepancy (raw units) | 1.819e-11 |

Individual target-specific checks appear in [the reproduction audit CSV](data/model_reproduction_audit.csv).

## 9. Source files and limitations

- The original regression predictions, coefficients and comparative performance are in [`../../results/experiment_v2/`](../../results/experiment_v2/) (relative to the project root: `results/experiment_v2/`).
- The full coefficient diagnostics, exact test values, month-level residuals, per-target autocorrelations and error rankings remain in the [data directory](data/).
- Figures are in [figures/](figures/), generated by `analyze_section8.py`.
- Do not interpret a predictive association as a causal effect. The Lasso uncertainty summaries and approximate null-bootstrap p-values are exploratory and are not formal post-selection inference. The auxiliary OLS/HAC table refers to a different estimator.
- The complete 264-row CSV is deliberately linked rather than printed as an excessively long Markdown table.
