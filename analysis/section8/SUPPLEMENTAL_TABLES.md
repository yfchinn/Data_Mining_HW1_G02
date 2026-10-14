# Section 8 — Supplemental diagnostic tables

These tables are generated from existing `data/*.csv` results, without re-fitting or selecting models.
Counts refer to test observations in the rolling Lasso H experiment unless otherwise noted.
For concise core results see [TABLES.md](TABLES.md); for all 24 coefficient tables see [ALL_24_TARGETS_TABLES.md](ALL_24_TARGETS_TABLES.md).

## 1. Largest 24 absolute forecast errors

**Signed error = actual minus predicted** (passengers). Positive means underprediction.

| Rank | Month | Station / direction | Actual | Predicted | Signed error | Absolute error | APE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2025-01 | Taipei exit | 1,356,368 | 1,520,266 | -163,898 | 163,898 | 12.08% |
| 2 | 2024-10 | Taipei exit | 1,337,731 | 1,470,904 | -133,173 | 133,173 | 9.96% |
| 3 | 2024-10 | Taipei entry | 1,295,528 | 1,426,832 | -131,304 | 131,304 | 10.14% |
| 4 | 2025-10 | Zuoying entry | 1,002,974 | 872,204 | +130,770 | 130,770 | 13.04% |
| 5 | 2025-02 | Zuoying exit | 790,950 | 913,648 | -122,698 | 122,698 | 15.51% |
| 6 | 2025-08 | Taipei exit | 1,569,094 | 1,447,520 | +121,574 | 121,574 | 7.75% |
| 7 | 2024-10 | Zuoying exit | 784,347 | 903,896 | -119,549 | 119,549 | 15.24% |
| 8 | 2024-10 | Zuoying entry | 785,666 | 902,362 | -116,696 | 116,696 | 14.85% |
| 9 | 2025-10 | Zuoying exit | 1,003,900 | 887,415 | +116,485 | 116,485 | 11.60% |
| 10 | 2025-10 | Taichung exit | 1,240,862 | 1,125,734 | +115,128 | 115,128 | 9.28% |
| 11 | 2025-08 | Taipei entry | 1,516,365 | 1,407,356 | +109,009 | 109,009 | 7.19% |
| 12 | 2025-03 | Taichung exit | 1,169,069 | 1,062,119 | +106,950 | 106,950 | 9.15% |
| 13 | 2026-03 | Chiayi entry | 378,728 | 273,358 | +105,370 | 105,370 | 27.82% |
| 14 | 2025-03 | Taichung entry | 1,181,489 | 1,076,541 | +104,948 | 104,948 | 8.88% |
| 15 | 2025-01 | Taichung entry | 1,076,326 | 1,179,889 | -103,563 | 103,563 | 9.62% |
| 16 | 2026-03 | Chiayi exit | 372,328 | 271,547 | +100,781 | 100,781 | 27.07% |
| 17 | 2024-12 | Taipei exit | 1,606,770 | 1,506,035 | +100,735 | 100,735 | 6.27% |
| 18 | 2025-10 | Taipei entry | 1,467,682 | 1,367,886 | +99,796 | 99,796 | 6.80% |
| 19 | 2025-10 | Taichung entry | 1,237,635 | 1,140,761 | +96,874 | 96,874 | 7.83% |
| 20 | 2024-09 | Taipei entry | 1,333,457 | 1,427,946 | -94,489 | 94,489 | 7.09% |
| 21 | 2026-03 | Taipei exit | 1,586,252 | 1,493,450 | +92,802 | 92,802 | 5.85% |
| 22 | 2024-10 | Taichung exit | 1,058,226 | 1,149,038 | -90,812 | 90,812 | 8.58% |
| 23 | 2025-02 | Taoyuan exit | 831,176 | 741,915 | +89,261 | 89,261 | 10.74% |
| 24 | 2025-12 | Taipei exit | 1,582,453 | 1,493,845 | +88,608 | 88,608 | 5.60% |

## 2. Largest 24 percentage forecast errors

| Rank | Month | Station / direction | Actual | Predicted | Signed error | APE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2026-03 | Chiayi entry | 378,728 | 273,358 | +105,370 | 27.82% |
| 2 | 2026-03 | Chiayi exit | 372,328 | 271,547 | +100,781 | 27.07% |
| 3 | 2025-02 | Changhua exit | 67,370 | 82,931 | -15,561 | 23.10% |
| 4 | 2025-02 | Yunlin exit | 123,603 | 151,473 | -27,870 | 22.55% |
| 5 | 2025-02 | Chiayi exit | 231,299 | 276,210 | -44,911 | 19.42% |
| 6 | 2025-02 | Tainan exit | 384,303 | 452,373 | -68,070 | 17.71% |
| 7 | 2025-02 | Nangang entry | 296,596 | 343,744 | -47,148 | 15.90% |
| 8 | 2025-02 | Zuoying exit | 790,950 | 913,648 | -122,698 | 15.51% |
| 9 | 2025-10 | Chiayi exit | 309,874 | 261,819 | +48,055 | 15.51% |
| 10 | 2024-10 | Zuoying exit | 784,347 | 903,896 | -119,549 | 15.24% |
| 11 | 2024-10 | Zuoying entry | 785,666 | 902,362 | -116,696 | 14.85% |
| 12 | 2025-10 | Chiayi entry | 308,442 | 263,749 | +44,693 | 14.49% |
| 13 | 2025-02 | Changhua entry | 74,199 | 84,889 | -10,690 | 14.41% |
| 14 | 2026-02 | Chiayi exit | 315,203 | 269,876 | +45,327 | 14.38% |
| 15 | 2025-05 | Miaoli exit | 121,816 | 104,685 | +17,131 | 14.06% |
| 16 | 2025-03 | Nangang exit | 325,272 | 280,201 | +45,071 | 13.86% |
| 17 | 2025-01 | Chiayi exit | 305,374 | 263,114 | +42,260 | 13.84% |
| 18 | 2025-02 | Yunlin entry | 137,419 | 156,345 | -18,926 | 13.77% |
| 19 | 2025-12 | Banqiao entry | 541,593 | 468,717 | +72,876 | 13.46% |
| 20 | 2026-02 | Chiayi entry | 312,359 | 270,728 | +41,631 | 13.33% |
| 21 | 2025-08 | Nangang exit | 342,010 | 296,431 | +45,579 | 13.33% |
| 22 | 2025-10 | Nangang entry | 400,641 | 347,413 | +53,228 | 13.29% |
| 23 | 2025-10 | Zuoying entry | 1,002,974 | 872,204 | +130,770 | 13.04% |
| 24 | 2026-04 | Chiayi exit | 324,761 | 283,331 | +41,430 | 12.76% |

## 3. Residual serial dependence for each target

Exploratory three-lag Ljung–Box p-values are descriptive diagnostics based on only 24 test months; they are not independent hypothesis tests of 24 unrelated series.

| Target | Lag-1 signed residual correlation | Ljung–Box (3 lags), exploratory p |
| --- | ---: | ---: |
| Banqiao entry | -0.048 | 0.9838 |
| Changhua entry | -0.212 | 0.7351 |
| Chiayi entry | +0.191 | 0.692 |
| Hsinchu entry | -0.047 | 0.6872 |
| Miaoli entry | -0.152 | 0.7659 |
| Nangang entry | -0.278 | 0.4386 |
| Taichung entry | -0.091 | 0.4986 |
| Tainan entry | +0.016 | 0.9596 |
| Taipei entry | +0.002 | 0.6589 |
| Taoyuan entry | -0.274 | 0.5865 |
| Yunlin entry | -0.084 | 0.3436 |
| Zuoying entry | -0.217 | 0.7512 |
| Banqiao exit | -0.104 | 0.8589 |
| Changhua exit | -0.282 | 0.4406 |
| Chiayi exit | +0.042 | 0.8926 |
| Hsinchu exit | -0.042 | 0.8341 |
| Miaoli exit | -0.212 | 0.4014 |
| Nangang exit | -0.199 | 0.1123 |
| Taichung exit | -0.055 | 0.8327 |
| Tainan exit | -0.022 | 0.9944 |
| Taipei exit | -0.114 | 0.2919 |
| Taoyuan exit | -0.338 | 0.3713 |
| Yunlin exit | -0.146 | 0.3506 |
| Zuoying exit | -0.262 | 0.4946 |

## 4. Paired baseline B versus holiday-duration H, by test month

Positive B − H means H has smaller mean absolute error for that month (passengers); these are descriptive held-out comparisons.

| Month | B − H MAE improvement | H mean absolute error | B mean absolute error |
| --- | ---: | ---: | ---: |
| 2024-07 | +191 | 26,973 | 27,164 |
| 2024-08 | -797 | 22,106 | 21,309 |
| 2024-09 | +94 | 20,667 | 20,761 |
| 2024-10 | +245 | 45,674 | 45,920 |
| 2024-11 | +437 | 20,637 | 21,074 |
| 2024-12 | +112 | 31,079 | 31,191 |
| 2025-01 | +1,311 | 33,264 | 34,575 |
| 2025-02 | +424 | 35,028 | 35,451 |
| 2025-03 | -826 | 32,277 | 31,451 |
| 2025-04 | +341 | 9,519 | 9,859 |
| 2025-05 | +340 | 17,594 | 17,934 |
| 2025-06 | -188 | 17,713 | 17,526 |
| 2025-07 | +108 | 11,462 | 11,570 |
| 2025-08 | +119 | 48,247 | 48,366 |
| 2025-09 | -573 | 18,291 | 17,718 |
| 2025-10 | +1,768 | 56,553 | 58,321 |
| 2025-11 | -973 | 26,208 | 25,235 |
| 2025-12 | -583 | 44,592 | 44,009 |
| 2026-01 | +243 | 19,143 | 19,385 |
| 2026-02 | +2,923 | 26,822 | 29,746 |
| 2026-03 | -884 | 43,683 | 42,799 |
| 2026-04 | +179 | 27,172 | 27,350 |
| 2026-05 | +69 | 21,696 | 21,765 |
| 2026-06 | +198 | 22,232 | 22,430 |

## 5. Per-target final-month reproduction audit

Numbers measure agreement between refitted frozen models and the original experiment outputs.

| Target | Lasso α | Train months | Maximum coefficient difference (raw units) | Forecast difference (passengers) |
| --- | ---: | ---: | ---: | ---: |
| Banqiao entry | 10000 | 36 | 6.594e-12 | 5.821e-11 |
| Changhua entry | 1000 | 36 | 1.251e-12 | 0.000e+00 |
| Chiayi entry | 10000 | 36 | 2.331e-12 | 0.000e+00 |
| Hsinchu entry | 10000 | 36 | 6.821e-13 | 0.000e+00 |
| Miaoli entry | 1000 | 36 | 2.046e-12 | 0.000e+00 |
| Nangang entry | 10000 | 36 | 9.714e-17 | 0.000e+00 |
| Taichung entry | 10000 | 36 | 9.095e-13 | 0.000e+00 |
| Tainan entry | 10000 | 36 | 2.290e-16 | 0.000e+00 |
| Taipei entry | 10000 | 36 | 6.821e-13 | 0.000e+00 |
| Taoyuan entry | 10000 | 36 | 8.327e-17 | 0.000e+00 |
| Yunlin entry | 1000 | 36 | 5.912e-12 | 0.000e+00 |
| Zuoying entry | 10000 | 36 | 2.046e-12 | 1.164e-10 |
| Banqiao exit | 10000 | 36 | 2.387e-12 | 0.000e+00 |
| Changhua exit | 1000 | 36 | 1.023e-12 | 0.000e+00 |
| Chiayi exit | 10000 | 36 | 2.274e-13 | 0.000e+00 |
| Hsinchu exit | 10000 | 36 | 4.547e-13 | 0.000e+00 |
| Miaoli exit | 1000 | 36 | 3.411e-12 | 1.455e-11 |
| Nangang exit | 10000 | 36 | 1.110e-16 | 0.000e+00 |
| Taichung exit | 10000 | 36 | 3.638e-12 | 0.000e+00 |
| Tainan exit | 10000 | 36 | 2.217e-12 | 0.000e+00 |
| Taipei exit | 10000 | 36 | 1.819e-11 | 0.000e+00 |
| Taoyuan exit | 10000 | 36 | 2.274e-12 | 0.000e+00 |
| Yunlin exit | 1000 | 36 | 3.638e-12 | 0.000e+00 |
| Zuoying exit | 10000 | 36 | 4.093e-12 | 0.000e+00 |

## 6. Full machine-readable result files

- The [complete final model feature audit](data/ALL_24_targets_11_features_lastfit_Lasso_OlsHAC_VIF.csv) has 264 feature-target entries.
- The [complete held-out predictions](data/frozen_rolling_lasso_H_test_predictions_with_errors.csv) have 576 target-month observations.
- The [11 Taipei Lasso restricted-null p-values](data/taipei_lasso_conditional_null_bootstrap_pvalues.csv) are exploratory fixed-penalty approximations, not formal selective inference.

**Statistical caveat:** The bootstrap sensitivity intervals do not establish formal post-selection coverage, and auxiliary OLS/HAC coefficients and tests concern a separate estimator. Observed shared residuals do not identify causal failure mechanisms.
