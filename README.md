# Data_Mining_HW1_G02

## Links

- Document: [Google Docs](https://docs.google.com/document/d/17EDonnW9b7wmKxxSu9W65ORkYWyFXTFzqIjIo54BFNc/edit?usp=sharing)
- Report: [Overleaf](https://www.overleaf.com/5219884191pcbkyktfpgxp#06cc55)

## How to Use

### 1. Download the Repository

Open a terminal in the folder where you want to save the project, then run:

```bash
git clone https://github.com/yfchinn/Data_Mining_HW1_G02.git
cd Data_Mining_HW1_G02
```

Cloning downloads the project and sets up its connection to GitHub. Run this step only once. For later sessions, open a terminal inside the cloned folder.

### 2. Set Your Commit Identity

Inside the cloned folder, run these commands once, replacing the examples with your own name and email:

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

### 3. Get the Latest Changes

Before you start editing, make sure your previous work is committed, then run:

```bash
git switch main
git pull --rebase origin main
```

### 4. Commit and Push Your Changes

After editing, check which files changed:

```bash
git status
git diff
```

Stage the files you want to share, replacing `path/to/your-file` with an actual file or folder path. Use quotes around paths containing spaces:

```bash
git add "path/to/your-file"
git commit -m "Describe your changes"
git pull --rebase origin main
git push origin main
```

Repeat `git add` for additional files before committing. The final pull includes any updates your teammates pushed while you were working.

## Experiment v2: train on operating history

On branch `experiment-v2`, `OLS_static_lag.py`, `OLS_rolling.py`,
`lasso_static_lag.py`, and `lasso_rolling.py` share the operating-history rules
in `station_history.py`. Run these same scripts to write **new** outputs under
`results/experiment_v2/{ols_static_lag,ols_rolling,lasso_static_lag,lasso_rolling}/`.
The original results and baseline `OLS.py` / `lasso.py` remain historical v1
experiments; those two baseline scripts do not apply the new rules.

- Use documented opening months from `preprocessing.OPENING_MONTHS`; stations
  already operating at the dataset start use January 2007. Retain partial
  opening months because the observations are monthly.
- Mask pre-opening targets before computing lags and rolling means, retaining
  the monthly index. Genuine zero observations after opening remain valid.
- Keep the same ten features and July 2024–June 2026 test months. Twelve months
  of operating history are needed before the first complete feature row.
- Keep the four static CV date boundaries. Skip a target's fold if it has
  fewer than 24 complete training rows or incomplete validation features.
  This minimum is an experiment rule, not a guarantee of statistical adequacy.
  Record every used/skipped fold and training dates in `fold_audit.csv`.
- Nangang, Miaoli, Changhua, and Yunlin use folds 3 and 4. Other stations use
  all four. Lasso selects alpha only across the target's eligible folds.
- Rolling validation remains July 2022–June 2024. A candidate window must
  have its full requested history for every target/month; it is not shortened
  for younger stations. Candidates 36 and 60 pass; 120 is excluded because
  late-opening stations lack that history. See `window_eligibility.csv`.
  Each model chooses its common window by validation MAE on all 24 targets;
  Lasso also selects alpha per target. Monthly coefficient reports include
  the actual training start, end, and sample count.
- Test remains one step ahead using previously observed monthly usage.

Full-line monthly **entry** test results (24 months), saved in
`results/experiment_v2/model_comparison.csv`:

| Model | Selected rolling window | MAE | MAPE |
| --- | ---: | ---: | ---: |
| Static OLS + Lag | — | 346,424 | 5.031% |
| Static Lasso + Lag | — | 357,798 | 5.159% |
| Rolling OLS | 60 | 403,149 | 6.003% |
| Rolling Lasso | 36 | 297,206 | 4.312% |
| Seasonal naive (t-12) | — | 368,317 | 5.319% |

Window choices are based on validation, not the test results. V2 changes both
the eligible history and validation/window selection; its difference from v1
cannot be attributed to removal of pre-opening zeros alone.

### Static and rolling holiday experiments

Run `uv run --no-project python static_holiday_experiment.py` to compare raw
holiday durations and raw-plus-log features with v2 static and rolling OLS
and Lasso. The script keeps its original filename and report prefix.

- `raw`: `long_holiday_days_started` and `long_holiday_days_ended`.
- `raw_log1p`: both raw columns **plus** a separate `log1p` column for each.
  `log1p(0) = 0`, so months without qualifying holidays remain valid.
- Both omit `long_holiday_count`. The former `long_holiday_days_in_month` was
  removed from the dataset. Other features are month index, month sine/cosine,
  days in month, and five target-history features (11 inputs for raw, 13 for
  raw-plus-log).

Static runs use the same eligible folds and opening-date rule as the v2
baseline; coefficients remain fixed throughout the test period. Rolling runs
validate 36/60-month windows on July 2022–June 2024 (120 is ineligible for
late-opening stations), then refit monthly in test. Each model/variant chooses
one common window by mean station validation MAE; Lasso selects alpha per
target/window. Window and alpha choices are frozen before test evaluation.
All experiments forecast one month ahead on July 2024–June 2026 using previously
observed usage.

The consolidated report is `results/experiment_v2/static_holiday_comparison.csv`.
It now contains **both static and rolling** results, their corresponding v2
baselines, and seasonal naive (t-12). Positive
`mae_improvement_vs_baseline_pct` means lower test MAE than the same method/model
with the original holiday count. The `static_holiday_` station metrics, CV
metrics, predictions, and coefficient files are overwritten together and
include a `method` column. These replace the earlier log-only experiment;
static raw results are retained as a reference by rerunning them.

Full-line monthly entry test results (MAE in persons/month):

| Method/model | Holiday inputs | Window | MAE | MAPE | MAE improvement vs own v2 baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| Static OLS | Baseline count | — | 346,424 | 5.031% | — |
| Static OLS | Raw pair | — | 342,757 | 4.973% | 1.06% |
| Static OLS | Raw pair + log1p pair | — | 335,782 | 4.871% | 3.07% |
| Static Lasso | Baseline count | — | 357,798 | 5.159% | — |
| Static Lasso | Raw pair | — | 347,630 | 5.021% | 2.84% |
| Static Lasso | Raw pair + log1p pair | — | 346,760 | 5.007% | 3.09% |
| Rolling OLS | Baseline count | 60 | 403,149 | 6.003% | — |
| Rolling OLS | Raw pair | 60 | 395,402 | 5.897% | 1.92% |
| Rolling OLS | Raw pair + log1p pair | 60 | 402,401 | 5.994% | 0.19% |
| Rolling Lasso | Baseline count | 36 | 297,206 | 4.312% | — |
| Rolling Lasso | Raw pair | 36 | 296,902 | 4.306% | 0.10% |
| Rolling Lasso | Raw pair + log1p pair | 36 | 297,806 | 4.321% | -0.20% |
| Seasonal naive | Last year's month | — | 368,317 | 5.319% | — |

The static raw-plus-log variant improves both validation and test MAE compared
with the raw pair. Rolling Lasso's raw-pair improvement over baseline is very
small. Rolling OLS's holiday variants remain worse than seasonal naive on test;
their validation MAE also exceeds the OLS baseline's validation MAE.

CV scores are parameter-selection scores rather than independent estimates of
the selected model's performance. Comparisons with the v2 baseline change
holiday representation and may change rolling window and Lasso alpha. Calendar
data are retrospective: revisions published after a forecast month would not
have been known at that time.

Run `uv run --no-project python summarize_station_results.py` to compare
per-station entry and exit errors against each model and the seasonal naive
forecast (previous year's same month). The generated
`results/experiment_v2/station_method_report.md` includes each station's MAPE;
`station_method_comparison.csv` contains MAE, RMSE, R², and MAPE for every
target and setting; `station_method_summary.csv` averages each metric equally
over the 24 entry/exit targets. These station averages are different from the
full-line entry totals above.

Run the history and leakage checks with:

```bash
uv run --no-project python -m unittest test_station_history
```

## Historical v1: Lasso experiments aligned with OLS

Reference: `origin/feature/experiment-v1`, commit
`960ab07b3e6d414bd44e830dcb90367994b0d929` (`OLS_static_lag.py`, `OLS_rolling.py`).
The processed monthly data is identical on both branches.

```bash
uv pip install -r requirements.txt
uv run --no-project python lasso_static_lag.py
uv run --no-project python lasso_rolling.py
```

Both use ten features: `month_index`, `month_sin`, `month_cos`, `days_in_month`,
`long_holiday_count`, the target's `lag_1`, `lag_3`, `lag_12`, and means of its
previous 3 and 12 months. Rolling means are shifted before aggregation.
`lasso_features.py` shares feature generation, metrics and reporting.

These are **one-month-ahead evaluations**: actual usage of preceding months
is available at each forecast date. Test dates are July 2024–June 2026.
No current-month or future target is used in features, fitting, scaling or tuning.

| Script | Fitting and validation | Reports |
| --- | --- | --- |
| `lasso_static_lag.py` | Four expanding folds of 24 months, using OLS fold boundaries. Choose each target's alpha by mean validation MAE, then fit once through June 2024. Update lag features with observed history during test, keeping coefficients fixed. | `results/lasso_static_lag/` |
| `lasso_rolling.py` | Monthly refits on the latest 36/60/120 available training months. Validate on July 2022–June 2024; choose alpha per target/window, then choose one common window by average station MAE. Freeze the window and alphas for test. | `results/lasso_rolling/` |

Alpha candidates are `1, 10, 100, 1000, 10000, 100000, 1000000`.
Each fit uses `StandardScaler` fitted only on its own training rows.
Reported coefficients are converted back to original feature units.
Validation scores are used for parameter selection, rather than providing an
independent estimate of the selected model's generalization error.
MAE, RMSE, R2 and MAPE definitions match OLS (MAPE excludes zero actuals).
Total comparison sums entry predictions only and uses the same t-12 baseline.
Monthly availability of observed usage is assumed, as in OLS.

Total monthly entry results on the same 24 test months are saved in
`results/lasso_ols_comparison.csv`. Rolling Lasso selects a 120-month window,
the same as the reference OLS run.

| Model | MAE (persons/month) | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| Static OLS + Lag | 342,157 | 400,706 | 4.974% |
| Static Lasso + Lag | 341,364 | 408,066 | 4.941% |
| Rolling OLS | 357,708 | 409,961 | 5.242% |
| Rolling Lasso | 352,912 | 409,957 | 5.140% |
| Seasonal naive (t-12) | 368,317 | 445,376 | 5.319% |

Lasso improves MAE/MAPE slightly; static Lasso has worse RMSE than static OLS.
These results do not establish a large or statistically significant advantage.

Known limitation: these runs retain pre-opening station zeros, matching the
reference OLS experiments. Early static validation folds have insufficient
post-opening history for Nangang, Miaoli, Changhua, and Yunlin. Excluding
pre-opening rows and skipping folds with insufficient valid history has been
discussed but is not implemented in these results.

The original `lasso.py` remains unchanged. Historical lag-only and holiday
experiment reports remain in `reports/`. Their scripts are available at commit
`2a0c3ca`; the static/rolling scripts replace them. Historical fixed-horizon
scores are not a controlled comparison with the new one-step forecasts.

## Official calendar data (historical experiment)

### Audit of 2008–2011 and annual totals

The monthly file counts **only runs of at least three consecutive days off**
on the DGPA government-office calendar. The [2008 DGPA announcement](https://www.dgpa.gov.tw/information?pid=1395&uid=2)
lists exactly three: February 6–11 (6 days), April 4–6 (3), and October 10–12
(3). For 2009, the [original DGPA announcement](https://www.dgpa.gov.tw/information?pid=1353&uid=82)
lists January 24–February 1 (9) and May 28–31 (4); a
[later DGPA notice](https://www.dgpa.gov.tw/information?pid=2274&uid=82)
confirms January 1–4 (4) was also a break. Thus January 2009 has
`started=13` (4 + 9) and `ended=4`; February has `ended=9`.

The XLS special-observance fill previously marked May 1, 2010, May 1, 2011,
and September 3, 2011 as working days even though they were weekends. This is
fixed in `data/official_holiday_days.csv`. They do not form or extend any
three-day break, so `data/official_long_holiday_periods.csv` and the monthly
holiday-duration features are unchanged. The parser now checks **every year's
day-off total** against DGPA's
[annual totals table](https://www.dgpa.gov.tw/FileConversion?filename=dgpa%2Ffiles%2F202506%2Fcbc12c92-ad33-4bf2-a96e-c5d36fbd23fa.doc&name=%E5%85%A8%E5%B9%B4%E6%94%BE%E5%81%87%E6%97%A5%E6%95%B8%E4%B8%80%E8%A6%BD%E8%A1%A8.doc&nfix=pdf).
The corrected 2010/2011 totals are 112/115 days. `test_official_calendar.py`
also verifies daily coverage, every maximal three-day break, and the 2008/2009
date lists. The existing model predictions are unaffected by this daily-only
correction.

### Calendar correction

The 2007 XLS parser previously reversed the fills for adjusted days off (51)
and makeup workdays (43). These are now corrected against the legend and
footnotes in the [official 2007 calendar](https://www.dgpa.gov.tw/information?pid=4970&uid=30).
All 2007–2016 XLS legends are now checked during parsing. The corrected 2007
breaks ending that year are December 30, 2006–January 1 (3 days), February
17–25 (9), April 5–8 (4), June 16–19 (4), and September 22–25 (4).

The daily calendar, break intervals, and `data/processed_calendar` have been
rebuilt. Only eight daily classifications and the three holiday-duration
features for February, April, June, and September 2007 changed. All 112 break
intervals and 234 monthly aggregates were verified against daily records.
Original `data/processed`, `lasso.py`, and existing model reports/results are
preserved. Historical reports using the three duration features reflect the
old calendar and have not been rerun; the current OLS-aligned Lasso models
use the original holiday count and are unaffected by this correction.

The original `lasso.py` reads `data/processed/merged_monthly.csv` and writes to
`reports/lasso/`. Its data and reports remain separate from this experiment.

The new `build_official_holidays.py` downloads government office calendars from
[DGPA historical work calendars](https://www.dgpa.gov.tw/informationlist?uid=30)
(2007–2016 XLS, plus December 30–31, 2006 for the first cross-year break) and the [government open-data calendar](https://data.gov.tw/dataset/14718)
(2017–2026 CSV). It saves daily office day-off flags in
`data/official_holiday_days.csv` and periods of at least three consecutive days
off in `data/official_long_holiday_periods.csv`. Weekend make-up workdays are
treated as workdays. The XLS parser uses the official files' cell formatting;
when refreshing those files, inspect any changes to their styles.

```bash
uv pip install -r requirements.txt
uv run --no-project python build_official_holidays.py
uv run --no-project python preprocessing.py --output-dir data/processed_calendar
```

`preprocessing.py` retains `long_holiday_count` from the original CSV and adds:

| Column | Meaning |
| --- | --- |
| `long_holiday_days_started` | Full duration of every qualifying break that starts in the month, summed. |
| `long_holiday_days_ended` | Full duration of every qualifying break that ends in the month, summed. |

For a December 30–January 2 break, `started` adds 4 to December, `ended` adds
4 to January. The static holiday experiment uses these two columns; the
default static/rolling experiments use the original holiday count.

`long_holiday_days_in_month` and its missingness indicator were removed from
the exported datasets, dictionary, and preprocessing metadata in experiment v2.
Rebuilding the data now produces only the two duration columns above. The
better two-column results also exclude the old holiday count, so that comparison
does not isolate the effect of removing `in_month` alone.

The office calendar is a proxy for passenger holidays, not the holiday schedule
of every traveler. The test is retrospective: calendar revisions announced after
July 2024 may appear in the held-out period's features. A strict forecast made
in July 2024 would need the calendar version known then.
