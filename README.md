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

## Lasso experiments aligned with OLS

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
| `long_holiday_days_in_month` | Days of qualifying breaks that actually fall in the month. |

For a December 30–January 2 break, `started` adds 4 to December, `ended` adds
4 to January, and `in_month` adds 2 to each month. The OLS-aligned Lasso
experiments use the original holiday count, not these three columns.

The office calendar is a proxy for passenger holidays, not the holiday schedule
of every traveler. The test is retrospective: calendar revisions announced after
July 2024 may appear in the held-out period's features. A strict forecast made
in July 2024 would need the calendar version known then.
