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
The original `OLS.py` / `lasso.py` are historical scripts and do not apply
these operating-history rules.

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

The `static_holiday_` station metrics, CV metrics, predictions, and coefficient
files are overwritten together and include a `method` column. The earlier
log-only experiment was replaced; static raw results are retained by rerunning
them.

CV scores are parameter-selection scores rather than independent estimates of
the selected model's performance. Comparisons with the v2 baseline change
holiday representation and may change rolling window and Lasso alpha. Calendar
data are retrospective: revisions published after a forecast month would not
have been known at that time.

Run `uv run --no-project python summarize_station_results.py` after the four
baseline scripts and `static_holiday_experiment.py` to compare
per-station entry and exit errors against each model and the seasonal naive
forecast (previous year's same month). The generated
`results/experiment_v2/station_method_report.md` includes each station's MAPE;
`station_method_comparison.csv` contains MAE, RMSE, R², and MAPE for every
target and setting; `station_method_summary.csv` averages each metric equally
over the 24 entry/exit targets. Positive
`mae_improvement_vs_seasonal_pct` means lower per-target MAE than the same
station's previous-year-same-month forecast.

| Setting | Equal-weight MAE (persons) | Equal-weight MAPE |
| --- | ---: | ---: |
| Seasonal naive (t-12) | 36,318 | 7.42% |
| Static OLS, baseline / raw / raw + log1p | 33,481 / 32,419 / 31,985 | 6.29% / 6.06% / 5.93% |
| Static Lasso, baseline / raw / raw + log1p | 34,229 / 32,598 / 32,798 | 7.18% / 6.37% / 6.39% |
| Rolling OLS, baseline / raw / raw + log1p | 39,704 / 37,301 / 37,983 | 7.25% / 6.89% / 7.06% |
| Rolling Lasso, baseline / raw / raw + log1p | 28,455 / 28,276 / 28,352 | 5.50% / 5.42% / 5.43% |

These values first calculate each target's error over 24 test months and then
average the 24 entry/exit targets equally. See the generated station report for
RMSE, R², and individual stations.

To rebuild the v2 outputs from the committed processed data, run from the
repository root:

```bash
uv pip install -r requirements.txt
uv run --no-project python OLS_static_lag.py
uv run --no-project python lasso_static_lag.py
uv run --no-project python OLS_rolling.py
uv run --no-project python lasso_rolling.py
uv run --no-project python static_holiday_experiment.py
uv run --no-project python summarize_station_results.py
```

The model scripts write station predictions and metrics. The final script
creates the equal-weight station comparison and t-12 reference. It does not
need the discarded whole-network output files.

Run the history and leakage checks with:

```bash
uv run --no-project python -m unittest test_station_history
```

Historical v1 outputs have been removed from this branch. They remain available
in Git history and on `feature/experiment-v1`.

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
rebuilt. Only eight daily classifications and the two holiday-duration
features for February, April, June, and September 2007 changed. All 112 break
intervals and 234 monthly aggregates were verified against daily records.
Original `data/processed` and `lasso.py` are retained for reference. The v2
default models use the original holiday count and are unaffected by this
calendar correction.

The original `lasso.py` reads `data/processed/merged_monthly.csv`; it is not
part of the v2 workflow.

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
4 to January. The static and rolling holiday experiments use these two columns; the
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
