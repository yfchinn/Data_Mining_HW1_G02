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

## Lasso experiment: official calendar and 24-month lag

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
uv run --no-project python lasso_lag24.py
uv run --no-project python lasso_lag24_only.py
uv run --no-project python lasso_lag12.py
```

`preprocessing.py` retains `long_holiday_count` from the original CSV and adds:

| Column | Meaning |
| --- | --- |
| `long_holiday_days_started` | Full duration of every qualifying break that starts in the month, summed. |
| `long_holiday_days_ended` | Full duration of every qualifying break that ends in the month, summed. |
| `long_holiday_days_in_month` | Days of qualifying breaks that actually fall in the month. |

For a December 30–January 2 break, `started` adds 4 to December, `ended` adds
4 to January, and `in_month` adds 2 to each month. The new Lasso adds each
station's own monthly usage from 24 months earlier. It discards the first 24
training months, uses three expanding 24-month validation folds, and holds out
July 2024–June 2026 as test data. The lag for every test month comes from before
the test period. Metrics and predictions go to `reports/lasso_lag24/`.
`lasso_lag24_only.py` uses the original five features plus the same lag, without
the new holiday-duration columns. It reads the original processed data and
writes to `reports/lasso_lag24_only/`. Both lag experiments use the same split
and validation folds.
`lasso_lag12.py` uses the original five features plus a 12-month station lag.
It keeps the same 24-month validation and test windows. For months 13–24 in
each window, the lag comes from the model's earlier predictions in that window;
actual validation/test targets are never used as inputs. Its output is in
`reports/lasso_lag12_recursive/`.

The office calendar is a proxy for passenger holidays, not the holiday schedule
of every traveler. The test is retrospective: calendar revisions announced after
July 2024 may appear in the held-out period's features. A strict forecast made
in July 2024 would need the calendar version known then.
