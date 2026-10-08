"""
執行python preprocessing.py，即可在 data/processed/ 取得處理後的資料，原始 CSV 不會被修改。
亦可直接讀取使用 data/processed/merged_monthly.csv。
hsr_passengers 是該月份的旅客人數，可作為預測目標。
所有資料統一使用 2007 年 1 月至 2026 年 7 月（含首尾），共 235 個月。
日期以西元每月第一天表示整個月份。

將車站尚未啟用的缺值填為 0。
月份週期用 sin/cos 做encoding。

由於資料是否需縮放是取決於模型的選擇，因此這支程式不直接縮放完整資料，而是提供縮放函式供引用。
請用 from preprocessing import fit_scaler, scale_features 匯入標準化函式，
然後用 
parameters = fit_scaler(train_features, columns)
scale_features(data, parameters)
完成標準化，可以套用到訓練、測試及未來資料。

由於剔除異常值須根據train/test split的結果 (train不可看到test的部分來剔除outliers)，因此這支程式不直接剔除異常值，而是提供剔除異常值的函式 fit_outlier_bounds 與 remove_outliers 供引用。
先按時間切分資料，再只使用訓練資料估計界線。
預設採用所有訓練月份的 1.5 IQR 界線，也可指定 method="sigma"，使用訓練資料的平均值 ± 3 個樣本標準差。

from preprocessing import fit_outlier_bounds, remove_outliers

data = pd.read_csv("data/processed/merged_monthly.csv", parse_dates=["date"])
data = data.sort_values("date").set_index("date")
# 示範切分
train = data.iloc[:-24].copy()
test = data.iloc[-24:].copy()

# 明確選擇要檢查的欄位。此例只依訓練期目標值篩選訓練月份；
# hsr_passengers 不可作為預測同月份 hsr_passengers 的輸入特徵
columns = ["hsr_passengers"]
bounds = fit_outlier_bounds(train, columns)  # 預設 method="iqr", multiplier=1.5
clean_train, removed_train = remove_outliers(train, bounds)
print(f"保留 {len(clean_train)} 列，剔除 {len(removed_train)} 列")
print(removed_train[columns])

- multiplier 可調整界線倍數。
- 方法、倍數及檢查欄位只在訓練期選擇，不能根據測試結果調整。
- 使用 TimeSeriesSplit 時，每折只以該折訓練部分估計界線並剔除訓練列，驗證列保留。
- 多欄位聯合篩選可能刪除很多月份，應記錄剔除比例，並以相同驗證期間比較有／無剔除的效果。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSIS_START = "2007-01-01"
ANALYSIS_END = "2026-07-01"
MONTH_PATTERN = re.compile(r"^(\d{3})年(\d{2})月$")
MISSING_TOKENS = {"", "-", "--", "NA", "N/A", "NaN", "null"}
STATIONS = ["total", "nangang", "taipei", "banqiao", "taoyuan", "hsinchu",
            "miaoli", "taichung", "changhua", "yunlin", "chiayi", "tainan", "zuoying"]
OPENING_MONTHS = {"taipei": "2007-03-01", "miaoli": "2015-12-01",
                  "changhua": "2015-12-01", "yunlin": "2015-12-01",
                  "nangang": "2016-07-01"}
BUS_COMMON = ["route_km", "vehicles", "trips", "vehicle_km", "passengers",
              "passenger_km", "revenue_twd"]
BUS_CITY = ["route_km", "vehicles", "trips", "vehicle_km", "vehicle_days",
            "fuel_liters", "passengers", "daily_passengers", "passenger_km", "revenue_twd"]
DATASETS = {
    "bus": ("汽車客運概況.csv", [f"bus_{group}_{metric}"
            for group, metrics in [("total", BUS_COMMON), ("city", BUS_CITY),
                                   ("highway", BUS_COMMON)] for metric in metrics]),
    "hsr_service": ("高速鐵路行駛次數及行駛公里.csv", ["hsr_trips", "hsr_train_km"]),
    "hsr_usage": ("高速鐵路客運概況.csv", ["hsr_passengers", "hsr_daily_passengers",
        "hsr_passenger_km", "hsr_avg_trip_km", "hsr_occupancy_pct", "hsr_punctuality_pct"]),
    "hsr_stations": ("高速鐵路旅客人數─按進出站分.csv",
                     [f"hsr_{direction}_{station}" for direction in ["entry", "exit"]
                      for station in STATIONS]),
}
FLAG_COLUMNS = ["dataset", "date", "column", "value", "reason", "lower_bound", "upper_bound"]


def read_monthly(path: Path, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Use a CSV parser so quoted multiline footnotes remain single records."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    station_file = columns[0].startswith("hsr_entry_")
    labels = rows[1][2:] if station_file else rows[0][2:]
    if len(labels) != len(columns):
        raise ValueError(f"Unexpected header width in {path.name}")
    if station_file:
        labels = [f"{direction}_{label}" for direction in ["進站", "出站"]
                  for label in rows[1][2:15]]
        # Validate both station orders, rather than silently mislabeling a new source.
        if rows[1][2:15] != rows[1][15:28]:
            raise ValueError(f"Different entry/exit station order in {path.name}")
    records, excluded = [], []
    for row in rows:
        if not row:
            continue
        label = row[0].strip()
        match = MONTH_PATTERN.fullmatch(label)
        if not match:
            excluded.append(label)
            continue
        if len(row) != len(columns) + 2 or row[1].strip():
            raise ValueError(f"Unexpected monthly row structure: {path.name}: {label}")
        date = pd.Timestamp(int(match[1]) + 1911, int(match[2]), 1)
        values = []
        for column, token in zip(columns, row[2:]):
            token = token.strip()
            if token in MISSING_TOKENS:
                values.append(np.nan)
            else:
                try:
                    values.append(float(token.replace(",", "")))
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value: {path.name}: {label}: {column}") from exc
        records.append([date, *values])
    frame = pd.DataFrame(records, columns=["date", *columns])
    if frame.empty:
        raise ValueError(f"No monthly records in {path.name}")
    original_count = len(frame)
    frame = frame.drop_duplicates()
    if frame["date"].duplicated().any():
        raise ValueError(f"Conflicting duplicate months in {path.name}")
    duplicate_count = original_count - len(frame)
    frame = frame.set_index("date").sort_index()
    complete_index = pd.date_range(frame.index.min(), frame.index.max(), freq="MS", name="date")
    missing_months = complete_index.difference(frame.index)
    frame = frame.reindex(complete_index)
    metadata = {
        "source_file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_monthly_rows": original_count,
        "exact_duplicates_removed": duplicate_count,
        "missing_months": [d.strftime("%Y-%m-%d") for d in missing_months],
        "excluded_record_count": len(excluded),
        "source_notes": [row[0] for row in rows if row and
                         row[0].startswith(("說明", "資料來源", "單位", "最近一次"))],
        "columns": dict(zip(columns, labels)),
    }
    return frame, metadata


def missing_reason(column: str, date: pd.Timestamp, missing_months: set[str]) -> str:
    if date.strftime("%Y-%m-%d") in missing_months:
        return "missing_month"
    if column.startswith(("hsr_entry_", "hsr_exit_")):
        station = column.split("_", 2)[2]
        if station in OPENING_MONTHS and date < pd.Timestamp(OPENING_MONTHS[station]):
            return "station_not_open"
    first_recorded = {"bus_city_fuel_liters": "1992-01-01",
                      "bus_city_vehicle_days": "2001-01-01"}
    if column in first_recorded and date < pd.Timestamp(first_recorded[column]):
        return "not_yet_collected"
    return "unexpected_missing"


def inspect_quality(name: str, frame: pd.DataFrame, metadata: dict) -> tuple[list, list]:
    """Flag values only. Seasonal IQR uses at most five earlier same-month values."""
    missing, flags = [], []
    missing_months = set(metadata["missing_months"])
    for column in frame.columns:
        series = frame[column]
        for date in series.index[series.isna()]:
            missing.append({"dataset": name, "date": date, "column": column,
                            "reason": missing_reason(column, date, missing_months)})
        finite = series.notna() & np.isfinite(series)
        invalid = series.notna() & (~finite | (series < 0))
        if column.endswith("_pct"):
            invalid |= series > 100
        for date in series.index[invalid]:
            flags.append({"dataset": name, "date": date, "column": column,
                          "value": series.loc[date], "reason": "invalid_range",
                          "lower_bound": 0, "upper_bound": 100 if column.endswith("_pct") else None})
        valid = series.where(~invalid)
        grouped = valid.groupby(valid.index.month)
        q1 = grouped.transform(lambda s: s.shift(1).rolling(5, min_periods=3).quantile(0.25))
        q3 = grouped.transform(lambda s: s.shift(1).rolling(5, min_periods=3).quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier = finite & ~invalid & (iqr > 0) & ((series < lower) | (series > upper))
        for date in series.index[outlier]:
            flags.append({"dataset": name, "date": date, "column": column,
                          "value": series.loc[date], "reason": "historical_seasonal_iqr",
                          "lower_bound": lower.loc[date], "upper_bound": upper.loc[date]})
    return missing, flags


def encode_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode known calendar information without fitting parameters."""
    result = frame.copy()
    result["month_sin"] = np.sin(2 * np.pi * (result.index.month - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (result.index.month - 1) / 12)
    result["days_in_month"] = result.index.days_in_month
    return result


def _outlier_values(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Exclude original missing/imputed cells from statistical outlier checks."""
    values = frame[columns].astype(float).copy()
    for column in columns:
        indicator = f"{column}_is_missing"
        if indicator in frame:
            if not frame[indicator].isin([0, 1]).all():
                raise ValueError(f"{indicator} must contain only 0 or 1")
            values.loc[frame[indicator].eq(1), column] = np.nan
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Resolve infinite values before statistical outlier detection")
    return values


def fit_outlier_bounds(training_data: pd.DataFrame, columns: list[str],
                       method: str = "iqr", multiplier: float | None = None) -> dict:
    """Fit fixed per-column bounds using ONLY the caller's training partition.

    Default method is iqr: Q1 - 1.5 IQR, Q3 + 1.5 IQR.
    Optional sigma: mean +/- 3 sample standard deviations (ddof=1).
    Uses all training months together, not the historical seasonal report.
    Requires >=3 observed values per column. Zero-spread columns are skipped.
    Original missing flags must accompany data to exclude pre-opening zeros.
    The caller must split first and refit inside each CV training fold.
    """
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Supply a nonempty list of unique numeric columns")
    if any(c in {"date", "month_sin", "month_cos", "days_in_month"}
           or c.endswith("_is_missing") for c in columns):
        raise ValueError("Do not select dates, calendar encoding or missing indicators")
    if method not in {"sigma", "iqr"}:
        raise ValueError("method must be 'sigma' or 'iqr'")
    multiplier = (3.0 if method == "sigma" else 1.5) if multiplier is None else multiplier
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be finite and positive")
    values = _outlier_values(training_data, columns)
    if (values.count() < 3).any():
        raise ValueError("Each selected column needs at least three observed training values")
    if method == "sigma":
        low_center = high_center = values.mean()
        spread = values.std(ddof=1)
    else:
        low_center, high_center = values.quantile(0.25), values.quantile(0.75)
        spread = high_center - low_center
    lower, upper = low_center - multiplier * spread, high_center + multiplier * spread
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Nonfinite bounds; check the scale of training values")
    return {"method": method, "multiplier": float(multiplier), "columns": list(columns),
            "lower": lower.to_dict(), "upper": upper.to_dict(),
            "skipped_columns": spread.index[spread.eq(0)].tolist(),
            "observed_counts": values.count().to_dict()}


def remove_outliers(data: pd.DataFrame, bounds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept_rows, removed_rows), preserving columns, index and row order.

    Remove a whole row if ANY selected observed value is strictly outside its
    fitted bounds. Missing/imputed values and zero-spread columns do not trigger
    removal. Inputs are unchanged; no statistics are refitted here.
    Intended for training data only: keep validation/test rows for fair scoring.
    Build lag/rolling features BEFORE removal to preserve calendar alignment.
    """
    values = _outlier_values(data, bounds["columns"])
    flagged = values.lt(pd.Series(bounds["lower"])) | values.gt(pd.Series(bounds["upper"]))
    flagged.loc[:, bounds["skipped_columns"]] = False
    remove = flagged.any(axis=1)
    return data.loc[~remove].copy(), data.loc[remove].copy()


def fit_scaler(training_features: pd.DataFrame, columns: list[str]) -> dict:
    """Fit population-standard-deviation scaling on caller-selected training rows only.

    Missing values stay missing. Entirely missing or nonfinite columns are errors.
    Targets, dates, calendar encoding and indicator columns must not be selected.
    """
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Supply a nonempty list of unique continuous feature columns")
    forbidden = {"hsr_passengers", "date", "month_sin", "month_cos", "days_in_month"}
    if forbidden.intersection(columns) or any(c.endswith("_is_missing") for c in columns):
        raise ValueError("Do not scale the target, dates, calendar encoding or indicators")
    selected = training_features[columns].astype(float)
    if selected.empty or selected.isna().all().any():
        raise ValueError("Each selected feature needs at least one observed training value")
    if np.isinf(selected.to_numpy()).any():
        raise ValueError("Resolve nonfinite training values before scaling")
    means = selected.mean()
    scales = selected.std(ddof=0).replace(0, 1.0)
    return {"method": "standard", "columns": columns,
            "mean": means.to_dict(), "scale": scales.to_dict()}


def scale_features(features: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Apply previously fitted parameters without refitting or changing other columns."""
    result = features.copy()
    columns = parameters["columns"]
    result[columns] = (result[columns] - pd.Series(parameters["mean"])) / pd.Series(parameters["scale"])
    return result


def check_consistency(tables: dict[str, pd.DataFrame]) -> list[dict]:
    results = []

    def compare(label: str, actual: pd.Series, expected: pd.Series, tolerance: float = 0):
        pair = pd.concat([actual.rename("actual"), expected.rename("expected")], axis=1)
        available = pair.notna().all(axis=1) & np.isfinite(pair).all(axis=1)
        difference = pair["actual"] - pair["expected"]
        failures = available & (difference.abs() > tolerance)
        results.append({"check": label, "checked_months": int(available.sum()),
                        "unchecked_months": int((~available).sum()),
                        "mismatches": [{"date": d.strftime("%Y-%m-%d"),
                                        "difference": float(difference.loc[d])}
                                       for d in pair.index[failures]]})

    stations = tables["hsr_stations"]
    for direction in ["entry", "exit"]:
        station_values = stations[[f"hsr_{direction}_{s}" for s in STATIONS[1:]]].copy()
        # Only documented pre-opening cells count as zero for reconciliation.
        # The export also fills documented pre-opening missing cells with zero.
        for station, opening in OPENING_MONTHS.items():
            column = f"hsr_{direction}_{station}"
            mask = (station_values.index < pd.Timestamp(opening)) & station_values[column].isna()
            station_values.loc[mask, column] = 0
        compare(f"{direction}_station_sum", stations[f"hsr_{direction}_total"],
                station_values.sum(axis=1, min_count=len(STATIONS) - 1))
        compare(f"{direction}_total_vs_usage", stations[f"hsr_{direction}_total"],
                tables["hsr_usage"]["hsr_passengers"])
    bus = tables["bus"]
    for metric in BUS_COMMON:
        compare(f"bus_total_{metric}_vs_city_plus_highway", bus[f"bus_total_{metric}"],
                bus[f"bus_city_{metric}"] + bus[f"bus_highway_{metric}"],
                tolerance=0.02 if metric == "route_km" else 0)
    return results


def save_csv(frame: pd.DataFrame, path: Path, index: bool = True) -> None:
    frame.to_csv(path, index=index, encoding="utf-8-sig", date_format="%Y-%m-%d", na_rep="")


def run(input_dir: Path, output_dir: Path) -> dict:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory must differ from the raw data directory")
    tables, summaries, all_missing, all_flags = {}, {}, [], []
    for name, (filename, columns) in DATASETS.items():
        frame, metadata = read_monthly(input_dir / filename, columns)
        analysis_index = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="MS", name="date")
        if analysis_index.min() < frame.index.min() or analysis_index.max() > frame.index.max():
            raise ValueError(f"{name} does not cover the required analysis period")
        metadata["source_start"] = frame.index.min().strftime("%Y-%m-%d")
        metadata["source_end"] = frame.index.max().strftime("%Y-%m-%d")
        frame = frame.reindex(analysis_index)
        metadata["missing_months"] = [
            date for date in metadata["missing_months"]
            if ANALYSIS_START <= date <= ANALYSIS_END
        ]
        missing, flags = inspect_quality(name, frame, metadata)
        metadata.update({"rows": len(frame), "start": frame.index.min().strftime("%Y-%m-%d"),
                         "end": frame.index.max().strftime("%Y-%m-%d"),
                         "missing_cells": len(missing), "flagged_cells": len(flags)})
        metadata["imputed_zero_cells"] = sum(item["reason"] == "station_not_open" for item in missing)
        metadata["remaining_missing_cells"] = len(missing) - metadata["imputed_zero_cells"]
        tables[name], summaries[name] = frame, metadata
        all_missing.extend(missing)
        all_flags.extend(flags)

    # All tables use the same fixed analysis period.
    merged = tables["hsr_usage"].join(tables["hsr_service"], how="left", validate="one_to_one")
    merged = merged.join(tables["hsr_stations"], how="left", validate="one_to_one")
    merged = merged.join(tables["bus"], how="left", validate="one_to_one")
    for name in ["hsr_service", "hsr_stations", "bus"]:
        absent = merged.index.difference(tables[name].index)
        for date in absent:
            reason = "outside_source_coverage"
            if date > tables[name].index.max():
                reason = "not_yet_published"
            for column in tables[name].columns:
                all_missing.append({"dataset": "merged", "date": date, "column": column,
                                    "reason": reason})

    report = {
        "analysis_period": {"start": ANALYSIS_START, "end": ANALYSIS_END,
                            "months": len(merged)},
        "datasets": summaries,
        "merged": {"rows": len(merged), "target": "hsr_passengers",
                   "join": "left joins on HSR usage months", "scaled": False},
        "missing_policy": "Fill only station_not_open missing cells with zero. Preserve other missing values. missing_cells, missing_values.csv and _is_missing indicators describe pre-imputation missingness.",
        "outlier_policy": "Flag only. Same calendar month, previous five years, at least three valid observations; 1.5 IQR. Zero IQR is skipped.",
        "scaling_policy": "Deferred. Call fit_scaler on training features, then scale_features using those parameters.",
        "consistency_checks": check_consistency(tables),
        "definition_changes": [
            "HSR started on 2007-01-05; the first operating month is partial.",
            "Taipei opened on 2007-03-02; March 2007 is partial.",
            "HSR punctuality threshold changed from <10 minutes in 2007 to <5 minutes in 2008.",
            "Bus trip-count definitions and city data provider changed in 2010.",
            "Bus route distance, geographic coverage and passenger coverage changed in 2011.",
            "City bus coverage changed in September 2024; see source notes.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {**tables, "merged_monthly": merged}.items():
        encoded = encode_calendar(frame)
        indicators = frame.isna().astype("int8").add_suffix("_is_missing")
        for item in all_missing:
            if item["reason"] == "station_not_open" and (item["dataset"] == name or name == "merged_monthly"):
                encoded.loc[item["date"], item["column"]] = 0
        save_csv(encoded.join(indicators), output_dir / f"{name}.csv")
    save_csv(pd.DataFrame(all_missing, columns=["dataset", "date", "column", "reason"]),
             output_dir / "missing_values.csv", index=False)
    save_csv(pd.DataFrame(all_flags, columns=FLAG_COLUMNS), output_dir / "quality_flags.csv", index=False)
    dictionary = [{"dataset": name, "column": column, "source_label": label}
                  for name, metadata in summaries.items() for column, label in metadata["columns"].items()]
    save_csv(pd.DataFrame(dictionary), output_dir / "data_dictionary.csv", index=False)
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    report = run(args.input_dir, args.output_dir)
    for name, summary in report["datasets"].items():
        print(f"{name}: {summary['rows']} months, {summary['remaining_missing_cells']} remaining missing cells, "
              f"{summary['imputed_zero_cells']} cells filled with zero, "
              f"{summary['flagged_cells']} flagged cells")
    mismatches = sum(len(check["mismatches"]) for check in report["consistency_checks"])
    print(f"Consistency mismatches retained for review: {mismatches}")
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
