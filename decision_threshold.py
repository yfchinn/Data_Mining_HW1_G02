"""THSR station-level decision threshold analysis.

Usage:
  python decision_threshold.py --model "Static OLS=results/ols/test_predictions.csv" \\
      --model "OLS Lag=results/ols_lag/test_predictions.csv"
  python decision_threshold.py --model "Static OLS=results/ols/test_predictions.csv" --phase cv

Input long-form CSV columns: date, target, actual, predicted.
For --phase cv, also require fold. Never use test results to choose a threshold.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

THRESHOLDS = (5.0, 10.0, 15.0)

def read_predictions(spec, phase):
    if "=" not in spec:
        raise ValueError("Use --model 'Model Name=path/to/predictions.csv'")
    name, filename = spec.split("=", 1)
    df = pd.read_csv(filename, encoding="utf-8-sig")
    required = {"date", "target", "actual", "predicted"}
    if phase == "cv":
        required.add("fold")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename}: missing {sorted(missing)}. CV needs raw fold predictions, not cv_metrics.csv.")
    df = df[list(required)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for col in ("actual", "predicted"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[["actual", "predicted"]].to_numpy()).all():
        raise ValueError(f"{filename}: non-finite predictions")
    if (df["actual"] < 0).any():
        raise ValueError(f"{filename}: negative actual passenger counts")
    keys = ["date", "target"] + (["fold"] if phase == "cv" else [])
    if df.duplicated(keys).any():
        raise ValueError(f"{filename}: duplicate date/target records")
    df["model"] = name.strip()
    return df

def metrics(group):
    y = group["actual"].to_numpy(dtype=float)
    pred = group["predicted"].to_numpy(dtype=float)
    mean_actual = y.mean()
    err = pred - y
    mae = np.abs(err).mean()
    rmse = np.sqrt(np.square(err).mean())
    denom = np.square(y - mean_actual).sum()
    return pd.Series({
        "months": len(group),
        "mean_actual": mean_actual,
        "MAE": mae,
        "RMSE": rmse,
        "ME": err.mean(),  # positive: overestimation
        "NMAE_pct": 100 * mae / mean_actual if mean_actual > 0 else np.nan,
        "R2": 1 - np.square(err).sum() / denom if denom > 0 else np.nan,
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="Model Name=path/to/predictions.csv; repeat per model")
    parser.add_argument("--phase", choices=["test", "cv"], default="test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    frames = [read_predictions(spec, args.phase) for spec in args.model]
    if len({f["model"].iloc[0] for f in frames}) != len(frames):
        raise ValueError("Model names must be unique")

    # Check same targets and dates across models, within each fold when using CV.
    key = ["date", "target"] + (["fold"] if args.phase == "cv" else [])
    base = frames[0].set_index(key).sort_index()
    for frame in frames[1:]:
        other = frame.set_index(key).sort_index()
        if not base.index.equals(other.index):
            raise ValueError("Models have different target/date/fold coverage. Compare on identical observations.")
        if not np.allclose(base["actual"], other["actual"], rtol=0, atol=1e-6):
            raise ValueError("Actual values differ across model files")

    df = pd.concat(frames, ignore_index=True)
    group_cols = ["model", "target"] + (["fold"] if args.phase == "cv" else [])
    station = df.groupby(group_cols, sort=True).apply(metrics).reset_index()
    if station["NMAE_pct"].isna().any():
        raise ValueError("Some targets have zero mean actual demand; NMAE undefined.")

    for threshold in THRESHOLDS:
        station[f"pass_{int(threshold)}pct"] = station["NMAE_pct"] <= threshold

    # Macro-average across station-direction targets.
    if args.phase == "cv":
        fold_summary = station.groupby(["model", "fold"]).agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            NMAE_pct=("NMAE_pct", "mean"),
            **{f"pass_rate_{int(t)}pct": (f"pass_{int(t)}pct", "mean") for t in THRESHOLDS}
        ).reset_index()
        for t in THRESHOLDS:
            fold_summary[f"pass_rate_{int(t)}pct"] *= 100
        summary = fold_summary.groupby("model").agg(
            **{f"{col}_mean": (col, "mean") for col in fold_summary.columns if col not in ("model", "fold")},
            **{f"{col}_std": (col, "std") for col in fold_summary.columns if col not in ("model", "fold")}
        ).reset_index()
    else:
        fold_summary = None
        summary = station.groupby("model").agg(
            targets=("target", "nunique"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            NMAE_pct=("NMAE_pct", "mean"),
            ME=("ME", "mean"),
            **{f"pass_rate_{int(t)}pct": (f"pass_{int(t)}pct", "mean") for t in THRESHOLDS}
        ).reset_index()
        for t in THRESHOLDS:
            summary[f"pass_rate_{int(t)}pct"] *= 100

    output = Path(args.output or f"results/decision_threshold/{args.phase}")
    output.mkdir(parents=True, exist_ok=True)
    station.to_csv(output / "station_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "model_threshold_summary.csv", index=False, encoding="utf-8-sig")
    if fold_summary is not None:
        fold_summary.to_csv(output / "fold_threshold_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\\nPhase: {args.phase.upper()}")
    print("Thresholds: 5%, 10% (provisional primary), 15%")
    print(summary.round(3).to_string(index=False))
    print(f"\\nSaved to {output.resolve()}")
    if args.phase == "test":
        print("IMPORTANT: test results must not be used to choose/change the primary threshold.")

if __name__ == "__main__":
    main()
