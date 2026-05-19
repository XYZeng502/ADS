"""Experiment 01b: Conformal Prediction for Spend T+1 Intervals.

Wraps standard MSE XGBoost with MAPIE SplitConformalRegressor to produce
calibrated 80% prediction intervals. Compares coverage/width against
the quantile regression approach from exp_01.

Usage:
    python scripts/experiments/exp_01b_conformal.py --input daily_merged.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor

from app.experiments.core import (
    evaluate,
    compute_coverage,
    compute_interval_width,
    build_model,
    feature_cols,
    apply_holiday_transition_calibration,
)
from app.unified_daily import build_unified_daily
from app.prediction_artifacts import validate_merged_daily_csv

EXPERIMENT_NAME = "exp_01b_conformal"
TARGET_COVERAGE = 0.80  # desired confidence level


def _assign_spend_bucket(df: pd.DataFrame) -> pd.DataFrame:
    app_avg = df.groupby("应用ID")["y_true"].mean().rename("app_avg_spend")
    df = df.join(app_avg, on="应用ID")
    df["spend_bucket"] = pd.qcut(df["app_avg_spend"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    return df


def _train_conformal_predict(train_df, test_df, feats, weight_exponent, use_gpu):
    """Train XGBoost + MAPIE conformal, return predictions with intervals."""
    train = train_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    if train.empty or test.empty or len(train) < 50:
        return None

    x_train = train[feats].values
    y_train = train["target_t1_spend"].clip(lower=0.0).values
    x_test = test[feats].values
    y_test = test["target_t1_spend"].values
    sample_weight = np.power(train["消耗金额"].clip(lower=0).values, weight_exponent)

    # Split train into proper training + calibration
    split_idx = int(len(x_train) * 0.8)
    x_proper = x_train[:split_idx]
    x_calib = x_train[split_idx:]
    y_proper = y_train[:split_idx]
    y_calib = y_train[split_idx:]
    sw_proper = sample_weight[:split_idx]

    # Train base model
    model = build_model("XGBoost", use_gpu=use_gpu)
    if model is None:
        return None
    try:
        model.fit(x_proper, y_proper, sample_weight=sw_proper)
    except TypeError:
        model.fit(x_proper, y_proper)

    # Wrap with MAPIE conformal (pre-trained estimator, prefit=True is default)
    conformal = SplitConformalRegressor(
        estimator=model, confidence_level=TARGET_COVERAGE
    )
    conformal.conformalize(x_calib, y_calib)

    # Predict with intervals
    # y_pis shape: (n_samples, 2, n_confidence_levels)
    y_pred, y_pis = conformal.predict_interval(x_test)
    y_lower = y_pis[:, 0, 0]
    y_upper = y_pis[:, 1, 0]

    # Apply holiday calibration
    y_pred = apply_holiday_transition_calibration(test, y_pred)

    return pd.DataFrame({
        "日期": test["日期"].values,
        "应用ID": test["应用ID"].values,
        "y_true": y_test,
        "y_pred": np.clip(y_pred, 0.0, None),
        "y_lower": np.clip(y_lower, 0.0, None),
        "y_upper": np.clip(y_upper, 0.0, None),
        "model": "XGBoost_Conformal",
    })


def main():
    parser = argparse.ArgumentParser(description="Exp 01b: Conformal prediction intervals")
    parser.add_argument("--input", default="daily_merged.csv")
    parser.add_argument("--eval-recent-days", type=int, default=20)
    parser.add_argument("--weight-exponent", type=float, default=0.35)
    args = parser.parse_args()

    validate_merged_daily_csv(Path(args.input))

    out_dir = Path("outputs/experiments") / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _ = build_unified_daily(args.input, min_spend_train=2.0)
    feats = feature_cols("base")

    # Build expanding-window encodings
    model_df = df.sort_values(["应用ID", "日期"]).reset_index(drop=True).copy()
    grp_spend = model_df.groupby("应用ID")["target_t1_spend"]
    model_df["app_avg_spend"] = (
        grp_spend.cumsum().sub(model_df["target_t1_spend"])
        / grp_spend.cumcount().clip(lower=1)
    )
    model_df["app_avg_spend"] = model_df["app_avg_spend"].fillna(model_df["target_t1_spend"].median())
    if "roi_d1" in model_df.columns:
        grp_roi = model_df.groupby("应用ID")["roi_d1"]
        model_df["app_avg_roi_d1"] = (
            grp_roi.cumsum().sub(model_df["roi_d1"])
            / grp_spend.cumcount().clip(lower=1)
        )
        model_df["app_avg_roi_d1"] = model_df["app_avg_roi_d1"].fillna(model_df["roi_d1"].median())
        feats_all = feats + ["app_avg_spend", "app_avg_roi_d1"]
    else:
        feats_all = feats + ["app_avg_spend"]

    # Walk-forward loop
    dates = sorted(model_df["日期"].dropna().unique())
    cutoffs = list(dates[:-1])
    eval_recent = args.eval_recent_days
    if eval_recent > 0 and len(cutoffs) > eval_recent:
        cutoffs = cutoffs[-eval_recent:]

    all_preds = []
    n_cutoffs = len(cutoffs)
    for i, cutoff in enumerate(cutoffs):
        t_cutoff = pd.Timestamp(cutoff)
        next_day = cutoff + np.timedelta64(1, "D")
        train = model_df[model_df["日期"] <= t_cutoff].copy()
        test = model_df[model_df["日期"] == next_day].copy()
        if test.empty or train["日期"].nunique() < 20:
            continue
        print(f"  [{i+1}/{n_cutoffs}] cutoff={t_cutoff.date()} train={len(train)} test={len(test)}", flush=True)
        pred_df = _train_conformal_predict(
            train, test, feats_all, args.weight_exponent, use_gpu=False
        )
        if pred_df is not None and not pred_df.empty:
            all_preds.append(pred_df)

    if not all_preds:
        raise ValueError("No results produced. Check data and parameters.")

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df = _assign_spend_bucket(pred_df)

    y_true = pred_df["y_true"].values
    y_lower = pred_df["y_lower"].values
    y_upper = pred_df["y_upper"].values
    y_pred = pred_df["y_pred"].values

    coverage = compute_coverage(y_true, y_lower, y_upper)
    interval_width_median = float(np.median(compute_interval_width(y_lower, y_upper, y_pred)))
    point_metrics = evaluate(y_true, y_pred)

    # Per-bucket
    bucket_stats = {}
    for bucket in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        b = pred_df[pred_df["spend_bucket"] == bucket]
        if b.empty:
            continue
        bucket_stats[bucket] = {
            "n": int(len(b)),
            "coverage": round(compute_coverage(b["y_true"].values, b["y_lower"].values, b["y_upper"].values), 4),
            "interval_width_median": round(float(np.median(
                compute_interval_width(b["y_lower"].values, b["y_upper"].values, b["y_pred"].values)
            )), 4),
            "mape_pct": round(evaluate(b["y_true"].values, b["y_pred"].values)["mape_pct"], 2),
        }

    coverage_err = abs(coverage - TARGET_COVERAGE)
    verdict = "promising" if coverage_err <= 0.05 else ("inconclusive" if coverage_err <= 0.15 else "regress")

    report = {
        "experiment": EXPERIMENT_NAME,
        "method": "SplitConformalRegressor (MAPIE) with XGBoost base",
        "target_coverage": TARGET_COVERAGE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_recent_days": args.eval_recent_days,
            "weight_exponent": args.weight_exponent,
        },
        "metrics": {
            "coverage": round(coverage, 4),
            "coverage_error_pp": round(coverage_err * 100, 2),
            "interval_width_median": round(interval_width_median, 4),
            "mape_pct": round(point_metrics["mape_pct"], 2),
        },
        "by_spend_bucket": bucket_stats,
        "verdict": verdict,
    }

    pred_df.to_csv(out_dir / "predictions_XGBoost_Conformal.csv", index=False, encoding="utf-8-sig")
    (out_dir / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport saved to {out_dir / 'experiment_report.json'}")
    print(f"  Coverage: {coverage:.4f} (target: {TARGET_COVERAGE})")
    print(f"  Interval width (median): {interval_width_median:.4f}")
    print(f"  MAPE: {point_metrics['mape_pct']:.2f}%")
    for b, s in bucket_stats.items():
        print(f"  {b}: coverage={s['coverage']:.4f}, width={s['interval_width_median']:.4f}")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
