"""Experiment 01c: Conformalized Quantile Regression (CQR).

Combines quantile regression with conformal calibration — the direct
fix for the heteroscedasticity problem found in exp_01b.

Usage:
    python scripts/experiments/exp_01c_cqr.py --input daily_merged.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from mapie.regression import ConformalizedQuantileRegressor

from app.experiments.core import (
    evaluate,
    compute_coverage,
    compute_interval_width,
    build_model,
    feature_cols,
    apply_holiday_transition_calibration,
    assign_spend_bucket,
    compute_expanding_window_encodings,
    walk_forward_windows,
    coverage_verdict,
    SPEND_BUCKET_LABELS,
)
from app.unified_daily import build_unified_daily
from app.prediction_artifacts import validate_merged_daily_csv

EXPERIMENT_NAME = "exp_01c_cqr"
TARGET_COVERAGE = 0.80


def _train_cqr_predict(train_df, test_df, feats, weight_exponent, use_gpu, q4_confidence=0.85):
    train = train_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    if train.empty or test.empty or len(train) < 60:
        return None

    # Assign static spend buckets for per-bucket calibration
    train["_bucket"] = _assign_static_bucket_label(train)
    test["_bucket"] = _assign_static_bucket_label(test)

    x = train[feats].values
    y = train["target_t1_spend"].clip(lower=0.0).values
    x_test = test[feats].values
    y_test = test["target_t1_spend"].values
    sw = np.power(train["消耗金额"].clip(lower=0).values, weight_exponent)

    # Split: proper train (70%) / calibration (30%)
    n = len(x)
    n_proper = int(n * 0.7)
    x_proper, x_calib = x[:n_proper], x[n_proper:]
    y_proper, y_calib = y[:n_proper], y[n_proper:]
    sw_proper = sw[:n_proper]
    calib_buckets = train["_bucket"].values[n_proper:]

    # Train 3 quantile XGBoost models on ALL proper training data
    estimators = []
    for alpha in [0.05, 0.5, 0.95]:
        m = build_model("XGBoost", use_gpu=use_gpu,
                        objective="reg:quantileerror", quantile_alpha=alpha)
        if m is None:
            return None
        try:
            m.fit(x_proper, y_proper, sample_weight=sw_proper)
        except TypeError:
            m.fit(x_proper, y_proper)
        estimators.append(m)

    def _calibrate_and_predict(x_cal, y_cal, x_test_subset, conf_level):
        if len(x_cal) < 20:
            return None
        cqr = ConformalizedQuantileRegressor(
            estimator=estimators,
            confidence_level=conf_level,
            prefit=True,
        )
        cqr.conformalize(x_cal, y_cal)
        _, y_pis = cqr.predict_interval(x_test_subset)
        y_pis = y_pis.squeeze(-1)
        return y_pis

    # Per-bucket confidence levels: Q4 gets higher target
    bucket_conf = {"Q4_high": q4_confidence, "Q3": TARGET_COVERAGE,
                   "Q2": TARGET_COVERAGE, "Q1_low": TARGET_COVERAGE}

    n_test = len(x_test)
    y_pis_all = np.zeros((n_test, 2))

    y_pred_raw_all = estimators[1].predict(x_test)

    test_buckets = test["_bucket"].values
    buckets_used = set()

    for bucket in sorted(set(test_buckets)):
        calib_mask = calib_buckets == bucket
        test_mask = test_buckets == bucket

        if not calib_mask.any() or calib_mask.sum() < 20:
            continue

        x_cal_bucket = x_calib[calib_mask]
        y_cal_bucket = y_calib[calib_mask]
        x_test_bucket = x_test[test_mask]

        conf = bucket_conf.get(bucket, TARGET_COVERAGE)
        y_pis_bucket = _calibrate_and_predict(x_cal_bucket, y_cal_bucket, x_test_bucket, conf)
        if y_pis_bucket is not None:
            y_pis_all[test_mask] = y_pis_bucket
            buckets_used.add(bucket)

    # Fallback: global CQR for buckets without enough calibration samples
    missing_mask = ~np.array([b in buckets_used for b in test_buckets])
    if missing_mask.any():
        y_pis_global = _calibrate_and_predict(x_calib, y_calib, x_test[missing_mask], TARGET_COVERAGE)
        if y_pis_global is not None:
            y_pis_all[missing_mask] = y_pis_global

    y_pred = apply_holiday_transition_calibration(test, y_pred_raw_all)

    # Ensure intervals are non-crossing
    y_lower = np.min(y_pis_all, axis=1)
    y_upper = np.max(y_pis_all, axis=1)

    return pd.DataFrame({
        "日期": test["日期"].values,
        "应用ID": test["应用ID"].values,
        "y_true": y_test,
        "y_pred": np.clip(y_pred, 0.0, None),
        "y_lower": np.clip(y_lower, 0.0, None),
        "y_upper": np.clip(y_upper, 0.0, None),
        "model": "XGBoost_CQR_bucket",
    })


def _assign_static_bucket_label(df):
    """Assign static spend bucket labels Q1-Q4 based on app mean spend."""
    train_holdout = df["target_t1_spend"] if "target_t1_spend" in df.columns else df["消耗金额"]
    app_avg = df.groupby("应用ID")[train_holdout.name].transform("mean")
    return pd.qcut(app_avg, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])


def main():
    parser = argparse.ArgumentParser(description="Exp 01c: CQR intervals")
    parser.add_argument("--input", default="daily_merged.csv")
    parser.add_argument("--eval-recent-days", type=int, default=20)
    parser.add_argument("--weight-exponent", type=float, default=0.35)
    parser.add_argument("--q4-confidence", type=float, default=0.85,
                        help="Target coverage for Q4 high-spend bucket (default 0.85)")
    args = parser.parse_args()

    validate_merged_daily_csv(Path(args.input))

    out_dir = Path("outputs/experiments") / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _ = build_unified_daily(args.input, min_spend_train=2.0)
    feats = feature_cols("base")

    model_df, enc_feats = compute_expanding_window_encodings(df)
    feats_all = feats + enc_feats

    all_preds = []
    for train, test, _, _ in walk_forward_windows(model_df, args.eval_recent_days):
        pred_df = _train_cqr_predict(train, test, feats_all, args.weight_exponent, use_gpu=False, q4_confidence=args.q4_confidence)
        if pred_df is not None and not pred_df.empty:
            all_preds.append(pred_df)

    if not all_preds:
        raise ValueError("No results.")

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df = assign_spend_bucket(pred_df)

    y_true = pred_df["y_true"].values
    y_lower = pred_df["y_lower"].values
    y_upper = pred_df["y_upper"].values
    y_pred = pred_df["y_pred"].values

    coverage = compute_coverage(y_true, y_lower, y_upper)
    width_median = float(np.median(compute_interval_width(y_lower, y_upper, y_pred)))
    point_metrics = evaluate(y_true, y_pred)

    bucket_stats = {}
    for bucket in SPEND_BUCKET_LABELS:
        b = pred_df[pred_df["spend_bucket"] == bucket]
        if b.empty:
            continue
        bucket_stats[bucket] = {
            "n": int(len(b)),
            "coverage": round(compute_coverage(b["y_true"].values, b["y_lower"].values, b["y_upper"].values), 4),
            "interval_width_median": round(
                float(np.median(compute_interval_width(b["y_lower"].values, b["y_upper"].values, b["y_pred"].values))), 4
            ),
            "mape_pct": round(evaluate(b["y_true"].values, b["y_pred"].values)["mape_pct"], 2),
        }

    verdict = coverage_verdict(coverage, TARGET_COVERAGE)

    bucket_coverages = [s["coverage"] for s in bucket_stats.values()]
    het_score = float(np.std(bucket_coverages)) if len(bucket_coverages) > 1 else 0.0

    report = {
        "experiment": EXPERIMENT_NAME,
        "method": "ConformalizedQuantileRegressor (MAPIE CQR) + XGBoost",
        "target_coverage": TARGET_COVERAGE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"eval_recent_days": args.eval_recent_days, "weight_exponent": args.weight_exponent, "q4_confidence": args.q4_confidence},
        "metrics": {
            "coverage": round(coverage, 4),
            "coverage_error_pp": round(abs(coverage - TARGET_COVERAGE) * 100, 2),
            "interval_width_median": round(width_median, 4),
            "mape_pct": round(point_metrics["mape_pct"], 2),
            "heteroscedasticity_score": round(het_score, 4),
        },
        "by_spend_bucket": bucket_stats,
        "comparison": {
            "quantile_p05p95_coverage": 0.7458,
            "conformal_coverage": 0.7392,
            "conformal_het_score": "Q1:0.998 vs Q4:0.313",
        },
        "verdict": verdict,
    }

    pred_df.to_csv(out_dir / "predictions_XGBoost_CQR.csv", index=False, encoding="utf-8-sig")
    (out_dir / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {out_dir / 'experiment_report.json'}")
    print(f"  Coverage: {coverage:.4f} (target {TARGET_COVERAGE})")
    print(f"  Width (median): {width_median:.4f}")
    print(f"  MAPE: {point_metrics['mape_pct']:.2f}%")
    print(f"  Heteroscedasticity score: {het_score:.4f} (0=perfect balance)")
    for b, s in bucket_stats.items():
        print(f"  {b}: coverage={s['coverage']:.4f}  width={s['interval_width_median']:.4f}")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
