"""Experiment 01: Quantile Regression for Spend T+1 Prediction.

Trains 3 XGBoost quantile variants (alpha=0.1/0.5/0.9) alongside the standard
MSE XGBoost model. Evaluates calibration quality via coverage, pinball loss,
and normalized interval width -- broken down by spend bucket.

Usage:
    python scripts/experiments/exp_01_quantile.py --input daily_merged.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from app.experiments.core import (
    evaluate,
    evaluate_pinball,
    compute_coverage,
    compute_interval_width,
    run,
)
from app.unified_daily import build_unified_daily
from app.prediction_artifacts import validate_merged_daily_csv

EXPERIMENT_NAME = "exp_01_quantile"
QUANTILE_ALPHAS = [0.1, 0.5, 0.9]


def _make_quantile_variants(alphas: List[float]) -> List[Dict]:
    variants = []
    for a in alphas:
        label = f"q{int(a * 100)}"
        variants.append({
            "suffix": label,
            "overrides": {
                "objective": "reg:quantileerror",
                "quantile_alpha": a,
            },
        })
    return variants


def _assign_spend_bucket(df: pd.DataFrame) -> pd.DataFrame:
    app_avg = df.groupby("应用ID")["y_true"].mean().rename("app_avg_spend")
    df = df.join(app_avg, on="应用ID")
    df["spend_bucket"] = pd.qcut(df["app_avg_spend"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    return df


def _find_quantile_keys(columns: list) -> tuple:
    p10_key = None
    p50_key = None
    p90_key = None
    for col in columns:
        if "_q10" in col and "_log" not in col:
            p10_key = col
        elif "_q50" in col and "_log" not in col:
            p50_key = col
        elif "_q90" in col and "_log" not in col:
            p90_key = col
    return p10_key, p50_key, p90_key


def _compute_quantile_metrics(results: list) -> Dict:
    pred_dfs = {}
    for r in results:
        pred_dfs[r.model_name] = r.prediction_df.set_index(["日期", "应用ID"])

    p10_key, p50_key, p90_key = _find_quantile_keys(list(pred_dfs.keys()))

    if not all([p10_key, p50_key, p90_key]):
        return {"error": "Missing quantile predictions", "available_keys": list(pred_dfs.keys())}

    p10 = pred_dfs[p10_key]["y_pred"]
    p50 = pred_dfs[p50_key]["y_pred"]
    p90 = pred_dfs[p90_key]["y_pred"]
    y_true = pred_dfs[p10_key]["y_true"]

    coverage = compute_coverage(y_true.values, p10.values, p90.values)
    interval_width_median = float(np.median(compute_interval_width(p10.values, p90.values, p50.values)))

    pinball_scores = {}
    for a, key in [(0.1, p10_key), (0.5, p50_key), (0.9, p90_key)]:
        pinball_scores[f"alpha_{a}"] = evaluate_pinball(
            y_true.values, pred_dfs[key]["y_pred"].values, a
        )

    p50_metrics = evaluate(y_true.values, p50.values)

    return {
        "coverage_p10_p90": round(coverage, 4),
        "ideal_coverage": 0.80,
        "coverage_error_pp": round((coverage - 0.80) * 100, 2),
        "interval_width_median": round(interval_width_median, 4),
        "pinball_loss": {k: round(v, 6) for k, v in pinball_scores.items()},
        "p50_mape_pct": round(p50_metrics["mape_pct"], 2),
    }


def _bucket_metrics(results: list) -> Dict:
    pred_dfs = {}
    for r in results:
        pred_dfs[r.model_name] = r.prediction_df

    merged = None
    for key, df in pred_dfs.items():
        sub = df[["日期", "应用ID", "y_pred"]].rename(columns={"y_pred": f"pred_{key}"})
        if merged is None:
            merged = df[["日期", "应用ID", "y_true"]].copy()
            merged = merged.drop_duplicates()
        merged = merged.merge(sub, on=["日期", "应用ID"], how="inner")

    merged = _assign_spend_bucket(merged)

    p10_key, p50_key, p90_key = _find_quantile_keys(["pred_{}".format(r.model_name) for r in results])

    if not all([p10_key, p50_key, p90_key]):
        return {}

    bucket_stats = {}
    for bucket in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        b = merged[merged["spend_bucket"] == bucket]
        if b.empty:
            continue
        y = b["y_true"].values
        lo = b[p10_key].values
        md = b[p50_key].values
        hi = b[p90_key].values
        bucket_stats[bucket] = {
            "n": int(len(b)),
            "coverage": round(compute_coverage(y, lo, hi), 4),
            "interval_width_median": round(float(np.median(compute_interval_width(lo, hi, md))), 4),
            "p50_mape_pct": round(evaluate(y, md)["mape_pct"], 2),
        }

    return bucket_stats


def main():
    parser = argparse.ArgumentParser(description="Exp 01: Quantile regression for T+1 spend")
    parser.add_argument("--input", default="daily_merged.csv")
    parser.add_argument("--eval-recent-days", type=int, default=20)
    parser.add_argument("--weight-exponent", type=float, default=0.35)
    args = parser.parse_args()

    validate_merged_daily_csv(Path(args.input))

    out_dir = Path("outputs/experiments") / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df, clean_meta = build_unified_daily(args.input, min_spend_train=2.0)

    quantile_variants = _make_quantile_variants(QUANTILE_ALPHAS)

    print("=== Training baseline + quantile variants ===")
    results = run(
        df,
        min_train_days=20,
        alpha=0.4,
        use_log_target=True,
        feature_set="base",
        eval_recent_days=args.eval_recent_days,
        tree_models=["XGBoost"],
        skip_baseline_two_stage=True,
        use_gpu=False,
        weight_exponent=args.weight_exponent,
        extra_model_variants=quantile_variants,
    )

    if not results:
        raise ValueError("No results produced. Check data and parameters.")

    # Save predictions
    for r in results:
        filename = f"predictions_{r.model_name.replace('/', '_')}.csv"
        r.prediction_df.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")

    # Baseline metrics
    baseline_rows = []
    for r in results:
        if "_q" not in r.model_name:
            baseline_rows.append({
                "model": r.model_name,
                **r.metrics,
                "samples": int(len(r.prediction_df)),
            })
    baseline_df = pd.DataFrame(baseline_rows).sort_values("mape")
    baseline_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    # Quantile metrics
    quantile_metrics = _compute_quantile_metrics(results)
    bucket_stats = _bucket_metrics(results)

    # Verdict
    coverage_err = abs(quantile_metrics.get("coverage_error_pp", 999))
    if coverage_err <= 5:
        verdict = "promising"
    elif coverage_err <= 15:
        verdict = "inconclusive"
    else:
        verdict = "regress"

    report = {
        "experiment": EXPERIMENT_NAME,
        "baseline": "spend_v12_unified (MSE XGBoost)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_recent_days": args.eval_recent_days,
            "weight_exponent": args.weight_exponent,
            "quantile_alphas": QUANTILE_ALPHAS,
        },
        "baseline_best_mape_pct": float(baseline_df.iloc[0]["mape_pct"]) if len(baseline_df) > 0 else None,
        "quantile_metrics": quantile_metrics,
        "by_spend_bucket": bucket_stats,
        "verdict": verdict,
    }

    report_path = out_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print(f"  Coverage (P10-P90): {quantile_metrics.get('coverage_p10_p90', 'N/A')}")
    print(f"  Interval width (median): {quantile_metrics.get('interval_width_median', 'N/A')}")
    if bucket_stats:
        print(f"  Q4_high coverage: {bucket_stats.get('Q4_high', {}).get('coverage', 'N/A')}")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
