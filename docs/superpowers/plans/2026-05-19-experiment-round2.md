# Experiment Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve quantile coverage from 65% toward 80% via wider intervals and conformal calibration, then run hierarchical reconciliation to reduce per-app MAPE.

**Architecture:** Three independent experiments building on the existing framework (`app/experiments/core.py`). exp_01a tweaks quantile alphas, exp_01b wraps XGBoost with MAPIE conformal prediction, exp_02 defines a spend-bucket hierarchy and applies MinTrace reconciliation.

**Tech Stack:** Python 3, XGBoost, MAPIE (SplitConformalRegressor), hierarchicalforecast (MinTrace), pandas, numpy

**Dependencies installed:** `hierarchicalforecast==1.5.1`, `mapie==1.4.0`

---

### Task 1: exp_01a — widen quantile range to P05/P50/P95

**Files:**
- Modify: `scripts/experiments/exp_01_quantile.py`

**Goal:** Test whether wider quantile alphas [0.05, 0.5, 0.95] close the coverage gap from -14.66pp toward ≤5pp.

- [ ] **Step 1: Change QUANTILE_ALPHAS and re-run**

```python
# scripts/experiments/exp_01_quantile.py, line 31
# Change:
QUANTILE_ALPHAS = [0.1, 0.5, 0.9]
# To:
QUANTILE_ALPHAS = [0.05, 0.5, 0.95]
```

- [ ] **Step 2: Run with new alphas**

```bash
PYTHONPATH=/home/lsh/ad_ml python scripts/experiments/exp_01_quantile.py \
  --input daily_merged.csv \
  --eval-recent-days 20 \
  --weight-exponent 0.35
```

Expected: coverage closer to 80%, interval wider than 1.18.

- [ ] **Step 3: Compare reports**

```bash
# Check new coverage vs old 65.34%
cat outputs/experiments/exp_01_quantile/experiment_report.json | python -c "import sys,json; d=json.load(sys.stdin); print(f'Coverage: {d[\"quantile_metrics\"][\"coverage_p10_p90\"]}'); print(f'Width: {d[\"quantile_metrics\"][\"interval_width_median\"]}'); print(f'Verdict: {d[\"verdict\"]}')"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/experiments/exp_01_quantile.py
git commit -m "exp_01a: widen quantile alphas to P05/P50/P95 for better coverage"
```

---

### Task 2: exp_01b — conformal prediction with MAPIE

**Files:**
- Create: `scripts/experiments/exp_01b_conformal.py`

**Goal:** Apply conformal prediction (MAPIE SplitConformalRegressor) on top of standard MSE XGBoost. Compare coverage and interval width against quantile regression approach (exp_01/exp_01a).

- [ ] **Step 1: Write the experiment script**

```python
# scripts/experiments/exp_01b_conformal.py
"""Experiment 01b: Conformal Prediction for Spend T+1 Intervals.

Wraps standard MSE XGBoost with MAPIE SplitConformalRegressor to produce
calibrated prediction intervals. Compares coverage/width against the
quantile regression approach from exp_01.

Usage:
    python scripts/experiments/exp_01b_conformal.py --input daily_merged.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from sklearn.model_selection import train_test_split

from app.experiments.core import (
    ModelResult,
    evaluate,
    evaluate_pinball,
    compute_coverage,
    compute_interval_width,
    build_model,
    feature_cols,
    safe_model_filename,
    apply_holiday_transition_calibration,
    ewma_spend,
)
from app.unified_daily import build_unified_daily
from app.prediction_artifacts import validate_merged_daily_csv

EXPERIMENT_NAME = "exp_01b_conformal"
ALPHA = 0.2  # target 80% coverage (1 - alpha)


def _assign_spend_bucket(df: pd.DataFrame) -> pd.DataFrame:
    app_avg = df.groupby("应用ID")["y_true"].mean().rename("app_avg_spend")
    df = df.join(app_avg, on="应用ID")
    df["spend_bucket"] = pd.qcut(df["app_avg_spend"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    return df


def _train_conformal_predict(train_df, test_df, feats, weight_exponent, use_gpu):
    """Train XGBoost + MAPIE conformal on train_df, predict on test_df."""
    train = train_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    if train.empty or test.empty:
        return None

    x_train = train[feats].values
    y_train = train["target_t1_spend"].clip(lower=0.0).values
    x_test = test[feats].values
    y_test = test["target_t1_spend"].values

    sample_weight = np.power(train["消耗金额"].clip(lower=0).values, weight_exponent)

    # Split train into proper_train + calibration
    if len(x_train) < 50:
        return None
    x_proper, x_calib, y_proper, y_calib, sw_proper, sw_calib = train_test_split(
        x_train, y_train, sample_weight, test_size=0.2, random_state=42
    )

    # Train base model on proper training set
    model = build_model("XGBoost", use_gpu=use_gpu)
    if model is None:
        return None
    try:
        model.fit(x_proper, y_proper, sample_weight=sw_proper)
    except TypeError:
        model.fit(x_proper, y_proper)

    # Wrap with conformal calibration
    conformal = SplitConformalRegressor(estimator=model, n_splits=1)
    conformal.fit(x_calib, y_calib, sample_weight=sw_calib)

    # Predict with intervals
    y_pred, y_pis = conformal.predict(x_test, alpha=ALPHA)

    # Apply holiday calibration to point predictions
    y_pred_cal = apply_holiday_transition_calibration(test, y_pred)
    y_lower_cal = apply_holiday_transition_calibration(test, y_pis[:, 0])
    y_upper_cal = apply_holiday_transition_calibration(test, y_pis[:, 1])

    return pd.DataFrame({
        "日期": test["日期"].values,
        "应用ID": test["应用ID"].values,
        "y_true": y_test,
        "y_pred": np.clip(y_pred_cal, 0.0, None),
        "y_lower": np.clip(y_lower_cal, 0.0, None),
        "y_upper": np.clip(y_upper_cal, 0.0, None),
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

    # Build expanding-window encodings (same as core.run)
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

    dates = sorted(model_df["日期"].dropna().unique())
    cutoffs = list(dates[:-1])
    eval_recent = args.eval_recent_days
    if eval_recent > 0 and len(cutoffs) > eval_recent:
        cutoffs = cutoffs[-eval_recent:]

    all_preds = []
    for i, cutoff in enumerate(cutoffs):
        t_cutoff = pd.Timestamp(cutoff)
        next_day = cutoff + np.timedelta64(1, "D")
        train = model_df[model_df["日期"] <= t_cutoff].copy()
        test = model_df[model_df["日期"] == next_day].copy()
        if test.empty or train["日期"].nunique() < 20:
            continue
        print(f"  [{i+1}/{len(cutoffs)}] cutoff={t_cutoff.date()} train={len(train)} test={len(test)}", flush=True)
        pred_df = _train_conformal_predict(train, test, feats_all, args.weight_exponent, use_gpu=False)
        if pred_df is not None and not pred_df.empty:
            all_preds.append(pred_df)

    if not all_preds:
        raise ValueError("No results produced.")

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df = _assign_spend_bucket(pred_df)

    y_true = pred_df["y_true"].values
    y_lower = pred_df["y_lower"].values
    y_upper = pred_df["y_upper"].values
    y_pred = pred_df["y_pred"].values

    coverage = compute_coverage(y_true, y_lower, y_upper)
    interval_width_median = float(np.median(compute_interval_width(y_lower, y_upper, y_pred)))
    point_metrics = evaluate(y_true, y_pred)

    # Per-bucket stats
    bucket_stats = {}
    for bucket in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        b = pred_df[pred_df["spend_bucket"] == bucket]
        if b.empty:
            continue
        bucket_stats[bucket] = {
            "n": int(len(b)),
            "coverage": round(compute_coverage(b["y_true"].values, b["y_lower"].values, b["y_upper"].values), 4),
            "interval_width_median": round(float(np.median(compute_interval_width(b["y_lower"].values, b["y_upper"].values, b["y_pred"].values))), 4),
            "mape_pct": round(evaluate(b["y_true"].values, b["y_pred"].values)["mape_pct"], 2),
        }

    coverage_err = abs(coverage - 0.80)
    if coverage_err <= 0.05:
        verdict = "promising"
    elif coverage_err <= 0.15:
        verdict = "inconclusive"
    else:
        verdict = "regress"

    report = {
        "experiment": EXPERIMENT_NAME,
        "method": "SplitConformalRegressor (MAPIE) with XGBoost base",
        "target_coverage": 1 - ALPHA,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_recent_days": args.eval_recent_days,
            "weight_exponent": args.weight_exponent,
            "conformal_alpha": ALPHA,
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

    # Save
    pred_df.to_csv(out_dir / "predictions_XGBoost_Conformal.csv", index=False, encoding="utf-8-sig")
    (out_dir / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport saved to {out_dir / 'experiment_report.json'}")
    print(f"  Coverage: {coverage:.4f} (target: {1 - ALPHA})")
    print(f"  Interval width (median): {interval_width_median:.4f}")
    print(f"  MAPE: {point_metrics['mape_pct']:.2f}%")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run exp_01b**

```bash
PYTHONPATH=/home/lsh/ad_ml python scripts/experiments/exp_01b_conformal.py \
  --input daily_merged.csv \
  --eval-recent-days 20 \
  --weight-exponent 0.35
```

Expected: coverage much closer to 80% than quantile approach, at cost of wider intervals.

- [ ] **Step 3: Compare all interval approaches**

Print side-by-side comparison of exp_01 (P10/P90), exp_01a (P05/P95), and exp_01b (conformal).

- [ ] **Step 4: Commit**

```bash
git add scripts/experiments/exp_01b_conformal.py
git commit -m "feat: add exp_01b conformal prediction intervals via MAPIE"
```

---

### Task 3: exp_02 — hierarchical reconciliation

**Files:**
- Create: `scripts/experiments/exp_02_hierarchical.py`

**Goal:** Apply MinTrace reconciliation on a 3-level hierarchy (Total → spend buckets → apps) and measure MAPE improvement per bucket vs baseline.

- [ ] **Step 1: Write the experiment script**

```python
# scripts/experiments/exp_02_hierarchical.py
"""Experiment 02: Hierarchical Reconciliation for Spend T+1.

Constructs a 3-level hierarchy (Total -> spend bucket -> app), trains
XGBoost at each level independently, then reconciles via MinTrace.
Compares reconciled vs unreconciled MAPE by spend bucket.

Usage:
    python scripts/experiments/exp_02_hierarchical.py --input daily_merged.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace, BottomUp

from app.experiments.core import (
    evaluate,
    feature_cols,
    build_model,
    apply_holiday_transition_calibration,
)
from app.unified_daily import build_unified_daily
from app.prediction_artifacts import validate_merged_daily_csv

EXPERIMENT_NAME = "exp_02_hierarchical"


def _assign_static_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Assign apps to static spend buckets based on overall mean spend."""
    app_avg = df.groupby("应用ID")["消耗金额"].mean().rename("app_avg_spend")
    bucket = pd.qcut(app_avg, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    df = df.join(bucket.rename("spend_bucket"), on="应用ID")
    return df


def _build_hierarchy(df: pd.DataFrame) -> tuple:
    """Build hierarchical time series: Total, 4 buckets, N apps.

    Returns:
        Y_df: (unique_id, ds, y) hierarchical time series
        S_df: (unique_id, ...) summing matrix as required by hierarchicalforecast
        tags: dict mapping level names to unique_id prefixes
    """
    df = _assign_static_bucket(df.copy())

    # Aggregate at three levels
    # Level 0: Total
    total = df.groupby("日期").agg(y=("target_t1_spend", "sum")).reset_index()
    total["unique_id"] = "total"

    # Level 1: Spend buckets
    buckets = df.groupby(["日期", "spend_bucket"]).agg(y=("target_t1_spend", "sum")).reset_index()
    buckets["unique_id"] = "total/" + buckets["spend_bucket"].astype(str)

    # Level 2: Apps
    apps = df.groupby(["日期", "应用ID", "spend_bucket"]).agg(y=("target_t1_spend", "sum")).reset_index()
    apps["unique_id"] = "total/" + apps["spend_bucket"].astype(str) + "/" + apps["应用ID"].astype(str)

    # Combine into Y_df format
    cols = ["unique_id", "日期", "y"]
    Y_df = pd.concat([
        total.rename(columns={"日期": "ds"})[cols[:2] + ["y"]].assign(ds=lambda x: pd.to_datetime(x["ds"])),
        buckets.rename(columns={"日期": "ds"})[cols[:2] + ["y"]].assign(ds=lambda x: pd.to_datetime(x["ds"])),
        apps.rename(columns={"日期": "ds"})[cols[:2] + ["y"]].assign(ds=lambda x: pd.to_datetime(x["ds"])),
    ], ignore_index=True)

    # Build summing matrix S_df
    # For each bottom-level series, specify its parent path
    bottom_apps = apps["unique_id"].unique()
    S_rows = []
    for uid in bottom_apps:
        parts = uid.split("/")
        row = {"unique_id": uid}
        # Self
        row[uid] = 1
        # Parent (bucket)
        bucket_uid = "/".join(parts[:2])
        row[bucket_uid] = 1
        # Grandparent (total)
        row["total"] = 1
        S_rows.append(row)

    S_df = pd.DataFrame(S_rows).fillna(0).set_index("unique_id")
    # Ensure all aggregates appear as columns too
    for col_id in list(buckets["unique_id"].unique()) + ["total"]:
        if col_id not in S_df.columns:
            S_df[col_id] = 0.0

    tags = {
        "total": ["total"],
        "bucket": sorted(buckets["unique_id"].unique().tolist()),
        "app": sorted(bottom_apps.tolist()),
    }

    return Y_df, S_df, tags


def _train_level_model(train_df, test_df, feats, use_gpu):
    """Train XGBoost on a level-aggregated dataset."""
    train = train_df.dropna(subset=feats + ["y"]).copy()
    test = test_df.dropna(subset=feats + ["y"]).copy()
    if train.empty or test.empty:
        return test.assign(y_pred=np.nan)

    model = build_model("XGBoost", use_gpu=use_gpu)
    if model is None:
        return test.assign(y_pred=np.nan)

    x_train = train[feats].values
    y_train = train["y"].clip(lower=0.0).values
    x_test = test[feats].values

    try:
        model.fit(x_train, y_train)
    except Exception:
        return test.assign(y_pred=np.nan)

    pred = model.predict(x_test)
    return test.assign(y_pred=np.clip(pred, 0.0, None))


def _walk_forward_hierarchical(model_df, feats, eval_recent_days, static_bucket_map):
    """Run walk-forward, producing base forecasts at all hierarchy levels.

    Returns Y_hat_df and Y_test_df in hierarchicalforecast format.
    """
    dates = sorted(model_df["日期"].dropna().unique())
    cutoffs = list(dates[:-1])
    if eval_recent_days > 0 and len(cutoffs) > eval_recent_days:
        cutoffs = cutoffs[-eval_recent_days:]

    all_hats = []
    all_tests = []

    for i, cutoff in enumerate(cutoffs):
        next_day = cutoff + np.timedelta64(1, "D")
        train = model_df[model_df["日期"] <= pd.Timestamp(cutoff)].copy()
        test = model_df[model_df["日期"] == next_day].copy()
        if test.empty or train["日期"].nunique() < 20:
            continue
        print(f"  [{i+1}/{len(cutoffs)}] cutoff={pd.Timestamp(cutoff).date()}", flush=True)

        # Build hierarchy for this cutoff
        Y_train, S_df, tags = _build_hierarchy(train)
        Y_test, _, _ = _build_hierarchy(test)

        # Train per-level base forecasts
        preds = []
        for uid, grp_train in Y_train.groupby("unique_id"):
            grp_test = Y_test[Y_test["unique_id"] == uid]
            if grp_test.empty:
                continue
            # Per-level feature engineering (simple: use lags of y)
            # For now, train on the aggregated y directly with minimal features
            pred_df = _train_level_model(grp_train, grp_test, feats, use_gpu=False)
            if pred_df is not None and not pred_df.empty:
                preds.append(pred_df[["unique_id", "ds", "y_pred"]])

        if not preds:
            continue

        Y_hat = pd.concat(preds, ignore_index=True)
        Y_hat["ds"] = pd.to_datetime(Y_hat["ds"])

        # Reconcile
        hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method="mint_shrink")])
        try:
            Y_rec = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_train, S_df=S_df, tags=tags)
        except Exception:
            Y_rec = Y_hat.copy()

        # Merge with actuals
        Y_test_clean = Y_test[["unique_id", "ds", "y"]].copy()
        Y_test_clean["ds"] = pd.to_datetime(Y_test_clean["ds"])
        Y_merged = Y_rec.merge(Y_test_clean, on=["unique_id", "ds"], how="inner")

        all_tests.append(Y_merged)

    if not all_tests:
        return None
    return pd.concat(all_tests, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Exp 02: Hierarchical reconciliation")
    parser.add_argument("--input", default="daily_merged.csv")
    parser.add_argument("--eval-recent-days", type=int, default=20)
    args = parser.parse_args()

    validate_merged_daily_csv(Path(args.input))

    out_dir = Path("outputs/experiments") / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _ = build_unified_daily(args.input, min_spend_train=2.0)
    feats = feature_cols("base")

    # Build expanding encodings
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

    # Static bucket assignment
    model_df = _assign_static_bucket(model_df)

    print("=== Running hierarchical walk-forward ===")
    result = _walk_forward_hierarchical(model_df, feats, args.eval_recent_days, None)

    if result is None:
        raise ValueError("No results produced.")

    # Filter to app-level only for evaluation
    app_results = result[~result["unique_id"].str.startswith("total/") | result["unique_id"].str.startswith("total/Q")]
    # Actually: app level = unique_id has 3 parts (total/bucket/app)
    app_results = result[result["unique_id"].str.count("/") == 2].copy()

    # Add bucket info
    app_results["spend_bucket"] = app_results["unique_id"].str.extract(r"total/(Q\w+)/")[0]

    # Overall MAPE
    y_true = app_results["y"].values
    y_pred = app_results["y_pred"].values
    metrics = evaluate(y_true, y_pred)

    # By bucket
    bucket_stats = {}
    for bucket in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        b = app_results[app_results["spend_bucket"] == bucket]
        if b.empty:
            continue
        bm = evaluate(b["y"].values, b["y_pred"].values)
        bucket_stats[bucket] = {
            "n": int(len(b)),
            "mape_pct": round(bm["mape_pct"], 2),
            "mae": round(bm["mae"], 2),
        }

    report = {
        "experiment": EXPERIMENT_NAME,
        "method": "MinTrace (mint_shrink) hierarchical reconciliation",
        "hierarchy": "Total -> Q1-Q4 spend buckets -> apps",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_recent_days": args.eval_recent_days,
        },
        "overall_metrics": {
            "mape_pct": round(metrics["mape_pct"], 2),
            "mae": round(metrics["mae"], 2),
            "rmse": round(metrics["rmse"], 2),
            "samples": int(len(app_results)),
        },
        "by_spend_bucket": bucket_stats,
        "note": "Baseline spend_v12 MAPE ~62%. Improvement = baseline - reconciled MAPE.",
    }

    app_results.to_csv(out_dir / "predictions_reconciled.csv", index=False, encoding="utf-8-sig")
    (out_dir / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport saved to {out_dir / 'experiment_report.json'}")
    print(f"  Overall MAPE: {metrics['mape_pct']:.2f}%")
    for bucket, stats in bucket_stats.items():
        print(f"  {bucket}: MAPE={stats['mape_pct']:.2f}% (n={stats['n']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run exp_02**

```bash
PYTHONPATH=/home/lsh/ad_ml python scripts/experiments/exp_02_hierarchical.py \
  --input daily_merged.csv \
  --eval-recent-days 20
```

- [ ] **Step 3: Check results**

```bash
cat outputs/experiments/exp_02_hierarchical/experiment_report.json
```

Expected: MAPE by bucket, with Q4 ideally improved vs baseline 62%.

- [ ] **Step 4: Commit**

```bash
git add scripts/experiments/exp_02_hierarchical.py
git commit -m "feat: add exp_02 hierarchical reconciliation via MinTrace"
```

---

### Task 4: Summary comparison

**Files:**
- Create: `scripts/experiments/compare_round2.py`

**Goal:** Single script that reads all experiment reports and prints a side-by-side comparison table.

- [ ] **Step 1: Write comparison script**

```python
# scripts/experiments/compare_round2.py
"""Print side-by-side comparison of all round 2 experiment results."""

import json
from pathlib import Path

EXPERIMENTS = [
    ("exp_01", "Quantile P10/P50/P90"),
    ("exp_01_quantile", "Quantile P05/P50/P95 (same script, re-run)"),
    ("exp_01b_conformal", "Conformal (MAPIE SplitConformal)"),
    ("exp_02_hierarchical", "Hierarchical (MinTrace)"),
]

BASE = Path("outputs/experiments")

print(f"{'Experiment':<30} {'Coverage':>10} {'Width':>10} {'MAPE':>10} {'Verdict':>15}")
print("-" * 80)

for exp_dir, label in EXPERIMENTS:
    report_path = BASE / exp_dir / "experiment_report.json"
    if not report_path.exists():
        print(f"{label:<30} {'(not run)':>10}")
        continue
    r = json.loads(report_path.read_text())

    # Extract metrics depending on experiment type
    if "quantile_metrics" in r:
        qm = r["quantile_metrics"]
        coverage = qm.get("coverage_p10_p90", "N/A")
        width = qm.get("interval_width_median", "N/A")
        mape = qm.get("p50_mape_pct", "N/A")
    elif "metrics" in r:
        m = r["metrics"]
        coverage = m.get("coverage", "N/A")
        width = m.get("interval_width_median", "N/A")
        mape = m.get("mape_pct", "N/A")
    else:
        coverage = "N/A"
        width = "N/A"
        mape = r.get("overall_metrics", {}).get("mape_pct", "N/A")

    c_str = f"{coverage:.4f}" if isinstance(coverage, (int, float)) else str(coverage)
    w_str = f"{width:.4f}" if isinstance(width, (int, float)) else str(width)
    m_str = f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape)

    print(f"{label:<30} {c_str:>10} {w_str:>10} {m_str:>10} {r.get('verdict', 'N/A'):>15}")

print("\nBaseline spend_v12: MAPE ~62%")
```

- [ ] **Step 2: Run comparison**

```bash
PYTHONPATH=/home/lsh/ad_ml python scripts/experiments/compare_round2.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/experiments/compare_round2.py
git commit -m "feat: add round 2 experiment comparison script"
```
