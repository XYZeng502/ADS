"""Experiment 02: Hierarchical Reconciliation for Spend T+1.

Tests whether MinTrace reconciliation improves per-app MAPE by reconciling
INCOHERENT base forecasts:
  - App level: XGBoost per-app model (via tree_predict)
  - Bucket/Total level: N-day trailing average of aggregated spend

Incoherence is necessary: forecasts that are already coherent (bottom-up sum)
are unchanged by reconciliation regardless of method (OLS, mint_shrink, etc.).

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
from hierarchicalforecast.methods import MinTrace

from app.experiments.core import evaluate, feature_cols, tree_predict
from app.unified_daily import build_unified_daily
from app.prediction_artifacts import validate_merged_daily_csv

EXPERIMENT_NAME = "exp_02_hierarchical"


def _assign_static_bucket(df: pd.DataFrame) -> pd.DataFrame:
    app_avg = df.groupby("应用ID")["消耗金额"].mean().rename("app_avg_spend")
    bucket = pd.qcut(app_avg, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    return df.join(bucket.rename("spend_bucket"), on="应用ID")


def _build_S_df(bottom_apps: pd.DataFrame) -> pd.DataFrame:
    """Summing matrix: rows = all hierarchy nodes, cols = bottom nodes (apps)."""
    bottom = bottom_apps[["应用ID", "spend_bucket"]].drop_duplicates()
    bottom_uids = sorted(f"total/{b}/{a}" for _, (a, b) in bottom.iterrows())
    buckets = sorted(bottom["spend_bucket"].unique())
    all_nodes = ["total"] + [f"total/{b}" for b in buckets] + bottom_uids
    n_b = len(bottom_uids)

    S_arr = np.zeros((len(all_nodes), n_b), dtype=np.float64)
    for j, buid in enumerate(bottom_uids):
        bu = "/".join(buid.split("/")[:2])
        S_arr[0, j] = 1.0
        S_arr[all_nodes.index(bu), j] = 1.0
        S_arr[all_nodes.index(buid), j] = 1.0

    S_df = pd.DataFrame(S_arr, columns=bottom_uids)
    S_df.insert(0, "unique_id", all_nodes)
    return S_df


def _trailing_avg(series: pd.Series, window: int) -> float:
    """One-step-ahead forecast: mean of last `window` observations."""
    vals = series.dropna().values
    if len(vals) < 1:
        return 0.0
    if len(vals) < window:
        return float(vals[-1])
    return float(vals[-window:].mean())


def main():
    parser = argparse.ArgumentParser(description="Exp 02: Hierarchical reconciliation")
    parser.add_argument("--input", default="daily_merged.csv")
    parser.add_argument("--eval-recent-days", type=int, default=20)
    parser.add_argument("--weight-exponent", type=float, default=0.35)
    parser.add_argument("--trailing-window", type=int, default=7)
    args = parser.parse_args()

    validate_merged_daily_csv(Path(args.input))
    out_dir = Path("outputs/experiments") / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _ = build_unified_daily(args.input, min_spend_train=2.0)
    feats = feature_cols("base")

    # Expanding-window encodings (same as core.run)
    model_df = df.sort_values(["应用ID", "日期"]).reset_index(drop=True).copy()
    grp_spend = model_df.groupby("应用ID")["target_t1_spend"]
    model_df["app_avg_spend"] = (
        grp_spend.cumsum().sub(model_df["target_t1_spend"])
        / grp_spend.cumcount().clip(lower=1)
    ).fillna(model_df["target_t1_spend"].median())
    if "roi_d1" in model_df.columns:
        grp_roi = model_df.groupby("应用ID")["roi_d1"]
        model_df["app_avg_roi_d1"] = (
            grp_roi.cumsum().sub(model_df["roi_d1"])
            / grp_spend.cumcount().clip(lower=1)
        ).fillna(model_df["roi_d1"].median())
        feats_all = feats + ["app_avg_spend", "app_avg_roi_d1"]
    else:
        feats_all = feats + ["app_avg_spend"]

    model_df = _assign_static_bucket(model_df)

    dates = sorted(model_df["日期"].dropna().unique())
    all_dates = [pd.Timestamp(d) for d in dates]
    cutoffs = list(dates[:-1])
    if args.eval_recent_days > 0 and len(cutoffs) > args.eval_recent_days:
        cutoffs = cutoffs[-args.eval_recent_days:]

    all_comparisons = []
    cutoff_results = 0

    for i, cutoff in enumerate(cutoffs):
        t_cutoff = pd.Timestamp(cutoff)
        next_day = pd.Timestamp(cutoff + np.timedelta64(1, "D"))

        train = model_df[model_df["日期"] <= t_cutoff].copy()
        test = model_df[model_df["日期"] == next_day].copy()
        if test.empty or train["日期"].nunique() < 20:
            continue

        print(f"  [{i+1}/{len(cutoffs)}] cutoff={t_cutoff.date()} "
              f"train_days={train['日期'].nunique()} test_apps={test['应用ID'].nunique()}", flush=True)

        # ---------------------------------------------------------------
        # 1. Per-app XGBoost out-of-sample predictions
        # ---------------------------------------------------------------
        pred_df = tree_predict("XGBoost", train, test, feats_all,
                               use_log_target=True, use_gpu=False,
                               weight_exponent=args.weight_exponent)
        if pred_df.empty:
            continue

        # Filter test to only apps with predictions
        pred_map = pred_df.set_index("应用ID")["y_pred"]
        test = test[test["应用ID"].isin(pred_map.index)].copy()
        if test.empty:
            continue

        # ---------------------------------------------------------------
        # 2. Trailing-average forecasts for total & bucket levels
        #    (independent of XGBoost — creates incoherence)
        # ---------------------------------------------------------------
        train_total_ts = train.groupby("日期")["target_t1_spend"].sum()
        total_fcst = _trailing_avg(train_total_ts, args.trailing_window)

        bucket_fcsts: Dict[str, float] = {}
        for bucket in test["spend_bucket"].unique():
            b_ts = train[train["spend_bucket"] == bucket].groupby("日期")["target_t1_spend"].sum()
            bucket_fcsts[bucket] = _trailing_avg(b_ts, args.trailing_window)

        # ---------------------------------------------------------------
        # 3. Build Y_hat_df (base forecasts at all 3 levels)
        # ---------------------------------------------------------------
        hat_recs = [{"unique_id": "total", "ds": next_day, "y_pred": total_fcst}]

        for bucket, bf in bucket_fcsts.items():
            hat_recs.append({"unique_id": f"total/{bucket}", "ds": next_day, "y_pred": bf})

        for aid in test["应用ID"].unique():
            app_bucket = test[test["应用ID"] == aid]["spend_bucket"].iloc[0]
            hat_recs.append({
                "unique_id": f"total/{app_bucket}/{aid}",
                "ds": next_day,
                "y_pred": pred_map.get(aid, 0.0),
            })

        Y_hat_df = pd.DataFrame(hat_recs)
        if len(Y_hat_df) < 3:
            continue

        # ---------------------------------------------------------------
        # 4. Build S_df and tags
        # ---------------------------------------------------------------
        S_df = _build_S_df(test)
        node_ids = S_df["unique_id"].tolist()
        tags = {
            "total": ["total"],
            "bucket": [u for u in node_ids if u.count("/") == 1],
            "app": [u for u in node_ids if u.count("/") == 2],
        }

        # ---------------------------------------------------------------
        # 5. Reconcile via OLS (does not require balanced in-sample panel)
        # ---------------------------------------------------------------
        try:
            hrec = HierarchicalReconciliation(
                reconcilers=[MinTrace(method="ols")]
            )
            rec = hrec.reconcile(
                Y_hat_df=Y_hat_df,
                Y_df=None,
                S_df=S_df,
                tags=tags,
            )
        except Exception as e:
            print(f"    Reconciliation failed: {e}")
            continue

        # ---------------------------------------------------------------
        # 6. Extract reconciled app-level predictions
        # ---------------------------------------------------------------
        app_ids = tags["app"]
        rec_apps = rec[rec["unique_id"].isin(app_ids)].copy()
        if rec_apps.empty:
            continue

        skip = {"unique_id", "ds", "y_pred"}
        pc = [c for c in rec_apps.columns if c not in skip]
        if not pc:
            continue

        rec_apps["应用ID"] = rec_apps["unique_id"].str.extract(r"/(\d+)$")[0].astype(int)
        rec_apps["y_pred_reconciled"] = rec_apps[pc[0]].fillna(0.0)

        # ---------------------------------------------------------------
        # 7. Build comparison
        # ---------------------------------------------------------------
        common = sorted(set(test["应用ID"].unique()) & set(rec_apps["应用ID"].unique()))
        rows = []
        for aid in common:
            amask = test["应用ID"] == aid
            rows.append({
                "日期": next_day.date(),
                "应用ID": aid,
                "spend_bucket": test.loc[amask, "spend_bucket"].iloc[0],
                "y_true": test.loc[amask, "target_t1_spend"].iloc[0],
                "y_pred": pred_map.get(aid, np.nan),
                "y_pred_reconciled": rec_apps[rec_apps["应用ID"] == aid]["y_pred_reconciled"].iloc[0],
            })
        comp = pd.DataFrame(rows).dropna()
        if not comp.empty:
            all_comparisons.append(comp)
            cutoff_results += 1

    # -------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------
    if not all_comparisons:
        raise ValueError("No results produced. Check data and parameters.")

    comp_df = pd.concat(all_comparisons, ignore_index=True)

    y_true = comp_df["y_true"].values
    y_unrec = comp_df["y_pred"].values
    y_rec = comp_df["y_pred_reconciled"].values

    unrec_metrics = evaluate(y_true, y_unrec)
    rec_metrics = evaluate(y_true, y_rec)

    bucket_stats: Dict[str, dict] = {}
    for bucket in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        b = comp_df[comp_df["spend_bucket"] == bucket]
        if b.empty:
            continue
        u = evaluate(b["y_true"].values, b["y_pred"].values)
        r = evaluate(b["y_true"].values, b["y_pred_reconciled"].values)
        bucket_stats[bucket] = {
            "n": int(len(b)),
            "unreconciled_mape_pct": round(u["mape_pct"], 2),
            "reconciled_mape_pct": round(r["mape_pct"], 2),
            "mape_improvement_pp": round(u["mape_pct"] - r["mape_pct"], 2),
        }

    impr = unrec_metrics["mape_pct"] - rec_metrics["mape_pct"]
    verdict = "promising" if impr > 2 else ("inconclusive" if impr > -2 else "regress")

    report = {
        "experiment": EXPERIMENT_NAME,
        "method": (
            "MinTrace(ols) on incoherent forecasts: "
            "XGBoost(apps) + trailing_avg(total/bucket)"
        ),
        "hierarchy": "Total -> Q1-Q4 spend buckets -> apps",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_recent_days": args.eval_recent_days,
            "weight_exponent": args.weight_exponent,
            "trailing_window": args.trailing_window,
        },
        "overall_metrics": {
            "unreconciled_mape_pct": round(unrec_metrics["mape_pct"], 2),
            "reconciled_mape_pct": round(rec_metrics["mape_pct"], 2),
            "mape_improvement_pp": round(impr, 2),
            "samples": int(len(comp_df)),
        },
        "by_spend_bucket": bucket_stats,
        "verdict": verdict,
    }

    comp_df.to_csv(out_dir / "predictions_comparison.csv", index=False, encoding="utf-8-sig")
    (out_dir / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport saved to {out_dir / 'experiment_report.json'}")
    print(f"  Cutoffs: {cutoff_results}/{len(cutoffs)}")
    print(f"  Unreconciled MAPE: {unrec_metrics['mape_pct']:.2f}%  (XGBoost apps)")
    print(f"  Reconciled MAPE:   {rec_metrics['mape_pct']:.2f}%  (MinTrace OLS)")
    print(f"  Improvement:       {impr:+.2f}pp")
    for b, s in bucket_stats.items():
        print(f"  {b}: {s['unreconciled_mape_pct']:.2f}% -> "
              f"{s['reconciled_mape_pct']:.2f}% ({s['mape_improvement_pp']:+.2f}pp)")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
