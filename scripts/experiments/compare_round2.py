# scripts/experiments/compare_round2.py
"""Print side-by-side comparison of all round 2 experiment results."""

import json
from pathlib import Path

EXPERIMENTS = [
    ("exp_01_quantile", "Quantile P05/P50/P95"),
    ("exp_01b_conformal", "Conformal (MAPIE SplitConformal)"),
    ("exp_02_hierarchical", "Hierarchical (MinTrace)"),
]

BASE = Path("outputs/experiments")

print(f"{'Experiment':<35} {'Coverage':>10} {'Width':>10} {'MAPE':>10} {'Verdict':>15}")
print("-" * 85)

for exp_dir, label in EXPERIMENTS:
    report_path = BASE / exp_dir / "experiment_report.json"
    if not report_path.exists():
        print(f"{label:<35} {'(not run)':>10}")
        continue
    r = json.loads(report_path.read_text())

    if "quantile_metrics" in r:
        qm = r["quantile_metrics"]
        coverage = qm.get("coverage", "N/A")
        width = qm.get("interval_width_median", "N/A")
        mape = qm.get("median_mape_pct", "N/A")
    elif "metrics" in r:
        m = r["metrics"]
        coverage = m.get("coverage", "N/A")
        width = m.get("interval_width_median", "N/A")
        mape = m.get("mape_pct", "N/A")
    elif "overall_metrics" in r:
        coverage = "N/A"
        width = "N/A"
        om = r["overall_metrics"]
        mape = f"{om.get('unreconciled_mape_pct', 'N/A')}→{om.get('reconciled_mape_pct', 'N/A')}"
    else:
        coverage, width, mape = "N/A", "N/A", "N/A"

    c_str = f"{coverage:.4f}" if isinstance(coverage, (int, float)) else str(coverage)
    w_str = f"{width:.4f}" if isinstance(width, (int, float)) else str(width)
    m_str = f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape)

    print(f"{label:<35} {c_str:>10} {w_str:>10} {m_str:>10} {r.get('verdict', 'N/A'):>15}")

print()
print("Baseline spend_v12_unified: XGBoost_log MAPE 58.62%, no intervals")
print()

# Per-bucket comparison for interval approaches
print("=" * 85)
print("Per-Bucket Coverage Comparison (P05/P50/P95 vs Conformal)")
print("-" * 85)
print(f"{'Bucket':<12} {'Quantile Cov':>14} {'Quantile W':>12} {'Conformal Cov':>15} {'Conformal W':>14}")
print("-" * 85)

quantile_buckets = {}
conformal_buckets = {}

for exp_dir, _ in EXPERIMENTS[:2]:
    report_path = BASE / exp_dir / "experiment_report.json"
    if report_path.exists():
        r = json.loads(report_path.read_text())
        buckets = r.get("by_spend_bucket", {})
        if exp_dir == "exp_01_quantile":
            quantile_buckets = buckets
        else:
            conformal_buckets = buckets

for bucket in ["Q1_low", "Q2", "Q3", "Q4_high"]:
    qb = quantile_buckets.get(bucket, {})
    cb = conformal_buckets.get(bucket, {})
    print(
        f"{bucket:<12} "
        f"{qb.get('coverage', 'N/A'):>14} "
        f"{qb.get('interval_width_median', 'N/A'):>12} "
        f"{cb.get('coverage', 'N/A'):>15} "
        f"{cb.get('interval_width_median', 'N/A'):>14}"
    )
