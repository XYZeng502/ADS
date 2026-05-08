import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _load_baseline(path: Path) -> Tuple[Dict, Dict[Tuple[str, str], Dict[str, float]], Dict[str, Dict[str, float]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    by_flow = {(r["flow"], r["model"]): r for r in obj.get("by_flow", [])}
    overall = {r["model"]: r for r in obj.get("overall", [])}
    return obj, by_flow, overall


def _compare(
    candidate_overall: pd.DataFrame,
    candidate_flow: pd.DataFrame,
    baseline_overall: Dict[str, Dict[str, float]],
    baseline_flow: Dict[Tuple[str, str], Dict[str, float]],
    tolerance: Dict[str, float],
) -> Tuple[List[Dict], List[Dict]]:
    overall_diffs: List[Dict] = []
    for _, row in candidate_overall.iterrows():
        base = baseline_overall.get(row["model"])
        if not base:
            continue
        diff = {
            "scope": "overall",
            "model": row["model"],
            "samples_baseline": base["samples"],
            "samples_candidate": int(row.get("samples", 0)),
        }
        violated = False
        for metric, tol in (
            ("mae", tolerance.get("mae_relative_tolerance", 0.1)),
            ("rmse", tolerance.get("rmse_relative_tolerance", 0.1)),
            ("mape", tolerance.get("mape_relative_tolerance", 0.1)),
        ):
            base_v = float(base[metric])
            cand_v = float(row[metric])
            rel = (cand_v - base_v) / base_v if base_v > 1e-9 else 0.0
            diff[f"{metric}_baseline"] = base_v
            diff[f"{metric}_candidate"] = cand_v
            diff[f"{metric}_rel_change"] = rel
            if rel > tol:
                violated = True
        diff["violated"] = violated
        overall_diffs.append(diff)

    flow_diffs: List[Dict] = []
    for _, row in candidate_flow.iterrows():
        base = baseline_flow.get((row["flow"], row["model"]))
        if not base:
            continue
        diff = {
            "scope": "by_flow",
            "flow": row["flow"],
            "model": row["model"],
            "samples_baseline": base["samples"],
            "samples_candidate": int(row.get("samples", 0)),
        }
        violated = False
        for metric, tol in (
            ("mae", tolerance.get("mae_relative_tolerance", 0.1)),
            ("rmse", tolerance.get("rmse_relative_tolerance", 0.1)),
            ("mape", tolerance.get("mape_relative_tolerance", 0.1)),
        ):
            base_v = float(base[metric])
            cand_v = float(row[metric])
            rel = (cand_v - base_v) / base_v if base_v > 1e-9 else 0.0
            diff[f"{metric}_baseline"] = base_v
            diff[f"{metric}_candidate"] = cand_v
            diff[f"{metric}_rel_change"] = rel
            if rel > tol:
                violated = True
        diff["violated"] = violated
        flow_diffs.append(diff)

    return overall_diffs, flow_diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="对比当前指标与基线，超阈值视为业务调整带来劣化。")
    parser.add_argument(
        "--baseline",
        default="outputs/baseline_roi_d1_metrics.json",
        help="基线指标JSON路径",
    )
    parser.add_argument(
        "--candidate-dir",
        default="outputs/model_parallel_roi_d1_sanity_app_flow",
        help="包含 metrics_overall.csv / metrics_by_flow.csv 的目录",
    )
    parser.add_argument(
        "--out-report",
        default="outputs/metrics_drift_report.json",
        help="输出对比报告路径",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="任何一项相对劣化超阈值即非零退出（用于CI/护栏）",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    cand_dir = Path(args.candidate_dir)

    baseline_obj, base_flow, base_overall = _load_baseline(baseline_path)
    cand_overall = pd.read_csv(cand_dir / "metrics_overall.csv", encoding="utf-8-sig")
    cand_flow = pd.read_csv(cand_dir / "metrics_by_flow.csv", encoding="utf-8-sig")

    overall_diffs, flow_diffs = _compare(
        cand_overall,
        cand_flow,
        base_overall,
        base_flow,
        baseline_obj.get("tolerance", {}),
    )

    violated_count = sum(1 for r in overall_diffs + flow_diffs if r.get("violated"))
    report = {
        "baseline": str(baseline_path),
        "candidate_dir": str(cand_dir),
        "tolerance": baseline_obj.get("tolerance", {}),
        "violated_count": violated_count,
        "overall_diffs": overall_diffs,
        "flow_diffs": flow_diffs,
    }
    Path(args.out_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "violated_count": violated_count,
        "overall_violations": [d for d in overall_diffs if d["violated"]],
        "flow_violations": [d for d in flow_diffs if d["violated"]],
    }, ensure_ascii=False, indent=2))

    if args.fail_on_violation and violated_count > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
