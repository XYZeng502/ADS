#!/usr/bin/env python3
"""
应用级 T+1 消耗快速基线（与 run_parallel_models_spend_t1._ewma_spend 同构的逐日滚动 EWMA），
输出列与 Web / offline_backtest 读取的 spend 预测 CSV 一致（含 y_pred_fused）。

全量并行 spend 训练在大数据上极慢时，用本脚本先保证「推荐融合链路」与看板有可用预测。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="daily_merged.csv")
    parser.add_argument(
        "--output-dir",
        default="outputs/model_parallel_spend_t1_target_calendar_calibrated_app",
        help="写入 predictions_EWMA_quick.csv（统计基线，不冒充树模型；融合链路按 metrics 优先选真实模型）",
    )
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(args.alpha)

    raw = pd.read_csv(args.input, encoding="utf-8-sig")
    raw["日期"] = pd.to_datetime(raw["日期"], errors="coerce")
    raw = raw[raw["日期"].notna()]
    raw["消耗金额"] = pd.to_numeric(raw["消耗金额"], errors="coerce").fillna(0.0).clip(lower=0.0)
    g = (
        raw.groupby(["应用ID", "日期"], as_index=False)["消耗金额"]
        .sum()
        .sort_values(["应用ID", "日期"])
    )

    rows: list[dict] = []
    for app_id, sub in g.groupby("应用ID"):
        sub = sub.sort_values("日期")
        vals = sub["消耗金额"].to_numpy(dtype=float)
        ds = sub["日期"].to_numpy()
        if len(vals) < 2:
            continue
        for i in range(len(vals) - 1):
            s = float(vals[0])
            for j in range(1, i + 1):
                s = alpha * float(vals[j]) + (1.0 - alpha) * s
            d = pd.Timestamp(ds[i]).strftime("%Y-%m-%d")
            y_true = float(vals[i + 1])
            y_pred = float(s)
            rows.append(
                {
                    "日期": d,
                    "应用ID": str(app_id).strip(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_pred_fused": y_pred,
                    "model": "EWMA_quick",
                }
            )

    df = pd.DataFrame(rows)
    out_path = out_dir / "predictions_EWMA_quick.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    abs_err = (df["y_true"] - df["y_pred"]).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = abs_err / np.maximum(df["y_true"].to_numpy(), 1e-8)
    mape = float(np.nanmean(ape))
    mae = float(abs_err.mean())
    pd.DataFrame(
        [{"model": "EWMA_quick", "mae": mae, "mape": mape, "mape_pct": mape * 100, "samples": len(df)}]
    ).to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "target": "T+1 spend",
        "method": "app_level_walkforward_ewma",
        "alpha": alpha,
        "samples": int(len(df)),
        "output_csv": str(out_path),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
