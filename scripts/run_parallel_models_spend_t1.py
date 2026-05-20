import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.experiments.core import (
    ModelResult,
    evaluate,
    ewma_spend,
    feature_cols,
    safe_model_filename,
    run as _run,
)
from app.prediction_artifacts import validate_merged_daily_csv
from app.unified_daily import build_unified_daily


def _extract_app_id(entity_id: str) -> str:
    s = str(entity_id)
    return s.split("|")[0] if "|" in s else s


def _evaluate_with_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    base = evaluate(y_true, y_pred)
    wmape = float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-8))
    base["wmape"] = wmape
    base["wmape_pct"] = wmape * 100.0
    return base


def _build_reconciled_prediction(
    split_pred_df: pd.DataFrame,
    app_pred_df: pd.DataFrame,
    min_entities: int,
    split_weight: float,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    split = split_pred_df.copy()
    app = app_pred_df.copy()
    split["日期"] = pd.to_datetime(split["日期"], errors="coerce")
    app["日期"] = pd.to_datetime(app["日期"], errors="coerce")
    split["应用ID"] = split["应用ID"].astype(str)
    app["应用ID"] = app["应用ID"].astype(str)

    split["app_id"] = split["应用ID"].map(_extract_app_id)
    split_agg = (
        split.groupby(["日期", "app_id"], as_index=False)
        .agg(split_y_true=("y_true", "sum"), split_y_pred=("y_pred", "sum"), split_entity_cnt=("应用ID", "nunique"))
        .rename(columns={"app_id": "应用ID"})
    )
    pair = app.merge(split_agg, on=["日期", "应用ID"], how="inner")
    if pair.empty:
        raise ValueError("融合失败：app 基线预测与分流预测没有可对齐的 (日期, 应用ID) 交集。")

    w = max(0.0, min(float(split_weight), 1.0))
    use_split = (pair["split_entity_cnt"].values >= int(min_entities)).astype(float)
    app_pred = pair["y_pred"].values
    split_pred = pair["split_y_pred"].values
    fused_pred = use_split * (w * split_pred + (1.0 - w) * app_pred) + (1.0 - use_split) * app_pred

    out = pair[["日期", "应用ID"]].copy()
    out["y_true"] = pair["y_true"].values
    out["y_pred_app"] = app_pred
    out["y_pred_split_agg"] = split_pred
    out["split_entity_cnt"] = pair["split_entity_cnt"].values
    out["y_pred_fused"] = fused_pred

    y_true = out["y_true"].values
    app_metrics = _evaluate_with_wmape(y_true, out["y_pred_app"].values)
    split_metrics = _evaluate_with_wmape(y_true, out["y_pred_split_agg"].values)
    fused_metrics = _evaluate_with_wmape(y_true, out["y_pred_fused"].values)
    report = {
        "samples": int(len(out)),
        "rule": f"if split_entity_cnt>={int(min_entities)}: fused={w:.2f}*split + {1.0-w:.2f}*app else app",
        "app_metrics": app_metrics,
        "split_agg_metrics": split_metrics,
        "fused_metrics": fused_metrics,
    }
    return out, report


def _build_online_reconciled_prediction(
    split_pred_df: pd.DataFrame,
    app_pred_df: pd.DataFrame,
    *,
    min_history_days: int,
    min_entities_grid: List[int],
    split_weight_grid: List[float],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    split = split_pred_df.copy()
    app = app_pred_df.copy()
    split["日期"] = pd.to_datetime(split["日期"], errors="coerce")
    app["日期"] = pd.to_datetime(app["日期"], errors="coerce")
    split["应用ID"] = split["应用ID"].astype(str)
    app["应用ID"] = app["应用ID"].astype(str)
    split["app_id"] = split["应用ID"].map(_extract_app_id)

    split_agg = (
        split.groupby(["日期", "app_id"], as_index=False)
        .agg(split_y_true=("y_true", "sum"), split_y_pred=("y_pred", "sum"), split_entity_cnt=("应用ID", "nunique"))
        .rename(columns={"app_id": "应用ID"})
    )
    pair = app.merge(split_agg, on=["日期", "应用ID"], how="inner").sort_values(["日期", "应用ID"])
    if pair.empty:
        raise ValueError("在线融合失败：app 基线预测与分流预测没有可对齐的 (日期, 应用ID) 交集。")

    def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-8))

    frames: list[pd.DataFrame] = []
    choices: list[dict[str, float | int | str]] = []
    dates = sorted(pair["日期"].dropna().unique())
    for d in dates:
        hist = pair[pair["日期"] < d].copy()
        cur = pair[pair["日期"] == d].copy()
        if len(hist["日期"].unique()) < min_history_days:
            best_k = 10**9
            best_w = 0.0
        else:
            best_score: float | None = None
            best_k = min_entities_grid[0]
            best_w = split_weight_grid[0]
            y_hist = hist["y_true"].values
            for k in min_entities_grid:
                use_split = (hist["split_entity_cnt"].values >= k).astype(float)
                for w in split_weight_grid:
                    w2 = max(0.0, min(float(w), 1.0))
                    pred = (
                        use_split * (w2 * hist["split_y_pred"].values + (1.0 - w2) * hist["y_pred"].values)
                        + (1.0 - use_split) * hist["y_pred"].values
                    )
                    score = _wmape(y_hist, pred)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_k = int(k)
                        best_w = float(w2)

        use_cur = (cur["split_entity_cnt"].values >= best_k).astype(float)
        cur = cur.copy()
        cur["reconcile_min_entities"] = int(best_k)
        cur["reconcile_split_weight"] = float(best_w)
        cur["y_pred_online_fused"] = (
            use_cur * (best_w * cur["split_y_pred"].values + (1.0 - best_w) * cur["y_pred"].values)
            + (1.0 - use_cur) * cur["y_pred"].values
        )
        frames.append(cur)
        choices.append(
            {
                "date": str(pd.Timestamp(d).date()),
                "min_entities": int(best_k),
                "split_weight": float(best_w),
                "samples": int(len(cur)),
            }
        )

    out = pd.concat(frames, ignore_index=True)
    result = out[["日期", "应用ID"]].copy()
    result["y_true"] = out["y_true"].values
    result["y_pred_app"] = out["y_pred"].values
    result["y_pred_split_agg"] = out["split_y_pred"].values
    result["split_entity_cnt"] = out["split_entity_cnt"].values
    result["reconcile_min_entities"] = out["reconcile_min_entities"].values
    result["reconcile_split_weight"] = out["reconcile_split_weight"].values
    result["y_pred_fused"] = out["y_pred_online_fused"].values

    y_true = result["y_true"].values
    report: Dict[str, object] = {
        "samples": int(len(result)),
        "mode": "online_calibrated",
        "min_history_days": int(min_history_days),
        "app_metrics": _evaluate_with_wmape(y_true, result["y_pred_app"].values),
        "split_agg_metrics": _evaluate_with_wmape(y_true, result["y_pred_split_agg"].values),
        "fused_metrics": _evaluate_with_wmape(y_true, result["y_pred_fused"].values),
        "choices_tail": choices[-5:],
    }
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description="并行预测 T+1 spend，并输出与 roi_d1 同口径百分比误差")
    parser.add_argument("--input", default="daily_20260421_120112.csv")
    parser.add_argument("--output-dir", default="outputs/model_parallel_spend_t1")
    parser.add_argument("--min-train-days", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--use-log-target", action="store_true", help="增加 log1p(spend) 分支")
    parser.add_argument("--weight-exponent", type=float, default=0.35, help="样本权重指数：spend^exponent (0=等权, 0.35=平衡, 0.5=sqrt, 1=线性)")
    parser.add_argument(
        "--split-col",
        default="",
        help="（已废弃-统一底表不区分分流）",
    )
    parser.add_argument(
        "--clean-group-cols",
        default="",
        help="（已废弃-统一底表内置清洗）",
    )
    parser.add_argument(
        "--product-key-cols",
        default="",
        help="（已废弃-统一底表内置实体定义）",
    )
    parser.add_argument(
        "--min-target-spend-train",
        type=float,
        default=0.0,
        help="训练集过滤：仅保留 target_t1_spend >= 该阈值的样本",
    )
    parser.add_argument(
        "--min-target-spend-eval",
        type=float,
        default=0.0,
        help="评估集过滤：仅统计 target_t1_spend >= 该阈值的样本",
    )
    parser.add_argument(
        "--min-spend-train",
        type=float,
        default=0.01,
        help="训练最低日消耗阈值（过滤低消耗噪声样本）",
    )
    parser.add_argument(
        "--reconcile-with-app-baseline",
        action="store_true",
        help="启用分流回聚合融合：需要提供 app 维度基线预测 CSV（列包含 日期/应用ID/y_true/y_pred）",
    )
    parser.add_argument(
        "--reconcile-app-prediction-csv",
        default="",
        help="app 维度基线预测 CSV（建议使用 outputs/model_parallel_spend_t1_current/predictions_LightGBM_log.csv）",
    )
    parser.add_argument(
        "--reconcile-model",
        default="LightGBM_log",
        help="用于融合的分流模型名（对应 predictions_{model}.csv）",
    )
    parser.add_argument(
        "--reconcile-min-entities",
        type=int,
        default=6,
        help="低样本回退阈值：应用日下分流实体数低于该值时回退 app 预测",
    )
    parser.add_argument(
        "--reconcile-split-weight",
        type=float,
        default=0.5,
        help="分流预测在融合中的权重（0-1）",
    )
    parser.add_argument(
        "--reconcile-mode",
        choices=["fixed", "online"],
        default="fixed",
        help="融合模式：fixed 使用固定阈值/权重；online 使用历史日期在线选择阈值/权重",
    )
    parser.add_argument(
        "--reconcile-min-history-days",
        type=int,
        default=5,
        help="online 融合最少历史评估天数；不足时回退 app 基线",
    )
    parser.add_argument(
        "--feature-set",
        choices=["base", "calendar_pacing"],
        default="base",
        help="特征集：base 为当前稳定特征；calendar_pacing 额外加入月末/节假日/滚动节奏特征",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="新数据快速出数：默认仅近 40 个 walk-forward 截止日 + 仅 ExtraTrees + 跳过 Baseline/TwoStage（可用 --eval-recent-days / --tree-models 覆盖）",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="LightGBM 使用 GPU 加速（需支持 CUDA 的 LightGBM 版本）",
    )
    parser.add_argument(
        "--eval-recent-days",
        type=int,
        default=0,
        help=">0 时仅对最近 N 个训练截止日做预测步（大幅加速）；0 表示不截断",
    )
    parser.add_argument(
        "--tree-models",
        default="",
        help="逗号分隔覆盖默认四树模型；与应用级 T+1 融合建议 --split-col \"\" --product-key-cols 应用ID",
    )
    parser.add_argument(
        "--target-type",
        choices=["absolute", "ratio"],
        default="absolute",
        help="预测目标类型：absolute=绝对spend值，ratio=t+1/t成长比例",
    )
    args = parser.parse_args()

    validate_merged_daily_csv(Path(args.input))

    eval_recent: Optional[int] = int(args.eval_recent_days) if args.eval_recent_days > 0 else None
    tree_override: Optional[List[str]] = [x.strip() for x in str(args.tree_models).split(",") if x.strip()] or None
    if args.fast:
        if eval_recent is None:
            eval_recent = 40
        if not tree_override:
            tree_override = ["ExtraTrees"]
    if not tree_override:
        tree_override = ["XGBoost", "CatBoost"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, clean_meta = build_unified_daily(
        args.input,
        min_spend_train=2.0,
    )
    if args.target_type == "ratio":
        df["target_t1_spend"] = df["target_spend_ratio_t1"]
    results = _run(
        df,
        min_train_days=args.min_train_days,
        alpha=args.alpha,
        use_log_target=args.use_log_target,
        feature_set=args.feature_set,
        min_target_spend_train=float(args.min_target_spend_train),
        min_target_spend_eval=float(args.min_target_spend_eval),
        eval_recent_days=eval_recent,
        tree_models=tree_override,
        use_gpu=bool(args.use_gpu),
        weight_exponent=args.weight_exponent,
    )
    if not results:
        raise ValueError("未得到 spend 预测结果，请检查数据和参数。")

    metric_rows = []
    for r in results:
        metric_rows.append({"model": r.model_name, **r.metrics, "samples": int(len(r.prediction_df))})
        r.prediction_df.to_csv(out_dir / f"predictions_{safe_model_filename(r.model_name)}.csv", index=False, encoding="utf-8-sig")
    metric_df = pd.DataFrame(metric_rows).sort_values("mape")
    metric_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    reconcile_report = None
    if args.reconcile_with_app_baseline:
        if not args.reconcile_app_prediction_csv:
            raise ValueError("启用 --reconcile-with-app-baseline 时必须提供 --reconcile-app-prediction-csv。")
        split_file = out_dir / f"predictions_{safe_model_filename(args.reconcile_model)}.csv"
        if not split_file.exists():
            raise ValueError(f"未找到分流模型预测文件: {split_file}")
        app_file = Path(args.reconcile_app_prediction_csv)
        if not app_file.exists():
            raise ValueError(f"未找到 app 基线预测文件: {app_file}")
        split_pred = pd.read_csv(split_file, encoding="utf-8-sig")
        app_pred = pd.read_csv(app_file, encoding="utf-8-sig")
        if args.reconcile_mode == "online":
            fused_df, reconcile_report = _build_online_reconciled_prediction(
                split_pred_df=split_pred,
                app_pred_df=app_pred,
                min_history_days=max(int(args.reconcile_min_history_days), 1),
                min_entities_grid=list(range(1, 9)),
                split_weight_grid=[i / 10 for i in range(0, 11)],
            )
        else:
            fused_df, reconcile_report = _build_reconciled_prediction(
                split_pred_df=split_pred,
                app_pred_df=app_pred,
                min_entities=max(int(args.reconcile_min_entities), 1),
                split_weight=float(args.reconcile_split_weight),
            )
        fused_df.to_csv(out_dir / "predictions_reconciled.csv", index=False, encoding="utf-8-sig")
        (out_dir / "reconcile_report.json").write_text(
            json.dumps(reconcile_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = {
        "target": "T+1 spend",
        "unified_data": True,
        "fast_mode": bool(args.fast),
        "eval_recent_days": eval_recent,
        "tree_models": tree_override,
        "clean_meta": clean_meta,
        "min_spend_train": 2.0,
        "feature_set": args.feature_set,
        "feature_count": len(feature_cols(args.feature_set)),
        "use_gpu": bool(args.use_gpu),
        "app_encoding": "expanding_target",
        "best_model_by_mape": str(metric_df.iloc[0]["model"]),
        "best_mape": float(metric_df.iloc[0]["mape"]),
        "best_mape_pct": float(metric_df.iloc[0]["mape_pct"]),
        "reconcile_enabled": bool(args.reconcile_with_app_baseline),
        "reconcile_model": args.reconcile_model,
        "reconcile_min_entities": int(args.reconcile_min_entities),
        "reconcile_split_weight": float(args.reconcile_split_weight),
        "reconcile_mode": args.reconcile_mode,
        "reconcile_min_history_days": int(args.reconcile_min_history_days),
        "reconcile_report": reconcile_report,
        "output_dir": str(out_dir),
        "files": {
            "metrics": str(out_dir / "metrics_summary.csv"),
            "predictions": [str(out_dir / f"predictions_{safe_model_filename(r.model_name)}.csv") for r in results],
            "predictions_reconciled": str(out_dir / "predictions_reconciled.csv") if reconcile_report else None,
            "reconcile_report": str(out_dir / "reconcile_report.json") if reconcile_report else None,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
