"""
预测产物路径解析：Web 与 offline_backtest 共用，避免 glob/硬编码文件名导致新数据重训后看板仍指向旧模型或错文件。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional


MERGED_DAILY_REQUIRED_COLS: List[str] = [
    "应用ID",
    "日期",
    "消耗金额",
    "首日广告收入",
    "曝光量",
    "点击量",
    "下载量",
    "激活人数(快应用新增用户数)",
]


def validate_merged_daily_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header:
        raise ValueError(f"CSV 无表头: {path}")
    header_set = {str(c).strip() for c in header}
    missing = [c for c in MERGED_DAILY_REQUIRED_COLS if c not in header_set]
    if missing:
        raise ValueError(f"CSV 缺少并行训练/回放所需列 {missing}，当前列: {sorted(header_set)[:40]}...")


def safe_model_filename(model_name: str) -> str:
    return str(model_name).replace("/", "_")


def latest_metric_model_name(metrics_path: Optional[Path], default_model: str) -> str:
    if metrics_path is None or not metrics_path.exists():
        return default_model
    try:
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return default_model
        model = str(rows[0].get("model", "")).strip()
        return model or default_model
    except OSError:
        return default_model


def pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def resolve_roi_predictions_csv(roi_dir: Optional[Path]) -> Optional[Path]:
    if roi_dir is None or not roi_dir.is_dir():
        return None
    metrics = roi_dir / "metrics_summary.csv"
    best = latest_metric_model_name(metrics, "ExtraTrees_log")
    primary = roi_dir / f"predictions_{safe_model_filename(best)}.csv"
    if primary.exists():
        return primary
    return pick_first_existing(sorted(roi_dir.glob("predictions_*.csv")))


def resolve_spend_predictions_csv(spend_dir: Optional[Path]) -> Optional[Path]:
    """
    优先 metrics_summary 最优模型；其次融合/常用树模型；再 EWMA 快导出（非学习模型，仅兜底）。
    """
    if spend_dir is None or not spend_dir.is_dir():
        return None
    metrics = spend_dir / "metrics_summary.csv"
    best = latest_metric_model_name(metrics, "ExtraTrees_log")
    candidates = [
        spend_dir / f"predictions_{safe_model_filename(best)}.csv",
        spend_dir / "predictions_reconciled.csv",
        spend_dir / "predictions_LightGBM_log.csv",
        spend_dir / "predictions_GBDT_log.csv",
        spend_dir / "predictions_ExtraTrees_log.csv",
        spend_dir / "predictions_EWMA_quick.csv",
    ]
    p = pick_first_existing(candidates)
    if p is not None:
        return p
    return pick_first_existing(sorted(spend_dir.glob("predictions_*.csv")))
