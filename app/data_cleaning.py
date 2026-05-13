"""
统一数据清洗：供所有训练/预测/回放脚本复用，避免每个脚本各自重复清洗逻辑。

清洗步骤：
  1. 日期列 → datetime
  2. 业务数值列 → pd.to_numeric + fillna(0)
  3. 删无效行（指定数值列整行加总为 0）
  4. 多键去重（保留第一条）
  5. 按日期升序排序

额外提供：
  - validate_merged_daily_csv 的列校验（与 prediction_artifacts 一致）
  - 应用-日期聚合（当前训练脚本所需粒度）
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

# 完整的业务数值列（包含变现累积列，即使当前训练只用部分列，也应清洗）
_NUMERIC_COLS_ALL: List[str] = [
    "消耗金额",
    "曝光量",
    "点击量",
    "下载量",
    "激活人数(快应用新增用户数)",
    "注册人数",
    "首日广告收入",
    "3日累计变现金额",
    "7日累计变现金额",
    "30日累计变现金额",
    "买量广告收入",
]

# 去重 + 无效行判断使用的维度键
_DEDUP_KEYS: List[str] = [
    "日期",
    "应用ID",
    "广告主ID",
    "计划ID",
    "广告组ID",
    "创意ID",
]

# 训练所需列校验（与 prediction_artifacts.MERGED_DAILY_REQUIRED_COLS 一致）
_REQUIRED_COLS: List[str] = [
    "应用ID",
    "日期",
    "消耗金额",
    "首日广告收入",
    "曝光量",
    "点击量",
    "下载量",
    "激活人数(快应用新增用户数)",
]


def clean(
    path: Path,
    dedup_keys: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """主清洗入口，返回清洗后的 DataFrame（保留细粒度维度列）。"""
    dedup_keys = dedup_keys or _DEDUP_KEYS
    numeric_cols = numeric_cols or _NUMERIC_COLS_ALL

    df = pd.read_csv(str(path), encoding="utf-8-sig")

    # 1. 日期列
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df[df["日期"].notna()].copy()

    # 2. 业务数值列 → numeric + fillna(0)
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 3. 删无效行：指定数值列加总为 0 的行（无投放无回收）
    _zero_mask_cols = [c for c in numeric_cols if c in df.columns]
    if _zero_mask_cols:
        row_sum = df[_zero_mask_cols].sum(axis=1)
        df = df[row_sum > 0].copy()

    # 4. 去重
    avail_keys = [k for k in dedup_keys if k in df.columns]
    if avail_keys:
        df = df.drop_duplicates(subset=avail_keys, keep=keep).copy()

    # 5. 排序
    df = df.sort_values("日期").reset_index(drop=True)

    return df


def validate_columns(df: pd.DataFrame, required: Optional[List[str]] = None) -> None:
    """校验 DataFrame 是否包含训练所需列。"""
    required = required or _REQUIRED_COLS
    cols = set(df.columns)
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"缺少必要列 {missing}，当前列: {sorted(cols)}"
        )


def to_app_daily(
    df: pd.DataFrame,
    agg_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """聚合到应用-日期粒度（当前训练脚本的输入格式）。"""
    agg_cols = agg_cols or [
        "消耗金额",
        "首日广告收入",
        "曝光量",
        "点击量",
        "下载量",
        "激活人数(快应用新增用户数)",
    ]
    avail_agg = [c for c in agg_cols if c in df.columns]
    return (
        df.groupby(["应用ID", "日期"], as_index=False)[avail_agg]
        .sum()
        .sort_values(["应用ID", "日期"])
    )


def validate_csv(path: Path, required: Optional[List[str]] = None) -> dict:
    """轻量级 CSV 探查：返回行数/日期范围/应用数/列校验结果。"""
    required = required or _REQUIRED_COLS
    # 快速读表头
    header = pd.read_csv(str(path), encoding="utf-8-sig", nrows=0).columns.tolist()
    missing = [c for c in required if c not in header]

    # 快速统计
    df = pd.read_csv(
        str(path),
        encoding="utf-8-sig",
        usecols=lambda c: c in required + ["日期"],
    )
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    dates = df["日期"].dropna()

    return {
        "file": str(path),
        "rows": len(df),
        "date_min": str(dates.min().date()) if len(dates) > 0 else None,
        "date_max": str(dates.max().date()) if len(dates) > 0 else None,
        "apps": int(df["应用ID"].nunique()) if "应用ID" in df.columns else 0,
        "missing_cols": missing,
        "ok": len(missing) == 0,
    }
