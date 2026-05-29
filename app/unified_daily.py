"""
共享数据管线：为 ROI 和 Spend 两个预测模型提供完全一致的训练底表。

保证：
- 相同的 (应用ID, 日期) 样本
- 相同的过滤口径（app 级零消耗剔除 + min 30 天 + 行级 spend >= 2.0）
- 相同的日历/节假日特征
- 各自模型增量添加目标相关 lag/target 列
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from app.config import settings
from app.core.calendar import CalendarService

CALENDAR = CalendarService()

DEFAULT_CLEAN_GROUP_COLS = [
    "应用ID",
    "推广流量",
    "推广流量名称",
    "流量场景",
    "流量场景名称",
    "创意规格",
    "创意规格名称",
]

# ---- calendar helpers ----
def _is_holiday(d: date) -> bool:
    return CALENDAR.is_holiday(d)


def _is_rest_day(d: date) -> int:
    return int(CALENDAR.is_rest_day(d))


def _holiday_id(d: date) -> int:
    return int(CALENDAR.features_for_day(d)["holiday_id"])


def _holiday_seq_index(d: date) -> int:
    return int(CALENDAR.features_for_day(d)["holiday_seq_index"])


def _holiday_days_remaining(d: date) -> int:
    return int(CALENDAR.features_for_day(d)["holiday_days_remaining"])


def _holiday_window_len(d: date) -> int:
    return int(CALENDAR.features_for_day(d)["holiday_window_len"])


def _is_last_holiday_day(d: date) -> int:
    return int(CALENDAR.features_for_day(d)["is_last_holiday_day"])


def _is_first_workday_after_holiday(d: date) -> int:
    return int(CALENDAR.features_for_day(d)["is_first_workday_after_holiday"])


def _days_to_next_holiday(d: date) -> int:
    return CALENDAR.days_to_next_holiday(d)


def _days_since_prev_holiday(d: date) -> int:
    return CALENDAR.days_since_prev_holiday(d)


def _build_composition_features(raw: pd.DataFrame) -> pd.DataFrame:
    comp_dims = {
        "traffic": "推广流量名称",
        "scene": "流量场景名称",
        "creative": "创意规格名称",
        "billing": "计费方式",
    }
    available_dims = {k: v for k, v in comp_dims.items() if v in raw.columns}
    if not available_dims:
        return pd.DataFrame(columns=["应用ID", "日期"])

    frames = []
    for dim_key, dim_col in available_dims.items():
        spend = raw.groupby(["应用ID", "日期", dim_col])["消耗金额"].sum().reset_index()
        total = spend.groupby(["应用ID", "日期"])["消耗金额"].transform("sum")
        spend["share"] = np.where(total > 1e-8, spend["消耗金额"] / total, 0.0)
        agg = (
            spend.groupby(["应用ID", "日期"])
            .agg(**{
                f"{dim_key}_top1_pct": ("share", "max"),
                f"{dim_key}_hhi": ("share", lambda s: float((s**2).sum())),
            })
            .reset_index()
        )
        pos = spend[spend["消耗金额"] > 1e-8]
        agg2 = pos.groupby(["应用ID", "日期"])[dim_col].nunique().reset_index()
        agg2.columns = ["应用ID", "日期", f"{dim_key}_n_unique"]
        agg = agg.merge(agg2, on=["应用ID", "日期"], how="left")
        agg[f"{dim_key}_n_unique"] = agg[f"{dim_key}_n_unique"].fillna(0).astype(int)
        frames.append(agg)

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["应用ID", "日期"], how="outer")
    for c in result.columns:
        if c not in ("应用ID", "日期"):
            result[c] = result[c].fillna(0.0 if "_pct" in c or "_hhi" in c else 0)
    return result


def build_unified_daily(
    input_csv: str,
    min_spend_train: float = 5.0,
    min_app_days: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """构建 ROI / Spend 共享的 应用-日期 底表。

    返回 (df, meta)，其中 df 的每一行是唯一的 (应用ID, 日期)。
    """
    from app.data_cleaning import clean as _clean_csv

    raw = _clean_csv(Path(input_csv))

    # 阶段1：细粒度清洗（全时段总消耗为 0 的实体剔除）
    clean_key_cols = [c for c in DEFAULT_CLEAN_GROUP_COLS if c in raw.columns]
    clean_key = raw[clean_key_cols[0]].astype(str).fillna("")
    for c in clean_key_cols[1:]:
        clean_key = clean_key + "|" + raw[c].astype(str).fillna("")
    clean_spend = (
        pd.DataFrame({"clean_key": clean_key, "消耗金额": raw["消耗金额"]})
        .groupby("clean_key", as_index=False)["消耗金额"]
        .sum()
        .rename(columns={"消耗金额": "total_spend"})
    )
    valid_clean_keys = set(clean_spend[clean_spend["total_spend"] > 0]["clean_key"].tolist())
    clean_before = int(clean_spend.shape[0])
    raw = raw[clean_key.isin(valid_clean_keys)].copy()
    clean_after = len(valid_clean_keys)

    # 阶段2：聚合到 应用-日期
    comp_df = _build_composition_features(raw)
    agg_cols = ["消耗金额", "首日广告收入", "曝光量", "点击量", "下载量", "激活人数(快应用新增用户数)"]
    df = (
        raw.groupby(["应用ID", "日期"], as_index=False)[agg_cols]
        .sum()
        .sort_values(["应用ID", "日期"])
    )
    df = df.merge(comp_df, on=["应用ID", "日期"], how="left")
    for c in comp_df.columns:
        if c not in ("应用ID", "日期") and c in df.columns:
            df[c] = df[c].fillna(0)

    # 阶段3：过滤
    # 3a：app 级 — 全周期消耗为 0
    app_total_spend = df.groupby("应用ID")["消耗金额"].sum()
    zero_spend_apps = set(app_total_spend[app_total_spend <= 0].index)
    if zero_spend_apps:
        df = df[~df["应用ID"].isin(zero_spend_apps)].copy()

    # 3b：app 级 — 历史天数不足
    min_app_days = max(min_app_days, settings.app_last_day_min_train_days, 1)
    app_day_counts = df.groupby("应用ID")["日期"].nunique()
    valid_apps = set(app_day_counts[app_day_counts >= min_app_days].index)
    apps_before = int(df["应用ID"].nunique())
    df = df[df["应用ID"].isin(valid_apps)].copy()
    apps_after = int(df["应用ID"].nunique())

    # 3c：行级 — 消耗 < min_spend_train 的行过滤
    rows_before = len(df)
    df = df[df["消耗金额"] >= min_spend_train].copy()
    rows_after = len(df)

    # ---- 基础特征 ----
    df["roi_d1"] = np.where(df["消耗金额"] > 0, df["首日广告收入"] / df["消耗金额"], np.nan)
    df["act_per_spend"] = np.where(
        df["消耗金额"] > 0, df["激活人数(快应用新增用户数)"] / df["消耗金额"], np.nan
    )
    df["rev_per_act_d1"] = np.where(
        df["激活人数(快应用新增用户数)"] > 0,
        df["首日广告收入"] / df["激活人数(快应用新增用户数)"],
        np.nan,
    )
    df["ctr"] = np.where(df["曝光量"] > 0, df["点击量"] / df["曝光量"], 0.0)
    df["cvr_dl"] = np.where(df["点击量"] > 0, df["下载量"] / df["点击量"], 0.0)
    df["cvr_act"] = np.where(df["下载量"] > 0, df["激活人数(快应用新增用户数)"] / df["下载量"], 0.0)

    # ---- 日期特征 ----
    df["dow"] = df["日期"].dt.weekday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_fri_sat"] = df["dow"].isin([4, 5]).astype(int)
    df["is_holiday"] = df["日期"].dt.date.map(_is_holiday).astype(int)
    df["is_rest_day"] = df["日期"].dt.date.map(_is_rest_day).astype(int)
    df["dom"] = df["日期"].dt.day
    df["month"] = df["日期"].dt.month
    df["week_of_month"] = ((df["dom"] - 1) // 7 + 1).clip(1, 5)
    df["days_in_month"] = df["日期"].dt.days_in_month
    df["days_to_month_end"] = df["days_in_month"] - df["dom"]
    df["month_progress"] = df["dom"] / df["days_in_month"].replace(0, np.nan)
    df["is_month_start_3d"] = (df["dom"] <= 3).astype(int)
    df["is_month_end_3d"] = (df["days_to_month_end"] <= 2).astype(int)
    df["is_month_end_7d"] = (df["days_to_month_end"] <= 6).astype(int)
    df["is_month_end_10d"] = (df["days_to_month_end"] <= 9).astype(int)
    df["days_to_next_holiday"] = df["日期"].dt.date.map(_days_to_next_holiday).clip(upper=30)
    df["days_since_prev_holiday"] = df["日期"].dt.date.map(_days_since_prev_holiday).clip(upper=30)
    df["is_pre_holiday_3d"] = df["days_to_next_holiday"].between(1, 3).astype(int)
    df["is_post_holiday_3d"] = df["days_since_prev_holiday"].between(1, 3).astype(int)
    df["is_summer_winter_break"] = df["month"].isin([1, 2, 7, 8]).astype(int)
    # ROI 独有：pre/post holiday 单天
    target_dates_tmp = df["日期"].dt.date.map(lambda d: d + timedelta(days=1))
    df["pre_holiday"] = target_dates_tmp.map(_is_holiday).astype(int)
    df["post_holiday"] = df["日期"].dt.date.map(lambda d: int(_is_holiday(d - timedelta(days=1))))

    df["holiday_id"] = df["日期"].dt.date.map(_holiday_id).astype(int)
    df["holiday_seq_index"] = df["日期"].dt.date.map(_holiday_seq_index).astype(int)
    df["holiday_days_remaining"] = df["日期"].dt.date.map(_holiday_days_remaining).astype(int)
    df["holiday_window_len"] = df["日期"].dt.date.map(_holiday_window_len).astype(int)
    df["is_last_holiday_day"] = df["日期"].dt.date.map(_is_last_holiday_day).astype(int)

    # ---- T+1 目标日特征 ----
    target_dates = df["日期"].dt.date.map(lambda d: d + timedelta(days=1))
    df["target_dow"] = target_dates.map(lambda d: d.weekday()).astype(int)
    df["target_is_weekend"] = target_dates.map(lambda d: d.weekday() >= 5).astype(int)
    df["target_is_rest_day"] = target_dates.map(lambda d: int(CALENDAR.is_rest_day(d))).astype(int)
    df["target_is_adjusted_workday"] = target_dates.map(lambda d: int(CALENDAR.is_adjusted_workday(d))).astype(int)
    df["target_is_holiday"] = target_dates.map(_is_holiday).astype(int)
    df["target_holiday_id"] = target_dates.map(_holiday_id).astype(int)
    df["target_holiday_seq_index"] = target_dates.map(_holiday_seq_index).astype(int)
    df["target_holiday_days_remaining"] = target_dates.map(_holiday_days_remaining).astype(int)
    df["target_holiday_window_len"] = target_dates.map(_holiday_window_len).astype(int)
    df["target_is_last_holiday_day"] = target_dates.map(_is_last_holiday_day).astype(int)
    df["target_days_to_next_holiday"] = target_dates.map(_days_to_next_holiday).clip(upper=30)
    df["target_days_since_prev_holiday"] = target_dates.map(_days_since_prev_holiday).clip(upper=30)
    df["target_is_pre_holiday_3d"] = df["target_days_to_next_holiday"].between(1, 3).astype(int)
    df["target_is_post_holiday_1d"] = (df["target_days_since_prev_holiday"] == 1).astype(int)
    df["target_is_post_holiday_3d"] = df["target_days_since_prev_holiday"].between(1, 3).astype(int)
    df["target_is_first_workday_after_holiday"] = target_dates.map(_is_first_workday_after_holiday).astype(int)

    # ---- 滚动/滞后特征 ----
    g = df.groupby("应用ID")
    for lag in [1, 2, 3, 7]:
        df[f"spend_lag_{lag}"] = g["消耗金额"].shift(lag)
        # 两个命名约定：roi_lag（spend 脚本用） 和 roi_d1_lag（ROI 脚本用）
        roi_shifted = g["roi_d1"].shift(lag)
        df[f"roi_lag_{lag}"] = roi_shifted
        df[f"roi_d1_lag_{lag}"] = roi_shifted
        df[f"act_per_spend_lag_{lag}"] = g["act_per_spend"].shift(lag)
        df[f"rev_per_act_d1_lag_{lag}"] = g["rev_per_act_d1"].shift(lag)

    for window in [3, 7]:
        shifted_spend = g["消耗金额"].shift(1)
        df[f"spend_roll_mean_{window}"] = (
            shifted_spend.groupby(df["应用ID"])
            .rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"spend_roll_std_{window}"] = (
            shifted_spend.groupby(df["应用ID"])
            .rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        )

    # ---- MTD 特征 ----
    month_key = df["日期"].dt.to_period("M")
    df["mtd_spend"] = df.groupby(["应用ID", month_key])["消耗金额"].cumsum()
    df["mtd_revenue"] = df.groupby(["应用ID", month_key])["首日广告收入"].cumsum()
    df["mtd_roi_d1"] = np.where(df["mtd_spend"] > 1e-8, df["mtd_revenue"] / df["mtd_spend"], 0.0)

    # ---- 变化率特征 ----
    df["spend_ratio_1d"] = np.where(
        df["spend_lag_1"] > 1e-8, df["消耗金额"] / df["spend_lag_1"] - 1.0, 0.0
    )
    # ROI 独有：3日变化率
    spend_lag_3_mean = g["消耗金额"].rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    df["spend_ratio_3d"] = np.where(
        spend_lag_3_mean > 1e-8, df["消耗金额"] / spend_lag_3_mean - 1.0, np.nan
    )

    # ---- ROI 波动率特征 ----
    df["roi_d1_std_7"] = g["roi_d1"].rolling(7, min_periods=3).std().reset_index(level=0, drop=True)
    roi_q75 = g["roi_d1"].rolling(7, min_periods=3).quantile(0.75).reset_index(level=0, drop=True)
    roi_q25 = g["roi_d1"].rolling(7, min_periods=3).quantile(0.25).reset_index(level=0, drop=True)
    df["roi_d1_iqr_7"] = roi_q75 - roi_q25
    df["roi_d1_range_7"] = (
        g["roi_d1"].rolling(7, min_periods=3).max().reset_index(level=0, drop=True)
        - g["roi_d1"].rolling(7, min_periods=3).min().reset_index(level=0, drop=True)
    )

    # ---- T+1 目标 ----
    df["target_t1_spend"] = g["消耗金额"].shift(-1)
    df["target_t1_roi_d1"] = g["roi_d1"].shift(-1)
    df["target_t1_act_per_spend"] = g["act_per_spend"].shift(-1)
    df["target_t1_rev_per_act_d1"] = g["rev_per_act_d1"].shift(-1)
    df["target_spend_ratio_t1"] = np.where(
        df["消耗金额"] > 1e-8,
        df["target_t1_spend"] / df["消耗金额"],
        1.0,
    )

    # ---- 交互特征：休息日 × 消耗模式 ----
    df["spend_x_is_rest_day"] = df["消耗金额"] * df["is_rest_day"]
    df["spend_lag1_x_target_rest"] = df["spend_lag_1"] * df["target_is_rest_day"]
    df["spend_lag1_x_target_holiday"] = df["spend_lag_1"] * df["target_is_holiday"]
    df["spend_ratio_x_target_rest"] = df["spend_ratio_1d"] * df["target_is_rest_day"]
    df["roi_lag1_x_target_rest"] = df["roi_lag_1"] * df["target_is_rest_day"]

    # ---- app label encoding ----
    df["app_label"] = df["应用ID"].astype("category").cat.codes

    # ---- 清理 ----
    fill_cols = [c for c in df.columns if "lag_" in c or "roll_" in c or "ratio_" in c
                 or c in ("month_progress", "mtd_roi_d1", "roi_d1_std_7", "roi_d1_iqr_7",
                          "roi_d1_range_7", "act_per_spend", "rev_per_act_d1")]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta = {
        "clean_groups_before": clean_before,
        "clean_groups_after": clean_after,
        "dropped_zero_spend_clean_groups": max(clean_before - clean_after, 0),
        "min_app_days": min_app_days,
        "min_spend_train": min_spend_train,
        "apps_before_day_filter": apps_before,
        "apps_after_day_filter": apps_after,
        "apps_filtered_by_days": max(apps_before - apps_after, 0),
        "rows_before_spend_filter": rows_before,
        "rows_after_spend_filter": rows_after,
        "rows_filtered_by_spend": max(rows_before - rows_after, 0),
        "final_apps": int(df["应用ID"].nunique()),
        "final_rows": len(df),
    }
    return df, meta
