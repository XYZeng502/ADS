import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.core.calendar import CalendarService

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


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
DEFAULT_PRODUCT_KEY_COLS = ["应用ID", "推广流量名称"]


@dataclass
class ModelResult:
    model_name: str
    metrics: Dict[str, float]
    prediction_df: pd.DataFrame


def _is_holiday(d: date) -> bool:
    return CALENDAR.is_holiday(d)


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


def _normalize_cols(raw_cols: List[str]) -> List[str]:
    seen = set()
    cols: List[str] = []
    for c in raw_cols:
        c2 = str(c).strip()
        if not c2 or c2 in seen:
            continue
        seen.add(c2)
        cols.append(c2)
    return cols


def _build_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    out = df[cols[0]].astype(str).fillna("")
    for c in cols[1:]:
        out = out + "|" + df[c].astype(str).fillna("")
    return out


def _build_app_daily(
    input_csv: str,
    split_col: str,
    clean_group_cols: List[str],
    product_key_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    raw = pd.read_csv(input_csv, encoding="utf-8-sig")
    raw["日期"] = pd.to_datetime(raw["日期"], errors="coerce")
    raw = raw[raw["日期"].notna()].copy()
    for c in ["消耗金额", "首日广告收入", "曝光量", "点击量", "下载量", "激活人数(快应用新增用户数)"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    split_col = str(split_col or "").strip()
    clean_group_cols = _normalize_cols(clean_group_cols or DEFAULT_CLEAN_GROUP_COLS)
    product_key_cols = _normalize_cols(product_key_cols or DEFAULT_PRODUCT_KEY_COLS)
    needed = sorted(set(clean_group_cols + product_key_cols + ([split_col] if split_col else [])))
    for c in needed:
        if c not in raw.columns:
            raise ValueError(f"输入数据缺少聚合字段: {c}")
        raw[c] = raw[c].fillna("未知").astype(str)

    # 阶段1：细粒度清洗（全时段总消耗为0的实体剔除）
    clean_key = _build_key(raw, clean_group_cols)
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

    # 阶段2：按训练实体聚合（并可再按 split_col 分流）
    entity_cols = list(dict.fromkeys(product_key_cols + ([split_col] if split_col else [])))
    group_cols = entity_cols + ["日期"]
    df = (
        raw.groupby(group_cols, as_index=False)[["消耗金额", "首日广告收入", "曝光量", "点击量", "下载量", "激活人数(快应用新增用户数)"]]
        .sum()
        .sort_values(group_cols)
    )
    df["entity_id"] = _build_key(df, entity_cols)
    # 下游逻辑统一仍以 应用ID 作为实体键（此处映射到训练实体）
    df["应用ID"] = df["entity_id"]
    if split_col:
        df["split_value"] = df[split_col].astype(str).fillna("未知")
    else:
        df["split_value"] = "ALL"
    df["roi_d1"] = np.where(df["消耗金额"] > 0, df["首日广告收入"] / df["消耗金额"], 0.0)
    df["ctr"] = np.where(df["曝光量"] > 0, df["点击量"] / df["曝光量"], 0.0)
    df["cvr_dl"] = np.where(df["点击量"] > 0, df["下载量"] / df["点击量"], 0.0)
    df["cvr_act"] = np.where(df["下载量"] > 0, df["激活人数(快应用新增用户数)"] / df["下载量"], 0.0)
    df["dow"] = df["日期"].dt.weekday
    df["is_fri_sat"] = df["dow"].isin([4, 5]).astype(int)
    df["is_holiday"] = df["日期"].dt.date.map(_is_holiday).astype(int)
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
    df["holiday_id"] = df["日期"].dt.date.map(_holiday_id).astype(int)
    df["holiday_seq_index"] = df["日期"].dt.date.map(_holiday_seq_index).astype(int)
    df["holiday_days_remaining"] = df["日期"].dt.date.map(_holiday_days_remaining).astype(int)
    df["holiday_window_len"] = df["日期"].dt.date.map(_holiday_window_len).astype(int)
    df["is_last_holiday_day"] = df["日期"].dt.date.map(_is_last_holiday_day).astype(int)

    # T+1 spend 预测应使用“目标日”的日历属性，尤其是假期结束后的回落。
    target_dates = df["日期"].dt.date.map(lambda d: d + timedelta(days=1))
    df["target_dow"] = target_dates.map(lambda d: d.weekday()).astype(int)
    df["target_is_weekend"] = target_dates.map(lambda d: d.weekday() >= 5).astype(int)
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

    g = df.groupby("应用ID")
    for lag in [1, 2, 3, 7]:
        df[f"spend_lag_{lag}"] = g["消耗金额"].shift(lag)
        df[f"roi_lag_{lag}"] = g["roi_d1"].shift(lag)
    for window in [3, 7]:
        shifted_spend = g["消耗金额"].shift(1)
        df[f"spend_roll_mean_{window}"] = shifted_spend.groupby(df["应用ID"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"spend_roll_std_{window}"] = shifted_spend.groupby(df["应用ID"]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
    month_key = df["日期"].dt.to_period("M")
    df["mtd_spend"] = df.groupby(["应用ID", month_key])["消耗金额"].cumsum()
    df["mtd_revenue"] = df.groupby(["应用ID", month_key])["首日广告收入"].cumsum()
    df["mtd_roi_d1"] = np.where(df["mtd_spend"] > 1e-8, df["mtd_revenue"] / df["mtd_spend"], 0.0)
    df["spend_ratio_1d"] = np.where(df["spend_lag_1"] > 1e-8, df["消耗金额"] / df["spend_lag_1"] - 1.0, 0.0)
    df["target_t1_spend"] = g["消耗金额"].shift(-1)

    lag_cols = [c for c in df.columns if "lag_" in c or "roll_" in c] + [
        "spend_ratio_1d",
        "month_progress",
        "mtd_roi_d1",
    ]
    for c in lag_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta = {
        "clean_groups_before": clean_before,
        "clean_groups_after": clean_after,
        "dropped_zero_spend_clean_groups": max(clean_before - clean_after, 0),
        "train_entities_after_clean": int(df["应用ID"].nunique()),
    }
    return df, meta


BASE_FEATURE_COLS = [
        "消耗金额",
        "roi_d1",
        "ctr",
        "cvr_dl",
        "cvr_act",
        "dow",
        "is_fri_sat",
        "is_holiday",
        "dom",
        "month",
        "week_of_month",
        "days_to_month_end",
        "month_progress",
        "is_month_start_3d",
        "is_month_end_3d",
        "is_month_end_7d",
        "is_month_end_10d",
        "days_to_next_holiday",
        "days_since_prev_holiday",
        "is_pre_holiday_3d",
        "is_post_holiday_3d",
        "is_summer_winter_break",
        "holiday_id",
        "holiday_seq_index",
        "holiday_days_remaining",
        "holiday_window_len",
        "is_last_holiday_day",
        "target_dow",
        "target_is_weekend",
        "target_is_holiday",
        "target_holiday_id",
        "target_holiday_seq_index",
        "target_holiday_days_remaining",
        "target_holiday_window_len",
        "target_is_last_holiday_day",
        "target_days_to_next_holiday",
        "target_days_since_prev_holiday",
        "target_is_pre_holiday_3d",
        "target_is_post_holiday_1d",
        "target_is_post_holiday_3d",
        "target_is_first_workday_after_holiday",
        "spend_lag_1",
        "spend_lag_2",
        "spend_lag_3",
        "spend_lag_7",
        "spend_roll_mean_3",
        "spend_roll_std_3",
        "spend_roll_mean_7",
        "spend_roll_std_7",
        "roi_lag_1",
        "roi_lag_2",
        "roi_lag_3",
        "roi_lag_7",
        "spend_ratio_1d",
    ]

CALENDAR_PACING_FEATURE_COLS = [
    "week_of_month",
    "days_to_month_end",
    "month_progress",
    "is_month_start_3d",
    "is_month_end_3d",
    "is_month_end_7d",
    "is_month_end_10d",
    "days_to_next_holiday",
    "days_since_prev_holiday",
    "is_pre_holiday_3d",
    "is_post_holiday_3d",
    "is_summer_winter_break",
    "holiday_id",
    "holiday_seq_index",
    "holiday_days_remaining",
    "holiday_window_len",
    "is_last_holiday_day",
    "target_dow",
    "target_is_weekend",
    "target_is_holiday",
    "target_holiday_id",
    "target_holiday_seq_index",
    "target_holiday_days_remaining",
    "target_holiday_window_len",
    "target_is_last_holiday_day",
    "target_days_to_next_holiday",
    "target_days_since_prev_holiday",
    "target_is_pre_holiday_3d",
    "target_is_post_holiday_1d",
    "target_is_post_holiday_3d",
    "target_is_first_workday_after_holiday",
    "spend_roll_mean_3",
    "spend_roll_std_3",
    "spend_roll_mean_7",
    "spend_roll_std_7",
    "mtd_spend",
    "mtd_revenue",
    "mtd_roi_d1",
]


def _feature_cols(feature_set: str = "base") -> List[str]:
    if feature_set == "calendar_pacing":
        return list(dict.fromkeys(BASE_FEATURE_COLS + CALENDAR_PACING_FEATURE_COLS))
    return BASE_FEATURE_COLS


def _build_model(name: str) -> Optional[object]:
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
    if name == "GBDT":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.04, max_depth=3, random_state=42)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
    if name == "LightGBM":
        if LGBMRegressor is None:
            return None
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    return None


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > 1e-8
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if np.any(mask) else np.nan
    return {"mae": mae, "rmse": rmse, "mape": mape, "mape_pct": mape * 100.0 if np.isfinite(mape) else np.nan}


def _safe_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_")


def _extract_app_id(entity_id: str) -> str:
    s = str(entity_id)
    return s.split("|")[0] if "|" in s else s


def _evaluate_with_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    base = _evaluate(y_true, y_pred)
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


def _ewma_spend(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    preds = []
    for app_id, g_test in test_df.groupby("应用ID"):
        vals = train_df[train_df["应用ID"] == app_id].sort_values("日期")["消耗金额"].dropna().values
        if len(vals) == 0:
            p = np.nan
        else:
            s = vals[0]
            for v in vals[1:]:
                s = alpha * v + (1 - alpha) * s
            p = s
        for _, r in g_test.iterrows():
            preds.append({"日期": r["日期"], "应用ID": r["应用ID"], "y_true": r["target_t1_spend"], "y_pred": p, "model": "EWMA"})
    return pd.DataFrame(preds)


def _tree_predict(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feats: List[str], use_log_target: bool) -> pd.DataFrame:
    train = train_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    model = _build_model(name)
    if model is None:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    x_train = train[feats]
    y_train = train["target_t1_spend"].clip(lower=0.0)
    x_test = test[feats]
    y_test = test["target_t1_spend"].values
    label = name
    if use_log_target:
        y_train = np.log1p(y_train)
        label = f"{name}_log"
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if use_log_target:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0.0, None)
    pred = _apply_holiday_transition_calibration(test, pred)
    return pd.DataFrame({"日期": test["日期"].values, "应用ID": test["应用ID"].values, "y_true": y_test, "y_pred": pred, "model": label})


def _apply_holiday_transition_calibration(test_df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(pred, dtype=float).copy()
    current_spend = pd.to_numeric(test_df["消耗金额"], errors="coerce").fillna(0.0).to_numpy()
    roll3 = pd.to_numeric(test_df.get("spend_roll_mean_3", 0.0), errors="coerce").fillna(0.0).to_numpy()
    recent_ref = np.maximum(current_spend, roll3)

    last_holiday_day = pd.to_numeric(test_df.get("target_is_last_holiday_day", 0), errors="coerce").fillna(0).to_numpy() == 1
    first_workday_after_holiday = (
        pd.to_numeric(test_df.get("target_is_first_workday_after_holiday", 0), errors="coerce").fillna(0).to_numpy() == 1
    )

    # 假期最后一天通常延续假期内投放强度，避免被普通工作日/周内均值过度拉低。
    adjusted[last_holiday_day] = np.maximum(adjusted[last_holiday_day], recent_ref[last_holiday_day] * 0.80)
    # 节后首个工作日常出现预算回撤，给模型外推值加上业务上限。
    adjusted[first_workday_after_holiday] = np.minimum(
        adjusted[first_workday_after_holiday],
        recent_ref[first_workday_after_holiday] * 0.45,
    )
    return np.clip(adjusted, 0.0, None)


def _baseline_residual_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    train = train_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    train["dow_target"] = (train["日期"] + pd.to_timedelta(1, unit="D")).dt.weekday
    test["dow_target"] = (test["日期"] + pd.to_timedelta(1, unit="D")).dt.weekday

    # App + target_dow baseline
    g = train.groupby(["应用ID", "dow_target"], as_index=False)["target_t1_spend"].mean().rename(columns={"target_t1_spend": "baseline"})
    global_base = float(train["target_t1_spend"].median())
    train = train.merge(g, on=["应用ID", "dow_target"], how="left")
    test = test.merge(g, on=["应用ID", "dow_target"], how="left")
    train["baseline"] = train["baseline"].fillna(global_base)
    test["baseline"] = test["baseline"].fillna(global_base)

    # High-spend bucket model
    high_th = float(train["消耗金额"].quantile(0.8))
    train["is_high"] = (train["消耗金额"] >= high_th).astype(int)
    test["is_high"] = (test["消耗金额"] >= high_th).astype(int)

    train["residual"] = train["target_t1_spend"] - train["baseline"]
    pred_res = np.zeros(len(test), dtype=float)
    for flag in [0, 1]:
        tr = train[train["is_high"] == flag]
        te_idx = test.index[test["is_high"] == flag]
        if tr.empty or len(te_idx) == 0:
            continue
        model = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(tr[feats], tr["residual"])
        pred_res[test.index.get_indexer(te_idx)] = model.predict(test.loc[te_idx, feats])

    pred = np.clip(test["baseline"].values + pred_res, 0.0, None)
    return pd.DataFrame(
        {
            "日期": test["日期"].values,
            "应用ID": test["应用ID"].values,
            "y_true": test["target_t1_spend"].values,
            "y_pred": pred,
            "model": "BaselineResidual",
        }
    )


def _two_stage_regime_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    train = train_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    q1 = float(train["target_t1_spend"].quantile(0.4))
    q2 = float(train["target_t1_spend"].quantile(0.8))

    def bucket(v: float) -> int:
        if v <= q1:
            return 0
        if v <= q2:
            return 1
        return 2

    train["regime"] = train["target_t1_spend"].map(bucket)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train[feats], train["regime"])
    reg_pred = clf.predict(test[feats])

    # bucket-specific regressors on log spend
    models = {}
    for b in [0, 1, 2]:
        tr = train[train["regime"] == b]
        if len(tr) < 30:
            continue
        m = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        m.fit(tr[feats], np.log1p(tr["target_t1_spend"].clip(lower=0.0)))
        models[b] = m

    global_model = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    global_model.fit(train[feats], np.log1p(train["target_t1_spend"].clip(lower=0.0)))

    preds = []
    x_test = test[feats]
    for i, b in enumerate(reg_pred):
        m = models.get(int(b), global_model)
        pred_log = m.predict(x_test.iloc[[i]])[0]
        preds.append(float(np.expm1(pred_log)))
    pred = np.clip(np.array(preds, dtype=float), 0.0, None)
    return pd.DataFrame(
        {
            "日期": test["日期"].values,
            "应用ID": test["应用ID"].values,
            "y_true": test["target_t1_spend"].values,
            "y_pred": pred,
            "model": "TwoStageRegime",
        }
    )


def _run(
    df: pd.DataFrame,
    min_train_days: int,
    alpha: float,
    use_log_target: bool,
    feature_set: str = "base",
    min_target_spend_train: float = 0.0,
    min_target_spend_eval: float = 0.0,
) -> List[ModelResult]:
    feats = _feature_cols(feature_set)
    models = ["RandomForest", "GBDT", "ExtraTrees", "LightGBM"]
    preds_by_model: Dict[str, List[pd.DataFrame]] = {"EWMA": [], "BaselineResidual": [], "TwoStageRegime": []}
    for m in models:
        preds_by_model[m] = []
        if use_log_target:
            preds_by_model[f"{m}_log"] = []

    # 与 roi_d1 分流逻辑对齐：按 split_value 分桶后分别进行 walk-forward
    for _, bucket in df.groupby("split_value"):
        bucket = bucket.sort_values(["应用ID", "日期"]).copy()
        app_dummies = pd.get_dummies(bucket["应用ID"].astype(str), prefix="app")
        model_df = pd.concat([bucket.reset_index(drop=True), app_dummies.reset_index(drop=True)], axis=1)
        feats_all = feats + list(app_dummies.columns)
        dates = sorted(model_df["日期"].dropna().unique())
        for cutoff in dates[:-1]:
            train = model_df[model_df["日期"] <= cutoff].copy()
            test = model_df[model_df["日期"] == (cutoff + np.timedelta64(1, "D"))].copy()
            if min_target_spend_train > 0:
                train = train[train["target_t1_spend"] >= min_target_spend_train].copy()
            if min_target_spend_eval > 0:
                test = test[test["target_t1_spend"] >= min_target_spend_eval].copy()
            if test.empty or train["日期"].nunique() < min_train_days:
                continue
            ew = _ewma_spend(train, test, alpha)
            if not ew.empty:
                preds_by_model["EWMA"].append(ew)
            br = _baseline_residual_predict(train, test, feats_all)
            if not br.empty:
                preds_by_model["BaselineResidual"].append(br)
            ts = _two_stage_regime_predict(train, test, feats_all)
            if not ts.empty:
                preds_by_model["TwoStageRegime"].append(ts)
            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = {}
                for m in models:
                    futures[ex.submit(_tree_predict, m, train, test, feats_all, False)] = m
                    if use_log_target:
                        futures[ex.submit(_tree_predict, m, train, test, feats_all, True)] = f"{m}_log"
                for fut in as_completed(futures):
                    name = futures[fut]
                    pred_df = fut.result()
                    if not pred_df.empty:
                        preds_by_model[name].append(pred_df)

    results: List[ModelResult] = []
    for name, frames in preds_by_model.items():
        if not frames:
            continue
        pred_df = pd.concat(frames, ignore_index=True).dropna(subset=["y_true", "y_pred"])
        if pred_df.empty:
            continue
        metrics = _evaluate(pred_df["y_true"].values, pred_df["y_pred"].values)
        results.append(ModelResult(model_name=name, metrics=metrics, prediction_df=pred_df))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="并行预测 T+1 spend，并输出与 roi_d1 同口径百分比误差")
    parser.add_argument("--input", default="daily_20260421_120112.csv")
    parser.add_argument("--output-dir", default="outputs/model_parallel_spend_t1")
    parser.add_argument("--min-train-days", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--use-log-target", action="store_true", help="增加 log1p(spend) 分支")
    parser.add_argument(
        "--split-col",
        default="推广流量名称",
        help="按该字段分流后训练 spend（例如：推广流量名称 或 流量场景名称）",
    )
    parser.add_argument(
        "--clean-group-cols",
        default=",".join(DEFAULT_CLEAN_GROUP_COLS),
        help="细粒度清洗字段（逗号分隔），仅用于清洗全时段零消耗实体",
    )
    parser.add_argument(
        "--product-key-cols",
        default=",".join(DEFAULT_PRODUCT_KEY_COLS),
        help="训练实体字段（逗号分隔），在清洗后按该维度聚合",
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
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_group_cols = _normalize_cols(str(args.clean_group_cols).split(","))
    product_key_cols = _normalize_cols(str(args.product_key_cols).split(","))
    df, clean_meta = _build_app_daily(
        args.input,
        split_col=args.split_col,
        clean_group_cols=clean_group_cols,
        product_key_cols=product_key_cols,
    )
    results = _run(
        df,
        min_train_days=args.min_train_days,
        alpha=args.alpha,
        use_log_target=args.use_log_target,
        feature_set=args.feature_set,
        min_target_spend_train=float(args.min_target_spend_train),
        min_target_spend_eval=float(args.min_target_spend_eval),
    )
    if not results:
        raise ValueError("未得到 spend 预测结果，请检查数据和参数。")

    metric_rows = []
    for r in results:
        metric_rows.append({"model": r.model_name, **r.metrics, "samples": int(len(r.prediction_df))})
        r.prediction_df.to_csv(out_dir / f"predictions_{_safe_model_filename(r.model_name)}.csv", index=False, encoding="utf-8-sig")
    metric_df = pd.DataFrame(metric_rows).sort_values("mape")
    metric_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    reconcile_report = None
    if args.reconcile_with_app_baseline:
        if not args.reconcile_app_prediction_csv:
            raise ValueError("启用 --reconcile-with-app-baseline 时必须提供 --reconcile-app-prediction-csv。")
        split_file = out_dir / f"predictions_{_safe_model_filename(args.reconcile_model)}.csv"
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
        "split_col": args.split_col,
        "clean_group_cols": clean_group_cols,
        "product_key_cols": product_key_cols,
        "clean_meta": clean_meta,
        "min_target_spend_train": float(args.min_target_spend_train),
        "min_target_spend_eval": float(args.min_target_spend_eval),
        "feature_set": args.feature_set,
        "feature_count": len(_feature_cols(args.feature_set)),
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
            "predictions": [str(out_dir / f"predictions_{_safe_model_filename(r.model_name)}.csv") for r in results],
            "predictions_reconciled": str(out_dir / "predictions_reconciled.csv") if reconcile_report else None,
            "reconcile_report": str(out_dir / "reconcile_report.json") if reconcile_report else None,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
