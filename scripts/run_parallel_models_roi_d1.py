import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.config import settings
from app.core.calendar import CalendarService
from app.prediction_artifacts import validate_merged_daily_csv

CALENDAR = CalendarService()

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


@dataclass
class ModelResult:
    model_name: str
    metrics: Dict[str, float]
    prediction_df: pd.DataFrame


def _to_log1p(y: pd.Series) -> pd.Series:
    return np.log1p(np.clip(y, a_min=0.0, a_max=None))


def _from_log1p(y: np.ndarray) -> np.ndarray:
    return np.expm1(y)


def _build_model(model_name: str, params: Optional[Dict[str, object]] = None) -> Optional[object]:
    p = params or {}
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=int(p.get("n_estimators", 300)),
            max_depth=int(p.get("max_depth", 8)),
            min_samples_leaf=int(p.get("min_samples_leaf", 5)),
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "GBDT":
        return GradientBoostingRegressor(
            n_estimators=int(p.get("n_estimators", 250)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            max_depth=int(p.get("max_depth", 3)),
            random_state=42,
        )
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=int(p.get("n_estimators", 400)),
            max_depth=int(p.get("max_depth", 10)),
            min_samples_leaf=int(p.get("min_samples_leaf", 4)),
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "HistGB":
        return HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
    if model_name == "LightGBM":
        if LGBMRegressor is None:
            return None
        return LGBMRegressor(
            n_estimators=int(p.get("n_estimators", 500)),
            learning_rate=float(p.get("learning_rate", 0.03)),
            num_leaves=int(p.get("num_leaves", 31)),
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    if model_name == "XGBoost":
        if XGBRegressor is None:
            return None
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
    return None


def _is_holiday(d: date) -> bool:
    return CALENDAR.is_holiday(d)


def _is_rest_day(d: date) -> int:
    return int(CALENDAR.is_rest_day(d))


def _is_pre_holiday(d: date) -> int:
    return 1 if _is_holiday(d + timedelta(days=1)) else 0


def _is_post_holiday(d: date) -> int:
    return 1 if _is_holiday(d - timedelta(days=1)) else 0


def _is_summer_winter_break(d: date) -> int:
    return 1 if d.month in (1, 2, 7, 8) else 0


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
    """为每个应用-日期计算子组构成特征（不增加行数）。"""
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


def _build_app_daily(
    input_csv: str,
    min_spend_train: float = 2.0,
    min_app_history_days: int = 0,
) -> pd.DataFrame:
    from app.data_cleaning import clean, to_app_daily as _to_app_daily, validate_columns as _validate_cols

    raw = clean(Path(input_csv))

    # 子组构成特征 — 在聚合前从 raw 计算
    comp_df = _build_composition_features(raw)
    app_day = _to_app_daily(raw)
    app_day = app_day.merge(comp_df, on=["应用ID", "日期"], how="left")
    for c in comp_df.columns:
        if c not in ("应用ID", "日期") and c in app_day.columns:
            app_day[c] = app_day[c].fillna(0)

    _validate_cols(app_day)

    # 清洗：过滤低消耗样本（ROI 噪声极大）
    if min_spend_train > 0:
        app_day = app_day[app_day["消耗金额"] >= min_spend_train].copy()

    # 清洗：过滤历史天数不足的应用（ROI 不稳定）
    if min_app_history_days <= 0:
        min_app_history_days = max(settings.app_last_day_min_train_days, 1)
    day_counts = app_day.groupby("应用ID")["日期"].nunique()
    keep_apps = day_counts[day_counts >= min_app_history_days].index
    apps_before = int(app_day["应用ID"].nunique())
    app_day = app_day[app_day["应用ID"].isin(keep_apps)].copy()
    apps_after = int(app_day["应用ID"].nunique())
    if apps_before > apps_after:
        print(f"[ROI] app filter: {apps_before} -> {apps_after} (min {min_app_history_days} days)")

    app_day["roi_d1"] = np.where(app_day["消耗金额"] > 0, app_day["首日广告收入"] / app_day["消耗金额"], np.nan)
    app_day["act_per_spend"] = np.where(
        app_day["消耗金额"] > 0, app_day["激活人数(快应用新增用户数)"] / app_day["消耗金额"], np.nan
    )
    app_day["rev_per_act_d1"] = np.where(
        app_day["激活人数(快应用新增用户数)"] > 0,
        app_day["首日广告收入"] / app_day["激活人数(快应用新增用户数)"],
        np.nan,
    )
    app_day["ctr"] = np.where(app_day["曝光量"] > 0, app_day["点击量"] / app_day["曝光量"], 0.0)
    app_day["cvr_dl"] = np.where(app_day["点击量"] > 0, app_day["下载量"] / app_day["点击量"], 0.0)
    app_day["cvr_act"] = np.where(app_day["下载量"] > 0, app_day["激活人数(快应用新增用户数)"] / app_day["下载量"], 0.0)
    app_day["dow"] = app_day["日期"].dt.weekday
    app_day["is_weekend"] = (app_day["dow"] >= 5).astype(int)
    # 业务口径：周五+周六常为特殊流量日
    app_day["is_fri_sat"] = app_day["dow"].isin([4, 5]).astype(int)
    app_day["dom"] = app_day["日期"].dt.day
    app_day["month"] = app_day["日期"].dt.month
    app_day["is_holiday"] = app_day["日期"].dt.date.map(_is_holiday).astype(int)
    app_day["is_rest_day"] = app_day["日期"].dt.date.map(_is_rest_day).astype(int)
    app_day["pre_holiday"] = app_day["日期"].dt.date.map(_is_pre_holiday).astype(int)
    app_day["post_holiday"] = app_day["日期"].dt.date.map(_is_post_holiday).astype(int)
    app_day["summer_winter_break"] = app_day["日期"].dt.date.map(_is_summer_winter_break).astype(int)

    # T+1 目标日节假日特征（与 spends 脚本口径一致）
    target_dates = app_day["日期"].dt.date.map(lambda d: d + timedelta(days=1))
    app_day["target_dow"] = target_dates.map(lambda d: d.weekday()).astype(int)
    app_day["target_is_weekend"] = target_dates.map(lambda d: d.weekday() >= 5).astype(int)
    app_day["target_is_holiday"] = target_dates.map(_is_holiday).astype(int)
    app_day["target_holiday_id"] = target_dates.map(_holiday_id).astype(int)
    app_day["target_holiday_seq_index"] = target_dates.map(_holiday_seq_index).astype(int)
    app_day["target_holiday_days_remaining"] = target_dates.map(_holiday_days_remaining).astype(int)
    app_day["target_holiday_window_len"] = target_dates.map(_holiday_window_len).astype(int)
    app_day["target_is_last_holiday_day"] = target_dates.map(_is_last_holiday_day).astype(int)
    app_day["target_days_to_next_holiday"] = target_dates.map(_days_to_next_holiday).clip(upper=30)
    app_day["target_days_since_prev_holiday"] = target_dates.map(_days_since_prev_holiday).clip(upper=30)
    app_day["target_is_pre_holiday_3d"] = app_day["target_days_to_next_holiday"].between(1, 3).astype(int)
    app_day["target_is_post_holiday_1d"] = (app_day["target_days_since_prev_holiday"] == 1).astype(int)
    app_day["target_is_post_holiday_3d"] = app_day["target_days_since_prev_holiday"].between(1, 3).astype(int)
    app_day["target_is_first_workday_after_holiday"] = target_dates.map(_is_first_workday_after_holiday).astype(int)

    # 机制特征：变化率 + 稳定度
    g = app_day.groupby("应用ID", group_keys=False)
    app_day["spend_lag_1_raw"] = g["消耗金额"].shift(1)
    app_day["spend_lag_3_mean_raw"] = g["消耗金额"].rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    app_day["spend_ratio_1d"] = np.where(
        app_day["spend_lag_1_raw"] > 1e-8, app_day["消耗金额"] / app_day["spend_lag_1_raw"] - 1.0, np.nan
    )
    app_day["spend_ratio_3d"] = np.where(
        app_day["spend_lag_3_mean_raw"] > 1e-8, app_day["消耗金额"] / app_day["spend_lag_3_mean_raw"] - 1.0, np.nan
    )

    app_day["roi_d1_std_7"] = g["roi_d1"].rolling(7, min_periods=3).std().reset_index(level=0, drop=True)
    roi_q75 = g["roi_d1"].rolling(7, min_periods=3).quantile(0.75).reset_index(level=0, drop=True)
    roi_q25 = g["roi_d1"].rolling(7, min_periods=3).quantile(0.25).reset_index(level=0, drop=True)
    app_day["roi_d1_iqr_7"] = roi_q75 - roi_q25
    app_day["roi_d1_range_7"] = (
        g["roi_d1"].rolling(7, min_periods=3).max().reset_index(level=0, drop=True)
        - g["roi_d1"].rolling(7, min_periods=3).min().reset_index(level=0, drop=True)
    )

    for c in ["act_per_spend", "rev_per_act_d1", "spend_ratio_1d", "spend_ratio_3d", "roi_d1_std_7", "roi_d1_iqr_7", "roi_d1_range_7"]:
        app_day[c] = app_day[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    app_day = app_day.drop(columns=["spend_lag_1_raw", "spend_lag_3_mean_raw"])
    return app_day


def _add_lags(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"roi_d1_lag_{lag}"] = out.groupby("应用ID")["roi_d1"].shift(lag)
        out[f"spend_lag_{lag}"] = out.groupby("应用ID")["消耗金额"].shift(lag)
        out[f"act_per_spend_lag_{lag}"] = out.groupby("应用ID")["act_per_spend"].shift(lag)
        out[f"rev_per_act_d1_lag_{lag}"] = out.groupby("应用ID")["rev_per_act_d1"].shift(lag)
    out["target_t1_roi_d1"] = out.groupby("应用ID")["roi_d1"].shift(-1)
    out["target_t1_act_per_spend"] = out.groupby("应用ID")["act_per_spend"].shift(-1)
    out["target_t1_rev_per_act_d1"] = out.groupby("应用ID")["rev_per_act_d1"].shift(-1)
    return out


def _get_base_feature_cols() -> List[str]:
    return [
        "消耗金额",
        "act_per_spend",
        "rev_per_act_d1",
        "ctr",
        "cvr_dl",
        "cvr_act",
        "spend_ratio_1d",
        "spend_ratio_3d",
        "roi_d1_std_7",
        "roi_d1_iqr_7",
        "roi_d1_range_7",
        "dow",
        "is_weekend",
        "is_fri_sat",
        "is_holiday",
        "pre_holiday",
        "post_holiday",
        "summer_winter_break",
        "dom",
        "month",
        "roi_d1_lag_1",
        "roi_d1_lag_2",
        "roi_d1_lag_3",
        "roi_d1_lag_7",
        "spend_lag_1",
        "spend_lag_2",
        "spend_lag_3",
        "spend_lag_7",
        "act_per_spend_lag_1",
        "act_per_spend_lag_2",
        "act_per_spend_lag_3",
        "act_per_spend_lag_7",
        "rev_per_act_d1_lag_1",
        "rev_per_act_d1_lag_2",
        "rev_per_act_d1_lag_3",
        "rev_per_act_d1_lag_7",
        # 子组构成特征
        "traffic_n_unique", "traffic_top1_pct", "traffic_hhi",
        "scene_n_unique", "scene_top1_pct", "scene_hhi",
        "creative_n_unique", "creative_top1_pct", "creative_hhi",
        "billing_n_unique", "billing_top1_pct", "billing_hhi",
        # 休息日特征（CalendarService + chinese_calendar）
        "is_rest_day",
        # T+1 目标日节假日特征
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
    ]


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > 1e-8
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if np.any(mask) else np.nan
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _ewma_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    preds = []
    for app_id, g_test in test_df.groupby("应用ID"):
        g_train = train_df[train_df["应用ID"] == app_id].sort_values("日期")
        vals = g_train["roi_d1"].dropna().values
        if len(vals) == 0:
            pred = np.nan
        else:
            s = vals[0]
            for v in vals[1:]:
                s = alpha * v + (1 - alpha) * s
            pred = s
        for _, row in g_test.iterrows():
            preds.append(
                {
                    "日期": row["日期"],
                    "应用ID": row["应用ID"],
                    "y_true": row["target_t1_roi_d1"],
                    "y_pred": pred,
                    "model": "EWMA",
                }
            )
    return pd.DataFrame(preds)


def _tree_predict(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_hard_sample_weight: bool = False,
    model_params: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    train = train_df.dropna(subset=feature_cols + ["target_t1_roi_d1"]).copy()
    test = test_df.dropna(subset=feature_cols + ["target_t1_roi_d1"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    model = _build_model(model_name, model_params)
    if model is None:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    x_train = train[feature_cols]
    y_train = train["target_t1_roi_d1"]
    x_test = test[feature_cols]
    y_test = test["target_t1_roi_d1"]

    if use_hard_sample_weight:
        train = train.copy()
        app_scale = train.groupby("应用ID", as_index=False)["消耗金额"].median().rename(columns={"消耗金额": "app_median_spend"})
        train = train.merge(app_scale, on="应用ID", how="left")
        app_rank = train["app_median_spend"].fillna(0.0).rank(method="first")
        spend_rank = train["消耗金额"].fillna(0.0).rank(method="first")
        train["app_scale_bucket"] = pd.qcut(app_rank, q=4, labels=["Q1_small", "Q2", "Q3", "Q4_large"])
        train["spend_bucket"] = pd.qcut(spend_rank, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        sample_weight = np.ones(len(train), dtype=float)
        sample_weight += (train["spend_bucket"].astype(str) == "Q1_low").astype(float) * 0.6
        sample_weight += (train["app_scale_bucket"].astype(str) == "Q2").astype(float) * 0.5
        x_train = train[feature_cols]
        y_train = train["target_t1_roi_d1"]
        try:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        except TypeError:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return pd.DataFrame(
        {
            "日期": test["日期"].values,
            "应用ID": test["应用ID"].values,
            "y_true": y_test.values,
            "y_pred": pred,
            "model": model_name,
        }
    )


def _tree_predict_split(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_log_target: bool = False,
    model_params: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    train = train_df.dropna(subset=feature_cols + ["target_t1_roi_d1"]).copy()
    test = test_df.dropna(subset=feature_cols + ["target_t1_roi_d1"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    spend_threshold = float(train["消耗金额"].quantile(0.25))
    train_low = train[train["消耗金额"] <= spend_threshold].copy()
    train_high = train[train["消耗金额"] > spend_threshold].copy()
    test_low = test[test["消耗金额"] <= spend_threshold].copy()
    test_high = test[test["消耗金额"] > spend_threshold].copy()

    pred_parts = []
    for part_name, tr, te in [("low", train_low, test_low), ("high", train_high, test_high)]:
        if tr.empty or te.empty:
            continue
        model = _build_model(model_name, model_params)
        if model is None:
            continue
        x_train = tr[feature_cols]
        y_train = tr["target_t1_roi_d1"]
        if use_log_target:
            y_train = _to_log1p(y_train)
        model.fit(x_train, y_train)
        pred = model.predict(te[feature_cols])
        if use_log_target:
            pred = np.clip(_from_log1p(pred), a_min=0.0, a_max=None)
        pred_parts.append(
            pd.DataFrame(
                {
                    "日期": te["日期"].values,
                    "应用ID": te["应用ID"].values,
                    "y_true": te["target_t1_roi_d1"].values,
                    "y_pred": pred,
                    "model": f"{model_name}_{'split_log' if use_log_target else 'split'}",
                    "segment": part_name,
                }
            )
        )
    if not pred_parts:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    out = pd.concat(pred_parts, ignore_index=True)
    return out.drop(columns=["segment"])


def _ewma_predict_log(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    preds = []
    for app_id, g_test in test_df.groupby("应用ID"):
        g_train = train_df[train_df["应用ID"] == app_id].sort_values("日期")
        vals = g_train["roi_d1"].dropna().values
        if len(vals) == 0:
            pred = np.nan
        else:
            lv = np.log1p(np.clip(vals, a_min=0.0, a_max=None))
            s = lv[0]
            for v in lv[1:]:
                s = alpha * v + (1 - alpha) * s
            pred = float(np.expm1(s))
        for _, row in g_test.iterrows():
            preds.append(
                {
                    "日期": row["日期"],
                    "应用ID": row["应用ID"],
                    "y_true": row["target_t1_roi_d1"],
                    "y_pred": pred,
                    "model": "EWMA_log",
                }
            )
    return pd.DataFrame(preds)


def _tree_predict_log(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_hard_sample_weight: bool = False,
    model_params: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    train = train_df.dropna(subset=feature_cols + ["target_t1_roi_d1"]).copy()
    test = test_df.dropna(subset=feature_cols + ["target_t1_roi_d1"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    model = _build_model(model_name, model_params)
    if model is None:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    x_train = train[feature_cols]
    y_train = _to_log1p(train["target_t1_roi_d1"])
    x_test = test[feature_cols]
    y_test = test["target_t1_roi_d1"]

    if use_hard_sample_weight:
        train = train.copy()
        app_scale = train.groupby("应用ID", as_index=False)["消耗金额"].median().rename(columns={"消耗金额": "app_median_spend"})
        train = train.merge(app_scale, on="应用ID", how="left")
        app_rank = train["app_median_spend"].fillna(0.0).rank(method="first")
        spend_rank = train["消耗金额"].fillna(0.0).rank(method="first")
        train["app_scale_bucket"] = pd.qcut(app_rank, q=4, labels=["Q1_small", "Q2", "Q3", "Q4_large"])
        train["spend_bucket"] = pd.qcut(spend_rank, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        sample_weight = np.ones(len(train), dtype=float)
        sample_weight += (train["spend_bucket"].astype(str) == "Q1_low").astype(float) * 0.6
        sample_weight += (train["app_scale_bucket"].astype(str) == "Q2").astype(float) * 0.5
        x_train = train[feature_cols]
        y_train = _to_log1p(train["target_t1_roi_d1"])
        try:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        except TypeError:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)
    pred_log = model.predict(x_test)
    pred = _from_log1p(pred_log)

    # Duan's smearing: log-transform 还原偏差校正
    if len(train) > 10:
        train_pred_log = model.predict(x_train)
        train_residuals_log = y_train.to_numpy() - train_pred_log
        smearing = float(np.mean(np.exp(train_residuals_log)))
        # smearing 应接近 1.0；偏离过大说明 log 空间拟合差，不强制校正
        if 0.8 <= smearing <= 1.25:
            pred = pred * smearing

    pred = np.clip(pred, a_min=0.0, a_max=None)
    return pd.DataFrame(
        {
            "日期": test["日期"].values,
            "应用ID": test["应用ID"].values,
            "y_true": y_test.values,
            "y_pred": pred,
            "model": f"{model_name}_log",
        }
    )


def _tree_predict_decomp(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_log_component: bool = False,
    model_params: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    need_cols = feature_cols + ["target_t1_roi_d1", "target_t1_act_per_spend", "target_t1_rev_per_act_d1"]
    train = train_df.dropna(subset=need_cols).copy()
    test = test_df.dropna(subset=need_cols).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    model_a = _build_model(model_name, model_params)
    model_r = _build_model(model_name, model_params)
    if model_a is None or model_r is None:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    x_train = train[feature_cols]
    x_test = test[feature_cols]
    y_a = train["target_t1_act_per_spend"].clip(lower=0.0)
    y_r = train["target_t1_rev_per_act_d1"].clip(lower=0.0)

    if use_log_component:
        y_a = _to_log1p(y_a)
        y_r = _to_log1p(y_r)

    model_a.fit(x_train, y_a)
    model_r.fit(x_train, y_r)
    pred_a = model_a.predict(x_test)
    pred_r = model_r.predict(x_test)
    if use_log_component:
        pred_a = _from_log1p(pred_a)
        pred_r = _from_log1p(pred_r)

    pred = np.clip(pred_a, a_min=0.0, a_max=None) * np.clip(pred_r, a_min=0.0, a_max=None)
    return pd.DataFrame(
        {
            "日期": test["日期"].values,
            "应用ID": test["应用ID"].values,
            "y_true": test["target_t1_roi_d1"].values,
            "y_pred": pred,
            "model": f"{model_name}_{'decomp_log' if use_log_component else 'decomp'}",
        }
    )


def _date_bucket_label(df: pd.DataFrame) -> pd.Series:
    return np.where(df["is_holiday"] == 1, "holiday", np.where(df["is_fri_sat"] == 1, "fri_sat", "normal"))


def _tree_predict_date_bucket(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_log_target: bool = False,
    model_params: Optional[Dict[str, object]] = None,
    min_bucket_samples: int = 40,
) -> pd.DataFrame:
    train = train_df.dropna(subset=feature_cols + ["target_t1_roi_d1"]).copy()
    test = test_df.dropna(subset=feature_cols + ["target_t1_roi_d1"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])

    train["date_bucket"] = _date_bucket_label(train)
    test["date_bucket"] = _date_bucket_label(test)

    global_model = _build_model(model_name, model_params)
    if global_model is None:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    y_train_global = train["target_t1_roi_d1"]
    if use_log_target:
        y_train_global = _to_log1p(y_train_global)
    global_model.fit(train[feature_cols], y_train_global)

    bucket_models: Dict[str, object] = {}
    for b, g in train.groupby("date_bucket"):
        if len(g) < min_bucket_samples:
            continue
        m = _build_model(model_name, model_params)
        if m is None:
            continue
        yb = g["target_t1_roi_d1"]
        if use_log_target:
            yb = _to_log1p(yb)
        m.fit(g[feature_cols], yb)
        bucket_models[str(b)] = m

    preds = []
    for b, g in test.groupby("date_bucket"):
        model = bucket_models.get(str(b), global_model)
        pred = model.predict(g[feature_cols])
        if use_log_target:
            pred = np.clip(_from_log1p(pred), a_min=0.0, a_max=None)
        preds.append(
            pd.DataFrame(
                {
                    "日期": g["日期"].values,
                    "应用ID": g["应用ID"].values,
                    "y_true": g["target_t1_roi_d1"].values,
                    "y_pred": pred,
                    "model": f"{model_name}_{'date_bucket_log' if use_log_target else 'date_bucket'}",
                }
            )
        )
    if not preds:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    return pd.concat(preds, ignore_index=True)


def _build_error_bucket_summary(pred_df: pd.DataFrame, app_daily_df: pd.DataFrame) -> pd.DataFrame:
    merged = pred_df.merge(
        app_daily_df[["日期", "应用ID", "消耗金额", "is_holiday", "is_fri_sat"]],
        on=["日期", "应用ID"],
        how="left",
    )
    merged["abs_err"] = (merged["y_true"] - merged["y_pred"]).abs()
    merged["sq_err"] = (merged["y_true"] - merged["y_pred"]) ** 2

    app_scale = app_daily_df.groupby("应用ID", as_index=False)["消耗金额"].median().rename(columns={"消耗金额": "app_median_spend"})
    merged = merged.merge(app_scale, on="应用ID", how="left")

    merged["spend_bucket"] = pd.qcut(
        merged["消耗金额"].fillna(0.0).rank(method="first"),
        q=4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    )
    merged["app_scale_bucket"] = pd.qcut(
        merged["app_median_spend"].fillna(0.0).rank(method="first"),
        q=4,
        labels=["Q1_small", "Q2", "Q3", "Q4_large"],
    )

    rows = []
    bucket_dims = ["is_holiday", "is_fri_sat", "spend_bucket", "app_scale_bucket"]
    for dim in bucket_dims:
        for val, g in merged.groupby(dim, dropna=False):
            if g.empty:
                continue
            rows.append(
                {
                    "bucket_dim": dim,
                    "bucket_value": str(val),
                    "samples": int(len(g)),
                    "mae": float(g["abs_err"].mean()),
                    "rmse": float(np.sqrt(g["sq_err"].mean())),
                    "mape": float(np.mean(np.abs((g["y_true"] - g["y_pred"]) / np.clip(g["y_true"], 1e-8, None)))),
                    "mean_y_true": float(g["y_true"].mean()),
                    "mean_y_pred": float(g["y_pred"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["bucket_dim", "rmse"], ascending=[True, False])


def _hard_sample_weight_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    app_scale = out.groupby("应用ID", as_index=False)["消耗金额"].median().rename(columns={"消耗金额": "app_median_spend"})
    out = out.merge(app_scale, on="应用ID", how="left")
    app_rank = out["app_median_spend"].fillna(0.0).rank(method="first")
    spend_rank = out["消耗金额"].fillna(0.0).rank(method="first")
    out["app_scale_bucket"] = pd.qcut(app_rank, q=4, labels=["Q1_small", "Q2", "Q3", "Q4_large"])
    out["spend_bucket"] = pd.qcut(spend_rank, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    out["hard_weight"] = 1.0
    out.loc[out["spend_bucket"].astype(str) == "Q1_low", "hard_weight"] += 1.0
    out.loc[out["app_scale_bucket"].astype(str) == "Q2", "hard_weight"] += 1.0
    return out


def _weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    err = (y_true - y_pred) ** 2
    return float(np.sqrt(np.sum(w * err) / np.sum(w)))


def _tune_params_for_hard_buckets(
    model_name: str,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Optional[Dict[str, object]]:
    if model_name not in {"ExtraTrees", "GBDT"}:
        return None
    work = df.dropna(subset=feature_cols + ["target_t1_roi_d1"]).copy().sort_values("日期")
    if work["日期"].nunique() < 30:
        return None
    dates = sorted(work["日期"].unique())
    split_idx = int(len(dates) * 0.75)
    split_idx = min(max(split_idx, 20), len(dates) - 5)
    cutoff = dates[split_idx]
    train = work[work["日期"] <= cutoff].copy()
    valid = work[work["日期"] > cutoff].copy()
    if train.empty or valid.empty:
        return None
    valid = _hard_sample_weight_frame(valid)

    if model_name == "ExtraTrees":
        grid = [
            {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 4},
            {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 4},
            {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 3},
            {"n_estimators": 700, "max_depth": 10, "min_samples_leaf": 2},
        ]
    else:
        grid = [
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
            {"n_estimators": 300, "learning_rate": 0.04, "max_depth": 3},
            {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4},
            {"n_estimators": 450, "learning_rate": 0.03, "max_depth": 3},
        ]

    best_params = None
    best_score = float("inf")
    for p in grid:
        model = _build_model(model_name, p)
        if model is None:
            continue
        model.fit(train[feature_cols], train["target_t1_roi_d1"])
        pred = model.predict(valid[feature_cols])
        score = _weighted_rmse(valid["target_t1_roi_d1"].values, pred, valid["hard_weight"].values)
        if score < best_score:
            best_score = score
            best_params = p
    return best_params


def _run_backtest_parallel(
    df: pd.DataFrame,
    alpha: float,
    min_train_days: int = 20,
    eval_recent_days: Optional[int] = None,
    include_log_target: bool = False,
    include_hard_weight_models: bool = False,
    include_split_models: bool = False,
    include_decomp_models: bool = False,
    selected_tree_models: Optional[List[str]] = None,
    tune_hard_buckets: bool = True,
    retune_frequency_days: int = 7,
    include_date_bucket_models: bool = False,
) -> List[ModelResult]:
    all_dates = sorted(df["日期"].dropna().unique())
    candidate_cutoffs = all_dates[:-1]
    if eval_recent_days is not None and eval_recent_days > 0 and len(candidate_cutoffs) > eval_recent_days:
        candidate_cutoffs = candidate_cutoffs[-eval_recent_days:]
    tree_model_names = selected_tree_models or ["RandomForest", "GBDT", "ExtraTrees", "HistGB", "LightGBM", "XGBoost"]
    model_specs: List[Tuple[str, str, Optional[Dict[str, object]]]] = [(m, m, None) for m in tree_model_names]
    base_feature_cols = _get_base_feature_cols()
    preds_by_model = {"EWMA": []}
    for m, _, _ in model_specs:
        preds_by_model[m] = []
    if tune_hard_buckets:
        for m in tree_model_names:
            preds_by_model[f"{m}_hard_tuned"] = []
    if include_hard_weight_models:
        for m, _, _ in model_specs:
            preds_by_model[f"{m}_hard"] = []
        if tune_hard_buckets:
            for m in tree_model_names:
                preds_by_model[f"{m}_hard_tuned_hard"] = []
    if include_split_models:
        for m in tree_model_names:
            preds_by_model[f"{m}_split"] = []
        if tune_hard_buckets:
            for m in tree_model_names:
                preds_by_model[f"{m}_hard_tuned_split"] = []
    if include_log_target:
        preds_by_model["EWMA_log"] = []
        for m, _, _ in model_specs:
            preds_by_model[f"{m}_log"] = []
        if tune_hard_buckets:
            for m in tree_model_names:
                preds_by_model[f"{m}_hard_tuned_log"] = []
        if include_hard_weight_models:
            for m, _, _ in model_specs:
                preds_by_model[f"{m}_log_hard"] = []
            if tune_hard_buckets:
                for m in tree_model_names:
                    preds_by_model[f"{m}_hard_tuned_log_hard"] = []
        if include_split_models:
            for m, _, _ in model_specs:
                preds_by_model[f"{m}_split_log"] = []
            if tune_hard_buckets:
                for m in tree_model_names:
                    preds_by_model[f"{m}_hard_tuned_split_log"] = []
    if include_decomp_models:
        for m, _, _ in model_specs:
            preds_by_model[f"{m}_decomp"] = []
        if tune_hard_buckets:
            for m in tree_model_names:
                preds_by_model[f"{m}_hard_tuned_decomp"] = []
        if include_log_target:
            for m, _, _ in model_specs:
                preds_by_model[f"{m}_decomp_log"] = []
            if tune_hard_buckets:
                for m in tree_model_names:
                    preds_by_model[f"{m}_hard_tuned_decomp_log"] = []
    if include_date_bucket_models:
        for m, _, _ in model_specs:
            preds_by_model[f"{m}_date_bucket"] = []
        if tune_hard_buckets:
            for m in tree_model_names:
                preds_by_model[f"{m}_hard_tuned_date_bucket"] = []
        if include_log_target:
            for m, _, _ in model_specs:
                preds_by_model[f"{m}_date_bucket_log"] = []
            if tune_hard_buckets:
                for m in tree_model_names:
                    preds_by_model[f"{m}_hard_tuned_date_bucket_log"] = []

    # one-hot app id
    app_dummies = pd.get_dummies(df["应用ID"].astype(str), prefix="app")
    model_df = pd.concat([df.reset_index(drop=True), app_dummies.reset_index(drop=True)], axis=1)
    feature_cols = base_feature_cols + list(app_dummies.columns)

    tuned_params_cache: Dict[str, Dict[str, object]] = {}
    last_retune_cutoff = None
    n_total = len(candidate_cutoffs)
    for i, cutoff in enumerate(candidate_cutoffs):
        train = model_df[model_df["日期"] <= cutoff].copy()
        test = model_df[model_df["日期"] == (cutoff + np.timedelta64(1, "D"))].copy()
        if test.empty:
            print(f"  [{i+1}/{n_total}] cutoff={pd.Timestamp(cutoff).date()} skip (no test)")
            continue

        train_days = train["日期"].nunique()
        if train_days < min_train_days:
            print(f"  [{i+1}/{n_total}] cutoff={pd.Timestamp(cutoff).date()} skip (train_days={train_days} < {min_train_days})")
            continue

        print(f"  [{i+1}/{n_total}] cutoff={pd.Timestamp(cutoff).date()} train={train_days}d test_apps={test['应用ID'].nunique()}")

        if tune_hard_buckets:
            need_retune = last_retune_cutoff is None
            if not need_retune and retune_frequency_days > 0:
                days_since = int((pd.Timestamp(cutoff) - pd.Timestamp(last_retune_cutoff)).days)
                need_retune = days_since >= retune_frequency_days
            if need_retune:
                new_cache = dict(tuned_params_cache)
                for m in tree_model_names:
                    p = _tune_params_for_hard_buckets(m, train, feature_cols)
                    if p is not None:
                        new_cache[m] = p
                tuned_params_cache = new_cache
                last_retune_cutoff = cutoff

        model_specs_for_cutoff = list(model_specs)
        if tune_hard_buckets:
            for m in tree_model_names:
                model_specs_for_cutoff.append((f"{m}_hard_tuned", m, tuned_params_cache.get(m)))

        # EWMA
        ewma_pred = _ewma_predict(train_df=train, test_df=test, alpha=alpha)
        if not ewma_pred.empty:
            preds_by_model["EWMA"].append(ewma_pred)
        if include_log_target:
            ewma_log_pred = _ewma_predict_log(train_df=train, test_df=test, alpha=alpha)
            if not ewma_log_pred.empty:
                preds_by_model["EWMA_log"].append(ewma_log_pred)

        # Tree models parallel
        available_models = [(label, base, p) for label, base, p in model_specs_for_cutoff if _build_model(base, p) is not None]
        with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
            futures = {
                ex.submit(_tree_predict, base, train, test, feature_cols, False, p): label for label, base, p in available_models
            }
            for fut in as_completed(futures):
                name = futures[fut]
                pred_df = fut.result()
                if not pred_df.empty:
                    pred_df["model"] = name
                    preds_by_model[name].append(pred_df)
        if include_hard_weight_models:
            with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
                futures = {
                    ex.submit(_tree_predict, base, train, test, feature_cols, True, p): label
                    for label, base, p in available_models
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    pred_df = fut.result()
                    if not pred_df.empty:
                        pred_df["model"] = f"{name}_hard"
                        preds_by_model[f"{name}_hard"].append(pred_df)
        if include_log_target:
            with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
                futures = {
                    ex.submit(_tree_predict_log, base, train, test, feature_cols, False, p): label
                    for label, base, p in available_models
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    pred_df = fut.result()
                    if not pred_df.empty:
                        pred_df["model"] = f"{name}_log"
                        preds_by_model[f"{name}_log"].append(pred_df)
            if include_hard_weight_models:
                with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
                    futures = {
                        ex.submit(_tree_predict_log, base, train, test, feature_cols, True, p): label
                        for label, base, p in available_models
                    }
                    for fut in as_completed(futures):
                        name = futures[fut]
                        pred_df = fut.result()
                        if not pred_df.empty:
                            pred_df["model"] = f"{name}_log_hard"
                            preds_by_model[f"{name}_log_hard"].append(pred_df)
        if include_split_models:
            with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
                futures = {
                    ex.submit(_tree_predict_split, base, train, test, feature_cols, False, p): label
                    for label, base, p in available_models
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    pred_df = fut.result()
                    if not pred_df.empty:
                        pred_df["model"] = f"{name}_split"
                        preds_by_model[f"{name}_split"].append(pred_df)
            if include_log_target:
                with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
                    futures = {
                        ex.submit(_tree_predict_split, base, train, test, feature_cols, True, p): label
                        for label, base, p in available_models
                    }
                    for fut in as_completed(futures):
                        name = futures[fut]
                        pred_df = fut.result()
                        if not pred_df.empty:
                            pred_df["model"] = f"{name}_split_log"
                            preds_by_model[f"{name}_split_log"].append(pred_df)
        if include_decomp_models:
            with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
                futures = {
                    ex.submit(_tree_predict_decomp, base, train, test, feature_cols, False, p): label
                    for label, base, p in available_models
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    pred_df = fut.result()
                    if not pred_df.empty:
                        pred_df["model"] = f"{name}_decomp"
                        preds_by_model[f"{name}_decomp"].append(pred_df)
            if include_log_target:
                with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
                    futures = {
                        ex.submit(_tree_predict_decomp, base, train, test, feature_cols, True, p): label
                        for label, base, p in available_models
                    }
                    for fut in as_completed(futures):
                        name = futures[fut]
                        pred_df = fut.result()
                        if not pred_df.empty:
                            pred_df["model"] = f"{name}_decomp_log"
                            preds_by_model[f"{name}_decomp_log"].append(pred_df)
        if include_date_bucket_models:
            with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
                futures = {
                    ex.submit(_tree_predict_date_bucket, base, train, test, feature_cols, False, p): label
                    for label, base, p in available_models
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    pred_df = fut.result()
                    if not pred_df.empty:
                        pred_df["model"] = f"{name}_date_bucket"
                        preds_by_model[f"{name}_date_bucket"].append(pred_df)
            if include_log_target:
                with ThreadPoolExecutor(max_workers=max(2, len(available_models))) as ex:
                    futures = {
                        ex.submit(_tree_predict_date_bucket, base, train, test, feature_cols, True, p): label
                        for label, base, p in available_models
                    }
                    for fut in as_completed(futures):
                        name = futures[fut]
                        pred_df = fut.result()
                        if not pred_df.empty:
                            pred_df["model"] = f"{name}_date_bucket_log"
                            preds_by_model[f"{name}_date_bucket_log"].append(pred_df)

    results: List[ModelResult] = []
    for model_name, frames in preds_by_model.items():
        if not frames:
            continue
        pred_df = pd.concat(frames, ignore_index=True).dropna(subset=["y_true", "y_pred"])
        if pred_df.empty:
            continue
        metrics = _evaluate(pred_df["y_true"].values, pred_df["y_pred"].values)
        results.append(ModelResult(model_name=model_name, metrics=metrics, prediction_df=pred_df))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="并行运行 EWMA + 树模型预测 T+1 roi_d1")
    parser.add_argument("--input", default="daily_20260421_120112.csv")
    parser.add_argument("--output-dir", default="outputs/model_parallel_roi_d1")
    parser.add_argument("--alpha", type=float, default=0.4, help="EWMA alpha")
    parser.add_argument("--min-train-days", type=int, default=20)
    parser.add_argument("--use-log-target", action="store_true", help="并行增加 log1p(roi_d1) 目标训练分支")
    parser.add_argument("--use-hard-sample-weight", action="store_true", help="并行增加困难样本加权模型分支")
    parser.add_argument("--use-split-models", action="store_true", help="并行增加低消耗分层模型分支")
    parser.add_argument("--use-decomp-models", action="store_true", help="并行增加两阶段分解模型分支")
    parser.add_argument(
        "--models",
        default="RandomForest,GBDT,ExtraTrees,HistGB,LightGBM,XGBoost",
        help="逗号分隔的树模型列表，如 ExtraTrees,GBDT",
    )
    parser.add_argument(
        "--tune-hard-buckets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用困难桶自动寻参（默认开启，可用 --no-tune-hard-buckets 关闭）",
    )
    parser.add_argument("--retune-frequency-days", type=int, default=7, help="自动重寻参频率（天），默认每7天")
    parser.add_argument("--use-date-bucket-models", action="store_true", help="并行增加日期分桶模型分支")
    parser.add_argument("--min-spend-train", type=float, default=2.0, help="训练最低日消耗阈值（过滤低消耗噪声 ROI，默认 2 元）")
    parser.add_argument("--min-app-history-days", type=int, default=0, help="应用最少历史天数（如 20 可过滤稀疏应用）")
    parser.add_argument("--eval-recent-days", type=int, default=0, help="仅评估最近 N 天（0=全部），用于统一口径")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_merged_daily_csv(Path(args.input))
    app_daily = _build_app_daily(
        args.input,
        min_spend_train=args.min_spend_train,
        min_app_history_days=args.min_app_history_days,
    )
    app_daily = _add_lags(app_daily, lags=[1, 2, 3, 7])

    selected_tree_models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    eval_recent = int(args.eval_recent_days) if args.eval_recent_days > 0 else None
    results = _run_backtest_parallel(
        df=app_daily,
        alpha=args.alpha,
        min_train_days=args.min_train_days,
        eval_recent_days=eval_recent,
        include_log_target=args.use_log_target,
        include_hard_weight_models=args.use_hard_sample_weight,
        include_split_models=args.use_split_models,
        include_decomp_models=args.use_decomp_models,
        selected_tree_models=selected_tree_models,
        tune_hard_buckets=args.tune_hard_buckets,
        retune_frequency_days=args.retune_frequency_days,
        include_date_bucket_models=args.use_date_bucket_models,
    )
    if not results:
        raise ValueError("未得到有效模型结果，请检查数据覆盖与训练窗口。")

    metric_rows = []
    for r in results:
        metric_rows.append({"model": r.model_name, **r.metrics, "samples": int(len(r.prediction_df))})
        r.prediction_df.to_csv(out_dir / f"predictions_{r.model_name}.csv", index=False, encoding="utf-8-sig")

    metric_df = pd.DataFrame(metric_rows).sort_values("rmse")
    metric_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    all_pred = pd.concat([r.prediction_df for r in results], ignore_index=True)
    hard_eval = _build_error_bucket_summary(all_pred[all_pred["model"] == metric_df.iloc[0]["model"]], app_daily)
    hard_eval.to_csv(out_dir / "error_bucket_summary.csv", index=False, encoding="utf-8-sig")

    merged_all = all_pred.merge(app_daily[["日期", "应用ID", "消耗金额"]], on=["日期", "应用ID"], how="left")
    merged_all = _hard_sample_weight_frame(merged_all)
    hard_rows = []
    for m, g in merged_all.groupby("model"):
        if g.empty:
            continue
        hard_rows.append(
            {
                "model": m,
                "hard_weighted_rmse": _weighted_rmse(g["y_true"].values, g["y_pred"].values, g["hard_weight"].values),
                "hard_weighted_mae": float(np.average(np.abs(g["y_true"] - g["y_pred"]), weights=g["hard_weight"])),
                "samples": int(len(g)),
            }
        )
    hard_metric_df = pd.DataFrame(hard_rows).sort_values("hard_weighted_rmse")
    hard_metric_df.to_csv(out_dir / "metrics_hard_focus_summary.csv", index=False, encoding="utf-8-sig")
    best_model_name = str(metric_df.iloc[0]["model"])
    best_hard_model_name = str(hard_metric_df.iloc[0]["model"]) if not hard_metric_df.empty else best_model_name

    report = {
        "target": "T+1 roi_d1",
        "min_spend_train": float(args.min_spend_train),
        "min_app_history_days": int(args.min_app_history_days),
        "include_log_target": bool(args.use_log_target),
        "include_hard_sample_weight": bool(args.use_hard_sample_weight),
        "include_split_models": bool(args.use_split_models),
        "include_decomp_models": bool(args.use_decomp_models),
        "selected_tree_models": selected_tree_models,
        "tune_hard_buckets": bool(args.tune_hard_buckets),
        "retune_frequency_days": int(args.retune_frequency_days),
        "include_date_bucket_models": bool(args.use_date_bucket_models),
        "models": metric_rows,
        "best_model_by_rmse": best_model_name,
        "best_model_by_hard_focus_rmse": best_hard_model_name,
        "output_dir": str(out_dir),
        "files": {
            "metrics": str(out_dir / "metrics_summary.csv"),
            "metrics_hard_focus": str(out_dir / "metrics_hard_focus_summary.csv"),
            "error_bucket_summary": str(out_dir / "error_bucket_summary.csv"),
            "predictions": [str(out_dir / f"predictions_{r.model_name}.csv") for r in results],
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
