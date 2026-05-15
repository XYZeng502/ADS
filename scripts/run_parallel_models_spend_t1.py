import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.config import settings
from app.prediction_artifacts import validate_merged_daily_csv
from app.unified_daily import build_unified_daily

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None


@dataclass
class ModelResult:
    model_name: str
    metrics: Dict[str, float]
    prediction_df: pd.DataFrame


BASE_FEATURE_COLS = [
        "消耗金额",
        "roi_d1",
        "ctr",
        "cvr_dl",
        "cvr_act",
        "dow",
        "is_fri_sat",
        "is_holiday",
        "is_rest_day",
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
        "target_is_rest_day",
        "target_is_adjusted_workday",
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
        # 子组构成特征（流量/场景/创意/计费分布的丰富度与集中度）
        "traffic_n_unique", "traffic_top1_pct", "traffic_hhi",
        "scene_n_unique", "scene_top1_pct", "scene_hhi",
        "creative_n_unique", "creative_top1_pct", "creative_hhi",
        "billing_n_unique", "billing_top1_pct", "billing_hhi",
        # 交互特征 + app label
        "spend_x_is_rest_day",
        "spend_lag1_x_target_rest",
        "spend_lag1_x_target_holiday",
        "spend_ratio_x_target_rest",
        "app_label",
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
    "target_is_rest_day",
    "target_is_adjusted_workday",
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
    "spend_x_is_rest_day",
    "spend_lag1_x_target_rest",
    "spend_lag1_x_target_holiday",
    "spend_ratio_x_target_rest",
    "app_label",
]


def _feature_cols(feature_set: str = "base") -> List[str]:
    if feature_set == "calendar_pacing":
        return list(dict.fromkeys(BASE_FEATURE_COLS + CALENDAR_PACING_FEATURE_COLS))
    return BASE_FEATURE_COLS


def _build_model(name: str, use_gpu: bool = False) -> Optional[object]:
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
    if name == "GBDT":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.04, max_depth=3, random_state=42)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=500, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
    if name == "LightGBM":
        if LGBMRegressor is None:
            return None
        kwargs: dict = dict(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
        if use_gpu:
            kwargs["device"] = "cuda"
        return LGBMRegressor(**kwargs)
    if name == "XGBoost":
        if XGBRegressor is None:
            return None
        xgb_kwargs = dict(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
        return XGBRegressor(**xgb_kwargs)
    if name == "CatBoost":
        if CatBoostRegressor is None:
            return None
        cb_kwargs = dict(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=0,
            thread_count=4,
        )
        return CatBoostRegressor(**cb_kwargs)
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


def _tree_predict(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feats: List[str], use_log_target: bool, use_gpu: bool = False) -> pd.DataFrame:
    train = train_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    model = _build_model(name, use_gpu=use_gpu)
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
    sample_weight = np.log1p(train["消耗金额"].clip(lower=0).values)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if use_log_target:
        log_resid = y_train - model.predict(x_train)
        smearing = float(np.clip(np.exp(log_resid).mean(), 0.8, 1.25))
        pred = np.expm1(pred) * smearing
    pred = np.clip(pred, 0.0, None)
    pred = _apply_holiday_transition_calibration(test, pred)
    return pd.DataFrame({"日期": test["日期"].values, "应用ID": test["应用ID"].values, "y_true": y_test, "y_pred": pred, "model": label})


def _apply_holiday_transition_calibration(test_df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(pred, dtype=float).copy()
    current_spend = pd.to_numeric(test_df["消耗金额"], errors="coerce").fillna(0.0).to_numpy()
    roll3 = pd.to_numeric(test_df.get("spend_roll_mean_3", 0.0), errors="coerce").fillna(0.0).to_numpy()
    roll7 = pd.to_numeric(test_df.get("spend_roll_mean_7", 0.0), errors="coerce").fillna(0.0).to_numpy()
    recent_ref = np.maximum(current_spend, roll3)
    wider_ref = np.maximum(recent_ref, roll7)

    last_holiday_day = pd.to_numeric(test_df.get("target_is_last_holiday_day", 0), errors="coerce").fillna(0).to_numpy() == 1
    first_workday_after_holiday = (
        pd.to_numeric(test_df.get("target_is_first_workday_after_holiday", 0), errors="coerce").fillna(0).to_numpy() == 1
    )

    adjusted[last_holiday_day] = np.maximum(adjusted[last_holiday_day], recent_ref[last_holiday_day] * 0.80)
    adjusted[first_workday_after_holiday] = np.minimum(
        adjusted[first_workday_after_holiday],
        recent_ref[first_workday_after_holiday] * 0.45,
    )

    # Mid-holiday weekday floor: model treats holiday weekdays (Mon-Tue) as normal
    # weekdays and underestimates. Weekend holidays are already predicted well.
    # Only apply floor when the target is a holiday weekday that is NOT the last day.
    target_is_holiday = pd.to_numeric(test_df.get("target_is_holiday", 0), errors="coerce").fillna(0).to_numpy() == 1
    target_is_weekend = pd.to_numeric(test_df.get("target_is_weekend", 0), errors="coerce").fillna(0).to_numpy() == 1
    target_is_rest_day = pd.to_numeric(test_df.get("target_is_rest_day", 0), errors="coerce").fillna(0).to_numpy() == 1
    mid_holiday_weekday = target_is_holiday & ~target_is_weekend & ~last_holiday_day & ~first_workday_after_holiday
    adjusted[mid_holiday_weekday] = np.maximum(adjusted[mid_holiday_weekday], wider_ref[mid_holiday_weekday] * 0.90)

    # Post-holiday weekend calibration: use target_is_rest_day to exclude 调休补班
    days_since_holiday = pd.to_numeric(test_df.get("target_days_since_prev_holiday", 99), errors="coerce").fillna(99).to_numpy()

    # First real rest day after holiday (days 3-4): strongest retaliatory surge
    early_ph_rest = target_is_rest_day & ~target_is_holiday & (days_since_holiday >= 3) & (days_since_holiday <= 4)
    adjusted[early_ph_rest] = np.maximum(adjusted[early_ph_rest], wider_ref[early_ph_rest] * 0.90)
    adjusted[early_ph_rest] = np.minimum(adjusted[early_ph_rest], wider_ref[early_ph_rest] * 1.35)

    # Rest of post-holiday week (days 5-7): normalizing
    late_ph_rest = target_is_rest_day & ~target_is_holiday & (days_since_holiday >= 5) & (days_since_holiday <= 7)
    adjusted[late_ph_rest] = np.maximum(adjusted[late_ph_rest], wider_ref[late_ph_rest] * 0.80)
    adjusted[late_ph_rest] = np.minimum(adjusted[late_ph_rest], wider_ref[late_ph_rest] * 1.10)

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
    clf = LogisticRegression(max_iter=4000, solver="lbfgs")
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
    *,
    eval_recent_days: Optional[int] = None,
    tree_models: Optional[List[str]] = None,
    skip_baseline_two_stage: bool = False,
    use_gpu: bool = False,
) -> List[ModelResult]:
    feats = _feature_cols(feature_set)
    models = list(tree_models) if tree_models else ["RandomForest", "GBDT", "ExtraTrees", "LightGBM"]
    preds_by_model: Dict[str, List[pd.DataFrame]] = {"EWMA": [], "BaselineResidual": [], "TwoStageRegime": []}
    for m in models:
        preds_by_model[m] = []
        if use_log_target:
            preds_by_model[f"{m}_log"] = []

    _enc_app_avg_spend = "app_avg_spend"
    _enc_app_avg_roi = "app_avg_roi_d1"
    # 统一底表已无 split_value；直接使用全量 df
    model_df = df.sort_values(["应用ID", "日期"]).reset_index(drop=True).copy()
    model_df["split_value"] = "ALL"
    # 向量化 expanding-window 目标编码替代 app dummies（无泄漏）
    grp_spend = model_df.groupby("应用ID")["target_t1_spend"]
    model_df[_enc_app_avg_spend] = (
        grp_spend.cumsum().sub(model_df["target_t1_spend"])
        / grp_spend.cumcount().clip(lower=1)
    )
    model_df[_enc_app_avg_spend] = model_df[_enc_app_avg_spend].fillna(model_df["target_t1_spend"].median())
    if "roi_d1" in model_df.columns:
        grp_roi = model_df.groupby("应用ID")["roi_d1"]
        model_df[_enc_app_avg_roi] = (
            grp_roi.cumsum().sub(model_df["roi_d1"])
            / grp_spend.cumcount().clip(lower=1)
        )
        model_df[_enc_app_avg_roi] = model_df[_enc_app_avg_roi].fillna(model_df["roi_d1"].median())
        enc_feats = [_enc_app_avg_spend, _enc_app_avg_roi]
    else:
        enc_feats = [_enc_app_avg_spend]
    feats_all = feats + enc_feats

    dates = sorted(model_df["日期"].dropna().unique())
    cutoffs = list(dates[:-1])
    if eval_recent_days is not None and eval_recent_days > 0 and len(cutoffs) > eval_recent_days:
        cutoffs = cutoffs[-eval_recent_days:]
    n_cutoffs = len(cutoffs)
    for i, cutoff in enumerate(cutoffs):
        t_cutoff = pd.Timestamp(cutoff)
        next_day = cutoff + np.timedelta64(1, "D")
        train = model_df[model_df["日期"] <= t_cutoff].copy()
        test = model_df[model_df["日期"] == next_day].copy()
        if min_target_spend_train > 0:
            train = train[train["target_t1_spend"] >= min_target_spend_train].copy()
        if min_target_spend_eval > 0:
            test = test[test["target_t1_spend"] >= min_target_spend_eval].copy()
        if test.empty or train["日期"].nunique() < min_train_days:
            continue
        print(f"  [{i+1}/{n_cutoffs}] cutoff={t_cutoff.date()} train={len(train)} test={len(test)}", flush=True)
        ew = _ewma_spend(train, test, alpha)
        if not ew.empty:
            preds_by_model["EWMA"].append(ew)
        if not skip_baseline_two_stage:
            br = _baseline_residual_predict(train, test, feats_all)
            if not br.empty:
                preds_by_model["BaselineResidual"].append(br)
            ts = _two_stage_regime_predict(train, test, feats_all)
            if not ts.empty:
                preds_by_model["TwoStageRegime"].append(ts)
        max_workers = max(2, min(8, len(models) * (2 if use_log_target else 1)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for m in models:
                futures[ex.submit(_tree_predict, m, train, test, feats_all, False, use_gpu)] = m
                if use_log_target:
                    futures[ex.submit(_tree_predict, m, train, test, feats_all, True, use_gpu)] = f"{m}_log"
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
    skip_bs = True
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
        skip_baseline_two_stage=skip_bs,
        use_gpu=bool(args.use_gpu),
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
        "unified_data": True,
        "fast_mode": bool(args.fast),
        "eval_recent_days": eval_recent,
        "tree_models": tree_override,
        "skip_baseline_two_stage": skip_bs,
        "clean_meta": clean_meta,
        "min_spend_train": 2.0,
        "feature_set": args.feature_set,
        "feature_count": len(_feature_cols(args.feature_set)),
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
            "predictions": [str(out_dir / f"predictions_{_safe_model_filename(r.model_name)}.csv") for r in results],
            "predictions_reconciled": str(out_dir / "predictions_reconciled.csv") if reconcile_report else None,
            "reconcile_report": str(out_dir / "reconcile_report.json") if reconcile_report else None,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
