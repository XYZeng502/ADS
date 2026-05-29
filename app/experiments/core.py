"""Shared core functions for model training and evaluation.

Extracted from scripts/run_parallel_models_spend_t1.py so experiment scripts
can import and reuse the walk-forward training pipeline.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    "消耗金额", "roi_d1", "ctr", "cvr_dl", "cvr_act",
    "dow", "is_fri_sat", "is_holiday", "is_rest_day",
    "dom", "month", "week_of_month",
    "days_to_month_end", "month_progress",
    "is_month_start_3d", "is_month_end_3d", "is_month_end_7d", "is_month_end_10d",
    "days_to_next_holiday", "days_since_prev_holiday",
    "is_pre_holiday_3d", "is_post_holiday_3d",
    "is_summer_winter_break",
    "holiday_id", "holiday_seq_index", "holiday_days_remaining", "holiday_window_len",
    "is_last_holiday_day",
    "target_dow", "target_is_weekend", "target_is_rest_day", "target_is_adjusted_workday",
    "target_is_holiday", "target_holiday_id", "target_holiday_seq_index",
    "target_holiday_days_remaining", "target_holiday_window_len",
    "target_is_last_holiday_day",
    "target_days_to_next_holiday", "target_days_since_prev_holiday",
    "target_is_pre_holiday_3d", "target_is_post_holiday_1d", "target_is_post_holiday_3d",
    "target_is_first_workday_after_holiday",
    "spend_lag_1", "spend_lag_2", "spend_lag_3", "spend_lag_7",
    "spend_roll_mean_3", "spend_roll_std_3", "spend_roll_mean_7", "spend_roll_std_7",
    "roi_lag_1", "roi_lag_2", "roi_lag_3", "roi_lag_7",
    "spend_ratio_1d",
    "traffic_n_unique", "traffic_top1_pct", "traffic_hhi",
    "scene_n_unique", "scene_top1_pct", "scene_hhi",
    "creative_n_unique", "creative_top1_pct", "creative_hhi",
    "billing_n_unique", "billing_top1_pct", "billing_hhi",
    "spend_x_is_rest_day", "spend_lag1_x_target_rest",
    "spend_lag1_x_target_holiday", "spend_ratio_x_target_rest",
    "app_label",
]

CALENDAR_PACING_FEATURE_COLS = [
    "week_of_month", "days_to_month_end", "month_progress",
    "is_month_start_3d", "is_month_end_3d", "is_month_end_7d", "is_month_end_10d",
    "days_to_next_holiday", "days_since_prev_holiday",
    "is_pre_holiday_3d", "is_post_holiday_3d", "is_summer_winter_break",
    "holiday_id", "holiday_seq_index", "holiday_days_remaining", "holiday_window_len",
    "is_last_holiday_day",
    "target_dow", "target_is_weekend", "target_is_rest_day", "target_is_adjusted_workday",
    "target_is_holiday", "target_holiday_id", "target_holiday_seq_index",
    "target_holiday_days_remaining", "target_holiday_window_len",
    "target_is_last_holiday_day",
    "target_days_to_next_holiday", "target_days_since_prev_holiday",
    "target_is_pre_holiday_3d", "target_is_post_holiday_1d", "target_is_post_holiday_3d",
    "target_is_first_workday_after_holiday",
    "spend_roll_mean_3", "spend_roll_std_3", "spend_roll_mean_7", "spend_roll_std_7",
    "mtd_spend", "mtd_revenue", "mtd_roi_d1",
    "spend_x_is_rest_day", "spend_lag1_x_target_rest",
    "spend_lag1_x_target_holiday", "spend_ratio_x_target_rest",
    "app_label",
]

SPEND_BUCKET_LABELS = ["Q1_low", "Q2", "Q3", "Q4_high"]


def feature_cols(feature_set: str = "base") -> List[str]:
    if feature_set == "calendar_pacing":
        return list(dict.fromkeys(BASE_FEATURE_COLS + CALENDAR_PACING_FEATURE_COLS))
    return BASE_FEATURE_COLS


def safe_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_")


def build_model(name: str, use_gpu: bool = False, **overrides):
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
        )
    if name == "GBDT":
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.04, max_depth=3, random_state=42
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=500, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
        )
    if name == "LightGBM":
        if LGBMRegressor is None:
            return None
        kwargs = dict(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1,
        )
        if use_gpu:
            kwargs["device"] = "cuda"
        kwargs.update(overrides)
        return LGBMRegressor(**kwargs)
    if name == "XGBoost":
        if XGBRegressor is None:
            return None
        kwargs = dict(
            n_estimators=500, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=42,
            n_jobs=4, enable_categorical=True,
        )
        if use_gpu:
            kwargs["device"] = "cuda"
        kwargs.update(overrides)
        return XGBRegressor(**kwargs)
    if name == "CatBoost":
        if CatBoostRegressor is None:
            return None
        kwargs = dict(
            iterations=500, learning_rate=0.03, depth=6,
            random_seed=42, verbose=0, thread_count=4,
        )
        kwargs.update(overrides)
        return CatBoostRegressor(**kwargs)
    return None


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > 1e-8
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
        if np.any(mask) else np.nan
    )
    wmape = float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-8))
    return {"mae": mae, "rmse": rmse, "mape": mape, "mape_pct": mape * 100.0 if np.isfinite(mape) else np.nan,
            "wmape": wmape, "wmape_pct": wmape * 100.0}


def evaluate_pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, alpha * errors, (alpha - 1) * errors)))


def compute_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def compute_interval_width(lower: np.ndarray, upper: np.ndarray, median: np.ndarray) -> np.ndarray:
    safe_median = np.where(median > 1e-8, median, 1.0)
    return (upper - lower) / safe_median


def apply_holiday_transition_calibration(test_df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(pred, dtype=float).copy()

    # Batch-convert all columns to numpy once
    def _col(name, default=0.0, as_bool=False):
        arr = pd.to_numeric(test_df.get(name, default), errors="coerce").fillna(default).to_numpy()
        return arr == 1 if as_bool else arr

    current_spend = _col("消耗金额")
    roll3 = _col("spend_roll_mean_3")
    roll7 = _col("spend_roll_mean_7")
    recent_ref = np.maximum(current_spend, roll3)
    wider_ref = np.maximum(recent_ref, roll7)

    last_holiday_day = _col("target_is_last_holiday_day", as_bool=True)
    first_workday_after_holiday = _col("target_is_first_workday_after_holiday", as_bool=True)

    adjusted[last_holiday_day] = np.maximum(adjusted[last_holiday_day], recent_ref[last_holiday_day] * 0.80)
    adjusted[first_workday_after_holiday] = np.minimum(
        adjusted[first_workday_after_holiday],
        recent_ref[first_workday_after_holiday] * 0.45,
    )

    target_is_holiday = _col("target_is_holiday", as_bool=True)
    target_is_weekend = _col("target_is_weekend", as_bool=True)
    mid_holiday_weekday = target_is_holiday & ~target_is_weekend & ~last_holiday_day & ~first_workday_after_holiday
    adjusted[mid_holiday_weekday] = np.maximum(adjusted[mid_holiday_weekday], wider_ref[mid_holiday_weekday] * 0.90)

    target_is_rest_day = _col("target_is_rest_day", as_bool=True)
    days_since_holiday = _col("target_days_since_prev_holiday", default=99.0)

    early_ph_rest = target_is_rest_day & ~target_is_holiday & (days_since_holiday >= 3) & (days_since_holiday <= 4)
    adjusted[early_ph_rest] = np.maximum(adjusted[early_ph_rest], wider_ref[early_ph_rest] * 0.90)
    adjusted[early_ph_rest] = np.minimum(adjusted[early_ph_rest], wider_ref[early_ph_rest] * 1.35)

    late_ph_rest = target_is_rest_day & ~target_is_holiday & (days_since_holiday >= 5) & (days_since_holiday <= 7)
    adjusted[late_ph_rest] = np.maximum(adjusted[late_ph_rest], wider_ref[late_ph_rest] * 0.80)
    adjusted[late_ph_rest] = np.minimum(adjusted[late_ph_rest], wider_ref[late_ph_rest] * 1.10)

    return np.clip(adjusted, 0.0, None)


def tree_predict(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 feats: List[str], use_log_target: bool, use_gpu: bool = False,
                 weight_exponent: float = 0.35, model_overrides: Optional[Dict] = None) -> pd.DataFrame:
    train = train_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_spend"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    model = build_model(name, use_gpu=use_gpu, **(model_overrides or {}))
    if model is None:
        return pd.DataFrame(columns=["日期", "应用ID", "y_true", "y_pred", "model"])
    x_train = train[feats].copy()
    y_train = train["target_t1_spend"].clip(lower=0.0)
    x_test = test[feats].copy()
    y_test = test["target_t1_spend"].values
    label = name
    if use_log_target:
        y_train = np.log1p(y_train)
        label = f"{name}_log"

    cat_cols = [c for c in ["app_label"] if c in x_train.columns]
    fit_kwargs = {}
    if name == "CatBoost" and cat_cols:
        for c in cat_cols:
            x_train[c] = x_train[c].astype("category")
            x_test[c] = x_test[c].astype("category")
        fit_kwargs["cat_features"] = cat_cols

    sample_weight = np.power(train["消耗金额"].clip(lower=0).values, weight_exponent)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight, **fit_kwargs)
    except TypeError:
        model.fit(x_train, y_train, **fit_kwargs)
    pred = model.predict(x_test)
    if use_log_target:
        log_resid = y_train - model.predict(x_train)
        smearing = float(np.clip(np.exp(log_resid).mean(), 0.8, 1.25))
        pred = np.expm1(pred) * smearing
    pred = np.clip(pred, 0.0, None)
    pred = apply_holiday_transition_calibration(test, pred)
    return pd.DataFrame({
        "日期": test["日期"].values, "应用ID": test["应用ID"].values,
        "y_true": y_test, "y_pred": pred, "model": label,
    })


def ewma_spend(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
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
            preds.append({
                "日期": r["日期"], "应用ID": r["应用ID"],
                "y_true": r["target_t1_spend"], "y_pred": p, "model": "EWMA",
            })
    return pd.DataFrame(preds)


def assign_spend_bucket(df: pd.DataFrame, value_col: str = "y_true") -> pd.DataFrame:
    """Assign Q1-Q4 spend buckets based on per-app mean of value_col."""
    app_avg = df.groupby("应用ID")[value_col].mean().rename("_app_avg_spend")
    df = df.join(app_avg, on="应用ID")
    df["spend_bucket"] = pd.qcut(df["_app_avg_spend"], q=4, labels=SPEND_BUCKET_LABELS)
    return df


def compute_expanding_window_encodings(df: pd.DataFrame) -> tuple:
    """Build expanding-window target encodings (no leakage).

    Returns (model_df, enc_feats) where enc_feats lists the new column names.
    """
    spend_col = "target_t1_spend"
    model_df = df.sort_values(["应用ID", "日期"]).reset_index(drop=True).copy()
    grp_spend = model_df.groupby("应用ID")[spend_col]
    model_df["app_avg_spend"] = (
        grp_spend.cumsum().sub(model_df[spend_col])
        / grp_spend.cumcount().clip(lower=1)
    )
    model_df["app_avg_spend"] = model_df["app_avg_spend"].fillna(model_df[spend_col].median())
    enc_feats = ["app_avg_spend"]
    if "roi_d1" in model_df.columns:
        grp_roi = model_df.groupby("应用ID")["roi_d1"]
        model_df["app_avg_roi_d1"] = (
            grp_roi.cumsum().sub(model_df["roi_d1"])
            / grp_spend.cumcount().clip(lower=1)
        )
        model_df["app_avg_roi_d1"] = model_df["app_avg_roi_d1"].fillna(model_df["roi_d1"].median())
        enc_feats.append("app_avg_roi_d1")
    return model_df, enc_feats


def coverage_verdict(coverage: float, target: float = 0.80,
                     promising_pp: float = 5.0, inconclusive_pp: float = 15.0) -> str:
    """Classify experiment result based on coverage error in percentage points."""
    error = abs((coverage - target) * 100)
    if error <= promising_pp:
        return "promising"
    elif error <= inconclusive_pp:
        return "inconclusive"
    return "regress"


def walk_forward_windows(model_df, eval_recent_days=20, min_train_days=20):
    """Generator yielding (train, test, cutoff_date) for walk-forward evaluation."""
    dates = sorted(model_df["日期"].dropna().unique())
    cutoffs = list(dates[:-1])
    if eval_recent_days > 0 and len(cutoffs) > eval_recent_days:
        cutoffs = cutoffs[-eval_recent_days:]
    n = len(cutoffs)
    for i, cutoff in enumerate(cutoffs):
        t_cutoff = pd.Timestamp(cutoff)
        next_day = cutoff + np.timedelta64(1, "D")
        train = model_df[model_df["日期"] <= t_cutoff]
        test = model_df[model_df["日期"] == next_day]
        if test.empty or train["日期"].nunique() < min_train_days:
            continue
        print(f"  [{i+1}/{n}] cutoff={t_cutoff.date()} train={len(train)} test={len(test)}", flush=True)
        yield train, test, t_cutoff, next_day


def run(df: pd.DataFrame, min_train_days: int, alpha: float,
        use_log_target: bool, feature_set: str = "base",
        min_target_spend_train: float = 0.0, min_target_spend_eval: float = 0.0,
        *, eval_recent_days: Optional[int] = None,
        tree_models: Optional[List[str]] = None,
        use_gpu: bool = False,
        weight_exponent: float = 0.35,
        model_overrides: Optional[Dict] = None,
        extra_model_variants: Optional[List[Dict]] = None) -> List[ModelResult]:
    feats = feature_cols(feature_set)
    models = list(tree_models) if tree_models else ["RandomForest", "GBDT", "ExtraTrees", "LightGBM"]
    preds_by_model: Dict[str, List[pd.DataFrame]] = {"EWMA": []}
    for m in models:
        preds_by_model[m] = []
        if use_log_target:
            preds_by_model[f"{m}_log"] = []
    if extra_model_variants:
        for variant in extra_model_variants:
            for m in models:
                preds_by_model[f"{m}_{variant['suffix']}"] = []
                if use_log_target:
                    preds_by_model[f"{m}_log_{variant['suffix']}"] = []

    _enc_app_avg_spend = "app_avg_spend"
    _enc_app_avg_roi = "app_avg_roi_d1"
    model_df = df.sort_values(["应用ID", "日期"]).reset_index(drop=True).copy()
    model_df["split_value"] = "ALL"
    grp_spend = model_df.groupby("应用ID")["target_t1_spend"]
    model_df[_enc_app_avg_spend] = (
        grp_spend.cumsum().sub(model_df["target_t1_spend"])
        / grp_spend.cumcount().clip(lower=1)
    )
    model_df[_enc_app_avg_spend] = model_df[_enc_app_avg_spend].fillna(
        model_df["target_t1_spend"].median()
    )
    if "roi_d1" in model_df.columns:
        grp_roi = model_df.groupby("应用ID")["roi_d1"]
        model_df[_enc_app_avg_roi] = (
            grp_roi.cumsum().sub(model_df["roi_d1"])
            / grp_spend.cumcount().clip(lower=1)
        )
        model_df[_enc_app_avg_roi] = model_df[_enc_app_avg_roi].fillna(
            model_df["roi_d1"].median()
        )
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
        ew = ewma_spend(train, test, alpha)
        if not ew.empty:
            preds_by_model["EWMA"].append(ew)
        max_workers = max(2, min(8, len(models) * (2 if use_log_target else 1)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for m in models:
                futures[ex.submit(tree_predict, m, train, test, feats_all, False, use_gpu, weight_exponent, model_overrides)] = m
                if use_log_target:
                    futures[ex.submit(tree_predict, m, train, test, feats_all, True, use_gpu, weight_exponent, model_overrides)] = f"{m}_log"
                if extra_model_variants:
                    for variant in extra_model_variants:
                        merged_overrides = {**(model_overrides or {}), **variant.get("overrides", {})}
                        futures[ex.submit(tree_predict, m, train, test, feats_all, False, use_gpu, weight_exponent, merged_overrides)] = f"{m}_{variant['suffix']}"
                        if use_log_target:
                            futures[ex.submit(tree_predict, m, train, test, feats_all, True, use_gpu, weight_exponent, merged_overrides)] = f"{m}_log_{variant['suffix']}"
            for fut in as_completed(futures):
                name = futures[fut]
                pred_df = fut.result()
                if not pred_df.empty:
                    preds_by_model[name].append(pred_df)

    results = []
    for name, frames in preds_by_model.items():
        if not frames:
            continue
        pred_df = pd.concat(frames, ignore_index=True).dropna(subset=["y_true", "y_pred"])
        if pred_df.empty:
            continue
        metrics = evaluate(pred_df["y_true"].values, pred_df["y_pred"].values)
        results.append(ModelResult(model_name=name, metrics=metrics, prediction_df=pred_df))
    return results


def build_cqr_prediction(pred_q05: pd.DataFrame,
                         pred_q50: pd.DataFrame,
                         pred_q95: pd.DataFrame,
                         weight_exponent: float = 0.35,
                         q4_confidence: float = 0.85,
                         random_state: int = 42) -> pd.DataFrame:
    """Post-hoc CQR calibration on quantile variant predictions.

    Takes three quantile XGBoost prediction DataFrames (from
    extra_model_variants with alpha=0.05/0.5/0.95), applies per-bucket
    CQR calibration, and returns a single DataFrame with calibrated
    point predictions and prediction intervals.

    Args:
        pred_q05: DataFrame with columns [日期, 应用ID, y_true, y_pred, model]
        pred_q50: DataFrame with columns [日期, 应用ID, y_true, y_pred, model]
        pred_q95: DataFrame with columns [日期, 应用ID, y_true, y_pred, model]
        q4_confidence: Target coverage for Q4 bucket (others use 0.80)

    Returns:
        DataFrame with columns [日期, 应用ID, y_true, y_pred, y_lower, y_upper, model]
    """
    from mapie.regression import ConformalizedQuantileRegressor

    df_q05 = pred_q05.copy()
    df_q50 = pred_q50.copy()
    df_q95 = pred_q95.copy()

    # Assign static spend buckets based on overall y_true mean per app
    all_y_true = pd.concat([
        df_q05[["日期", "应用ID", "y_true"]],
        df_q50[["日期", "应用ID", "y_true"]],
        df_q95[["日期", "应用ID", "y_true"]],
    ])
    app_avg = all_y_true.groupby("应用ID")["y_true"].mean()
    bucket_map = pd.qcut(app_avg, q=4, labels=SPEND_BUCKET_LABELS)

    for df_part in [df_q05, df_q50, df_q95]:
        df_part["_bucket"] = df_part["应用ID"].map(bucket_map)

    bucket_conf = {b: (q4_confidence if b == "Q4_high" else 0.80)
                   for b in SPEND_BUCKET_LABELS}

    results = []
    for dt in sorted(df_q05["日期"].unique()):
        mask05 = df_q05["日期"] == dt
        mask50 = df_q50["日期"] == dt
        mask95 = df_q95["日期"] == dt

        sub05 = df_q05[mask05].set_index("应用ID").sort_index()
        sub50 = df_q50[mask50].set_index("应用ID").sort_index()
        sub95 = df_q95[mask95].set_index("应用ID").sort_index()

        common_apps = sub05.index.intersection(sub50.index)
        common_apps = common_apps.intersection(sub95.index)
        if len(common_apps) < 5:
            continue

        n = len(common_apps)
        y_lower_all = np.full(n, np.nan)
        y_upper_all = np.full(n, np.nan)
        y_pred_all = sub50.loc[common_apps, "y_pred"].values
        y_true_all = sub05.loc[common_apps, "y_true"].values
        buckets_all = sub05.loc[common_apps, "_bucket"].values

        q_preds = np.column_stack([
            sub05.loc[common_apps, "y_pred"].values,
            sub50.loc[common_apps, "y_pred"].values,
            sub95.loc[common_apps, "y_pred"].values,
        ])
        # Ensure non-crossing
        q_preds = np.sort(q_preds, axis=1)

        for bucket in SPEND_BUCKET_LABELS:
            bm = buckets_all == bucket
            if bm.sum() < 5:
                continue

            q_bucket = q_preds[bm]
            y_bucket = y_true_all[bm]
            bucket_indices = np.where(bm)[0]

            try:
                cqr = ConformalizedQuantileRegressor(
                    estimator=None,
                    confidence_level=bucket_conf[bucket],
                )
                # Calibrate and predict on all samples — no split, no gaps
                cqr.fit(q_bucket, y_bucket)
                _, y_pis = cqr.predict(q_bucket)
                y_pis = y_pis.squeeze(-1)
                y_lower_all[bucket_indices] = np.min(y_pis, axis=1)
                y_upper_all[bucket_indices] = np.max(y_pis, axis=1)
            except Exception:
                y_lower_all[bucket_indices] = q_bucket[:, 0]
                y_upper_all[bucket_indices] = q_bucket[:, 2]

        valid = ~np.isnan(y_lower_all) & ~np.isnan(y_upper_all)
        if valid.sum() == 0:
            continue

        results.append(pd.DataFrame({
            "日期": dt,
            "应用ID": common_apps[valid],
            "y_true": y_true_all[valid],
            "y_pred": np.clip(y_pred_all[valid], 0.0, None),
            "y_lower": np.clip(y_lower_all[valid], 0.0, None),
            "y_upper": np.clip(y_upper_all[valid], 0.0, None),
            "model": "XGBoost_CQR",
        }))

    if not results:
        raise ValueError("CQR calibration produced no results.")

    return pd.concat(results, ignore_index=True)
