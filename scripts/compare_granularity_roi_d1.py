import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from app.experiments.core import evaluate
from sklearn.linear_model import Ridge


ENTITY_KEYS = {
    "app": ["应用ID"],
    "plan": ["应用ID", "计划ID"],
    "adgroup": ["应用ID", "计划ID", "广告组ID"],
}


def _load_and_aggregate(input_csv: str, level: str) -> pd.DataFrame:
    raw = pd.read_csv(input_csv, encoding="utf-8-sig")
    raw["日期"] = pd.to_datetime(raw["日期"], errors="coerce")
    raw = raw[raw["日期"].notna()].copy()
    for c in ["消耗金额", "首日广告收入", "曝光量", "点击量", "下载量", "激活人数(快应用新增用户数)"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    gcols = ENTITY_KEYS[level] + ["日期"]
    df = (
        raw.groupby(gcols, as_index=False)[["消耗金额", "首日广告收入", "曝光量", "点击量", "下载量", "激活人数(快应用新增用户数)"]]
        .sum()
        .sort_values(gcols)
    )
    df["roi_d1"] = np.where(df["消耗金额"] > 0, df["首日广告收入"] / df["消耗金额"], np.nan)
    df["ctr"] = np.where(df["曝光量"] > 0, df["点击量"] / df["曝光量"], 0.0)
    df["cvr_dl"] = np.where(df["点击量"] > 0, df["下载量"] / df["点击量"], 0.0)
    df["cvr_act"] = np.where(df["下载量"] > 0, df["激活人数(快应用新增用户数)"] / df["下载量"], 0.0)
    df["act_per_spend"] = np.where(df["消耗金额"] > 0, df["激活人数(快应用新增用户数)"] / df["消耗金额"], 0.0)
    df["rev_per_act_d1"] = np.where(df["激活人数(快应用新增用户数)"] > 0, df["首日广告收入"] / df["激活人数(快应用新增用户数)"], 0.0)
    df["dow"] = df["日期"].dt.weekday
    df["is_fri_sat"] = df["dow"].isin([4, 5]).astype(int)
    df["dom"] = df["日期"].dt.day
    df["month"] = df["日期"].dt.month
    return df


def _add_lags(df: pd.DataFrame, entity_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(entity_cols)
    for lag in [1, 2, 3, 7]:
        out[f"roi_d1_lag_{lag}"] = g["roi_d1"].shift(lag)
        out[f"spend_lag_{lag}"] = g["消耗金额"].shift(lag)
        out[f"act_per_spend_lag_{lag}"] = g["act_per_spend"].shift(lag)
        out[f"rev_per_act_d1_lag_{lag}"] = g["rev_per_act_d1"].shift(lag)
    out["spend_ratio_1d"] = np.where(out["spend_lag_1"] > 1e-8, out["消耗金额"] / out["spend_lag_1"] - 1.0, 0.0)
    out["spend_ratio_3d"] = np.where(out["spend_lag_3"] > 1e-8, out["消耗金额"] / out["spend_lag_3"] - 1.0, 0.0)
    out["entity_active_days_14"] = (
        g["roi_d1"].rolling(14, min_periods=1).count().reset_index(level=list(range(len(entity_cols))), drop=True)
    )
    out["entity_roi_std_14"] = (
        g["roi_d1"].rolling(14, min_periods=3).std().reset_index(level=list(range(len(entity_cols))), drop=True).fillna(0.0)
    )
    out["target_t1_roi_d1"] = g["roi_d1"].shift(-1)
    for c in out.columns:
        if c.endswith(("lag_1", "lag_2", "lag_3", "lag_7")) or c in ["spend_ratio_1d", "spend_ratio_3d"]:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _feature_cols() -> List[str]:
    return [
        "消耗金额",
        "ctr",
        "cvr_dl",
        "cvr_act",
        "act_per_spend",
        "rev_per_act_d1",
        "spend_ratio_1d",
        "spend_ratio_3d",
        "dow",
        "is_fri_sat",
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
        "entity_active_days_14",
        "entity_roi_std_14",
    ]


def _fit_predict_walk(df: pd.DataFrame, level: str, min_train_days: int, min_entity_days: int, min_entity_median_spend: float) -> pd.DataFrame:
    entity_cols = ENTITY_KEYS[level]
    feat_cols = _feature_cols()
    stats = (
        df.groupby(entity_cols, as_index=False)
        .agg(entity_days=("日期", "nunique"), entity_median_spend=("消耗金额", "median"))
    )
    keep = stats[(stats["entity_days"] >= min_entity_days) & (stats["entity_median_spend"] >= min_entity_median_spend)].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["_k"] = 1
    tmp = df.merge(keep[entity_cols + ["_k"]], on=entity_cols, how="inner")
    df = tmp.drop(columns=["_k"])
    all_dates = sorted(df["日期"].dropna().unique())
    preds = []
    for cutoff in all_dates[:-1]:
        train = df[df["日期"] <= cutoff].dropna(subset=["target_t1_roi_d1"]).copy()
        test = df[df["日期"] == (cutoff + np.timedelta64(1, "D"))].dropna(subset=["target_t1_roi_d1"]).copy()
        if test.empty or train["日期"].nunique() < min_train_days:
            continue
        train = train.dropna(subset=feat_cols)
        test = test.dropna(subset=feat_cols)
        if train.empty or test.empty:
            continue
        model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[feat_cols], train["target_t1_roi_d1"])
        keep_cols = list(dict.fromkeys(entity_cols + ["应用ID", "日期", "target_t1_roi_d1", "消耗金额"]))
        test_pred = test[keep_cols].copy()
        test_pred["y_pred"] = np.clip(model.predict(test[feat_cols]), 0.0, None)
        test_pred["level"] = level
        preds.append(test_pred)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def _to_app_level_eval(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame(columns=["应用ID", "日期", "y_true", "y_pred", "level"])
    tmp = pred_df.copy()
    tmp["weighted_pred"] = tmp["y_pred"] * tmp["消耗金额"]
    agg = (
        tmp.groupby(["应用ID", "日期", "level"], as_index=False)
        .agg(
            spend_sum=("消耗金额", "sum"),
            pred_sum=("weighted_pred", "sum"),
            y_true=("target_t1_roi_d1", "mean"),
        )
    )
    agg["y_pred"] = np.where(agg["spend_sum"] > 0, agg["pred_sum"] / agg["spend_sum"], agg["y_true"])
    return agg[["应用ID", "日期", "y_true", "y_pred", "level"]]


def _metrics(df: pd.DataFrame, name: str) -> Dict[str, float]:
    y = df["y_true"].values
    p = df["y_pred"].values
    metrics = evaluate(y, p)
    return {
        "model": name,
        "samples": int(len(df)),
        **metrics,
    }


def _build_dynamic_fusion_series(
    comp_df: pd.DataFrame,
    source_col: str,
    min_train_days: int,
    retune_frequency_days: int,
    model_name: str,
    weight_min: float,
    weight_max: float,
    smooth_alpha: float,
) -> pd.DataFrame:
    work = comp_df[["应用ID", "日期", "y_true", "y_pred_app", source_col]].copy()
    work[source_col] = work[source_col].replace([np.inf, -np.inf], np.nan).fillna(work["y_pred_app"])
    work = work.sort_values(["日期", "应用ID"])
    all_dates = sorted(work["日期"].unique())
    best_w = 0.4
    last_retune_date = None
    rows = []
    for d in all_dates:
        hist = work[work["日期"] < d].copy()
        today = work[work["日期"] == d].copy()
        if today.empty:
            continue
        if hist["日期"].nunique() >= min_train_days:
            need_retune = last_retune_date is None
            if not need_retune and retune_frequency_days > 0:
                need_retune = int((pd.Timestamp(d) - pd.Timestamp(last_retune_date)).days) >= retune_frequency_days
            if need_retune:
                best_rmse = float("inf")
                for w in np.linspace(weight_min, weight_max, 21):
                    pred = (1.0 - w) * hist["y_pred_app"].values + w * hist[source_col].values
                    rmse = float(np.sqrt(mean_squared_error(hist["y_true"].values, pred)))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        candidate_w = float(w)
                # 周级平滑，避免权重跳变过大
                best_w = (1.0 - smooth_alpha) * best_w + smooth_alpha * candidate_w
                best_w = float(np.clip(best_w, weight_min, weight_max))
                last_retune_date = d
        today_out = today[["应用ID", "日期", "y_true"]].copy()
        today_out["y_pred"] = np.clip((1.0 - best_w) * today["y_pred_app"].values + best_w * today[source_col].values, 0.0, None)
        today_out["level"] = model_name
        today_out["weight_used"] = best_w
        rows.append(today_out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["应用ID", "日期", "y_true", "y_pred", "level", "weight_used"])


def _stacked_calibration_walk(
    base_df: pd.DataFrame,
    source_col: str,
    model_name: str,
    min_train_days: int,
    fallback_weight: float,
) -> pd.DataFrame:
    work = base_df[["应用ID", "日期", "y_true", "y_pred_app", source_col]].copy()
    work[source_col] = work[source_col].replace([np.inf, -np.inf], np.nan).fillna(work["y_pred_app"])
    work = work.sort_values(["日期", "应用ID"])
    all_dates = sorted(work["日期"].unique())
    out_rows = []
    for current_date in all_dates:
        train = work[work["日期"] < current_date].copy()
        test = work[work["日期"] == current_date].copy()
        if test.empty:
            continue
        if train["日期"].nunique() < min_train_days:
            # 冷启动阶段：用固定权重融合兜底，保证全日期有输出
            pred = (1.0 - fallback_weight) * test["y_pred_app"].values + fallback_weight * test[source_col].values
            pred = np.clip(pred, 0.0, None)
        else:
            x_train = train[["y_pred_app", source_col]]
            y_train = train["y_true"]
            x_test = test[["y_pred_app", source_col]]
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(x_train, y_train)
            pred = np.clip(ridge.predict(x_test), 0.0, None)
        part = test[["应用ID", "日期", "y_true"]].copy()
        part["y_pred"] = pred
        part["level"] = model_name
        out_rows.append(part)
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["应用ID", "日期", "y_true", "y_pred", "level"])


def main() -> None:
    parser = argparse.ArgumentParser(description="对比应用/计划/广告组粒度的 roi_d1 预测表现")
    parser.add_argument("--input", default="daily_20260421_120112.csv")
    parser.add_argument("--output-dir", default="outputs/granularity_compare_roi_d1")
    parser.add_argument("--min-train-days", type=int, default=20)
    parser.add_argument("--fusion-plan-weight", type=float, default=0.4)
    parser.add_argument("--fusion-adgroup-weight", type=float, default=0.4)
    parser.add_argument("--min-entity-days", type=int, default=10)
    parser.add_argument("--min-entity-median-spend", type=float, default=20.0)
    parser.add_argument("--fusion-retune-frequency-days", type=int, default=7)
    parser.add_argument("--dynamic-weight-min", type=float, default=0.3)
    parser.add_argument("--dynamic-weight-max", type=float, default=0.5)
    parser.add_argument("--dynamic-weight-smooth-alpha", type=float, default=0.6)
    parser.add_argument("--use-dynamic-fusion", action="store_true", help="启用动态融合（实验开关，默认关闭）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    app_df = _add_lags(_load_and_aggregate(args.input, "app"), ENTITY_KEYS["app"])
    plan_df = _add_lags(_load_and_aggregate(args.input, "plan"), ENTITY_KEYS["plan"])
    adg_df = _add_lags(_load_and_aggregate(args.input, "adgroup"), ENTITY_KEYS["adgroup"])

    app_pred = _to_app_level_eval(
        _fit_predict_walk(app_df, "app", args.min_train_days, args.min_entity_days, args.min_entity_median_spend)
    )
    plan_pred = _to_app_level_eval(
        _fit_predict_walk(plan_df, "plan", args.min_train_days, args.min_entity_days, args.min_entity_median_spend)
    )
    adg_pred = _to_app_level_eval(
        _fit_predict_walk(adg_df, "adgroup", args.min_train_days, args.min_entity_days, args.min_entity_median_spend)
    )

    base = app_pred.rename(columns={"y_pred": "y_pred_app"})[["应用ID", "日期", "y_true", "y_pred_app"]]
    comp = (
        base.merge(plan_pred[["应用ID", "日期", "y_pred"]].rename(columns={"y_pred": "y_pred_plan"}), on=["应用ID", "日期"], how="left")
        .merge(adg_pred[["应用ID", "日期", "y_pred"]].rename(columns={"y_pred": "y_pred_adgroup"}), on=["应用ID", "日期"], how="left")
        .fillna(0.0)
    )
    comp["y_pred_fusion_plan"] = (1 - args.fusion_plan_weight) * comp["y_pred_app"] + args.fusion_plan_weight * comp["y_pred_plan"]
    comp["y_pred_fusion_adgroup"] = (1 - args.fusion_adgroup_weight) * comp["y_pred_app"] + args.fusion_adgroup_weight * comp["y_pred_adgroup"]
    dynamic_plan = pd.DataFrame()
    dynamic_adg = pd.DataFrame()
    if args.use_dynamic_fusion:
        dynamic_plan = _build_dynamic_fusion_series(
            comp_df=comp,
            source_col="y_pred_plan",
            min_train_days=args.min_train_days,
            retune_frequency_days=args.fusion_retune_frequency_days,
            model_name="fusion_app_plan_dynamic",
            weight_min=args.dynamic_weight_min,
            weight_max=args.dynamic_weight_max,
            smooth_alpha=args.dynamic_weight_smooth_alpha,
        )
        dynamic_adg = _build_dynamic_fusion_series(
            comp_df=comp,
            source_col="y_pred_adgroup",
            min_train_days=args.min_train_days,
            retune_frequency_days=args.fusion_retune_frequency_days,
            model_name="fusion_app_adgroup_dynamic",
            weight_min=args.dynamic_weight_min,
            weight_max=args.dynamic_weight_max,
            smooth_alpha=args.dynamic_weight_smooth_alpha,
        )
    stacked_plan = _stacked_calibration_walk(
        comp, "y_pred_plan", "stacked_app_plan", args.min_train_days, args.fusion_plan_weight
    )
    stacked_adg = _stacked_calibration_walk(
        comp, "y_pred_adgroup", "stacked_app_adgroup", args.min_train_days, args.fusion_adgroup_weight
    )

    rows = [
        _metrics(app_pred, "app_direct"),
        _metrics(plan_pred, "plan_to_app"),
        _metrics(adg_pred, "adgroup_to_app"),
        _metrics(comp.rename(columns={"y_pred_fusion_plan": "y_pred"})[["y_true", "y_pred"]], "fusion_app_plan"),
        _metrics(comp.rename(columns={"y_pred_fusion_adgroup": "y_pred"})[["y_true", "y_pred"]], "fusion_app_adgroup"),
    ]
    if not dynamic_plan.empty:
        rows.append(_metrics(dynamic_plan, "fusion_app_plan_dynamic"))
    if not dynamic_adg.empty:
        rows.append(_metrics(dynamic_adg, "fusion_app_adgroup_dynamic"))
    if not stacked_plan.empty:
        rows.append(_metrics(stacked_plan, "stacked_app_plan"))
    if not stacked_adg.empty:
        rows.append(_metrics(stacked_adg, "stacked_app_adgroup"))
    metric_df = pd.DataFrame(rows).sort_values("rmse")
    metric_df.to_csv(out_dir / "granularity_metrics_summary.csv", index=False, encoding="utf-8-sig")
    if not dynamic_plan.empty:
        dynamic_plan.to_csv(out_dir / "fusion_plan_dynamic_daily.csv", index=False, encoding="utf-8-sig")
    if not dynamic_adg.empty:
        dynamic_adg.to_csv(out_dir / "fusion_adgroup_dynamic_daily.csv", index=False, encoding="utf-8-sig")

    comp.to_csv(out_dir / "granularity_predictions_joined.csv", index=False, encoding="utf-8-sig")
    report = {
        "output_dir": str(out_dir),
        "best_model_by_rmse": str(metric_df.iloc[0]["model"]),
        "default_recommended_strategy": "fusion_app_plan",
        "default_recommended_weight": {"app": 0.6, "plan": 0.4},
        "dynamic_fusion_enabled": bool(args.use_dynamic_fusion),
        "metrics_file": str(out_dir / "granularity_metrics_summary.csv"),
        "predictions_file": str(out_dir / "granularity_predictions_joined.csv"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
