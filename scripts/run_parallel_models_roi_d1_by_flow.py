import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from app.experiments.core import evaluate

DEFAULT_CLEAN_GROUP_COLS = [
    "应用ID",
    "推广流量",
    "推广流量名称",
    "流量场景",
    "流量场景名称",
    "创意规格",
    "创意规格名称",
]
DEFAULT_PRODUCT_KEY_COLS = ["应用ID", "广告主ID", "计划ID", "推广流量名称"]


def _build_model(model_name: str):
    if model_name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if model_name == "GBDT":
        return GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42)
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
    return None




def _normalize_cols(raw_cols: List[str]) -> List[str]:
    seen = set()
    cols: List[str] = []
    for c in raw_cols:
        c2 = c.strip()
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


def _load_daily_by_product_flow(
    input_csv: str,
    clean_group_cols: List[str],
    product_key_cols: List[str],
    split_col: str,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    raw = pd.read_csv(input_csv, encoding="utf-8-sig")
    raw["日期"] = pd.to_datetime(raw["日期"], errors="coerce")
    raw = raw[raw["日期"].notna()].copy()
    for c in sorted(set(clean_group_cols + product_key_cols + [split_col])):
        if c not in raw.columns:
            raise ValueError(f"输入数据缺少聚合字段: {c}")
        raw[c] = raw[c].fillna("未知").astype(str)
    for c in ["消耗金额", "首日广告收入", "曝光量", "点击量", "下载量", "激活人数(快应用新增用户数)"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    # 清洗键（按你指定字段用于“数据清洗”，不用于最终预测维度）
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

    # 最终训练/预测维度：产品维度（计划）+ 指定分桶字段
    gcols = list(dict.fromkeys(product_key_cols + [split_col] + ["日期"]))
    df = (
        raw.groupby(gcols, as_index=False)[
            ["消耗金额", "首日广告收入", "曝光量", "点击量", "下载量", "激活人数(快应用新增用户数)"]
        ]
        .sum()
        .sort_values(gcols)
    )
    entity_cols = list(dict.fromkeys(product_key_cols + [split_col]))
    df["entity_id"] = _build_key(df, entity_cols)
    df["split_value"] = df[split_col].fillna("未知").astype(str)

    df["roi_d1"] = np.where(df["消耗金额"] > 0, df["首日广告收入"] / df["消耗金额"], np.nan)
    df["act_per_spend"] = np.where(df["消耗金额"] > 0, df["激活人数(快应用新增用户数)"] / df["消耗金额"], np.nan)
    df["rev_per_act_d1"] = np.where(
        df["激活人数(快应用新增用户数)"] > 0, df["首日广告收入"] / df["激活人数(快应用新增用户数)"], np.nan
    )
    df["dow"] = df["日期"].dt.weekday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    meta = {
        "clean_groups_before": clean_before,
        "clean_groups_after": clean_after,
        "dropped_zero_spend_clean_groups": max(clean_before - clean_after, 0),
        "train_entities_after_clean": int(df["entity_id"].nunique()),
    }
    return df, meta


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gcols = ["entity_id"]
    for lag in [1, 2, 3, 7]:
        out[f"roi_d1_lag_{lag}"] = out.groupby(gcols)["roi_d1"].shift(lag)
        out[f"spend_lag_{lag}"] = out.groupby(gcols)["消耗金额"].shift(lag)
    out["target_t1_roi_d1"] = out.groupby(gcols)["roi_d1"].shift(-1)
    out["target_t1_act_per_spend"] = out.groupby(gcols)["act_per_spend"].shift(-1)
    out["target_t1_rev_per_act_d1"] = out.groupby(gcols)["rev_per_act_d1"].shift(-1)
    return out


def _feature_cols() -> List[str]:
    return [
        "消耗金额",
        "act_per_spend",
        "rev_per_act_d1",
        "dow",
        "is_weekend",
        "roi_d1_lag_1",
        "roi_d1_lag_2",
        "roi_d1_lag_3",
        "roi_d1_lag_7",
        "spend_lag_1",
        "spend_lag_2",
        "spend_lag_3",
        "spend_lag_7",
    ]


def _ewma_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    preds = []
    for entity_id, g_test in test_df.groupby("entity_id"):
        g_train = train_df[train_df["entity_id"] == entity_id].sort_values("日期")
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
                    "推广流量名称": row.get("推广流量名称", "未知"),
                    "split_value": row["split_value"],
                    "entity_id": row["entity_id"],
                    "y_true": row["target_t1_roi_d1"],
                    "y_pred": pred,
                    "model": "EWMA",
                }
            )
    return pd.DataFrame(preds)


def _tree_predict(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    train = train_df.dropna(subset=feats + ["target_t1_roi_d1"]).copy()
    test = test_df.dropna(subset=feats + ["target_t1_roi_d1"]).copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["日期", "应用ID", "推广流量名称", "split_value", "entity_id", "y_true", "y_pred", "model"])
    model = _build_model(model_name)
    if model is None:
        return pd.DataFrame(columns=["日期", "应用ID", "推广流量名称", "split_value", "entity_id", "y_true", "y_pred", "model"])
    model.fit(train[feats], train["target_t1_roi_d1"])
    pred = model.predict(test[feats])
    return pd.DataFrame(
        {
            "日期": test["日期"].values,
            "应用ID": test["应用ID"].values,
            "推广流量名称": test.get("推广流量名称", pd.Series(["未知"] * len(test))).values,
            "split_value": test["split_value"].values,
            "entity_id": test["entity_id"].values,
            "y_true": test["target_t1_roi_d1"].values,
            "y_pred": pred,
            "model": model_name,
        }
    )


def run_by_flow(
    input_csv: str,
    output_dir: str,
    alpha: float = 0.4,
    min_train_days: int = 20,
    models: Optional[List[str]] = None,
    clean_group_cols: Optional[List[str]] = None,
    product_key_cols: Optional[List[str]] = None,
    split_col: str = "推广流量名称",
) -> Dict[str, object]:
    clean_group_cols = clean_group_cols or DEFAULT_CLEAN_GROUP_COLS
    product_key_cols = product_key_cols or DEFAULT_PRODUCT_KEY_COLS
    df, clean_meta = _load_daily_by_product_flow(
        input_csv,
        clean_group_cols=clean_group_cols,
        product_key_cols=product_key_cols,
        split_col=split_col,
    )
    df = _add_lags(df)
    feats = _feature_cols()
    models = models or ["ExtraTrees", "GBDT"]

    pred_frames: List[pd.DataFrame] = []
    for flow_name, flow_df in df.groupby("split_value"):
        flow_df = flow_df.sort_values(["entity_id", "日期"]).copy()
        dates = sorted(flow_df["日期"].dropna().unique())
        if len(dates) <= min_train_days + 1:
            continue
        by_model: Dict[str, List[pd.DataFrame]] = {"EWMA": []}
        for m in models:
            by_model[m] = []

        for i in range(min_train_days, len(dates) - 1):
            cutoff = dates[i]
            pred_day = dates[i + 1]
            train = flow_df[(flow_df["日期"] <= cutoff)].copy()
            test = flow_df[(flow_df["日期"] == pred_day)].copy()
            if train.empty or test.empty:
                continue
            ew = _ewma_predict(train, test, alpha=alpha)
            if not ew.empty:
                by_model["EWMA"].append(ew)
            for m in models:
                tp = _tree_predict(m, train, test, feats)
                if not tp.empty:
                    by_model[m].append(tp)

        for m, arr in by_model.items():
            if not arr:
                continue
            p = pd.concat(arr, ignore_index=True)
            p["split_value"] = flow_name
            pred_frames.append(p)

    if not pred_frames:
        raise ValueError("无可用预测结果：请检查最小训练天数或数据覆盖。")

    pred_all = pd.concat(pred_frames, ignore_index=True)
    pred_all = pred_all[pred_all["y_true"].notna() & pred_all["y_pred"].notna()].copy()

    rows = []
    for (flow, model), g in pred_all.groupby(["split_value", "model"]):
        metrics = evaluate(g["y_true"].values, g["y_pred"].values)
        rows.append(
            {
                "flow": flow,
                "model": model,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "samples": int(len(g)),
            }
        )
    metrics_by_flow = pd.DataFrame(rows).sort_values(["flow", "rmse", "mae"], ascending=[True, True, True])

    overall_rows = []
    for model, g in pred_all.groupby("model"):
        metrics = evaluate(g["y_true"].values, g["y_pred"].values)
        overall_rows.append(
            {
                "model": model,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "samples": int(len(g)),
            }
        )
    metrics_overall = pd.DataFrame(overall_rows).sort_values(["rmse", "mae"], ascending=[True, True])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions_by_flow.csv"
    flow_file = out_dir / "metrics_by_flow.csv"
    overall_file = out_dir / "metrics_overall.csv"
    pred_all.to_csv(pred_file, index=False, encoding="utf-8-sig")
    metrics_by_flow.to_csv(flow_file, index=False, encoding="utf-8-sig")
    metrics_overall.to_csv(overall_file, index=False, encoding="utf-8-sig")

    summary = {
        "input": input_csv,
        "clean_group_cols": clean_group_cols,
        "product_key_cols": product_key_cols,
        "split_col": split_col,
        **clean_meta,
        "flows": sorted([str(x) for x in pred_all["split_value"].dropna().unique().tolist()]),
        "models": sorted([str(x) for x in pred_all["model"].dropna().unique().tolist()]),
        "rows": int(len(pred_all)),
        "output": {
            "predictions_by_flow": str(pred_file),
            "metrics_by_flow": str(flow_file),
            "metrics_overall": str(overall_file),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="按推广流量拆分：同一应用分别训练 roi_d1 模型")
    parser.add_argument("--input", default="daily_20260421_120112.csv")
    parser.add_argument("--output-dir", default="outputs/model_parallel_roi_d1_by_flow")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--min-train-days", type=int, default=20)
    parser.add_argument("--models", default="ExtraTrees,GBDT,RandomForest", help="逗号分隔树模型")
    parser.add_argument(
        "--clean-group-cols",
        default="应用ID,推广流量,推广流量名称,流量场景,流量场景名称,创意规格,创意规格名称",
        help="仅用于清洗“全时段消耗为0”实体的字段列表（逗号分隔，支持重复字段自动去重）",
    )
    parser.add_argument(
        "--product-key-cols",
        default="应用ID,推广流量名称",
        help="实际训练/预测实体字段（默认稳定配置：应用ID+推广流量名称）",
    )
    parser.add_argument("--split-col", default="推广流量名称", help="分桶字段（默认：推广流量名称）")
    args = parser.parse_args()

    model_list = [m.strip() for m in str(args.models).split(",") if m.strip()]
    clean_group_cols = _normalize_cols([c.strip() for c in str(args.clean_group_cols).split(",") if c.strip()])
    product_key_cols = _normalize_cols([c.strip() for c in str(args.product_key_cols).split(",") if c.strip()])
    summary = run_by_flow(
        input_csv=args.input,
        output_dir=args.output_dir,
        alpha=args.alpha,
        min_train_days=args.min_train_days,
        models=model_list,
        clean_group_cols=clean_group_cols,
        product_key_cols=product_key_cols,
        split_col=args.split_col.strip(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

