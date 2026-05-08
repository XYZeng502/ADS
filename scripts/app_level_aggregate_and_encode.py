import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "消耗金额",
    "首日广告收入",
    "3日累计变现金额",
    "7日累计变现金额",
    "30日累计变现金额",
    "买量广告收入",
    "曝光量",
    "点击量",
    "下载量",
    "激活人数(快应用新增用户数)",
]

CATEGORICAL_ID_COLS = [
    "广告主ID",
    "计划ID",
    "广告组ID",
    "创意ID",
    "推广流量名称",
    "流量场景名称",
    "创意规格名称",
    "计费方式",
    "转化类型名称",
    "深度转化类型名称",
]


def _safe_ratio(num: pd.Series, den: pd.Series, zero_value: float = np.nan) -> pd.Series:
    out = pd.Series(zero_value, index=num.index, dtype=float)
    valid = den > 0
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df[df["日期"].notna()]
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df[c] = df[c].clip(lower=0.0)
    return df


def _dominant_category_by_spend(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    tmp = (
        df.groupby(["应用ID", "日期", cat_col], as_index=False)["消耗金额"]
        .sum()
        .sort_values(["应用ID", "日期", "消耗金额"], ascending=[True, True, False])
    )
    # 每个 app-date 保留消耗最高类别
    dom = tmp.drop_duplicates(subset=["应用ID", "日期"], keep="first")[["应用ID", "日期", cat_col]]
    dom = dom.rename(columns={cat_col: f"dom_{cat_col}"})
    return dom


def _build_app_daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    app_daily = (
        df.groupby(["应用ID", "日期"], as_index=False)[NUMERIC_COLS]
        .sum()
        .sort_values(["应用ID", "日期"])
    )

    app_daily["roi_d1"] = _safe_ratio(app_daily["首日广告收入"], app_daily["消耗金额"])
    app_daily["roi_d3"] = _safe_ratio(app_daily["3日累计变现金额"], app_daily["消耗金额"])
    app_daily["roi_d7"] = _safe_ratio(app_daily["7日累计变现金额"], app_daily["消耗金额"])
    app_daily["roi_d30"] = _safe_ratio(app_daily["30日累计变现金额"], app_daily["消耗金额"])
    app_daily["roi_buy"] = _safe_ratio(app_daily["买量广告收入"], app_daily["消耗金额"])
    app_daily["act_per_spend"] = _safe_ratio(app_daily["激活人数(快应用新增用户数)"], app_daily["消耗金额"])
    app_daily["rev_per_act_d1"] = _safe_ratio(
        app_daily["首日广告收入"], app_daily["激活人数(快应用新增用户数)"]
    )
    app_daily["ctr"] = _safe_ratio(app_daily["点击量"], app_daily["曝光量"], zero_value=0.0)
    app_daily["cvr_dl"] = _safe_ratio(app_daily["下载量"], app_daily["点击量"], zero_value=0.0)
    app_daily["cvr_act"] = _safe_ratio(
        app_daily["激活人数(快应用新增用户数)"], app_daily["下载量"], zero_value=0.0
    )
    app_daily["dow"] = app_daily["日期"].dt.weekday
    app_daily["is_weekend"] = (app_daily["dow"] >= 5).astype(int)
    app_daily["dom"] = app_daily["日期"].dt.day
    app_daily["month"] = app_daily["日期"].dt.month

    # D1 波动目标（按应用内逐日）
    app_daily["delta_roi_d1"] = app_daily.groupby("应用ID")["roi_d1"].diff()
    app_daily["abs_delta_roi_d1"] = app_daily["delta_roi_d1"].abs()
    return app_daily


def _svd_embedding_from_category_stats(
    df: pd.DataFrame,
    cat_col: str,
    n_dim: int,
) -> Tuple[pd.DataFrame, int]:
    """
    高基数类别做“统计向量->SVD压缩” embedding。
    不依赖外部深度学习库，适合离线特征工程。
    """
    stats = (
        df.groupby(cat_col, as_index=False)
        .agg(
            rows=("消耗金额", "count"),
            spend_sum=("消耗金额", "sum"),
            spend_mean=("消耗金额", "mean"),
            d1_rev_sum=("首日广告收入", "sum"),
            act_sum=("激活人数(快应用新增用户数)", "sum"),
        )
        .fillna(0.0)
    )
    stats["roi_d1_cat"] = np.where(stats["spend_sum"] > 0, stats["d1_rev_sum"] / stats["spend_sum"], 0.0)
    stats["act_per_spend_cat"] = np.where(stats["spend_sum"] > 0, stats["act_sum"] / stats["spend_sum"], 0.0)

    # 周内分布（7维）
    tmp = df[[cat_col, "日期", "消耗金额"]].copy()
    tmp["dow"] = tmp["日期"].dt.weekday
    dow = tmp.groupby([cat_col, "dow"], as_index=False)["消耗金额"].sum()
    pivot = dow.pivot(index=cat_col, columns="dow", values="消耗金额").fillna(0.0)
    pivot = pivot.div(pivot.sum(axis=1).replace(0, 1.0), axis=0)
    pivot = pivot.add_prefix("dow_share_").reset_index()

    feat = stats.merge(pivot, on=cat_col, how="left").fillna(0.0)
    feature_cols = [c for c in feat.columns if c != cat_col]
    X = feat[feature_cols].to_numpy(dtype=float)

    # 标准化
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd

    use_dim = max(1, min(n_dim, Xn.shape[1]))
    # 经济SVD
    U, S, _ = np.linalg.svd(Xn, full_matrices=False)
    E = U[:, :use_dim] * S[:use_dim]

    out = pd.DataFrame({cat_col: feat[cat_col].astype(str).values})
    for i in range(use_dim):
        out[f"emb_{cat_col}_{i+1}"] = E[:, i]
    return out, use_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="应用维度聚合 + one-hot/embedding 特征分析")
    parser.add_argument("--input", default="daily_20260421_120112.csv")
    parser.add_argument("--output-dir", default="outputs/app_feature_analysis")
    parser.add_argument("--onehot-max-card", type=int, default=8, help="唯一值<=阈值走one-hot")
    parser.add_argument("--embed-dim", type=int, default=4, help="高基数类别embedding维度上限")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, encoding="utf-8-sig")
    df = _clean(raw)
    app_daily = _build_app_daily_agg(df)

    # 加入每个类别的“消耗占优值”
    merged = app_daily.copy()
    dom_cols = []
    for c in CATEGORICAL_ID_COLS:
        dom = _dominant_category_by_spend(df, c)
        merged = merged.merge(dom, on=["应用ID", "日期"], how="left")
        dom_cols.append(f"dom_{c}")
    for c in dom_cols:
        merged[c] = merged[c].astype(str).fillna("NA")

    # 根据唯一值数决定编码方案
    cardinality = {c: int(merged[c].nunique(dropna=False)) for c in dom_cols}
    onehot_cols = [c for c in dom_cols if cardinality[c] <= args.onehot_max_card]
    embed_cols = [c for c in dom_cols if cardinality[c] > args.onehot_max_card]

    # one-hot
    onehot_df = pd.get_dummies(merged[onehot_cols], prefix=onehot_cols, dummy_na=False) if onehot_cols else pd.DataFrame(index=merged.index)

    # embedding
    embed_df = pd.DataFrame(index=merged.index)
    embed_dim_map: Dict[str, int] = {}
    for c in embed_cols:
        emb_map, used_dim = _svd_embedding_from_category_stats(
            df=merged.rename(columns={c: "cat_tmp"}),
            cat_col="cat_tmp",
            n_dim=args.embed_dim,
        )
        emb_map = emb_map.rename(columns={"cat_tmp": c})
        tmp = merged[[c]].astype(str).merge(emb_map, on=c, how="left")
        tmp = tmp.drop(columns=[c])
        tmp = tmp.rename(columns={col: col.replace("emb_cat_tmp_", f"emb_{c}_") for col in tmp.columns})
        embed_df = pd.concat([embed_df, tmp], axis=1)
        embed_dim_map[c] = used_dim

    numeric_feats = [
        "消耗金额",
        "首日广告收入",
        "3日累计变现金额",
        "7日累计变现金额",
        "30日累计变现金额",
        "激活人数(快应用新增用户数)",
        "roi_d1",
        "roi_d3",
        "roi_d7",
        "roi_d30",
        "act_per_spend",
        "rev_per_act_d1",
        "ctr",
        "cvr_dl",
        "cvr_act",
        "dow",
        "is_weekend",
        "dom",
        "month",
    ]
    model_df = pd.concat(
        [
            merged[["应用ID", "日期", "abs_delta_roi_d1", "delta_roi_d1"] + numeric_feats],
            onehot_df,
            embed_df,
        ],
        axis=1,
    )

    # 分析：与 D1 波动强度 abs_delta_roi_d1 相关性
    analysis = []
    target = "abs_delta_roi_d1"
    feature_cols = [c for c in model_df.columns if c not in ["应用ID", "日期", target, "delta_roi_d1"]]
    for c in feature_cols:
        s = model_df[[c, target]].dropna()
        if len(s) >= 12 and s[c].nunique() > 1:
            p = s[c].corr(s[target], method="pearson")
            sp = s[c].corr(s[target], method="spearman")
            analysis.append(
                {
                    "feature": c,
                    "pearson": float(p),
                    "spearman": float(sp),
                    "abs_pearson": float(abs(p)),
                    "abs_spearman": float(abs(sp)),
                }
            )
    corr_df = pd.DataFrame(analysis).sort_values("abs_pearson", ascending=False)

    # 保存输出
    app_daily.to_csv(out_dir / "app_daily_aggregated.csv", index=False, encoding="utf-8-sig")
    model_df.to_csv(out_dir / "app_feature_matrix_encoded.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(out_dir / "app_d1_volatility_feature_correlation.csv", index=False, encoding="utf-8-sig")

    report = {
        "input_file": args.input,
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(df)),
        "app_day_rows": int(len(app_daily)),
        "app_count": int(app_daily["应用ID"].nunique()),
        "date_count": int(app_daily["日期"].nunique()),
        "onehot_max_cardinality": args.onehot_max_card,
        "categorical_cardinality": cardinality,
        "onehot_features": onehot_cols,
        "embedding_features": embed_cols,
        "embedding_dim_used": embed_dim_map,
        "feature_matrix_shape": [int(model_df.shape[0]), int(model_df.shape[1])],
        "top_corr_features": corr_df.head(20).to_dict(orient="records"),
        "outputs": {
            "app_daily_aggregated": str(out_dir / "app_daily_aggregated.csv"),
            "feature_matrix": str(out_dir / "app_feature_matrix_encoded.csv"),
            "feature_correlation": str(out_dir / "app_d1_volatility_feature_correlation.csv"),
        },
    }
    (out_dir / "analysis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
