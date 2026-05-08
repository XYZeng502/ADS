import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NODE_DAYS = [1, 3, 7, 30]


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=num.index, dtype=float)
    m = den > 0
    out.loc[m] = num.loc[m] / den.loc[m]
    return out


def _normalize_quality_score(values: list[float]) -> float:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return 0.0
    # 小离散=高一致性。把CV映射到[0,1]分数。
    cv = float(np.mean(arr))
    score = 1.0 - min(max(cv / 0.6, 0.0), 1.0)
    return round(score, 4)


def load_cohort_table(csv_path: Path, as_of_date: str = "") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    rename_map = {
        "日期": "cohort_date",
        "应用ID": "app_id",
        "广告主ID": "advertiser_id",
        "计划ID": "plan_id",
        "消耗金额": "spend",
        "首日广告收入": "rev_d1",
        "3日累计变现金额": "rev_d3",
        "7日累计变现金额": "rev_d7",
        "30日累计变现金额": "rev_d30",
    }
    df = df.rename(columns=rename_map)
    df["cohort_date"] = pd.to_datetime(df["cohort_date"], errors="coerce")
    for c in ["spend", "rev_d1", "rev_d3", "rev_d7", "rev_d30"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df[df["cohort_date"].notna()].copy()

    if as_of_date:
        as_of = pd.to_datetime(as_of_date, errors="coerce")
        if pd.isna(as_of):
            raise ValueError("as_of_date 格式错误，应为 YYYY-MM-DD")
    else:
        as_of = df["cohort_date"].max()
    df["age_days"] = (as_of - df["cohort_date"]).dt.days + 1
    df["product_uid"] = df["advertiser_id"].astype(str) + ":" + df["plan_id"].astype(str)

    # cohort聚合：同产品同cohort日合并
    gcols = ["product_uid", "advertiser_id", "plan_id", "app_id", "cohort_date", "age_days"]
    cohort = df.groupby(gcols, as_index=False)[["spend", "rev_d1", "rev_d3", "rev_d7", "rev_d30"]].sum()
    cohort["roi_d1"] = _safe_ratio(cohort["rev_d1"], cohort["spend"])
    cohort["roi_d3"] = _safe_ratio(cohort["rev_d3"], cohort["spend"])
    cohort["roi_d7"] = _safe_ratio(cohort["rev_d7"], cohort["spend"])
    cohort["roi_d30"] = _safe_ratio(cohort["rev_d30"], cohort["spend"])
    return cohort


def analyze_consistency(
    cohort: pd.DataFrame,
    strict_age_gt: bool,
    min_mature_d30: int,
    max_cv_m30: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if strict_age_gt:
        valid_d1 = cohort["age_days"] > 1
        valid_d3 = cohort["age_days"] > 3
        valid_d7 = cohort["age_days"] > 7
        valid_d30 = cohort["age_days"] > 30
    else:
        valid_d1 = cohort["age_days"] >= 1
        valid_d3 = cohort["age_days"] >= 3
        valid_d7 = cohort["age_days"] >= 7
        valid_d30 = cohort["age_days"] >= 30

    work = cohort.copy()
    work.loc[~valid_d1, "roi_d1"] = np.nan
    work.loc[~valid_d3, "roi_d3"] = np.nan
    work.loc[~valid_d7, "roi_d7"] = np.nan
    work.loc[~valid_d30, "roi_d30"] = np.nan

    # 归一化成长倍率（相对D1），用于判断“曲线形状是否一致”
    work["m3"] = np.where((work["roi_d1"] > 0) & work["roi_d3"].notna(), work["roi_d3"] / work["roi_d1"], np.nan)
    work["m7"] = np.where((work["roi_d1"] > 0) & work["roi_d7"].notna(), work["roi_d7"] / work["roi_d1"], np.nan)
    work["m30"] = np.where((work["roi_d1"] > 0) & work["roi_d30"].notna(), work["roi_d30"] / work["roi_d1"], np.nan)

    rows = []
    for pid, g in work.groupby("product_uid"):
        def _cv(s: pd.Series) -> float:
            s = s.dropna()
            if len(s) < 2:
                return np.nan
            mu = float(s.mean())
            if mu <= 1e-8:
                return np.nan
            return float(s.std(ddof=0) / mu)

        cv_m3 = _cv(g["m3"])
        cv_m7 = _cv(g["m7"])
        cv_m30 = _cv(g["m30"])
        mature_d30 = int(g["m30"].notna().sum())
        p50_m3 = float(g["m3"].median()) if g["m3"].notna().any() else np.nan
        p50_m7 = float(g["m7"].median()) if g["m7"].notna().any() else np.nan
        p50_m30 = float(g["m30"].median()) if g["m30"].notna().any() else np.nan

        # 以模板中位曲线为基准，算样本偏差（越小越一致）
        template = np.array([1.0, p50_m3, p50_m7, p50_m30], dtype=float)
        errs = []
        for _, r in g.iterrows():
            vec = np.array([1.0, r["m3"], r["m7"], r["m30"]], dtype=float)
            m = np.isfinite(vec) & np.isfinite(template)
            if m.sum() >= 2:
                errs.append(float(np.mean(np.abs(vec[m] - template[m]))))
        mean_shape_mae = float(np.mean(errs)) if errs else np.nan

        consistency_score = _normalize_quality_score([v for v in [cv_m3, cv_m7, cv_m30] if np.isfinite(v)])
        template_usable = (mature_d30 >= min_mature_d30) and (np.isfinite(cv_m30) and cv_m30 <= max_cv_m30)

        rows.append(
            {
                "product_uid": pid,
                "cohort_count": int(len(g)),
                "mature_d30_count": mature_d30,
                "median_m3": p50_m3,
                "median_m7": p50_m7,
                "median_m30": p50_m30,
                "cv_m3": cv_m3,
                "cv_m7": cv_m7,
                "cv_m30": cv_m30,
                "mean_shape_mae": mean_shape_mae,
                "consistency_score": consistency_score,
                "template_usable": bool(template_usable),
            }
        )

    product_summary = pd.DataFrame(rows).sort_values(
        ["template_usable", "consistency_score", "mature_d30_count"], ascending=[False, False, False]
    )

    # 按应用维度再做一版一致性统计（回答“同应用曲线是否一致”）
    app_rows = []
    for app_id, g in work.groupby("app_id"):
        g = g.copy()
        g["m3"] = np.where((g["roi_d1"] > 0) & g["roi_d3"].notna(), g["roi_d3"] / g["roi_d1"], np.nan)
        g["m7"] = np.where((g["roi_d1"] > 0) & g["roi_d7"].notna(), g["roi_d7"] / g["roi_d1"], np.nan)
        g["m30"] = np.where((g["roi_d1"] > 0) & g["roi_d30"].notna(), g["roi_d30"] / g["roi_d1"], np.nan)

        def _cv(s: pd.Series) -> float:
            s = s.dropna()
            if len(s) < 2:
                return np.nan
            mu = float(s.mean())
            if mu <= 1e-8:
                return np.nan
            return float(s.std(ddof=0) / mu)

        cv_m3 = _cv(g["m3"])
        cv_m7 = _cv(g["m7"])
        cv_m30 = _cv(g["m30"])
        mature_d30 = int(g["m30"].notna().sum())
        score = _normalize_quality_score([v for v in [cv_m3, cv_m7, cv_m30] if np.isfinite(v)])
        app_template_usable = (mature_d30 >= min_mature_d30) and (np.isfinite(cv_m30) and cv_m30 <= max_cv_m30)
        app_rows.append(
            {
                "app_id": str(app_id),
                "cohort_count": int(len(g)),
                "mature_d30_count": mature_d30,
                "cv_m3": cv_m3,
                "cv_m7": cv_m7,
                "cv_m30": cv_m30,
                "consistency_score": score,
                "template_usable": bool(app_template_usable),
            }
        )
    app_summary = pd.DataFrame(app_rows).sort_values(
        ["template_usable", "consistency_score", "mature_d30_count"], ascending=[False, False, False]
    )
    return work, product_summary, app_summary


def plot_top_products(product_summary: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    top = product_summary.head(max(top_n, 0)).copy()
    if top.empty:
        return
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, top["median_m7"], width=0.2, label="median_m7")
    ax.bar(x, top["median_m30"], width=0.2, label="median_m30")
    ax.bar(x + 0.2, top["consistency_score"], width=0.2, label="consistency_score")
    ax.set_xticks(x)
    ax.set_xticklabels(top["product_uid"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Top Products: Curve Template Stability")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "top_products_curve_stability.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="同产品cohort成长曲线一致性分析（含成熟样本过滤）")
    parser.add_argument("--input", default="daily_20260421_120112.csv", help="输入CSV")
    parser.add_argument("--output-dir", default="outputs/cohort_curve_consistency", help="输出目录")
    parser.add_argument("--as-of-date", default="", help="计算age基准日期，默认用数据最大日期")
    parser.add_argument("--strict-age-gt", action="store_true", help="使用 age>n 判定有效；默认 age>=n")
    parser.add_argument("--min-mature-d30", type=int, default=20, help="模板可用最小成熟cohort数")
    parser.add_argument("--max-cv-m30", type=float, default=0.25, help="模板可用的m30最大CV阈值")
    parser.add_argument("--top-n-plot", type=int, default=20, help="绘图展示前N产品")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort_table(Path(args.input), as_of_date=args.as_of_date)
    work, summary, app_summary = analyze_consistency(
        cohort,
        strict_age_gt=args.strict_age_gt,
        min_mature_d30=args.min_mature_d30,
        max_cv_m30=args.max_cv_m30,
    )

    detail_file = out_dir / "cohort_curve_detail.csv"
    summary_file = out_dir / "product_curve_consistency_summary.csv"
    app_summary_file = out_dir / "app_curve_consistency_summary.csv"
    work.to_csv(detail_file, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    app_summary.to_csv(app_summary_file, index=False, encoding="utf-8-sig")
    plot_top_products(summary, out_dir, top_n=args.top_n_plot)

    usable = int(summary["template_usable"].sum()) if not summary.empty else 0
    app_usable = int(app_summary["template_usable"].sum()) if not app_summary.empty else 0
    report = {
        "input_file": args.input,
        "products": int(summary["product_uid"].nunique()) if not summary.empty else 0,
        "template_usable_products": usable,
        "template_usable_ratio": round(usable / max(len(summary), 1), 4),
        "apps": int(app_summary["app_id"].nunique()) if not app_summary.empty else 0,
        "template_usable_apps": app_usable,
        "template_usable_app_ratio": round(app_usable / max(len(app_summary), 1), 4),
        "global_median_app_cv_m30": float(app_summary["cv_m30"].median(skipna=True)) if not app_summary.empty else None,
        "global_median_cv_m30": float(summary["cv_m30"].median(skipna=True)) if not summary.empty else None,
        "global_median_consistency_score": float(summary["consistency_score"].median(skipna=True)) if not summary.empty else None,
        "age_valid_rule": "age>n" if args.strict_age_gt else "age>=n",
        "outputs": {
            "detail_csv": str(detail_file),
            "summary_csv": str(summary_file),
            "app_summary_csv": str(app_summary_file),
            "plot_png": str(out_dir / "top_products_curve_stability.png"),
        },
    }
    (out_dir / "analysis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

