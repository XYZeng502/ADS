import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=num.index, dtype=float)
    valid = den > 0
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def load_daily_product_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    rename_map = {
        "日期": "date",
        "应用ID": "app_id",
        "广告主ID": "advertiser_id",
        "计划ID": "product_id",
        "消耗金额": "spend",
        "首日广告收入": "rev_d1",
        "3日累计变现金额": "rev_d3",
        "7日累计变现金额": "rev_d7",
        "30日累计变现金额": "rev_d30",
        "买量广告收入": "rev_buy",
    }
    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = ["spend", "rev_d1", "rev_d3", "rev_d7", "rev_d30", "rev_buy"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df[c] = df[c].clip(lower=0.0)

    df = df[df["date"].notna()].copy()

    # 产品+日期聚合
    daily = (
        df.groupby(["product_id", "date"], as_index=False)[numeric_cols]
        .sum()
        .sort_values(["product_id", "date"])
    )
    daily["roi_d1"] = _safe_ratio(daily["rev_d1"], daily["spend"])
    daily["roi_d3"] = _safe_ratio(daily["rev_d3"], daily["spend"])
    daily["roi_d7"] = _safe_ratio(daily["rev_d7"], daily["spend"])
    daily["roi_d30"] = _safe_ratio(daily["rev_d30"], daily["spend"])
    daily["roi_buy"] = _safe_ratio(daily["rev_buy"], daily["spend"])
    return daily


def select_products(daily: pd.DataFrame, top_n: int, min_days: int) -> list[str]:
    stats = (
        daily.groupby("product_id", as_index=False)
        .agg(total_spend=("spend", "sum"), days=("date", "nunique"))
        .sort_values("total_spend", ascending=False)
    )
    stats = stats[stats["days"] >= min_days]
    if top_n > 0:
        stats = stats.head(top_n)
    return stats["product_id"].astype(str).tolist()


def plot_one_product(ts: pd.DataFrame, product_id: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ts["date"], ts["roi_d1"], label="ROI D1", linewidth=1.8)
    ax.plot(ts["date"], ts["roi_d3"], label="ROI D3", linewidth=1.8)
    ax.plot(ts["date"], ts["roi_d7"], label="ROI D7", linewidth=1.8)
    ax.plot(ts["date"], ts["roi_d30"], label="ROI D30", linewidth=1.8)
    ax.plot(ts["date"], ts["roi_buy"], label="ROI Buy", linewidth=1.8, linestyle="--")

    ax.set_title(f"Product {product_id} - Daily ROI Growth Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("ROI")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="按产品绘制每日ROI成长曲线")
    parser.add_argument("--input", default="daily_20260421_120112.csv", help="输入CSV路径")
    parser.add_argument("--output-dir", default="outputs/roi_curves", help="输出目录")
    parser.add_argument("--top-n", type=int, default=30, help="按消耗排序取前N产品；<=0表示全量")
    parser.add_argument("--min-days", type=int, default=7, help="最少出现天数阈值")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = load_daily_product_metrics(in_path)
    product_ids = select_products(daily, top_n=args.top_n, min_days=args.min_days)

    summary_rows = []
    for pid in product_ids:
        ts = daily[daily["product_id"].astype(str) == str(pid)].sort_values("date")
        png_name = f"product_{pid}_roi_curve.png"
        plot_one_product(ts, str(pid), out_dir / png_name)
        summary_rows.append(
            {
                "product_id": pid,
                "days": int(ts["date"].nunique()),
                "total_spend": float(ts["spend"].sum()),
                "avg_roi_d1": float(ts["roi_d1"].mean(skipna=True)),
                "avg_roi_d7": float(ts["roi_d7"].mean(skipna=True)),
                "avg_roi_d30": float(ts["roi_d30"].mean(skipna=True)),
                "curve_file": png_name,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("total_spend", ascending=False)
    summary_file = out_dir / "product_roi_curve_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print(f"输出产品数: {len(product_ids)}")
    print(f"图片目录: {out_dir}")
    print(f"汇总文件: {summary_file}")


if __name__ == "__main__":
    main()
