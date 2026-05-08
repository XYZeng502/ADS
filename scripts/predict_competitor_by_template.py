import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NODE_DAYS = np.array([1, 3, 7, 30], dtype=float)


def _parse_set(v: str) -> set[str]:
    if not v:
        return set()
    return {x.strip() for x in v.split(",") if x.strip()}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _monotone_cubic_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    n = len(x)
    if n == 1:
        return np.full_like(x_new, y[0], dtype=float)
    if n == 2:
        return np.interp(x_new, x, y, left=y[0], right=y[-1])

    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n, dtype=float)
    m[0] = delta[0]
    m[-1] = delta[-1]

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    y_new = np.empty_like(x_new, dtype=float)
    y_new[x_new <= x[0]] = y[0]
    y_new[x_new >= x[-1]] = y[-1]

    mask_mid = (x_new > x[0]) & (x_new < x[-1])
    xm = x_new[mask_mid]
    idx = np.searchsorted(x, xm, side="right") - 1
    idx = np.clip(idx, 0, n - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]
    hi = x1 - x0
    t = (xm - x0) / hi

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    y_new[mask_mid] = h00 * y0 + h10 * hi * m0 + h01 * y1 + h11 * hi * m1
    return y_new


def load_mature_cohorts(
    csv_path: Path,
    app_ids: set[str],
    advertiser_ids: set[str],
    plan_ids: set[str],
    min_spend: float,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df[df["日期"].notna()].copy()

    for c in ["消耗金额", "首日广告收入", "3日累计变现金额", "7日累计变现金额", "30日累计变现金额"]:
        df[c] = _to_num(df[c]).clip(lower=0.0)

    if app_ids:
        df = df[df["应用ID"].astype(str).isin(app_ids)]
    if advertiser_ids:
        df = df[df["广告主ID"].astype(str).isin(advertiser_ids)]
    if plan_ids:
        df = df[df["计划ID"].astype(str).isin(plan_ids)]

    as_of = df["日期"].max()
    df["age_days"] = (as_of - df["日期"]).dt.days + 1

    # 成熟样本：可观测到D30，且D1锚点可用
    mature = df[
        (df["age_days"] >= 30)
        & (df["消耗金额"] > min_spend)
        & (df["首日广告收入"] > 0)
    ].copy()
    return mature


def build_multiplier_template(mature: pd.DataFrame) -> dict:
    m3 = mature["3日累计变现金额"] / mature["首日广告收入"]
    m7 = mature["7日累计变现金额"] / mature["首日广告收入"]
    m30 = mature["30日累计变现金额"] / mature["首日广告收入"]

    mult = pd.DataFrame({"m1": 1.0, "m3": m3, "m7": m7, "m30": m30}).replace([np.inf, -np.inf], np.nan)
    mult = mult.dropna()
    mult = mult[(mult["m3"] > 0) & (mult["m7"] > 0) & (mult["m30"] > 0)].copy()

    # 单条样本单调修正：累计回收倍率不应下降
    arr = mult[["m1", "m3", "m7", "m30"]].to_numpy(dtype=float)
    arr = np.maximum.accumulate(arr, axis=1)
    mult[["m1", "m3", "m7", "m30"]] = arr

    template = {}
    for col in ["m1", "m3", "m7", "m30"]:
        v = mult[col].to_numpy()
        template[col] = {
            "p25": float(np.quantile(v, 0.25)),
            "p50": float(np.quantile(v, 0.50)),
            "p75": float(np.quantile(v, 0.75)),
            "mean": float(np.mean(v)),
        }
    template["sample_count"] = int(len(mult))
    return template


def _interp_curve(mult_nodes: np.ndarray, max_day: int = 30, interp_method: str = "monotone_cubic") -> np.ndarray:
    x = NODE_DAYS
    y = np.maximum.accumulate(mult_nodes)
    days = np.arange(1, max_day + 1, dtype=float)
    if interp_method == "linear":
        return np.interp(days, x, y, left=y[0], right=y[-1])
    return _monotone_cubic_interp(x, y, days)


def forecast_by_d1(template: dict, d1_roi: float, max_day: int = 30, interp_method: str = "monotone_cubic") -> dict:
    p50_nodes = np.array([template["m1"]["p50"], template["m3"]["p50"], template["m7"]["p50"], template["m30"]["p50"]])
    p25_nodes = np.array([template["m1"]["p25"], template["m3"]["p25"], template["m7"]["p25"], template["m30"]["p25"]])
    p75_nodes = np.array([template["m1"]["p75"], template["m3"]["p75"], template["m7"]["p75"], template["m30"]["p75"]])

    m_curve = _interp_curve(p50_nodes, max_day=max_day, interp_method=interp_method)
    low_curve = _interp_curve(p25_nodes, max_day=max_day, interp_method=interp_method)
    up_curve = _interp_curve(p75_nodes, max_day=max_day, interp_method=interp_method)

    days = np.arange(1, max_day + 1)
    roi_curve = d1_roi * m_curve
    roi_low = d1_roi * low_curve
    roi_up = d1_roi * up_curve

    return {
        "days": days,
        "roi_curve": roi_curve,
        "roi_low": roi_low,
        "roi_up": roi_up,
        "node_forecast": {
            "roi_d1": float(d1_roi * p50_nodes[0]),
            "roi_d3": float(d1_roi * p50_nodes[1]),
            "roi_d7": float(d1_roi * p50_nodes[2]),
            "roi_d30": float(d1_roi * p50_nodes[3]),
        },
        "node_forecast_p25": {
            "roi_d1": float(d1_roi * p25_nodes[0]),
            "roi_d3": float(d1_roi * p25_nodes[1]),
            "roi_d7": float(d1_roi * p25_nodes[2]),
            "roi_d30": float(d1_roi * p25_nodes[3]),
        },
        "node_forecast_p75": {
            "roi_d1": float(d1_roi * p75_nodes[0]),
            "roi_d3": float(d1_roi * p75_nodes[1]),
            "roi_d7": float(d1_roi * p75_nodes[2]),
            "roi_d30": float(d1_roi * p75_nodes[3]),
        },
    }


def save_outputs(
    output_dir: Path, name: str, template: dict, result: dict, d1_roi: float, filters: dict, interp_method: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_df = pd.DataFrame(
        {
            "day": result["days"],
            "roi_pred_p50": result["roi_curve"],
            "roi_pred_p25": result["roi_low"],
            "roi_pred_p75": result["roi_up"],
        }
    )
    curve_csv = output_dir / f"{name}_competitor_roi_curve.csv"
    curve_df.to_csv(curve_csv, index=False, encoding="utf-8-sig")

    meta = {
        "name": name,
        "d1_roi_anchor": d1_roi,
        "filters": filters,
        "template": template,
        "interp_method": interp_method,
        "forecast_nodes": result["node_forecast"],
        "forecast_nodes_p25": result["node_forecast_p25"],
        "forecast_nodes_p75": result["node_forecast_p75"],
        "curve_file": str(curve_csv),
    }
    meta_json = output_dir / f"{name}_competitor_roi_forecast.json"
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(result["days"], result["roi_curve"], color="#2563eb", linewidth=2.2, label="Predicted ROI (P50)")
    ax.fill_between(result["days"], result["roi_low"], result["roi_up"], color="#93c5fd", alpha=0.35, label="P25-P75 band")
    ax.set_title(f"Competitor ROI Forecast by Category Template ({name})")
    ax.set_xlabel("Cohort Age (day)")
    ax.set_ylabel("Cumulative ROI")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    png = output_dir / f"{name}_competitor_roi_curve.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    print(f"template sample_count: {template['sample_count']}")
    print(f"node forecast: {result['node_forecast']}")
    print(f"curve csv: {curve_csv}")
    print(f"plot png: {png}")
    print(f"meta json: {meta_json}")


def main():
    parser = argparse.ArgumentParser(description="基于同类cohort倍率模板预测竞品ROI成长曲线")
    parser.add_argument("--input", default="daily_20260421_120112.csv", help="输入CSV")
    parser.add_argument("--name", default="competitor_a", help="竞品名称标识")
    parser.add_argument("--d1-roi", type=float, required=True, help="竞品D1 ROI锚点，例如0.65")
    parser.add_argument("--app-ids", default="", help="模板样本应用ID列表，逗号分隔")
    parser.add_argument("--advertiser-ids", default="", help="模板样本广告主ID列表，逗号分隔")
    parser.add_argument("--plan-ids", default="", help="模板样本计划ID列表，逗号分隔")
    parser.add_argument("--min-spend", type=float, default=0.0, help="模板样本最小消耗过滤")
    parser.add_argument("--output-dir", default="outputs/competitor_forecast", help="输出目录")
    parser.add_argument("--max-day", type=int, default=30, help="输出曲线最大天数")
    parser.add_argument(
        "--interp-method",
        choices=["monotone_cubic", "linear"],
        default="monotone_cubic",
        help="曲线插值方法：monotone_cubic更平滑有弧度。",
    )
    args = parser.parse_args()

    app_ids = _parse_set(args.app_ids)
    advertiser_ids = _parse_set(args.advertiser_ids)
    plan_ids = _parse_set(args.plan_ids)

    mature = load_mature_cohorts(
        csv_path=Path(args.input),
        app_ids=app_ids,
        advertiser_ids=advertiser_ids,
        plan_ids=plan_ids,
        min_spend=args.min_spend,
    )
    if len(mature) < 20:
        raise ValueError(f"成熟样本过少（{len(mature)}），建议放宽过滤条件或降低min-spend。")

    template = build_multiplier_template(mature)
    result = forecast_by_d1(
        template,
        d1_roi=args.d1_roi,
        max_day=max(7, args.max_day),
        interp_method=args.interp_method,
    )
    save_outputs(
        output_dir=Path(args.output_dir),
        name=args.name,
        template=template,
        result=result,
        d1_roi=args.d1_roi,
        interp_method=args.interp_method,
        filters={
            "app_ids": sorted(list(app_ids)),
            "advertiser_ids": sorted(list(advertiser_ids)),
            "plan_ids": sorted(list(plan_ids)),
            "min_spend": args.min_spend,
            "max_day": args.max_day,
        },
    )


if __name__ == "__main__":
    main()
