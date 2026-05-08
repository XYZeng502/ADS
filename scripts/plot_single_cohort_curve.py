import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NODE_DAYS = np.array([1, 3, 7, 30], dtype=float)


def _to_float(v) -> float:
    try:
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _monotone_cubic_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Fritsch-Carlson 单调三次Hermite插值。
    特点：平滑、有弧度、保持单调，不会产生明显过冲。
    """
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

    y_mid = h00 * y0 + h10 * hi * m0 + h01 * y1 + h11 * hi * m1
    y_new[mask_mid] = y_mid
    return y_new


def _pick_cohort_row(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.row_index is not None:
        idx = int(args.row_index)
        if idx < 0 or idx >= len(df):
            raise ValueError(f"row_index超出范围: {idx}, 总行数={len(df)}")
        return df.iloc[idx]

    mask = pd.Series(True, index=df.index)
    if args.cohort_date:
        target_date = pd.to_datetime(args.cohort_date, errors="coerce")
        if pd.isna(target_date):
            raise ValueError("cohort_date格式错误，应为YYYY-MM-DD")
        mask &= df["日期"] == target_date
    if args.app_id:
        mask &= df["应用ID"].astype(str) == str(args.app_id)
    if args.advertiser_id:
        mask &= df["广告主ID"].astype(str) == str(args.advertiser_id)
    if args.plan_id:
        mask &= df["计划ID"].astype(str) == str(args.plan_id)
    if args.adgroup_id:
        mask &= df["广告组ID"].astype(str) == str(args.adgroup_id)
    if args.creative_id:
        mask &= df["创意ID"].astype(str) == str(args.creative_id)

    cand = df[mask].copy()
    if cand.empty:
        raise ValueError("未匹配到cohort，请检查筛选条件。")

    cand["消耗金额_num"] = pd.to_numeric(cand["消耗金额"], errors="coerce").fillna(0.0)
    cand["age_days"] = (df["日期"].max() - cand["日期"]).dt.days + 1
    if args.mature_only:
        mature_need = max(1, int(args.max_day))
        mature_cand = cand[cand["age_days"] >= mature_need]
        if mature_cand.empty:
            youngest = int(cand["age_days"].max()) if not cand.empty else 0
            raise ValueError(
                f"当前筛选结果无成熟样本（要求 age_days>={mature_need}，当前最大仅{youngest}）。"
            )
        cand = mature_cand
    # 若匹配多条，默认取消耗最大的一条，避免选到噪声样本
    return cand.sort_values("消耗金额_num", ascending=False).iloc[0]


def _build_curve(row: pd.Series, max_day: int = 30, interp_method: str = "monotone_cubic"):
    spend = _to_float(row.get("消耗金额"))
    if not np.isfinite(spend) or spend <= 0:
        raise ValueError("该cohort消耗金额<=0，无法计算ROI成长曲线。")

    rev_nodes = np.array(
        [
            _to_float(row.get("首日广告收入")),
            _to_float(row.get("3日累计变现金额")),
            _to_float(row.get("7日累计变现金额")),
            _to_float(row.get("30日累计变现金额")),
        ],
        dtype=float,
    )
    roi_nodes = rev_nodes / spend

    valid = np.isfinite(roi_nodes)
    if valid.sum() == 0:
        raise ValueError("D1/D3/D7/D30 均缺失，无法构建曲线。")

    x = NODE_DAYS[valid]
    y = roi_nodes[valid]

    # 对累计ROI做单调修正，避免数据噪声导致“回落”
    y = np.maximum.accumulate(y)

    days = np.arange(1, max_day + 1, dtype=float)
    if len(x) == 1:
        curve = np.full_like(days, y[0], dtype=float)
    else:
        if interp_method == "linear":
            curve = np.interp(days, x, y, left=y[0], right=y[-1])
        else:
            curve = _monotone_cubic_interp(x, y, days)

    node_map = {
        "roi_d1": float(roi_nodes[0]) if np.isfinite(roi_nodes[0]) else None,
        "roi_d3": float(roi_nodes[1]) if np.isfinite(roi_nodes[1]) else None,
        "roi_d7": float(roi_nodes[2]) if np.isfinite(roi_nodes[2]) else None,
        "roi_d30": float(roi_nodes[3]) if np.isfinite(roi_nodes[3]) else None,
    }
    return spend, days.astype(int), curve, node_map


def _plot_curve(days: np.ndarray, curve: np.ndarray, node_map: dict, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(days, curve, label="Simulated ROI curve", linewidth=2.2, color="#2563eb")

    # 标注关键节点
    node_points = {
        1: node_map.get("roi_d1"),
        3: node_map.get("roi_d3"),
        7: node_map.get("roi_d7"),
        30: node_map.get("roi_d30"),
    }
    for d, v in node_points.items():
        if v is not None and np.isfinite(v):
            ax.scatter([d], [v], color="#dc2626", s=40, zorder=3)
            ax.text(d, v, f" D{d}:{v:.3f}", fontsize=9, va="bottom")

    ax.set_title(title)
    ax.set_xlabel("Cohort Age (day)")
    ax.set_ylabel("Cumulative ROI")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="单cohort ROI成长曲线模拟")
    parser.add_argument("--input", default="daily_20260421_120112.csv", help="输入CSV")
    parser.add_argument("--output-dir", default="outputs/cohort_curves", help="输出目录")
    parser.add_argument("--row-index", type=int, default=None, help="直接按原始CSV行号选cohort（0-based）")
    parser.add_argument("--cohort-date", default="", help="cohort日期 YYYY-MM-DD")
    parser.add_argument("--app-id", default="", help="应用ID")
    parser.add_argument("--advertiser-id", default="", help="广告主ID")
    parser.add_argument("--plan-id", default="", help="计划ID")
    parser.add_argument("--adgroup-id", default="", help="广告组ID")
    parser.add_argument("--creative-id", default="", help="创意ID")
    parser.add_argument("--max-day", type=int, default=30, help="输出曲线最大天数")
    parser.add_argument(
        "--interp-method",
        choices=["monotone_cubic", "linear"],
        default="monotone_cubic",
        help="曲线插值方法：monotone_cubic更平滑有弧度，linear为线性。",
    )
    parser.add_argument(
        "--allow-immature",
        action="store_true",
        help="允许使用未成熟样本（默认关闭，即默认只做成熟样本）",
    )
    args = parser.parse_args()
    args.mature_only = not args.allow_immature

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")

    row = _pick_cohort_row(df, args)
    spend, days, curve, node_map = _build_curve(
        row,
        max_day=max(7, args.max_day),
        interp_method=args.interp_method,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_key = (
        f"{row.get('日期').date()}_app{row.get('应用ID')}_adv{row.get('广告主ID')}"
        f"_plan{row.get('计划ID')}_adg{row.get('广告组ID')}_cr{row.get('创意ID')}"
    )
    safe_name = cohort_key.replace(" ", "_").replace(":", "_")
    out_png = out_dir / f"cohort_curve_{safe_name}.png"
    out_csv = out_dir / f"cohort_curve_{safe_name}.csv"
    out_json = out_dir / f"cohort_curve_{safe_name}.json"

    curve_df = pd.DataFrame({"day": days, "roi_curve": curve})
    curve_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "cohort_key": {
            "date": str(row.get("日期").date()),
            "app_id": str(row.get("应用ID")),
            "advertiser_id": str(row.get("广告主ID")),
            "plan_id": str(row.get("计划ID")),
            "adgroup_id": str(row.get("广告组ID")),
            "creative_id": str(row.get("创意ID")),
        },
        "spend": float(spend),
        "roi_nodes": node_map,
        "m30_multiplier": (node_map["roi_d30"] / node_map["roi_d1"])
        if node_map["roi_d1"] not in (None, 0) and node_map["roi_d30"] is not None
        else None,
        "interp_method": args.interp_method,
        "curve_file": str(out_csv),
        "plot_file": str(out_png),
    }
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    title = (
        f"Cohort ROI Curve | {meta['cohort_key']['date']} | "
        f"app={meta['cohort_key']['app_id']} plan={meta['cohort_key']['plan_id']}"
    )
    _plot_curve(days, curve, node_map, out_png=out_png, title=title)

    print(f"选中cohort: {meta['cohort_key']}")
    print(f"spend: {meta['spend']:.2f}")
    print(f"roi_nodes: {meta['roi_nodes']}")
    print(f"曲线数据: {out_csv}")
    print(f"曲线图片: {out_png}")
    print(f"元信息: {out_json}")


if __name__ == "__main__":
    main()
