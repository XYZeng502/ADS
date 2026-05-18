#!/usr/bin/env python
"""每日买量收入预测：用释放曲线模型预测每日总收入，对比实际买量广告收入。"""
import csv
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from app.core.calendar import CalendarService

_CSV_PATH = Path("daily_merged.csv")
_OUTPUT_DIR = Path("outputs")
_CURVE_PATH = Path("outputs/per_app_release_curves.json")

# 默认倍率节点
_NODE_DAYS = [1, 3, 7, 30]
_NODE_MULT = [1.0, 1.1942, 1.3107, 1.4666]


def to_float(v: str) -> float:
    try:
        return float(v) if v not in ("", None) else 0.0
    except ValueError:
        return 0.0


def load_app_daily(csv_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    读取 CSV，按 (app_id, day) 聚合 spend / D1 / buy_revenue。
    返回: {app_id: {day_str: {"spend": ..., "d1_revenue": ..., "buy_revenue": ...}}}
    """
    app_daily: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"spend": 0.0, "d1_revenue": 0.0, "buy_revenue": 0.0})
    )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            app_id = row.get("应用ID", "")
            day = row.get("日期", "")
            if not app_id or not day:
                continue
            d = app_daily[app_id][day]
            d["spend"] += to_float(row.get("消耗金额", "0"))
            d["d1_revenue"] += to_float(row.get("首日广告收入", "0"))
            d["buy_revenue"] += to_float(row.get("买量广告收入", "0"))
    return app_daily


def _build_default_curve() -> Dict[int, float]:
    curve: Dict[int, float] = {}
    for d in range(1, 31):
        if d <= _NODE_DAYS[0]:
            curve[d] = _NODE_MULT[0]
        elif d >= _NODE_DAYS[-1]:
            curve[d] = _NODE_MULT[-1]
        else:
            for i in range(len(_NODE_DAYS) - 1):
                d0, d1 = _NODE_DAYS[i], _NODE_DAYS[i + 1]
                if d0 <= d <= d1:
                    y0, y1 = _NODE_MULT[i], _NODE_MULT[i + 1]
                    curve[d] = y0 + (y1 - y0) * (d - d0) / (d1 - d0)
                    break
    return curve


def load_curves(curve_path: Path) -> Dict[str, Dict[int, float]]:
    """加载 per-app 释放曲线。文件缺失返回空 dict。"""
    if not curve_path.exists():
        print(f"  [WARN] 曲线文件缺失: {curve_path}，全部使用默认曲线")
        return {}
    raw = json.loads(curve_path.read_text(encoding="utf-8"))
    return {
        app_id: {int(d): m for d, m in days.items()}
        for app_id, days in raw.items()
    }


def _curve_value(curve: Dict[int, float], age_day: int) -> float:
    if age_day <= 0:
        return 0.0
    max_day = max(curve.keys())
    age = min(age_day, max_day)
    return curve.get(age, curve[max_day])


def build_cohorts_until(
    app_daily: Dict[str, Dict[str, Dict[str, float]]],
    before_date: date,
) -> Dict[str, List[Tuple[date, float, float]]]:
    """从 CSV 构建每个 app 的历史 cohort（不含 before_date 当日）"""
    cohorts_map: Dict[str, List[Tuple[date, float, float]]] = {}
    for app_id, day_data in app_daily.items():
        sorted_days = sorted(day_data.keys())
        cohorts: List[Tuple[date, float, float]] = []
        for day_str in sorted_days:
            current_day = datetime.strptime(day_str, "%Y-%m-%d").date()
            if current_day >= before_date:
                break
            info = day_data[day_str]
            spend = info["spend"]
            d1_rev = info["d1_revenue"]
            if spend > 0 and d1_rev > 0:
                cohorts.append((current_day, spend, d1_rev / spend))
        if cohorts:
            cohorts_map[app_id] = cohorts
    return cohorts_map


def predict_single_day(
    target_date: date,
    cohorts_map: Dict[str, List[Tuple[date, float, float]]],
    day_inputs: List[Dict],
    curves: Dict[str, Dict[int, float]],
) -> List[Dict]:
    """在线预测单天买量收入。day_inputs: [{"应用ID": str, "spend": float, "d1_revenue": float}]"""
    default_curve = _build_default_curve()
    rows: List[Dict] = []

    inputs_by_app: Dict[str, Dict] = {}
    for inp in day_inputs:
        aid = inp.get("应用ID", "")
        if aid:
            inputs_by_app[aid] = inp

    for app_id, cohorts in cohorts_map.items():
        inp = inputs_by_app.get(app_id)
        if inp is None or inp["spend"] <= 0:
            continue

        new_spend = inp["spend"]
        new_d1_roi = inp["d1_revenue"] / new_spend

        all_cohorts = cohorts + [(target_date, new_spend, new_d1_roi)]
        curve = curves.get(app_id, default_curve)

        predicted = 0.0
        for cday, cspend, cd1 in all_cohorts:
            age_today = (target_date - cday).days + 1
            age_yesterday = age_today - 1
            f_today = _curve_value(curve, age_today)
            f_yesterday = _curve_value(curve, age_yesterday)
            delta = f_today - f_yesterday
            if delta > 0:
                predicted += cspend * cd1 * delta

        # 休息日校准：周末/节假日用户活跃度更高，释放曲线需按 day_type scale
        cal = CalendarService()
        day_type = cal.classify_day(target_date)
        predicted *= cal.get_scale_factors().get(day_type, 1.0)

        days_since_start = (target_date - cohorts[0][0]).days + 1

        rows.append({
            "应用ID": app_id,
            "y_pred": round(predicted, 2),
            "d1_roi": round(new_d1_roi, 4),
            "spend": new_spend,
            "d1_revenue": inp["d1_revenue"],
            "days_since_start": days_since_start,
            "target_date": target_date.isoformat(),
            "cohort_count": len(all_cohorts),
        })

    return rows


def predict_daily_revenues(
    app_daily: Dict[str, Dict[str, Dict[str, float]]],
    curves: Dict[str, Dict[int, float]],
    min_history_days: int = 20,
    t1_apps: set[str] | None = None,
) -> List[Dict]:
    """
    遍历每个 app 的每天，用释放曲线预测当日总收入。
    - min_history_days: app 最少历史天数（对齐 T+1 预测模块 min_train_days=20）
    - 仅输出 days_since_start > min_history_days 的行
    - 跳过 spend==0 且 buy_rev==0 的停投日
    """
    default_curve = _build_default_curve()
    rows: List[Dict] = []
    cal = CalendarService()

    for app_id, day_data in app_daily.items():
        if t1_apps is not None and app_id not in t1_apps:
            continue
        sorted_days = sorted(day_data.keys())
        if len(sorted_days) <= min_history_days:
            continue

        curve = curves.get(app_id, default_curve)
        app_start_day = datetime.strptime(sorted_days[0], "%Y-%m-%d").date()

        # 累积 cohort 列表: [(spend_day, spend, d1_roi)]
        cohorts: List[Tuple[date, float, float]] = []

        for day_str in sorted_days:
            current_day = datetime.strptime(day_str, "%Y-%m-%d").date()
            info = day_data[day_str]
            spend = info["spend"]
            d1_rev = info["d1_revenue"]
            buy_rev = info["buy_revenue"]
            days_since_start = (current_day - app_start_day).days + 1

            d1_roi = d1_rev / spend if spend > 0 else 0.0

            # 预测当日收入 = Σ 历史cohort × d1_roi × [F(age_today) - F(age_yesterday)]
            # 注意：当日 cohort 不在预测范围内（避免数据泄露：buy_revenue 包含当日 D1）
            predicted = 0.0
            for cday, cspend, cd1 in cohorts:
                age_today = (current_day - cday).days + 1
                age_yesterday = age_today - 1
                f_today = _curve_value(curve, age_today)
                f_yesterday = _curve_value(curve, age_yesterday)
                delta = f_today - f_yesterday
                if delta > 0:
                    predicted += cspend * cd1 * delta

            # 休息日校准
            day_type = cal.classify_day(current_day)
            predicted *= cal.get_scale_factors().get(day_type, 1.0)

            # 当日新 cohort 加入（预测之后再加入，仅作为后续天的历史数据）
            if spend > 0 and d1_roi > 0:
                cohorts.append((current_day, spend, d1_roi))

            if days_since_start <= min_history_days or (spend == 0.0 and buy_rev == 0.0):
                continue

            pred_carryover = round(predicted, 2)
            y_true_d1 = round(d1_rev, 2)
            y_true_carryover = round(buy_rev - d1_rev, 2)
            y_pred_combined = round(predicted + d1_rev, 2)

            rows.append({
                "日期": day_str,
                "应用ID": app_id,
                "y_true": round(buy_rev, 2),
                "y_pred": y_pred_combined,
                "y_true_d1": y_true_d1,
                "y_true_carryover": y_true_carryover,
                "y_pred_carryover": pred_carryover,
                "days_since_start": days_since_start,
            })

    return rows


def save_predictions(rows: List[Dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["日期", "应用ID", "y_true", "y_pred", "y_true_d1", "y_true_carryover", "y_pred_carryover", "days_since_start"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"预测结果已保存: {output_path} ({len(rows)} rows)")
    return output_path


def compute_mape(rows: List[Dict]) -> Dict[str, float]:
    """两种口径 MAPE：组合(D1已知) + Carryover(曲线预测)"""
    combined_apes: Dict[str, List[float]] = defaultdict(list)
    carryover_apes: Dict[str, List[float]] = defaultdict(list)

    for r in rows:
        if r["y_true"] > 0:
            # 组合 MAPE: D1已知场景，y_pred 包含实际D1
            ape_combined = abs(r["y_true"] - r["y_pred"]) / r["y_true"]
            combined_apes[r["应用ID"]].append(ape_combined)
        if r["y_true_carryover"] > 0:
            # Carryover MAPE: 仅评估曲线模型的尾量预测能力
            ape_co = abs(r["y_true_carryover"] - r["y_pred_carryover"]) / r["y_true_carryover"]
            carryover_apes[r["应用ID"]].append(ape_co)

    def _agg(app_apes):
        per_app = {}
        for app_id, apes in sorted(app_apes.items()):
            if apes:
                per_app[app_id] = round(sum(apes) / len(apes) * 100, 2)
        all_apes = [a for apes in app_apes.values() for a in apes]
        overall = round(sum(all_apes) / len(all_apes) * 100, 2) if all_apes else 0.0
        return overall, per_app

    combined_overall, combined_per_app = _agg(combined_apes)
    carryover_overall, carryover_per_app = _agg(carryover_apes)

    # D1 在总收入中的占比
    d1_total = sum(r["y_true_d1"] for r in rows)
    buy_total = sum(r["y_true"] for r in rows)
    d1_ratio = d1_total / buy_total * 100 if buy_total > 0 else 0.0

    print(f"\n===== 收入验证 MAPE（双口径） =====")
    print(f"D1 收入占比: {d1_ratio:.1f}% (精确已知)")
    print(f"组合 MAPE (D1已知场景): overall={combined_overall}%")
    print(f"Carryover MAPE (曲线预测): overall={carryover_overall}%")
    print(f"\nPer-app Carryover MAPE (top 5 by sample count):")
    sorted_apps = sorted(carryover_apes.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for app_id, apes in sorted_apps:
        print(f"  {app_id}: {carryover_per_app[app_id]}% ({len(apes)} samples)")
    return {
        "combined_overall": combined_overall,
        "carryover_overall": carryover_overall,
        "d1_ratio": round(d1_ratio, 1),
        "combined_per_app": combined_per_app,
        "carryover_per_app": carryover_per_app,
    }


def main():
    csv_path = _CSV_PATH
    output_path = _OUTPUT_DIR / "daily_revenue_predictions.csv"

    # 数据管道：入口健康检查
    from app.services.data_pipeline import check_data_health
    health = check_data_health(csv_path)
    print(f"数据健康: status={health.status}, last_date={health.last_date}, "
          f"days_behind={health.days_behind}, rows={health.row_count}, apps={health.app_count}")
    if not health.ok:
        print(f"  [ABORT] 数据不健康，拒绝执行。缺失列: {health.missing_cols}")
        return
    if health.status == "stale":
        print(f"  [WARN] 数据已滞后 {health.days_behind} 天，预测结果可能不反映当前状态")

    print("加载数据...")
    data = load_app_daily(csv_path)
    print(f"  {len(data)} apps")

    print("加载释放曲线...")
    curves = load_curves(_CURVE_PATH)
    print(f"  {len(curves)} per-app curves")

    # 加载 T+1 预测模块的 app 集合以对齐样本口径
    t1_apps: set[str] | None = None
    t1_path = Path(__file__).resolve().parent.parent / "outputs" / "model_parallel_spend_t1" / "predictions_XGBoost.csv"
    if t1_path.exists():
        t1_apps = set()
        with t1_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                a = row.get("应用ID", "")
                if a:
                    t1_apps.add(a)
        print(f"  {len(t1_apps)} T+1 apps for alignment")

    print("预测每日收入...")
    rows = predict_daily_revenues(data, curves, min_history_days=20, t1_apps=t1_apps)

    save_predictions(rows, output_path)
    compute_mape(rows)


if __name__ == "__main__":
    main()
