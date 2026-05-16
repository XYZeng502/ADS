#!/usr/bin/env python
"""每日买量收入预测：用释放曲线模型预测每日总收入，对比实际买量广告收入。"""
import csv
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

_CSV_PATH = Path("data/daily_merged.csv")
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


def predict_daily_revenues(
    app_daily: Dict[str, Dict[str, Dict[str, float]]],
    curves: Dict[str, Dict[int, float]],
) -> List[Dict]:
    """
    遍历每个 app 的每天，用释放曲线预测当日总收入。
    返回 list of dict，每行：日期, 应用ID, y_true, y_pred, days_since_start
    """
    default_curve = _build_default_curve()
    rows: List[Dict] = []

    for app_id, day_data in app_daily.items():
        curve = curves.get(app_id, default_curve)
        sorted_days = sorted(day_data.keys())
        app_start_day = sorted_days[0]

        # 累积 cohort 列表: [(spend_day, spend, d1_roi)]
        cohorts: List[Tuple[date, float, float]] = []

        for day_str in sorted_days:
            current_day = datetime.strptime(day_str, "%Y-%m-%d").date()
            info = day_data[day_str]
            spend = info["spend"]
            d1_rev = info["d1_revenue"]
            buy_rev = info["buy_revenue"]
            d1_roi = d1_rev / spend if spend > 0 else 0.0
            days_since_start = (current_day - datetime.strptime(app_start_day, "%Y-%m-%d").date()).days + 1

            # 当日新 cohort 加入
            if spend > 0 and d1_roi > 0:
                cohorts.append((current_day, spend, d1_roi))

            # 预测当日收入 = Σ cohort × d1_roi × [F(age_today) - F(age_yesterday)]
            predicted = 0.0
            for cday, cspend, cd1 in cohorts:
                age_today = (current_day - cday).days + 1
                age_yesterday = age_today - 1
                f_today = _curve_value(curve, age_today)
                f_yesterday = _curve_value(curve, age_yesterday)
                delta = f_today - f_yesterday
                if delta > 0:
                    predicted += cspend * cd1 * delta

            rows.append({
                "日期": day_str,
                "应用ID": app_id,
                "y_true": round(buy_rev, 2),
                "y_pred": round(predicted, 2),
                "days_since_start": days_since_start,
            })

    return rows
