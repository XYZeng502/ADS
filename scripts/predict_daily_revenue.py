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
