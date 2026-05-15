import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from app.config import settings
from app.prediction_artifacts import resolve_roi_predictions_csv, resolve_spend_predictions_csv
from app.schemas import ClientContext, ProductSlotMetrics, ProductState
from app.services.orchestrator import DailyOrchestrator
from app.services.risk import RiskService


DEFAULT_KPI = 1.05
DEFAULT_M30 = 1.44

_APP_LAST_DAY_CANONICAL_ROI_CHOICES = ("stable", "pred_only", "fused")


def _run_with_d1_calibration(orchestrator: DailyOrchestrator, context: ClientContext, enabled: bool):
    old = settings.d1_signal_calibration_enabled
    settings.d1_signal_calibration_enabled = enabled
    try:
        return orchestrator.run(context)
    finally:
        settings.d1_signal_calibration_enabled = old


@dataclass
class SlotAgg:
    spend: float = 0.0
    d1_revenue: float = 0.0
    d30_revenue: float = 0.0


def to_float(v: str) -> float:
    try:
        return float(v) if v not in ("", None) else 0.0
    except ValueError:
        return 0.0


def map_slot(flow_name: str) -> str:
    if flow_name == "自有流量":
        return "store"
    if flow_name == "联盟流量":
        return "union"
    return "smart"


def _load_release_multiplier_curve() -> Dict[int, float]:
    """
    读取日级释放倍率曲线（相对D1）。
    优先使用竞品模板输出的p50曲线；缺失时回退默认节点插值。
    """
    curve_path = Path("outputs/competitor_forecast/competitor_demo_curved_competitor_roi_curve.csv")
    if curve_path.exists():
        rows: Dict[int, float] = {}
        with curve_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    d = int(float(row.get("day", 0)))
                    roi = float(row.get("roi_pred_p50", 0.0))
                except (TypeError, ValueError):
                    continue
                if d > 0 and roi > 0:
                    rows[d] = roi
        if rows and 1 in rows:
            d1 = rows[1]
            curve = {}
            max_day = max(rows.keys())
            last = 1.0
            for d in range(1, max_day + 1):
                base = rows.get(d, rows.get(d - 1, d1))
                m = max(base / d1, last)
                curve[d] = m
                last = m
            return curve

    # 默认节点倍率曲线
    node_days = [1, 3, 7, 30]
    node_mult = [1.0, 1.1942, 1.3107, 1.4666]
    curve: Dict[int, float] = {}
    for d in range(1, 31):
        if d <= node_days[0]:
            curve[d] = node_mult[0]
            continue
        if d >= node_days[-1]:
            curve[d] = node_mult[-1]
            continue
        for i in range(len(node_days) - 1):
            d0, d1 = node_days[i], node_days[i + 1]
            if d0 <= d <= d1:
                y0, y1 = node_mult[i], node_mult[i + 1]
                ratio = (d - d0) / (d1 - d0)
                curve[d] = y0 + (y1 - y0) * ratio
                break
    return curve


def _curve_value(curve: Dict[int, float], age_day: int) -> float:
    """
    cumulative multiplier F(age):
    - age<=0: 0（尚未产生）
    - age>max_day: F(max_day)（>30天不再新增，累计值封顶）
    """
    if age_day <= 0:
        return 0.0
    max_day = max(curve.keys())
    age = min(age_day, max_day)
    return curve.get(age, curve[max_day])


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[idx]


def _pick_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _safe_date(s: str) -> Optional[date]:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _load_t1_prediction_rows(target: str) -> List[Dict[str, str]]:
    """
    读取预测模块输出（roi/spend），用于校准月末ROI输入。
    target: "roi" or "spend"
    """
    root = Path("outputs")
    if target == "roi":
        roi_dir = _pick_existing(
            [
                root / "model_parallel_roi_d1_v9_unified",
                root / "model_parallel_roi_d1_v8_001",
                root / "model_parallel_roi_d1_v7_001",
                root / "model_parallel_roi_d1_v7_aligned",
                root / "model_parallel_roi_d1_default_weekly_v3",
            ]
        )
        if roi_dir is None:
            return []
        pred_path = resolve_roi_predictions_csv(roi_dir) or Path("__missing__")
    else:
        spend_dir = _pick_existing(
            [
                root / "model_parallel_spend_t1_v12_unified",
                root / "model_parallel_spend_t1_v11_001",
                root / "model_parallel_spend_t1_v10_001",
                root / "model_parallel_spend_t1_v9_composition",
            ]
        )
        if spend_dir is None:
            return []
        pred_path = resolve_spend_predictions_csv(spend_dir) or Path("__missing__")

    if not pred_path.exists():
        return []
    try:
        with pred_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        return rows
    except Exception:
        return []


def _build_app_calibration_from_predictions(
    rows: List[Dict[str, str]],
    target_day: date,
    *,
    pred_keys: List[str],
    true_key: str = "y_true",
    app_key: str = "应用ID",
    lookback_days: int = 14,
    min_points: int = 3,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    返回：
    - app_bias_factor: 历史真实/预测 比值（用于校准）
    - app_t1_pred: 当天（date==target_day）预测值（用于融合）
    """
    by_app_hist: Dict[str, List[Tuple[date, float, float]]] = defaultdict(list)
    app_t1_pred: Dict[str, float] = {}
    pred_source_day = target_day - timedelta(days=1)
    start_day = pred_source_day - timedelta(days=lookback_days)

    for r in rows:
        app_id = str(r.get(app_key, "")).strip()
        d = _safe_date(str(r.get("日期", "")))
        if not app_id or d is None:
            continue
        pred = 0.0
        for k in pred_keys:
            val = to_float(r.get(k, "0"))
            if val > 0:
                pred = val
                break
        true_v = to_float(r.get(true_key, "0"))
        # 预测模块按 d -> d+1 组织，目标日 target_day 对应记录日期应为 target_day-1
        if d == pred_source_day and pred > 0:
            app_t1_pred[app_id] = pred
        # 历史偏差只用更早窗口，避免把目标日同日标签信息带入校准
        if d < pred_source_day and d >= start_day and pred > 0 and true_v > 0:
            by_app_hist[app_id].append((d, true_v, pred))

    app_bias_factor: Dict[str, float] = {}
    for app_id, hist in by_app_hist.items():
        if len(hist) < min_points:
            continue
        ratios = sorted([max(min(t / p, 1.35), 0.65) for _, t, p in hist if p > 0 and t > 0])
        if not ratios:
            continue
        app_bias_factor[app_id] = _quantile(ratios, 0.5)
    return app_bias_factor, app_t1_pred


def _stable_app_future_d1_roi(
    app_id: str,
    month_start: date,
    target_day: date,
    historical_daily: Dict[Tuple[str, str], Dict[str, float]],
) -> float:
    """
    月末应用级 ROI 用应用整体 D1 作为未来回收锚点。
    solver 可能只选择少量高 ROI 计划，如果直接用 solver 的组合 ROI，
    会把应用月末 ROI 主指标抬高到不可解释的水平。
    """
    same_month_rows: List[Tuple[date, float, float]] = []
    fallback_rows: List[Tuple[date, float, float]] = []
    for (a, day_str), info in historical_daily.items():
        if a != app_id:
            continue
        cday = datetime.strptime(day_str, "%Y-%m-%d").date()
        if cday >= target_day:
            continue
        spend = info.get("spend", 0.0)
        d1_roi = info.get("d1_roi", 0.0)
        if spend <= 0 or d1_roi <= 0:
            continue
        row = (cday, spend, d1_roi)
        fallback_rows.append(row)
        if cday >= month_start:
            same_month_rows.append(row)

    rows = same_month_rows or fallback_rows[-14:]
    if not rows:
        return 0.0

    total_spend = sum(spend for _, spend, _ in rows)
    mtd_roi = (
        sum(spend * d1_roi for _, spend, d1_roi in rows) / total_spend
        if total_spend > 0
        else 0.0
    )

    recent_rows = sorted(rows, key=lambda x: x[0])[-7:]
    recent_spend = sum(spend for _, spend, _ in recent_rows)
    recent_roi = (
        sum(spend * d1_roi for _, spend, d1_roi in recent_rows) / recent_spend
        if recent_spend > 0
        else mtd_roi
    )
    blended = 0.7 * mtd_roi + 0.3 * recent_roi

    daily_values = [d1_roi for _, spend, d1_roi in rows if spend > 0 and d1_roi > 0]
    low = _quantile(daily_values, 0.1)
    high = _quantile(daily_values, 0.9)
    if high > 0:
        blended = min(max(blended, low), high)
    return blended


def _app_month_roi_attribution(
    app_id: str,
    month_start: date,
    target_day: date,
    month_end_spend: float,
    month_end_revenue: float,
    cross_month_carryover_revenue: float,
    planned_daily_spend: float,
    app_future_d1_roi: float,
    historical_daily: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, float | str]:
    """
    应用月末 ROI 的业务归因。
    这里归因要和应用级月末 ROI 口径一致，不能继续沿用 solver 计划层的
    D1 归因，否则高 ROI 计划的波动会把所有应用都打成 D1_ANCHOR。
    """
    rows: List[Tuple[date, float, float]] = []
    for (a, day_str), info in historical_daily.items():
        if a != app_id:
            continue
        cday = datetime.strptime(day_str, "%Y-%m-%d").date()
        if cday < month_start or cday >= target_day:
            continue
        spend = info.get("spend", 0.0)
        d1_roi = info.get("d1_roi", 0.0)
        if spend > 0 and d1_roi > 0:
            rows.append((cday, spend, d1_roi))

    if rows:
        recent_rows = sorted(rows, key=lambda x: x[0])[-7:]
        recent_avg_spend = sum(spend for _, spend, _ in recent_rows) / len(recent_rows)
        d1_values = [d1_roi for _, _, d1_roi in rows]
        median_d1 = max(_quantile(d1_values, 0.5), 1e-6)
        d1_spread = (_quantile(d1_values, 0.8) - _quantile(d1_values, 0.2)) / median_d1
        d1_anchor_gap = abs(app_future_d1_roi - median_d1) / median_d1
    else:
        recent_avg_spend = planned_daily_spend
        d1_spread = 0.45
        d1_anchor_gap = 0.0

    spend_gap = abs(planned_daily_spend - recent_avg_spend) / max(recent_avg_spend, 1e-6)
    future_spend_share = (
        planned_daily_spend * max((month_end_spend - sum(spend for _, spend, _ in rows)) / max(planned_daily_spend, 1e-6), 0.0)
    ) / max(month_end_spend, 1e-6)
    # 三个分数做同量纲校准：预算变化天然有系统性基准，需降权；
    # D1 和曲线只在波动/长尾占比明显时才成为主因。
    raw_spend_score = spend_gap * 0.75 + future_spend_share * 0.35
    raw_d1_score = d1_spread * 0.55 + d1_anchor_gap * 0.65
    raw_curve_score = (cross_month_carryover_revenue / max(month_end_revenue, 1e-6)) * 3.0
    spend_score = min(raw_spend_score * 0.65, 1.0)
    d1_score = min(raw_d1_score * 1.8, 1.0)
    curve_score = min(raw_curve_score * 1.6, 1.0)

    score_map = {
        "SPEND": spend_score,
        "D1_ANCHOR": d1_score,
        "CURVE": curve_score,
    }
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    top_name, top_score = ranked[0]
    second_score = ranked[1][1]
    dominant = top_name if (top_score - second_score) >= 0.05 else "BALANCED"
    details = (
        f"spend={spend_score:.2f}, d1_anchor={d1_score:.2f}, curve={curve_score:.2f}; "
        "应用级口径：spend看预算节奏偏离，d1_anchor看应用D1波动/锚点偏离，curve看跨月长尾占比。"
    )
    return {
        "spend_factor_score": round(spend_score, 4),
        "d1_anchor_score": round(d1_score, 4),
        "curve_factor_score": round(curve_score, 4),
        "dominant_factor": dominant,
        "details": details,
    }


def _month_end_incremental_roi_for_app(
    app_id: str,
    month_start: date,
    month_end: date,
    target_day: date,
    month_spend_before_target: float,
    historical_daily: Dict[Tuple[str, str], Dict[str, float]],
    planned_daily_spend: float,
    planned_d1_roi: float,
    curve: Dict[int, float],
) -> Tuple[float, float, float, float, float]:
    """
    应用级月末ROI（cohort增量口径）：
    分子 = Σ cohort_spend * d1_roi * [F(age_end)-F(age_before_month_start)]
    分母 = 本月总消耗（已发生 + 预测剩余天）
    """
    incremental_revenue_this_month = 0.0
    incremental_revenue_30d_total = 0.0

    # 1) 历史cohort（target_day之前）贡献到本月末的增量
    for (a, day_str), info in historical_daily.items():
        if a != app_id:
            continue
        cday = datetime.strptime(day_str, "%Y-%m-%d").date()
        spend = info.get("spend", 0.0)
        d1_roi = info.get("d1_roi", 0.0)
        if spend <= 0 or d1_roi <= 0:
            continue

        age_end = (month_end - cday).days + 1
        age_before_month = (month_start - timedelta(days=1) - cday).days + 1
        delta_mult = _curve_value(curve, age_end) - _curve_value(curve, age_before_month)
        if delta_mult > 0:
            incremental_revenue_this_month += spend * d1_roi * delta_mult
        # 30天总回收（扣除月初前已实现部分），用于评估跨月回收潜力
        total_30d_mult = _curve_value(curve, 30) - _curve_value(curve, age_before_month)
        if total_30d_mult > 0:
            incremental_revenue_30d_total += spend * d1_roi * total_30d_mult

    # 2) target_day ~ month_end 的预测cohort增量
    future_days = max((month_end - target_day).days + 1, 1)
    for i in range(future_days):
        cday = target_day + timedelta(days=i)
        spend = planned_daily_spend
        d1_roi = planned_d1_roi
        if spend <= 0 or d1_roi <= 0:
            continue
        age_end = (month_end - cday).days + 1
        age_before_month = (month_start - timedelta(days=1) - cday).days + 1
        delta_mult = _curve_value(curve, age_end) - _curve_value(curve, age_before_month)
        if delta_mult > 0:
            incremental_revenue_this_month += spend * d1_roi * delta_mult
        total_30d_mult = _curve_value(curve, 30) - _curve_value(curve, age_before_month)
        if total_30d_mult > 0:
            incremental_revenue_30d_total += spend * d1_roi * total_30d_mult

    month_end_spend = month_spend_before_target + planned_daily_spend * future_days
    month_end_roi_this_month = incremental_revenue_this_month / month_end_spend if month_end_spend > 0 else 0.0
    month_end_roi_30d_total = incremental_revenue_30d_total / month_end_spend if month_end_spend > 0 else 0.0
    cross_month_carryover = max(incremental_revenue_30d_total - incremental_revenue_this_month, 0.0)
    return (
        month_end_spend,
        incremental_revenue_this_month,
        month_end_roi_this_month,
        month_end_roi_30d_total,
        cross_month_carryover,
    )


def load_aggregates(csv_path: Path):
    # day -> advertiser -> product(plan) -> slot -> SlotAgg
    nested = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(SlotAgg)))
    )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = row["日期"]
            adv = row["广告主ID"]
            product = row["计划ID"]
            slot = map_slot(row.get("推广流量名称", ""))

            agg: SlotAgg = nested[day][adv][product][slot]
            agg.spend += to_float(row.get("消耗金额", "0"))
            agg.d1_revenue += to_float(row.get("首日广告收入", "0"))
            agg.d30_revenue += to_float(row.get("30日累计变现金额", "0"))

    return nested


def load_aggregates_app_level(csv_path: Path):
    # day -> app -> advertiser -> product(plan) -> slot -> SlotAgg
    nested = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(SlotAgg))))
    )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = row["日期"]
            app = row["应用ID"]
            adv = row["广告主ID"]
            product = row["计划ID"]
            slot = map_slot(row.get("推广流量名称", ""))

            agg: SlotAgg = nested[day][app][adv][product][slot]
            agg.spend += to_float(row.get("消耗金额", "0"))
            agg.d1_revenue += to_float(row.get("首日广告收入", "0"))
            agg.d30_revenue += to_float(row.get("30日累计变现金额", "0"))
    return nested


def build_product_states(
    product_slot_aggs: Dict[str, Dict[str, SlotAgg]],
    prev_slot_spend: Dict[Tuple[str, str], float],
    d1_history: Dict[str, Deque[float]],
) -> List[ProductState]:
    products: List[ProductState] = []

    # 简单优先级：按产品当日总消耗排序，越大优先级越高
    spend_rank = []
    for product_id, slots in product_slot_aggs.items():
        total_spend = sum(a.spend for a in slots.values())
        spend_rank.append((total_spend, product_id))
    spend_rank.sort(reverse=True)
    priority_map = {pid: len(spend_rank) - idx for idx, (_, pid) in enumerate(spend_rank)}

    for product_id, slots in product_slot_aggs.items():
        slot_metrics: List[ProductSlotMetrics] = []
        product_spend = 0.0
        product_d1 = 0.0
        for slot, agg in slots.items():
            product_spend += agg.spend
            product_d1 += agg.d1_revenue

            d1_roi = agg.d1_revenue / agg.spend if agg.spend > 0 else 0.0
            m30 = agg.d30_revenue / agg.d1_revenue if agg.d1_revenue > 0 else DEFAULT_M30
            m30 = max(1.0, min(m30, 2.5))

            key = (product_id, slot)
            y_spend = prev_slot_spend.get(key, agg.spend * 0.8)
            cap = max(1000.0, agg.spend * 1.5, y_spend * 1.6)

            slot_metrics.append(
                ProductSlotMetrics(
                    slot=slot,  # type: ignore[arg-type]
                    predicted_d1_roi=round(d1_roi, 4),
                    m30_multiplier=round(m30, 4),
                    yesterday_spend=round(y_spend, 2),
                    cap=round(cap, 2),
                )
            )

        # 当前版本用历史均值作为预测，以便触发风控逻辑
        hist = list(d1_history[product_id])
        pred_hist = hist.copy()

        products.append(
            ProductState(
                product_id=product_id,
                product_name=f"计划{product_id}",
                priority=priority_map.get(product_id, 0),
                slots=slot_metrics,
                recent_d1_actual=hist,
                recent_d1_predicted=pred_hist,
            )
        )

        daily_product_d1 = product_d1 / product_spend if product_spend > 0 else 0.0
        d1_history[product_id].append(daily_product_d1)

    return products


def run_backtest(csv_path: Path, output_dir: Path, kpi: float) -> Dict:
    orchestrator = DailyOrchestrator()
    risk = RiskService()

    nested = load_aggregates(csv_path)
    all_days = sorted(nested.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d"))

    records = []

    # 每个广告主维护月内状态
    month_spend_so_far: Dict[Tuple[str, str], float] = defaultdict(float)
    month_rev_so_far: Dict[Tuple[str, str], float] = defaultdict(float)
    prev_day_total_spend: Dict[str, float] = defaultdict(float)
    prev_slot_spend: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    d1_histories: Dict[str, Dict[str, Deque[float]]] = defaultdict(
        lambda: defaultdict(lambda: deque(maxlen=3))
    )

    for day in all_days:
        month_tag = day[:7]
        advertisers = nested[day]
        for adv, product_slots in advertisers.items():
            day_total_spend = 0.0
            # 业务口径修正：不再使用买量广告收入（新老用户混合，参考意义弱）
            # 当前离线回测用“当日新增cohort的D1回收”作为可观测分子。
            day_total_actual_rev = 0.0
            for _, slots in product_slots.items():
                for _, agg in slots.items():
                    day_total_spend += agg.spend
                    day_total_actual_rev += agg.d1_revenue

            mkey = (adv, month_tag)
            products = build_product_states(
                product_slots,
                prev_slot_spend[adv],
                d1_histories[adv],
            )

            context = ClientContext(
                client_id=adv,
                client_name=f"广告主{adv}",
                date=datetime.strptime(day, "%Y-%m-%d").date(),
                month_spend_so_far=month_spend_so_far[mkey],
                month_revenue_so_far=month_rev_so_far[mkey],
                kpi_roi=kpi,
                yesterday_total_spend=(
                    prev_day_total_spend[adv] if prev_day_total_spend[adv] > 0 else day_total_spend
                ),
                products=products,
            )

            plan = orchestrator.run(context)
            today_pred_revenue = (
                sum(x.expected_d1_revenue for x in plan.suggestions)
                + plan.today_history_revenue_estimate
            )
            extra_alerts = risk.evaluate_traffic_anomaly(day_total_actual_rev, today_pred_revenue)

            actual_roi_today = day_total_actual_rev / day_total_spend if day_total_spend > 0 else 0.0
            planned_spend = sum(x.suggested_budget for x in plan.suggestions)

            records.append(
                {
                    "date": day,
                    "advertiser_id": adv,
                    "mode": plan.mode,
                    "products": len(products),
                    "actual_spend": round(day_total_spend, 2),
                    "actual_revenue": round(day_total_actual_rev, 2),
                    "actual_roi": round(actual_roi_today, 4),
                    "planned_spend": round(planned_spend, 2),
                    "predicted_month_end_roi": round(plan.month_end_roi_prediction, 4),
                    "predicted_month_end_spend": round(plan.month_end_spend_prediction, 2),
                    "alert_count": len(plan.alerts) + len(extra_alerts),
                }
            )

            # 更新状态
            month_spend_so_far[mkey] += day_total_spend
            month_rev_so_far[mkey] += day_total_actual_rev
            prev_day_total_spend[adv] = day_total_spend
            for pid, slots in product_slots.items():
                for slot, agg in slots.items():
                    prev_slot_spend[adv][(pid, slot)] = agg.spend

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / "offline_backtest_detail.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "advertiser_id",
                "mode",
                "products",
                "actual_spend",
                "actual_revenue",
                "actual_roi",
                "planned_spend",
                "predicted_month_end_roi",
                "predicted_month_end_spend",
                "alert_count",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    guard_cnt = sum(1 for r in records if r["mode"] == "GUARD")
    scale_cnt = sum(1 for r in records if r["mode"] == "SCALE")
    roi_warn_cnt = sum(1 for r in records if r["predicted_month_end_roi"] < kpi - 0.03)
    avg_actual_roi = sum(r["actual_roi"] for r in records) / len(records) if records else 0.0
    avg_pred_month_roi = (
        sum(r["predicted_month_end_roi"] for r in records) / len(records) if records else 0.0
    )
    avg_spend_gap = (
        sum((r["planned_spend"] - r["actual_spend"]) for r in records) / len(records)
        if records
        else 0.0
    )

    # 展示样例：按当天实际消耗排序取前10条
    sample_top_spend = sorted(records, key=lambda x: x["actual_spend"], reverse=True)[:10]

    summary = {
        "input_file": str(csv_path),
        "days": len(set(r["date"] for r in records)),
        "advertiser_days": len(records),
        "kpi": kpi,
        "mode_distribution": {"GUARD": guard_cnt, "SCALE": scale_cnt},
        "roi_warning_count": roi_warn_cnt,
        "avg_actual_roi": round(avg_actual_roi, 4),
        "avg_predicted_month_end_roi": round(avg_pred_month_roi, 4),
        "avg_planned_minus_actual_spend": round(avg_spend_gap, 2),
        "top_spend_samples": sample_top_spend,
        "detail_csv": str(detail_csv),
    }

    summary_file = output_dir / "offline_backtest_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_app_level_last_day_prediction(csv_path: Path, output_dir: Path, kpi: float) -> Dict:
    if settings.app_last_day_canonical_month_end_roi not in _APP_LAST_DAY_CANONICAL_ROI_CHOICES:
        raise ValueError(
            f"app_last_day_canonical_month_end_roi 必须是 {_APP_LAST_DAY_CANONICAL_ROI_CHOICES} 之一，"
            f"当前为 {settings.app_last_day_canonical_month_end_roi!r}"
        )
    orchestrator = DailyOrchestrator()
    risk = RiskService()

    nested = load_aggregates_app_level(csv_path)
    all_days = sorted(nested.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
    if len(all_days) < 2:
        raise ValueError("至少需要2天数据：前面用于训练，最后一天用于预测。")

    target_day = all_days[-1]
    train_days = all_days[:-1]
    target_month = target_day[:7]
    prev_day = all_days[-2]
    target_day_dt = datetime.strptime(target_day, "%Y-%m-%d").date()
    display_day_dt = target_day_dt + timedelta(days=1)  # 推荐日期指向「明天」
    month_start_dt = target_day_dt.replace(day=1)
    # next month first day - 1
    if month_start_dt.month == 12:
        month_end_dt = date(month_start_dt.year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end_dt = date(month_start_dt.year, month_start_dt.month + 1, 1) - timedelta(days=1)
    release_curve = _load_release_multiplier_curve()
    roi_pred_rows = _load_t1_prediction_rows("roi")
    spend_pred_rows = _load_t1_prediction_rows("spend")
    roi_bias, roi_t1_pred = _build_app_calibration_from_predictions(
        roi_pred_rows,
        target_day=target_day_dt,
        pred_keys=["y_pred"],
    )
    spend_bias, spend_t1_pred = _build_app_calibration_from_predictions(
        spend_pred_rows,
        target_day=target_day_dt,
        pred_keys=["y_pred_fused", "y_pred"],
    )

    # 训练统计（应用维度：同一应用下全部广告主统一建模）
    slot_stats = defaultdict(lambda: {"spend": 0.0, "d1": 0.0, "d30": 0.0, "max_spend": 0.0, "days": 0})
    product_d1_history = defaultdict(lambda: deque(maxlen=3))
    product_spend_total = defaultdict(float)
    month_spend_so_far = defaultdict(float)
    month_rev_so_far = defaultdict(float)
    app_daily_history: Dict[Tuple[str, str], Dict[str, float]] = {}

    for day in train_days:
        day_data = nested[day]
        for app_id, adv_map in day_data.items():
            app_day_spend = 0.0
            app_day_rev = 0.0
            for adv, prod_map in adv_map.items():
                for product_id, slot_map in prod_map.items():
                    product_uid = f"{adv}:{product_id}"
                    p_spend = 0.0
                    p_d1 = 0.0
                    for slot, agg in slot_map.items():
                        p_spend += agg.spend
                        p_d1 += agg.d1_revenue
                        app_day_spend += agg.spend
                        app_day_rev += agg.d1_revenue

                        key = (app_id, product_uid, slot)
                        slot_stats[key]["spend"] += agg.spend
                        slot_stats[key]["d1"] += agg.d1_revenue
                        slot_stats[key]["d30"] += agg.d30_revenue
                        slot_stats[key]["max_spend"] = max(slot_stats[key]["max_spend"], agg.spend)
                        slot_stats[key]["days"] += 1
                    product_spend_total[(app_id, product_uid)] += p_spend
                    d1_roi = p_d1 / p_spend if p_spend > 0 else 0.0
                    product_d1_history[(app_id, product_uid)].append(d1_roi)

            if day.startswith(target_month):
                month_spend_so_far[app_id] += app_day_spend
                month_rev_so_far[app_id] += app_day_rev

            app_daily_history[(app_id, day)] = {
                "spend": app_day_spend,
                "d1_roi": (app_day_rev / app_day_spend) if app_day_spend > 0 else 0.0,
            }

    # 统计每个应用的训练天数（用于剔除样本过少的不成熟应用）
    app_train_days: Dict[str, int] = {}
    for (a_id, _) in app_daily_history:
        app_train_days[a_id] = app_train_days.get(a_id, 0) + 1
    min_train_days = max(settings.app_last_day_min_train_days, 1)

    # 上一日消耗作为今日波动基线（应用维度）
    prev_slot_spend = defaultdict(float)
    prev_total_spend = defaultdict(float)
    for app_id, adv_map in nested[prev_day].items():
        for adv, prod_map in adv_map.items():
            for product_id, slot_map in prod_map.items():
                product_uid = f"{adv}:{product_id}"
                for slot, agg in slot_map.items():
                    prev_slot_spend[(app_id, product_uid, slot)] = agg.spend
                    prev_total_spend[app_id] += agg.spend

    # 最后一天实际数据（应用维度）：
    # 不使用买量广告收入，改用cohort口径可观测值（D1回收）
    target_actual = defaultdict(lambda: {"spend": 0.0, "revenue": 0.0})
    for app_id, adv_map in nested[target_day].items():
        for _, prod_map in adv_map.items():
            for _, slot_map in prod_map.items():
                for _, agg in slot_map.items():
                    target_actual[app_id]["spend"] += agg.spend
                    target_actual[app_id]["revenue"] += agg.d1_revenue

    predictions = []
    suggestion_rows = []

    # 只输出最后一天有投放的应用（每应用仅一条）
    for app_id in sorted(target_actual.keys()):
        if target_actual[app_id]["spend"] <= 0:
            continue
        if app_train_days.get(app_id, 0) < min_train_days:
            continue
        product_ids = [
            pid for (a, pid) in product_spend_total.keys() if a == app_id
        ]
        if not product_ids:
            continue

        # 产品优先级：训练期总消耗越大优先级越高
        rank = sorted(
            [(product_spend_total[(app_id, pid)], pid) for pid in set(product_ids)],
            reverse=True,
        )
        priority_map = {pid: len(rank) - idx for idx, (_, pid) in enumerate(rank)}

        products: List[ProductState] = []
        for _, product_id in rank:
            slot_metrics: List[ProductSlotMetrics] = []
            for slot in ("store", "union", "smart"):
                key = (app_id, product_id, slot)
                if key not in slot_stats:
                    continue
                st = slot_stats[key]
                spend = st["spend"]
                d1_roi = st["d1"] / spend if spend > 0 else 0.0
                m30 = st["d30"] / st["d1"] if st["d1"] > 0 else DEFAULT_M30
                m30 = max(1.0, min(m30, 2.5))
                avg_spend = spend / max(st["days"], 1)
                y_spend = prev_slot_spend.get(key, avg_spend)
                cap = max(1000.0, st["max_spend"] * 1.3, avg_spend * 1.8)
                slot_metrics.append(
                    ProductSlotMetrics(
                        slot=slot,  # type: ignore[arg-type]
                        predicted_d1_roi=round(d1_roi, 4),
                        m30_multiplier=round(m30, 4),
                        yesterday_spend=round(y_spend, 2),
                        cap=round(cap, 2),
                    )
                )
            if not slot_metrics:
                continue

            hist = list(product_d1_history[(app_id, product_id)])
            if not hist:
                hist = [0.0, 0.0, 0.0]
            product_name = f"计划{product_id}"
            products.append(
                ProductState(
                    product_id=product_id,
                    product_name=product_name,
                    priority=priority_map.get(product_id, 0),
                    slots=slot_metrics,
                    recent_d1_actual=hist,
                    recent_d1_predicted=hist.copy(),
                )
            )

        if not products:
            continue

        context = ClientContext(
            client_id=app_id,
            client_name=f"应用{app_id}",
            date=display_day_dt,
            month_spend_so_far=month_spend_so_far[app_id],
            month_revenue_so_far=month_rev_so_far[app_id],
            kpi_roi=kpi,
            yesterday_total_spend=max(prev_total_spend.get(app_id, 0.0), 1.0),
            products=products,
        )

        # A/B 对比：
        # A: 关闭D1锚点校准（原始锚点）
        # B: 开启D1锚点校准（当前线上默认）
        plan_a = _run_with_d1_calibration(orchestrator, context, enabled=False)
        plan_b = _run_with_d1_calibration(orchestrator, context, enabled=True)
        plan = plan_b
        pred_today_revenue = sum(x.expected_d1_revenue for x in plan.suggestions) + plan.today_history_revenue_estimate
        extra_alerts = risk.evaluate_traffic_anomaly(
            today_real_revenue=target_actual[app_id]["revenue"],
            today_predicted_revenue=pred_today_revenue,
        )
        all_alerts = [a.model_dump() for a in plan.alerts] + [a.model_dump() for a in extra_alerts]
        top_suggestions = sorted(plan.suggestions, key=lambda x: x.suggested_budget, reverse=True)[:8]
        solver_budget_b = round(sum(x.suggested_budget for x in plan_b.suggestions), 2)
        planned_total_budget = round(sum(x.suggested_budget for x in plan.suggestions), 2)
        planned_d1_total = sum(x.expected_d1_revenue for x in plan.suggestions)
        planned_d1_roi = planned_d1_total / planned_total_budget if planned_total_budget > 0 else 0.0
        spend_pred_calibrated = spend_t1_pred.get(app_id, 0.0) * spend_bias.get(app_id, 1.0)
        stable_app_future_d1_roi = _stable_app_future_d1_roi(
            app_id=app_id,
            month_start=month_start_dt,
            target_day=target_day_dt,
            historical_daily=app_daily_history,
        )
        roi_pred_calibrated = roi_t1_pred.get(app_id, 0.0) * roi_bias.get(app_id, 1.0)
        stable_anchor_d1_roi = (
            stable_app_future_d1_roi if stable_app_future_d1_roi > 0 else planned_d1_roi
        )
        fused_budget = solver_budget_b
        if spend_pred_calibrated > 0:
            fused_budget = round(max(0.65 * solver_budget_b + 0.35 * spend_pred_calibrated, 0.0), 2)
        fused_app_future_d1_roi = stable_app_future_d1_roi
        if roi_pred_calibrated > 0:
            fused_app_future_d1_roi = 0.75 * stable_app_future_d1_roi + 0.25 * roi_pred_calibrated
        month_end_d1_roi = fused_app_future_d1_roi if fused_app_future_d1_roi > 0 else planned_d1_roi
        pred_only_daily_spend = (
            round(max(spend_pred_calibrated, 0.0), 2)
            if spend_pred_calibrated > 0
            else solver_budget_b
        )
        pred_only_d1_roi = (
            roi_pred_calibrated if roi_pred_calibrated > 0 else stable_anchor_d1_roi
        )

        # ① 稳定锚点：求解器(B)日耗 + 仅历史稳定 D1 锚点
        (
            month_end_spend_cohort_stable,
            month_end_revenue_cohort_stable,
            month_end_roi_cohort_stable,
            month_end_roi_30d_total_stable,
            cross_month_carryover_revenue_stable,
        ) = _month_end_incremental_roi_for_app(
            app_id=app_id,
            month_start=month_start_dt,
            month_end=month_end_dt,
            target_day=target_day_dt,
            month_spend_before_target=month_spend_so_far[app_id],
            historical_daily=app_daily_history,
            planned_daily_spend=solver_budget_b,
            planned_d1_roi=stable_anchor_d1_roi,
            curve=release_curve,
        )
        # ② 纯预测模块：T+1 spend/roi_d1 校准值作为未来日耗与 D1（无 spend 预测时无法定义「纯预测日耗」，回退求解器日耗）
        (
            month_end_spend_cohort_pred_only,
            month_end_revenue_cohort_pred_only,
            month_end_roi_cohort_pred_only,
            month_end_roi_30d_total_pred_only,
            cross_month_carryover_revenue_pred_only,
        ) = _month_end_incremental_roi_for_app(
            app_id=app_id,
            month_start=month_start_dt,
            month_end=month_end_dt,
            target_day=target_day_dt,
            month_spend_before_target=month_spend_so_far[app_id],
            historical_daily=app_daily_history,
            planned_daily_spend=pred_only_daily_spend,
            planned_d1_roi=pred_only_d1_roi,
            curve=release_curve,
        )
        (
            month_end_spend_cohort_a,
            month_end_revenue_cohort_a,
            month_end_roi_cohort_a,
            month_end_roi_30d_total_a,
            cross_month_carryover_revenue_a,
        ) = _month_end_incremental_roi_for_app(
            app_id=app_id,
            month_start=month_start_dt,
            month_end=month_end_dt,
            target_day=target_day_dt,
            month_spend_before_target=month_spend_so_far[app_id],
            historical_daily=app_daily_history,
            planned_daily_spend=round(sum(x.suggested_budget for x in plan_a.suggestions), 2),
            planned_d1_roi=month_end_d1_roi,
            curve=release_curve,
        )
        (
            month_end_spend_cohort_b,
            month_end_revenue_cohort_b,
            month_end_roi_cohort_b,
            month_end_roi_30d_total_b,
            cross_month_carryover_revenue_b,
        ) = _month_end_incremental_roi_for_app(
            app_id=app_id,
            month_start=month_start_dt,
            month_end=month_end_dt,
            target_day=target_day_dt,
            month_spend_before_target=month_spend_so_far[app_id],
            historical_daily=app_daily_history,
            planned_daily_spend=solver_budget_b,
            planned_d1_roi=month_end_d1_roi,
            curve=release_curve,
        )
        # ③ 融合：保守加权（与既有线上实验一致）
        (
            month_end_spend_cohort_fused,
            month_end_revenue_cohort_fused,
            month_end_roi_cohort_fused,
            month_end_roi_30d_total_fused,
            cross_month_carryover_revenue_fused,
        ) = _month_end_incremental_roi_for_app(
            app_id=app_id,
            month_start=month_start_dt,
            month_end=month_end_dt,
            target_day=target_day_dt,
            month_spend_before_target=month_spend_so_far[app_id],
            historical_daily=app_daily_history,
            planned_daily_spend=fused_budget,
            planned_d1_roi=month_end_d1_roi,
            curve=release_curve,
        )

        _canon = settings.app_last_day_canonical_month_end_roi
        if _canon == "stable":
            month_end_spend_cohort_c = month_end_spend_cohort_stable
            month_end_revenue_cohort_c = month_end_revenue_cohort_stable
            month_end_roi_cohort_c = month_end_roi_cohort_stable
            month_end_roi_30d_total_c = month_end_roi_30d_total_stable
            cross_month_carryover_revenue_c = cross_month_carryover_revenue_stable
            planned_total_budget = solver_budget_b
            app_future_d1_roi = stable_anchor_d1_roi
        elif _canon == "pred_only":
            month_end_spend_cohort_c = month_end_spend_cohort_pred_only
            month_end_revenue_cohort_c = month_end_revenue_cohort_pred_only
            month_end_roi_cohort_c = month_end_roi_cohort_pred_only
            month_end_roi_30d_total_c = month_end_roi_30d_total_pred_only
            cross_month_carryover_revenue_c = cross_month_carryover_revenue_pred_only
            planned_total_budget = pred_only_daily_spend
            app_future_d1_roi = pred_only_d1_roi
        else:
            month_end_spend_cohort_c = month_end_spend_cohort_fused
            month_end_revenue_cohort_c = month_end_revenue_cohort_fused
            month_end_roi_cohort_c = month_end_roi_cohort_fused
            month_end_roi_30d_total_c = month_end_roi_30d_total_fused
            cross_month_carryover_revenue_c = cross_month_carryover_revenue_fused
            planned_total_budget = fused_budget
            app_future_d1_roi = fused_app_future_d1_roi

        month_end_roi_triad = {
            "canonical": settings.app_last_day_canonical_month_end_roi,
            "stable_anchor": {
                "planned_daily_spend": solver_budget_b,
                "planned_d1_roi": round(stable_anchor_d1_roi, 4),
                "month_end_roi_prediction": round(month_end_roi_cohort_stable, 4),
                "month_end_spend_prediction": round(month_end_spend_cohort_stable, 2),
                "month_end_revenue_prediction": round(month_end_revenue_cohort_stable, 2),
                "month_end_roi_30d_total": round(month_end_roi_30d_total_stable, 4),
                "cross_month_carryover_revenue": round(cross_month_carryover_revenue_stable, 2),
            },
            "pred_module_only": {
                "planned_daily_spend": pred_only_daily_spend,
                "planned_d1_roi": round(pred_only_d1_roi, 4),
                "month_end_roi_prediction": round(month_end_roi_cohort_pred_only, 4),
                "month_end_spend_prediction": round(month_end_spend_cohort_pred_only, 2),
                "month_end_revenue_prediction": round(month_end_revenue_cohort_pred_only, 2),
                "month_end_roi_30d_total": round(month_end_roi_30d_total_pred_only, 4),
                "cross_month_carryover_revenue": round(cross_month_carryover_revenue_pred_only, 2),
            },
            "fused": {
                "planned_daily_spend": fused_budget,
                "planned_d1_roi": round(month_end_d1_roi, 4),
                "month_end_roi_prediction": round(month_end_roi_cohort_fused, 4),
                "month_end_spend_prediction": round(month_end_spend_cohort_fused, 2),
                "month_end_revenue_prediction": round(month_end_revenue_cohort_fused, 2),
                "month_end_roi_30d_total": round(month_end_roi_30d_total_fused, 4),
                "cross_month_carryover_revenue": round(cross_month_carryover_revenue_fused, 2),
            },
        }
        app_roi_attr = _app_month_roi_attribution(
            app_id=app_id,
            month_start=month_start_dt,
            target_day=target_day_dt,
            month_end_spend=month_end_spend_cohort_c,
            month_end_revenue=month_end_revenue_cohort_c,
            cross_month_carryover_revenue=cross_month_carryover_revenue_c,
            planned_daily_spend=planned_total_budget,
            app_future_d1_roi=app_future_d1_roi,
            historical_daily=app_daily_history,
        )
        top_actions = " | ".join([f"{s.product_id}/{s.slot}:{s.suggested_budget:.2f}" for s in top_suggestions[:3]])

        suggestion_rows.append(
            {
                "target_day": target_day,
                "app_id": app_id,
                "mode": plan.mode,
                "canonical_month_end_roi_source": settings.app_last_day_canonical_month_end_roi,
                "actual_spend_last_day": round(target_actual[app_id]["spend"], 2),
                "actual_revenue_last_day": round(target_actual[app_id]["revenue"], 2),
                "planned_total_budget": planned_total_budget,
                "planned_daily_spend_stable_anchor": solver_budget_b,
                "planned_daily_spend_pred_only": pred_only_daily_spend,
                "planned_daily_spend_fused": fused_budget,
                "month_end_roi_prediction": round(month_end_roi_cohort_c, 4),
                "month_end_roi_stable_anchor": round(month_end_roi_cohort_stable, 4),
                "month_end_roi_pred_only": round(month_end_roi_cohort_pred_only, 4),
                "month_end_roi_fused": round(month_end_roi_cohort_fused, 4),
                "month_end_spend_prediction": round(month_end_spend_cohort_c, 2),
                "month_end_roi_prediction_a": round(month_end_roi_cohort_a, 4),
                "month_end_roi_prediction_b": round(month_end_roi_cohort_b, 4),
                "ab_roi_delta_b_minus_a": round(month_end_roi_cohort_b - month_end_roi_cohort_a, 4),
                "month_end_roi_30d_total": round(month_end_roi_30d_total_c, 4),
                "cross_month_carryover_revenue": round(cross_month_carryover_revenue_c, 2),
                "alert_count": len(all_alerts),
                "roi_dominant_factor": app_roi_attr["dominant_factor"],
                "roi_spend_factor_score": app_roi_attr["spend_factor_score"],
                "roi_d1_anchor_score": app_roi_attr["d1_anchor_score"],
                "roi_curve_factor_score": app_roi_attr["curve_factor_score"],
                "d1_anchor_raw_mean": round(plan.d1_anchor_raw_mean, 4),
                "d1_anchor_calibrated_mean": round(plan.d1_anchor_calibrated_mean, 4),
                "app_future_d1_roi": round(app_future_d1_roi, 4),
                "spend_t1_pred_calibrated": round(spend_pred_calibrated, 2),
                "roi_d1_t1_pred_calibrated": round(roi_pred_calibrated, 4),
                "top_actions": top_actions,
            }
        )

        predictions.append(
            {
                "target_day": target_day,
                "app_id": app_id,
                "mode": plan.mode,
                "canonical_month_end_roi_source": settings.app_last_day_canonical_month_end_roi,
                "month_end_roi_prediction": round(month_end_roi_cohort_c, 4),
                "month_end_roi_stable_anchor": round(month_end_roi_cohort_stable, 4),
                "month_end_roi_pred_only": round(month_end_roi_cohort_pred_only, 4),
                "month_end_roi_fused": round(month_end_roi_cohort_fused, 4),
                "month_end_spend_prediction": round(month_end_spend_cohort_c, 2),
                "month_end_revenue_prediction": round(month_end_revenue_cohort_c, 2),
                "month_end_roi_30d_total": round(month_end_roi_30d_total_c, 4),
                "cross_month_carryover_revenue": round(cross_month_carryover_revenue_c, 2),
                "ab_compare": {
                    "A_raw_anchor": {
                        "month_end_roi_prediction": round(month_end_roi_cohort_a, 4),
                        "month_end_spend_prediction": round(month_end_spend_cohort_a, 2),
                        "month_end_revenue_prediction": round(month_end_revenue_cohort_a, 2),
                        "month_end_roi_30d_total": round(month_end_roi_30d_total_a, 4),
                        "cross_month_carryover_revenue": round(cross_month_carryover_revenue_a, 2),
                        "d1_anchor_raw_mean": round(plan_a.d1_anchor_raw_mean, 4),
                        "d1_anchor_calibrated_mean": round(plan_a.d1_anchor_calibrated_mean, 4),
                    },
                    "B_calibrated_anchor": {
                        "month_end_roi_prediction": round(month_end_roi_cohort_b, 4),
                        "month_end_spend_prediction": round(month_end_spend_cohort_b, 2),
                        "month_end_revenue_prediction": round(month_end_revenue_cohort_b, 2),
                        "month_end_roi_30d_total": round(month_end_roi_30d_total_b, 4),
                        "cross_month_carryover_revenue": round(cross_month_carryover_revenue_b, 2),
                        "d1_anchor_raw_mean": round(plan_b.d1_anchor_raw_mean, 4),
                        "d1_anchor_calibrated_mean": round(plan_b.d1_anchor_calibrated_mean, 4),
                    },
                    "roi_delta_b_minus_a": round(month_end_roi_cohort_b - month_end_roi_cohort_a, 4),
                },
                "month_end_roi_triad": month_end_roi_triad,
                "today_history_revenue_estimate": plan.today_history_revenue_estimate,
                "actual_spend_last_day": round(target_actual[app_id]["spend"], 2),
                "actual_revenue_last_day": round(target_actual[app_id]["revenue"], 2),
                "planned_total_budget": planned_total_budget,
                "alert_count": len(all_alerts),
                "cohort_incremental_formula": "sum(spend*d1_roi*(F(age_end)-F(age_before_month))) / month_spend",
                "d1_anchor_raw_mean": round(plan.d1_anchor_raw_mean, 4),
                "d1_anchor_calibrated_mean": round(plan.d1_anchor_calibrated_mean, 4),
                "app_future_d1_roi": round(app_future_d1_roi, 4),
                "spend_t1_pred_calibrated": round(spend_pred_calibrated, 2),
                "roi_d1_t1_pred_calibrated": round(roi_pred_calibrated, 4),
                "roi_forecast_attribution": app_roi_attr,
                "alerts": all_alerts,
                "top_suggestions": [s.model_dump() for s in top_suggestions],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    avg_roi_a = (sum(p["ab_compare"]["A_raw_anchor"]["month_end_roi_prediction"] for p in predictions) / len(predictions)) if predictions else 0.0
    avg_roi_b = (sum(p["ab_compare"]["B_calibrated_anchor"]["month_end_roi_prediction"] for p in predictions) / len(predictions)) if predictions else 0.0
    target_apps_total = len(target_actual)
    filtered_apps = target_apps_total - len(predictions)
    summary = {
        "task": "app_level_last_day_prediction",
        "input_file": str(csv_path),
        "train_days": len(train_days),
        "target_day": target_day,
        "kpi": kpi,
        "canonical_month_end_roi_source": settings.app_last_day_canonical_month_end_roi,
        "min_train_days": min_train_days,
        "target_apps_total": target_apps_total,
        "filtered_apps": filtered_apps,
        "predicted_apps": len(predictions),
        "avg_predicted_month_end_roi": round(
            (sum(p["month_end_roi_prediction"] for p in predictions) / len(predictions)) if predictions else 0.0, 4
        ),
        "avg_month_end_roi_30d_total": round(
            (sum(p["month_end_roi_30d_total"] for p in predictions) / len(predictions)) if predictions else 0.0, 4
        ),
        "avg_cross_month_carryover_revenue": round(
            (sum(p["cross_month_carryover_revenue"] for p in predictions) / len(predictions)) if predictions else 0.0, 2
        ),
        "ab_compare": {
            "A_raw_anchor_avg_month_end_roi": round(avg_roi_a, 4),
            "B_calibrated_anchor_avg_month_end_roi": round(avg_roi_b, 4),
            "roi_delta_b_minus_a": round(avg_roi_b - avg_roi_a, 4),
        },
        "month_end_roi_triad_summary": {
            "avg_stable_anchor": round(
                (sum(p["month_end_roi_stable_anchor"] for p in predictions) / len(predictions))
                if predictions
                else 0.0,
                4,
            ),
            "avg_pred_only": round(
                (sum(p["month_end_roi_pred_only"] for p in predictions) / len(predictions))
                if predictions
                else 0.0,
                4,
            ),
            "avg_fused": round(
                (sum(p["month_end_roi_fused"] for p in predictions) / len(predictions)) if predictions else 0.0, 4
            ),
        },
        "avg_d1_anchor_raw_mean": round(
            (sum(p["d1_anchor_raw_mean"] for p in predictions) / len(predictions)) if predictions else 0.0, 4
        ),
        "avg_d1_anchor_calibrated_mean": round(
            (sum(p["d1_anchor_calibrated_mean"] for p in predictions) / len(predictions)) if predictions else 0.0, 4
        ),
        "predictions": predictions,
    }

    summary_file = output_dir / "app_level_last_day_prediction.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_file = output_dir / "app_level_last_day_suggestions.csv"
    with detail_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_day",
                "app_id",
                "mode",
                "actual_spend_last_day",
                "actual_revenue_last_day",
                "planned_total_budget",
                "canonical_month_end_roi_source",
                "planned_daily_spend_stable_anchor",
                "planned_daily_spend_pred_only",
                "planned_daily_spend_fused",
                "month_end_roi_prediction",
                "month_end_roi_stable_anchor",
                "month_end_roi_pred_only",
                "month_end_roi_fused",
                "month_end_spend_prediction",
                "month_end_roi_prediction_a",
                "month_end_roi_prediction_b",
                "ab_roi_delta_b_minus_a",
                "month_end_roi_30d_total",
                "cross_month_carryover_revenue",
                "alert_count",
                "roi_dominant_factor",
                "roi_spend_factor_score",
                "roi_d1_anchor_score",
                "roi_curve_factor_score",
                "d1_anchor_raw_mean",
                "d1_anchor_calibrated_mean",
                "app_future_d1_roi",
                "spend_t1_pred_calibrated",
                "roi_d1_t1_pred_calibrated",
                "top_actions",
            ],
        )
        writer.writeheader()
        writer.writerows(suggestion_rows)

    # 额外输出：按应用展示偏差主因排行（便于运营优先治理）
    attribution_file = output_dir / "app_level_roi_attribution_ranking.csv"
    with attribution_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_day",
                "app_id",
                "mode",
                "month_end_roi_prediction",
                "planned_total_budget",
                "roi_dominant_factor",
                "roi_spend_factor_score",
                "roi_d1_anchor_score",
                "roi_curve_factor_score",
                "attribution_details",
            ],
        )
        writer.writeheader()
        for p in predictions:
            attr = p.get("roi_forecast_attribution", {})
            writer.writerow(
                {
                    "target_day": p.get("target_day"),
                    "app_id": p.get("app_id"),
                    "mode": p.get("mode"),
                    "month_end_roi_prediction": p.get("month_end_roi_prediction"),
                    "planned_total_budget": p.get("planned_total_budget"),
                    "roi_dominant_factor": attr.get("dominant_factor"),
                    "roi_spend_factor_score": attr.get("spend_factor_score"),
                    "roi_d1_anchor_score": attr.get("d1_anchor_score"),
                    "roi_curve_factor_score": attr.get("curve_factor_score"),
                    "attribution_details": attr.get("details"),
                }
            )

    return {
        "summary_file": str(summary_file),
        "suggestion_file": str(detail_file),
        "attribution_ranking_file": str(attribution_file),
        "target_day": target_day,
        "predicted_apps": len(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="智能预算系统离线回放测试")
    parser.add_argument(
        "--task",
        default="backtest",
        choices=["backtest", "app_last_day"],
        help="任务类型：backtest=全量回放, app_last_day=应用层级最后一天预测",
    )
    parser.add_argument(
        "--input",
        default="daily_20260421_120112.csv",
        help="原始离线数据CSV路径",
    )
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--kpi", type=float, default=DEFAULT_KPI, help="KPI ROI（默认1.05）")
    args = parser.parse_args()

    if args.task == "backtest":
        summary = run_backtest(Path(args.input), Path(args.output_dir), args.kpi)
    else:
        summary = run_app_level_last_day_prediction(Path(args.input), Path(args.output_dir), args.kpi)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
