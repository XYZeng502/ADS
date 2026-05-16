import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from app.core.calendar import CalendarService
from app.schemas import ClientContext


class PredictorService:
    """
    预测服务骨架。
    先给出可用的启发式预测，后续可以替换为时序模型/因果模型。
    """
    _release_multiplier_curve: Dict[int, float] | None = None
    _curve_max_day: int = 30
    _per_app_curves: Dict[str, Dict[int, float]] | None = None
    _calendar: CalendarService | None = None
    _scale_factors: Dict[str, float] | None = None

    @classmethod
    def _get_calendar(cls) -> CalendarService:
        if cls._calendar is None:
            cls._calendar = CalendarService()
            cls._scale_factors = cls._calendar.get_scale_factors()
        return cls._calendar

    @classmethod
    def _get_day_scale(cls, d: date) -> float:
        cal = cls._get_calendar()
        day_type = cal.classify_day(d)
        return cls._scale_factors.get(day_type, 1.0)

    @staticmethod
    def _quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        vals = sorted(values)
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * q
        low = int(pos)
        high = min(low + 1, len(vals) - 1)
        frac = pos - low
        return vals[low] * (1 - frac) + vals[high] * frac

    @classmethod
    def _build_default_curve(cls) -> Dict[int, float]:
        """
        兜底倍率曲线（与现有业务经验一致）：
        D1=1.0, D3≈1.19, D7≈1.31, D30≈1.47，并做线性插值。
        """
        node_days = [1, 3, 7, 30]
        node_mult = [1.0, 1.1942, 1.3107, 1.4666]
        curve: Dict[int, float] = {}
        for d in range(1, cls._curve_max_day + 1):
            if d <= node_days[0]:
                curve[d] = node_mult[0]
            elif d >= node_days[-1]:
                curve[d] = node_mult[-1]
            else:
                # 简单线性插值，保证单调
                for i in range(len(node_days) - 1):
                    d0, d1 = node_days[i], node_days[i + 1]
                    if d0 <= d <= d1:
                        y0, y1 = node_mult[i], node_mult[i + 1]
                        ratio = (d - d0) / (d1 - d0)
                        curve[d] = y0 + (y1 - y0) * ratio
                        break
        return curve

    @classmethod
    def _load_release_curve(cls) -> Dict[int, float]:
        """
        优先读取竞品模板输出的日级ROI曲线，再归一化为倍率曲线：
        multiplier(day) = roi_pred_p50(day) / roi_pred_p50(day=1)
        若文件缺失则回退默认倍率曲线。
        """
        if cls._release_multiplier_curve is not None:
            return cls._release_multiplier_curve

        curve_path = Path("outputs/competitor_forecast/competitor_demo_curved_competitor_roi_curve.csv")
        if not curve_path.exists():
            cls._release_multiplier_curve = cls._build_default_curve()
            return cls._release_multiplier_curve

        rows: Dict[int, float] = {}
        with curve_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    day = int(float(row.get("day", 0)))
                    roi = float(row.get("roi_pred_p50", 0.0))
                except (TypeError, ValueError):
                    continue
                if day > 0 and roi > 0:
                    rows[day] = roi

        if not rows or 1 not in rows:
            cls._release_multiplier_curve = cls._build_default_curve()
            return cls._release_multiplier_curve

        d1 = rows[1]
        curve = {}
        max_day = max(rows.keys())
        for d in range(1, max_day + 1):
            if d in rows:
                curve[d] = max(rows[d] / d1, 1.0)
            else:
                # 缺失day时用上一个值平滑填充
                curve[d] = curve[d - 1]

        # 强制单调，避免极端噪声
        last = 1.0
        for d in sorted(curve.keys()):
            curve[d] = max(curve[d], last)
            last = curve[d]

        cls._release_multiplier_curve = curve
        cls._curve_max_day = max(curve.keys())
        return cls._release_multiplier_curve

    @classmethod
    def _release_multiplier(cls, day_window: int) -> float:
        curve = cls._load_release_curve()
        if day_window <= 1:
            return curve.get(1, 1.0)
        if day_window >= cls._curve_max_day:
            return curve.get(cls._curve_max_day, max(curve.values()))
        return curve.get(day_window, curve.get(day_window - 1, 1.0))

    @classmethod
    def _load_per_app_curves(cls) -> Dict[str, Dict[int, float]]:
        """
        加载 per-app 释放曲线文件。
        Fallback 链：per-app 曲线 -> 竞品模板 -> 默认曲线
        在 _release_multiplier_for_app 中按 app_id 查找。
        """
        if cls._per_app_curves is not None:
            return cls._per_app_curves

        curve_path = Path("outputs/per_app_release_curves.json")
        if not curve_path.exists():
            cls._per_app_curves = {}
            return cls._per_app_curves

        raw = json.loads(curve_path.read_text(encoding="utf-8"))
        # JSON keys 是字符串，转回 int
        cls._per_app_curves = {
            app_id: {int(d): m for d, m in days.items()}
            for app_id, days in raw.items()
        }
        return cls._per_app_curves

    @classmethod
    def _release_multiplier_for_app(cls, day_window: int, app_id: str | None) -> float:
        """按 app_id 查找 per-app 曲线，找不到则 fallback 全局曲线。"""
        if app_id:
            per_app = cls._load_per_app_curves()
            if app_id in per_app:
                curve = per_app[app_id]
                max_day = max(curve.keys())
                if day_window <= 1:
                    return curve.get(1, 1.0)
                if day_window >= max_day:
                    return curve.get(max_day, max(curve.values()))
                return curve.get(day_window, curve.get(day_window - 1, 1.0))
        # fallback 到现有全局曲线
        return cls._release_multiplier(day_window)

    @classmethod
    def _day_type_scale(cls, d: date) -> float:
        return cls._get_day_scale(d)

    @classmethod
    def _build_spend_trajectory(
        cls,
        planned_today_spend: float,
        remaining_days: int,
        scale_factor: float,
        current_date: date | None,
    ) -> List[float]:
        """
        生成“今日到月末”的日预算轨迹：
        - 若无日期信息，回退到原始固定比例外推
        - 若有日期信息，则按日类型（工作日/周末/节假日）做节奏分配
        """
        horizon = max(remaining_days, 1)
        if planned_today_spend <= 0:
            return [0.0] * horizon

        if current_date is None:
            if horizon == 1:
                return [planned_today_spend]
            future = planned_today_spend * scale_factor
            return [planned_today_spend] + [future] * (horizon - 1)

        # 以今天的scale作为基准，后续日期按相对scale修正，避免整体突变
        base_scale = max(scale_factor, 0.8)
        trajectory = []
        for i in range(horizon):
            d = current_date + timedelta(days=i)
            s = cls._day_type_scale(d)
            # 保留轻微平滑，避免日间跳变过大
            factor = 0.75 + 0.25 * (s / base_scale)
            trajectory.append(planned_today_spend * factor)
        return trajectory

    @classmethod
    def _trajectory_volatility_score(
        cls, planned_today_spend: float, remaining_days: int, scale_factor: float, current_date: date | None
    ) -> float:
        traj = cls._build_spend_trajectory(
            planned_today_spend=planned_today_spend,
            remaining_days=remaining_days,
            scale_factor=scale_factor,
            current_date=current_date,
        )
        if not traj:
            return 0.0
        mean_v = sum(traj) / len(traj)
        if mean_v <= 0:
            return 0.0
        # 简化CV评分并截断到[0,1]
        var = sum((x - mean_v) ** 2 for x in traj) / len(traj)
        cv = (var**0.5) / mean_v
        return max(0.0, min(cv * 2.5, 1.0))

    @classmethod
    def _d1_anchor_uncertainty_score(cls, context: ClientContext) -> float:
        recent_vals: List[float] = []
        for p in context.products:
            for x in p.recent_d1_actual:
                if x is not None and x > 0:
                    recent_vals.append(float(x))
        if len(recent_vals) < 3:
            return 0.45
        p20 = cls._quantile(recent_vals, 0.2)
        p80 = cls._quantile(recent_vals, 0.8)
        median = max(cls._quantile(recent_vals, 0.5), 1e-6)
        spread_ratio = (p80 - p20) / median
        return max(0.0, min(spread_ratio * 0.9, 1.0))

    @classmethod
    def _curve_uncertainty_score(cls) -> float:
        meta_path = Path("outputs/competitor_forecast/competitor_demo_curved_competitor_roi_forecast.json")
        if not meta_path.exists():
            return 0.35
        try:
            obj = json.loads(meta_path.read_text(encoding="utf-8"))
            p25 = obj.get("forecast_nodes_p25", {}).get("roi_d30")
            p75 = obj.get("forecast_nodes_p75", {}).get("roi_d30")
            p50 = obj.get("forecast_nodes", {}).get("roi_d30")
            if not p25 or not p75 or not p50:
                return 0.35
            spread = (float(p75) - float(p25)) / max(float(p50), 1e-6)
            return max(0.0, min(spread * 1.2, 1.0))
        except Exception:
            return 0.35

    @classmethod
    def _portfolio_d1_roi_anchor(cls, context: ClientContext) -> float:
        """
        估算应用级D1锚点：
        - 优先用各产品各版位的 yesterday_spend 作为权重
        - 融合 recent_d1_actual 的稳健分位数（P20/P50）以抑制乐观偏差
        """
        weighted_num = 0.0
        weighted_den = 0.0
        plain_vals: List[float] = []
        recent_vals: List[float] = []

        for p in context.products:
            for x in p.recent_d1_actual:
                if x is not None and x > 0:
                    recent_vals.append(float(x))
            for s in p.slots:
                v = max(s.predicted_d1_roi, 0.0)
                plain_vals.append(v)
                w = max(s.yesterday_spend, 0.0)
                if w > 0:
                    weighted_num += w * v
                    weighted_den += w

        predicted_anchor = (weighted_num / weighted_den) if weighted_den > 0 else (
            (sum(plain_vals) / len(plain_vals)) if plain_vals else 0.0
        )

        if not recent_vals:
            return predicted_anchor

        # 稳健锚点：用近期分位数抑制过乐观估计，P20用于风控，P50用于中枢。
        p20 = cls._quantile(recent_vals, 0.2)
        p50 = cls._quantile(recent_vals, 0.5)
        robust_recent = 0.6 * p20 + 0.4 * p50

        # 随着近期样本增多，逐步提高近期信息权重（上限40%）
        recent_weight = min(0.4, len(recent_vals) / 50.0)
        anchor = (1 - recent_weight) * predicted_anchor + recent_weight * robust_recent

        # 安全夹逼：避免被异常高值拉高，且不低于近期悲观分位的80%
        upper_cap = max(predicted_anchor, p50) * 1.05
        lower_cap = p20 * 0.8
        anchor = max(min(anchor, upper_cap), lower_cap)
        return max(anchor, 0.0)

    @classmethod
    def estimate_history_revenue_today(cls, context: ClientContext) -> float:
        """
        历史回收估计（升级版）：
        使用“cohort增量释放”思想估算今日历史贡献，替代固定比例法。
        近似假设：
        - 本月已发生日消耗按均值近似（无日级明细输入时）
        - 每日cohort的D1锚点采用应用级加权D1
        """
        elapsed_days = max(context.date.day - 1, 0)
        if elapsed_days <= 0 or context.month_spend_so_far <= 0:
            return 0.0

        avg_daily_spend = context.month_spend_so_far / elapsed_days
        d1_anchor = max(cls._portfolio_d1_roi_anchor(context), 0.0)
        if d1_anchor <= 0:
            return 0.0

        # 对于已发生cohort，今天对应其 age 为 2..elapsed_days+1
        # 今日增量 = spend * d1 * [F(age)-F(age-1)]
        history_revenue = 0.0
        for age in range(2, elapsed_days + 2):
            f_t = cls._release_multiplier_for_app(age, context.client_id)
            f_prev = cls._release_multiplier_for_app(age - 1, context.client_id)
            delta = max(f_t - f_prev, 0.0)
            history_revenue += avg_daily_spend * d1_anchor * delta

        return history_revenue

    @classmethod
    def estimate_roi_error_attribution(
        cls,
        context: ClientContext,
        planned_today_spend: float,
        remaining_days: int,
        scale_factor: float,
        current_date: date | None,
    ) -> Dict[str, float | str]:
        spend_score = cls._trajectory_volatility_score(
            planned_today_spend=planned_today_spend,
            remaining_days=remaining_days,
            scale_factor=scale_factor,
            current_date=current_date,
        )
        d1_score = cls._d1_anchor_uncertainty_score(context)
        curve_score = cls._curve_uncertainty_score()

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
            "分数越高表示该因素对月末ROI偏差贡献越大。"
        )
        return {
            "spend_factor_score": round(spend_score, 4),
            "d1_anchor_score": round(d1_score, 4),
            "curve_factor_score": round(curve_score, 4),
            "dominant_factor": dominant,
            "details": details,
        }

    @staticmethod
    def predict_month_end_roi(
        context: ClientContext,
        planned_today_spend: float,
        expected_today_revenue: float,
        remaining_days: int,
        scale_factor: float,
        current_date: date | None = None,
    ) -> Tuple[float, float]:
        """
        返回(月末消耗预测, 月末ROI预测)。
        """
        if planned_today_spend <= 0:
            month_end_spend = context.month_spend_so_far
            month_end_revenue = context.month_revenue_so_far
            month_end_roi = month_end_revenue / month_end_spend if month_end_spend > 0 else 0.0
            return month_end_spend, month_end_roi

        # 用当日预估回收反推D1锚点，随后按“距月末窗口”映射释放倍率。
        # 这样月末天数越短，可计入当月的ROI贡献越低，更贴近真实统计口径。
        d1_roi_anchor = max(expected_today_revenue / planned_today_spend, 0.0)
        day_spends = PredictorService._build_spend_trajectory(
            planned_today_spend=planned_today_spend,
            remaining_days=remaining_days,
            scale_factor=scale_factor,
            current_date=current_date,
        )
        horizon_days = len(day_spends)

        realized_revenue = 0.0
        for idx, spend in enumerate(day_spends):
            # idx=0 是今天；越靠后天数，距月末可释放窗口越短
            release_window = max(horizon_days - idx, 1)
            multiplier = PredictorService._release_multiplier_for_app(
                release_window, context.client_id
            )
            realized_revenue += spend * d1_roi_anchor * multiplier

        month_end_spend = context.month_spend_so_far + sum(day_spends)
        month_end_revenue = context.month_revenue_so_far + realized_revenue
        month_end_roi = month_end_revenue / month_end_spend if month_end_spend > 0 else 0.0
        return month_end_spend, month_end_roi
