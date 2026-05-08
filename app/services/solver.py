from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Literal

import pulp
from app.core.calendar import CalendarService
from app.config import settings
from app.schemas import ClientContext, DailyPlanItem, ProductState
from app.services.signal import resolve_product_d1_signal
from app.services.predictor import PredictorService


Mode = Literal["GUARD", "SCALE"]


def _slot_reference_d1_aggregate(product: ProductState) -> float:
    """产品内槽位 predicted_d1_roi 汇总：默认按昨日消耗加权，与信号层/组合锚点一致。"""
    slots = product.slots
    if not slots:
        return 0.0
    preds = [max(s.predicted_d1_roi, 0.0) for s in slots]
    if settings.solver_slot_d1_normalize_mode == "arithmetic_mean":
        return sum(preds) / len(preds)
    w_sum = sum(max(s.yesterday_spend, 0.0) for s in slots)
    if w_sum > 1e-12:
        return (
            sum(max(s.predicted_d1_roi, 0.0) * max(s.yesterday_spend, 0.0) for s in slots) / w_sum
        )
    return sum(preds) / len(preds)


@dataclass
class SolveResult:
    mode: Mode
    suggestions: List[DailyPlanItem]
    expected_today_spend: float
    expected_today_revenue: float


class BudgetSolver:
    """
    求解器骨架：
    - 当前：启发式分配，按边际收益排序并满足cap与波动约束
    - 后续：可替换成 Gurobi/PuLP 联合求解（产品 x 时间）
    """

    def __init__(self) -> None:
        self.calendar = CalendarService()

    def _heuristic_allocate(self, context: ClientContext, mode: Mode, day_scale: float) -> SolveResult:
        """LP失败时的兜底策略，保持系统可用。"""
        budget_ceiling = context.yesterday_total_spend * day_scale
        if mode == "SCALE":
            budget_ceiling *= 1.15
        else:
            budget_ceiling *= 0.95

        ranked = []
        for product in context.products:
            product_d1_anchor, _ = resolve_product_d1_signal(product)
            ref_slot_d1 = _slot_reference_d1_aggregate(product)
            scale = (product_d1_anchor / ref_slot_d1) if ref_slot_d1 > 1e-8 else 1.0
            scale = max(0.8, min(scale, 1.2))
            for slot in product.slots:
                adjusted_d1 = max(slot.predicted_d1_roi * scale, 0.0)
                long_term_roi = adjusted_d1 * slot.m30_multiplier
                score = long_term_roi + product.priority * 0.001
                ranked.append((score, product, slot, adjusted_d1))

        ranked.sort(key=lambda x: x[0], reverse=True)
        remain = budget_ceiling
        items: List[DailyPlanItem] = []
        total_spend = 0.0
        total_rev = 0.0
        for score, product, slot, adjusted_d1 in ranked:
            if remain <= 0:
                break
            slot_limit = min(slot.cap, max(slot.yesterday_spend * 1.4, slot.yesterday_spend + 100.0))
            alloc = min(remain, slot_limit)
            if mode == "GUARD" and adjusted_d1 * slot.m30_multiplier < context.kpi_roi * 0.95:
                alloc *= 0.4
            if alloc <= 0:
                continue
            expected_d1 = alloc * adjusted_d1
            total_spend += alloc
            total_rev += expected_d1
            remain -= alloc
            items.append(
                DailyPlanItem(
                    product_id=product.product_id,
                    product_name=product.product_name,
                    slot=slot.slot,
                    suggested_budget=round(alloc, 2),
                    expected_d1_revenue=round(expected_d1, 2),
                    expected_roi=round(adjusted_d1 * slot.m30_multiplier, 4),
                    decision_reason=f"heuristic score={score:.4f}, mode={mode}",
                )
            )
        return SolveResult(
            mode=mode,
            suggestions=items,
            expected_today_spend=round(total_spend, 2),
            expected_today_revenue=round(total_rev, 2),
        )

    def solve_daily(self, context: ClientContext, day_scale: float, current_date: date | None = None) -> SolveResult:
        current_roi = (
            context.month_revenue_so_far / context.month_spend_so_far
            if context.month_spend_so_far > 0
            else context.kpi_roi
        )
        mode: Mode = "SCALE" if current_roi >= context.kpi_roi + settings.release_surplus_threshold else "GUARD"
        if current_date is None:
            current_date = context.date

        horizon_days = max(30 - current_date.day + 1, 1)
        surplus_gap = max(current_roi - (context.kpi_roi + settings.release_surplus_threshold), 0.0)
        release_boost = 1.0
        if mode == "SCALE":
            # 盈余越高，可释放的预算越积极（但设上限防失控）
            release_boost += min(surplus_gap * 2.0, settings.lp_scale_release_extra_cap)
        today_budget_ceiling = context.yesterday_total_spend * day_scale * (1.15 if mode == "SCALE" else 0.95) * release_boost

        slot_records = []
        for product in context.products:
            product_d1_anchor, _ = resolve_product_d1_signal(product)
            ref_slot_d1 = _slot_reference_d1_aggregate(product)
            scale = (product_d1_anchor / ref_slot_d1) if ref_slot_d1 > 1e-8 else 1.0
            scale = max(0.8, min(scale, 1.2))
            for slot in product.slots:
                adjusted_d1 = max(slot.predicted_d1_roi * scale, 0.0)
                slot_records.append((product, slot, adjusted_d1))

        if not slot_records:
            return SolveResult(mode=mode, suggestions=[], expected_today_spend=0.0, expected_today_revenue=0.0)

        day_scales = []
        for t in range(horizon_days):
            d = current_date + timedelta(days=t)
            dt = self.calendar.classify_day(d)
            day_scales.append(self.calendar.get_scale_factors().get(dt, 1.0))

        # 联合优化：max Σ x[p,s,t]（加入时序折扣 + 今日/近端消耗软下限 + 弹性平滑）
        prob = pulp.LpProblem("portfolio_time_joint", pulp.LpMaximize)
        x_vars = {}
        coeff = {}
        time_weight = {}
        for idx, (product, slot, adjusted_d1) in enumerate(slot_records):
            for t in range(horizon_days):
                v = pulp.LpVariable(f"x_{idx}_{t}", lowBound=0)
                x_vars[(idx, t)] = v

                s = day_scales[t]
                relative_scale = s / max(day_scale, 0.8)
                day_cap = min(slot.cap, max(slot.yesterday_spend * 1.4, slot.yesterday_spend + 100.0) * relative_scale)
                prob += v <= max(day_cap, 0.0)

                release_window = max(horizon_days - t, 1)
                month_end_roi_per_spend = adjusted_d1 * PredictorService._release_multiplier(release_window)
                coeff[(idx, t)] = month_end_roi_per_spend
                time_weight[(idx, t)] = settings.lp_time_decay_factor**t

                # GUARD 模式抑制低效槽位
                if mode == "GUARD" and adjusted_d1 * slot.m30_multiplier < context.kpi_roi * 0.95:
                    prob += v <= max(day_cap * 0.4, 0.0)

                if t > 0:
                    ratio = day_scales[t] / max(day_scales[t - 1], 0.8)
                    up = settings.lp_smooth_up_factor * ratio
                    down = settings.lp_smooth_down_factor * ratio
                    tol = settings.lp_smooth_abs_tolerance
                    prev = x_vars[(idx, t - 1)]
                    # 自适应平滑：避免无业务理由的剧烈跳变
                    prob += v <= prev * up + tol
                    prob += v >= prev * down - tol

        # 当日总预算上限（执行层）
        prob += pulp.lpSum(x_vars[(idx, 0)] for idx in range(len(slot_records))) <= max(today_budget_ceiling, 0.0)

        # 月度ROI硬约束： (R0 + Σx*c) / (S0 + Σx) >= KPI
        rhs = context.kpi_roi * context.month_spend_so_far - context.month_revenue_so_far
        roi_lhs = pulp.lpSum(x_vars[(idx, t)] * (coeff[(idx, t)] - context.kpi_roi) for (idx, t) in x_vars)
        prob += roi_lhs >= rhs

        # 今日消耗软下限：在可行前提下尽量贴近运营执行节奏，避免过保守
        today_floor_ratio = settings.lp_today_floor_ratio_scale if mode == "SCALE" else settings.lp_today_floor_ratio_guard
        today_target_floor = max(today_budget_ceiling * today_floor_ratio, 0.0)
        today_sum = pulp.lpSum(x_vars[(idx, 0)] for idx in range(len(slot_records)))
        shortfall = pulp.LpVariable("today_shortfall", lowBound=0)
        prob += shortfall >= today_target_floor - today_sum

        near_shortfall = None
        if mode == "SCALE":
            k = min(max(settings.lp_near_term_days, 1), horizon_days)
            near_target = today_budget_ceiling * settings.lp_near_term_floor_ratio * k
            near_sum = pulp.lpSum(x_vars[(idx, t)] for idx in range(len(slot_records)) for t in range(k))
            near_shortfall = pulp.LpVariable("near_shortfall", lowBound=0)
            prob += near_shortfall >= near_target - near_sum

        total_future_spend = pulp.lpSum(x_vars.values())
        roi_surplus_excess = pulp.LpVariable("roi_surplus_excess", lowBound=0)
        target_gap = max(settings.lp_roi_surplus_target_gap, 0.0)
        # surplus_above_kpi = roi_lhs - rhs
        # 允许一定安全边际，超出部分计入惩罚
        prob += roi_surplus_excess >= (roi_lhs - rhs) - target_gap * (context.month_spend_so_far + total_future_spend)

        month_release_shortfall = pulp.LpVariable("month_release_shortfall", lowBound=0)
        release_floor = max(today_budget_ceiling * horizon_days * settings.lp_month_release_floor_ratio, 0.0)
        prob += month_release_shortfall >= release_floor - total_future_spend

        # 目标：最大化剩余期消耗（略偏向近期）并惩罚今日欠投放
        objective = (
            pulp.lpSum(x_vars[(idx, t)] * time_weight[(idx, t)] for (idx, t) in x_vars)
            - settings.lp_today_shortfall_penalty * shortfall
        )
        if near_shortfall is not None:
            objective -= settings.lp_near_term_shortfall_penalty * near_shortfall
        objective -= settings.lp_roi_surplus_penalty * roi_surplus_excess
        objective -= settings.lp_month_release_shortfall_penalty * month_release_shortfall
        prob += objective

        solve_status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[solve_status] not in {"Optimal", "Feasible"}:
            return self._heuristic_allocate(context=context, mode=mode, day_scale=day_scale)

        items: List[DailyPlanItem] = []
        total_spend = 0.0
        total_rev = 0.0
        for idx, (product, slot, adjusted_d1) in enumerate(slot_records):
            alloc = float(pulp.value(x_vars[(idx, 0)]) or 0.0)
            if alloc <= 1e-6:
                continue
            expected_d1 = alloc * adjusted_d1
            total_spend += alloc
            total_rev += expected_d1
            items.append(
                DailyPlanItem(
                    product_id=product.product_id,
                    product_name=product.product_name,
                    slot=slot.slot,
                    suggested_budget=round(alloc, 2),
                    expected_d1_revenue=round(expected_d1, 2),
                    expected_roi=round(adjusted_d1 * slot.m30_multiplier, 4),
                    decision_reason=f"joint_lp mode={mode}",
                )
            )

        return SolveResult(mode=mode, suggestions=items, expected_today_spend=round(total_spend, 2), expected_today_revenue=round(total_rev, 2))
