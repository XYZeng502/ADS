from datetime import date

from app.core.calendar import CalendarService
from app.config import settings
from app.schemas import (
    KPIAdjustmentImpact,
    ClientContext,
    DailyPlanResponse,
    ROIForecastAttribution,
    SlotBudgetSummary,
)
from app.services.business_trace import BusinessTraceService
from app.services.predictor import PredictorService
from app.services.rule_engine import RuleEngine
from app.services.risk import RiskService
from app.services.signal import resolve_product_d1_signal_with_meta
from app.services.solver import BudgetSolver


class DailyOrchestrator:
    def __init__(self) -> None:
        self.calendar = CalendarService()
        self.predictor = PredictorService()
        self.solver = BudgetSolver()
        self.risk = RiskService()
        self.rule_engine = RuleEngine()
        self.business_trace = BusinessTraceService()

    def run(self, context: ClientContext) -> DailyPlanResponse:
        day_type = self.calendar.classify_day(context.date)
        scale = self.calendar.get_scale_factors().get(day_type, 1.0)

        solve_result = self.solver.solve_daily(context, day_scale=scale, current_date=context.date)
        history_revenue = self.predictor.estimate_history_revenue_today(context)

        month_end_spend, month_end_roi = self.predictor.predict_month_end_roi(
            context=context,
            planned_today_spend=solve_result.expected_today_spend,
            expected_today_revenue=solve_result.expected_today_revenue + history_revenue,
            remaining_days=max(30 - context.date.day, 0),
            scale_factor=scale,
            current_date=context.date,
        )

        today_total_budget_target = round(sum(item.suggested_budget for item in solve_result.suggestions), 2)
        today_d1_revenue_target = round(sum(item.expected_d1_revenue for item in solve_result.suggestions), 2)
        slot_budget_map = {"store": 0.0, "union": 0.0, "smart": 0.0}
        for item in solve_result.suggestions:
            slot_budget_map[item.slot] += item.suggested_budget
        slot_summary = [
            SlotBudgetSummary(slot="store", suggested_budget=round(slot_budget_map["store"], 2)),
            SlotBudgetSummary(slot="union", suggested_budget=round(slot_budget_map["union"], 2)),
            SlotBudgetSummary(slot="smart", suggested_budget=round(slot_budget_map["smart"], 2)),
        ]

        alerts = self.risk.evaluate_product_risks(context)
        if month_end_roi < context.kpi_roi - settings.risk_month_roi_guard_buffer:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "category": "ROI_GUARD",
                    "message": f"预测月末ROI={month_end_roi:.2%} 低于KPI警戒线，建议缩减预算。",
                    "product_id": None,
                }
            )
            # 将字典转回模型，统一结构
            from app.schemas import RiskAlert

            alerts = [a if isinstance(a, RiskAlert) else RiskAlert(**a) for a in alerts]

        roi_gap = month_end_roi - context.kpi_roi
        if roi_gap >= settings.kpi_release_gap_threshold:
            action, delta = "释放预算", round(context.yesterday_total_spend * settings.kpi_release_budget_ratio, 2)
            msg = "月末ROI有明显盈余，可在风控约束下适度释放预算。"
        elif roi_gap <= -settings.kpi_guard_gap_threshold:
            action, delta = "缩减预算", round(-context.yesterday_total_spend * settings.kpi_guard_budget_ratio, 2)
            msg = "月末ROI低于KPI较多，建议进入保守模式并缩减预算。"
        else:
            action, delta = "维持", 0.0
            msg = "月末ROI接近KPI，建议维持当前节奏并滚动观测。"

        kpi_impact = KPIAdjustmentImpact(
            old_kpi=context.kpi_roi,
            projected_month_end_roi=round(month_end_roi, 4),
            roi_gap_vs_kpi=round(roi_gap, 4),
            suggested_action=action,
            suggested_budget_delta=delta,
            message=msg,
        )
        roi_attr = ROIForecastAttribution(
            **self.predictor.estimate_roi_error_attribution(
                context=context,
                planned_today_spend=solve_result.expected_today_spend,
                remaining_days=max(30 - context.date.day, 0),
                scale_factor=scale,
                current_date=context.date,
            )
        )
        diagnosis = self.risk.build_diagnosis(context)
        d1_meta = [resolve_product_d1_signal_with_meta(p)[2] for p in context.products]
        d1_raw_mean = round(sum(m.raw_signal for m in d1_meta) / len(d1_meta), 4) if d1_meta else 0.0
        d1_calibrated_mean = round(sum(m.calibrated_signal for m in d1_meta) / len(d1_meta), 4) if d1_meta else 0.0
        rule_hits = self.rule_engine.evaluate(
            kpi_roi=context.kpi_roi,
            month_end_roi=month_end_roi,
            remaining_days=max(30 - context.date.day, 0),
            day_type=day_type,
            diagnosis=diagnosis,
        )
        current_roi = (
            context.month_revenue_so_far / context.month_spend_so_far
            if context.month_spend_so_far > 0
            else context.kpi_roi
        )
        business_trace = self.business_trace.build(
            context=context,
            solve_result=solve_result,
            month_end_spend=month_end_spend,
            month_end_roi=month_end_roi,
            current_roi=current_roi,
            kpi_impact=kpi_impact,
            roi_attr=roi_attr,
            slot_summary=slot_summary,
            diagnosis=diagnosis,
            alerts=alerts,
            rule_hits=rule_hits,
            d1_raw_mean=d1_raw_mean,
            d1_calibrated_mean=d1_calibrated_mean,
        )

        notes = [
            f"day_type={day_type}, scale={scale}",
            f"d1_anchor_raw_mean={d1_raw_mean}, d1_anchor_calibrated_mean={d1_calibrated_mean}",
            f"solver_slot_d1_normalize_mode={settings.solver_slot_d1_normalize_mode}",
            "槽位缩放：产品内「参考 D1」与信号锚点对齐时使用消耗加权（与离线 roi_d1 脚本一致）；"
            "若需逐项复现旧算术平均，可把 settings.solver_slot_d1_normalize_mode 设为 arithmetic_mean。",
            "当前已启用产品×时间联合求解（PuLP），并可平滑迁移到 Gurobi。",
            "每日 T+1 数据回流后请重新调用 /v1/optimize/daily-plan 进行滚动修正。",
        ]

        return DailyPlanResponse(
            client_id=context.client_id,
            date=context.date,
            mode=solve_result.mode,
            month_cumulative_spend_forecast=round(month_end_spend, 2),
            month_total_scale_forecast=round(month_end_spend, 2),
            month_end_spend_prediction=round(month_end_spend, 2),
            month_roi_forecast=round(month_end_roi, 4),
            month_end_roi_prediction=round(month_end_roi, 4),
            today_d1_revenue_target=today_d1_revenue_target,
            today_total_budget_target=today_total_budget_target,
            today_slot_budget_summary=slot_summary,
            kpi_adjustment_impact=kpi_impact,
            roi_forecast_attribution=roi_attr,
            today_history_revenue_estimate=round(history_revenue, 2),
            suggestions=solve_result.suggestions,
            diagnosis=diagnosis,
            alerts=alerts,
            rule_hits=rule_hits,
            d1_anchor_raw_mean=d1_raw_mean,
            d1_anchor_calibrated_mean=d1_calibrated_mean,
            business_trace=business_trace,
            notes=notes,
        )

    def simulate_next_day(self, context: ClientContext, day: date) -> DailyPlanResponse:
        updated = context.model_copy(update={"date": day})
        return self.run(updated)
