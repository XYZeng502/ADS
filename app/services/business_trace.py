from app.schemas import (
    BusinessDecisionTrace,
    BusinessLayerStatus,
    BusinessLayerSummary,
    ClientContext,
    KPIAdjustmentImpact,
    ProductDiagnosis,
    ROIForecastAttribution,
    RiskAlert,
    RuleHit,
    SlotBudgetSummary,
)
from app.services.solver import SolveResult


def _status_by_gap(gap: float, warn_gap: float = 0.02, critical_gap: float = 0.05) -> BusinessLayerStatus:
    if gap <= -critical_gap:
        return "CRITICAL"
    if gap <= -warn_gap:
        return "WARN"
    return "OK"


class BusinessTraceService:
    """把系统计算结果整理成运营可读的业务决策链路。"""

    def build(
        self,
        *,
        context: ClientContext,
        solve_result: SolveResult,
        month_end_spend: float,
        month_end_roi: float,
        current_roi: float,
        kpi_impact: KPIAdjustmentImpact,
        roi_attr: ROIForecastAttribution,
        slot_summary: list[SlotBudgetSummary],
        diagnosis: list[ProductDiagnosis],
        alerts: list[RiskAlert],
        rule_hits: list[RuleHit],
        d1_raw_mean: float,
        d1_calibrated_mean: float,
    ) -> BusinessDecisionTrace:
        roi_gap = month_end_roi - context.kpi_roi
        kpi_status = _status_by_gap(roi_gap)
        forecast_status = "WARN" if roi_attr.dominant_factor in {"SPEND", "D1_ANCHOR", "CURVE"} and max(
            roi_attr.spend_factor_score,
            roi_attr.d1_anchor_score,
            roi_attr.curve_factor_score,
        ) >= 0.45 else "OK"
        if month_end_roi < context.kpi_roi:
            forecast_status = "CRITICAL" if roi_gap <= -0.05 else "WARN"

        risk_status: BusinessLayerStatus = "OK"
        if any(a.level == "CRITICAL" for a in alerts):
            risk_status = "CRITICAL"
        elif any(a.level == "WARN" for a in alerts):
            risk_status = "WARN"

        c_count = sum(1 for x in diagnosis if x.efficiency_level == "C")
        a_count = sum(1 for x in diagnosis if x.efficiency_level == "A")
        top_rule = rule_hits[0] if rule_hits else None

        slot_metrics = {f"{x.slot}_budget": round(x.suggested_budget, 2) for x in slot_summary}
        decision_status = risk_status if risk_status != "OK" else kpi_status

        return BusinessDecisionTrace(
            kpi_layer=BusinessLayerSummary(
                layer_id="kpi",
                title="目标/KPI层",
                status=kpi_status,
                primary_metrics={
                    "kpi_roi": round(context.kpi_roi, 4),
                    "current_month_roi": round(current_roi, 4),
                    "forecast_month_roi": round(month_end_roi, 4),
                    "roi_gap_vs_kpi": round(roi_gap, 4),
                },
                key_findings=[
                    f"当前累计ROI={current_roi:.2%}，预测月末ROI={month_end_roi:.2%}，KPI={context.kpi_roi:.2%}。",
                    f"KPI差距={roi_gap:.2%}，当前模式={solve_result.mode}。",
                ],
                recommended_actions=[kpi_impact.message],
            ),
            forecast_layer=BusinessLayerSummary(
                layer_id="forecast",
                title="预测层",
                status=forecast_status,
                primary_metrics={
                    "month_end_spend_prediction": round(month_end_spend, 2),
                    "month_end_roi_prediction": round(month_end_roi, 4),
                    "d1_anchor_raw_mean": round(d1_raw_mean, 4),
                    "d1_anchor_calibrated_mean": round(d1_calibrated_mean, 4),
                    "dominant_factor": roi_attr.dominant_factor,
                    "spend_factor_score": roi_attr.spend_factor_score,
                    "d1_anchor_score": roi_attr.d1_anchor_score,
                    "curve_factor_score": roi_attr.curve_factor_score,
                },
                key_findings=[
                    f"预测主因={roi_attr.dominant_factor}；{roi_attr.details}",
                    f"D1锚点从 raw={d1_raw_mean:.4f} 校准到 calibrated={d1_calibrated_mean:.4f}。",
                ],
                recommended_actions=["若预测主因为SPEND，优先检查分流消耗预测和近期预算执行节奏。"],
            ),
            solver_layer=BusinessLayerSummary(
                layer_id="solver",
                title="求解层",
                status="OK" if solve_result.expected_today_spend > 0 else "WARN",
                primary_metrics={
                    "mode": solve_result.mode,
                    "expected_today_spend": round(solve_result.expected_today_spend, 2),
                    "expected_today_d1_revenue": round(solve_result.expected_today_revenue, 2),
                    "suggestion_count": len(solve_result.suggestions),
                    **slot_metrics,
                },
                key_findings=[
                    f"今日建议总预算={solve_result.expected_today_spend:.2f}，建议条目数={len(solve_result.suggestions)}。",
                    f"求解模式={solve_result.mode}，按月末ROI约束控制消耗释放。",
                ],
                recommended_actions=[f"执行动作：{kpi_impact.suggested_action}，预算变化参考={kpi_impact.suggested_budget_delta:.2f}。"],
            ),
            risk_layer=BusinessLayerSummary(
                layer_id="risk",
                title="风险层",
                status=risk_status,
                primary_metrics={
                    "alert_count": len(alerts),
                    "critical_alert_count": sum(1 for a in alerts if a.level == "CRITICAL"),
                    "warn_alert_count": sum(1 for a in alerts if a.level == "WARN"),
                    "a_level_product_count": a_count,
                    "c_level_product_count": c_count,
                },
                key_findings=[a.message for a in alerts] or ["暂无关键风险预警。"],
                recommended_actions=["优先处理CRITICAL告警；若C档产品集中，建议预算向A/B档倾斜。"],
            ),
            decision_layer=BusinessLayerSummary(
                layer_id="decision",
                title="决策解释层",
                status=decision_status,
                primary_metrics={
                    "rule_hit_count": len(rule_hits),
                    "top_rule_priority": top_rule.priority if top_rule else 0,
                    "suggested_budget_delta": kpi_impact.suggested_budget_delta,
                },
                key_findings=[f"{r.rule_id}: {r.reason}" for r in rule_hits] or ["未命中强规则，按预测和求解结果滚动执行。"],
                recommended_actions=[r.action for r in rule_hits] or [kpi_impact.message],
            ),
        )
