from typing import List

from app.config import settings
from app.schemas import ProductDiagnosis, RuleHit


class RuleEngine:
    """
    P0-3 规则引擎（可配置）：
    - 输出可解释的命中规则（rule_id/优先级/动作/原因）
    - 不直接改写求解器结果，先做决策解释层，避免行为抖动
    """

    def evaluate(
        self,
        *,
        kpi_roi: float,
        month_end_roi: float,
        remaining_days: int,
        day_type: str,
        diagnosis: List[ProductDiagnosis],
    ) -> List[RuleHit]:
        hits: List[RuleHit] = []
        roi_gap = month_end_roi - kpi_roi

        if roi_gap >= settings.kpi_release_gap_threshold:
            hits.append(
                RuleHit(
                    rule_id="R1_SURPLUS_RELEASE",
                    priority=100,
                    action=f"建议释放预算 +{int(settings.rule_scale_release_ratio * 100)}%",
                    reason="月末ROI预测高于KPI阈值，存在盈余可释放空间。",
                    metric_snapshot={"roi_gap_vs_kpi": round(roi_gap, 4)},
                )
            )

        if roi_gap <= -settings.kpi_guard_gap_threshold:
            hits.append(
                RuleHit(
                    rule_id="R2_KPI_GUARD",
                    priority=120,
                    action=f"建议缩减预算 {int(settings.rule_guard_reduce_ratio * 100)}%",
                    reason="月末ROI预测低于KPI警戒阈值，需进入守护模式。",
                    metric_snapshot={"roi_gap_vs_kpi": round(roi_gap, 4)},
                )
            )

        if remaining_days <= settings.rule_month_end_days and roi_gap >= 0:
            hits.append(
                RuleHit(
                    rule_id="R3_MONTH_END_RELEASE_WINDOW",
                    priority=80,
                    action="建议月末盈余释放（跨月飞轮）",
                    reason="已进入月末窗口且仍有ROI盈余，适合可控放量。",
                    metric_snapshot={"remaining_days": float(remaining_days), "roi_gap_vs_kpi": round(roi_gap, 4)},
                )
            )

        c_count = sum(1 for d in diagnosis if d.efficiency_level == "C")
        total = max(len(diagnosis), 1)
        c_ratio = c_count / total
        if c_ratio >= settings.rule_c_level_ratio_threshold:
            hits.append(
                RuleHit(
                    rule_id="R4_LOW_EFFICIENCY_CLUSTER",
                    priority=90,
                    action="建议收缩低效产品并向A/B档倾斜",
                    reason="C档产品占比过高，组合效率风险上升。",
                    metric_snapshot={"c_level_ratio": round(c_ratio, 4)},
                )
            )

        if day_type in {"holiday", "shopping_festival", "summer_winter"}:
            hits.append(
                RuleHit(
                    rule_id="R5_TIME_WINDOW_BONUS",
                    priority=60,
                    action="建议提升高效槽位预算上限",
                    reason="当前处于流量高价值时间窗，应提高优质库存利用率。",
                    metric_snapshot={"day_scale_window": 1.0},
                )
            )

        hits.sort(key=lambda x: x.priority, reverse=True)
        return hits
