from typing import List

from app.config import settings
from app.schemas import ClientContext, ProductDiagnosis, RiskAlert
from app.services.signal import resolve_product_d1_signal


class RiskService:
    def evaluate_product_risks(self, context: ClientContext) -> List[RiskAlert]:
        alerts: List[RiskAlert] = []
        for p in context.products:
            if len(p.recent_d1_actual) < settings.risk_d1_drop_days or len(p.recent_d1_predicted) < settings.risk_d1_drop_days:
                continue
            actual = p.recent_d1_actual[-settings.risk_d1_drop_days :]
            pred = p.recent_d1_predicted[-settings.risk_d1_drop_days :]

            drops = 0
            for a, b in zip(actual, pred):
                if b > 0 and a < b * settings.risk_d1_drop_ratio:
                    drops += 1
            if drops == settings.risk_d1_drop_days:
                alerts.append(
                    RiskAlert(
                        level="WARN",
                        category="PRODUCT_D1_DROP",
                        product_id=p.product_id,
                        message=f"{p.product_name} 连续{settings.risk_d1_drop_days}天D1低于预测90%，建议自动降权。",
                    )
                )
        return alerts

    def evaluate_traffic_anomaly(self, today_real_revenue: float, today_predicted_revenue: float) -> List[RiskAlert]:
        if today_predicted_revenue <= 0:
            return []
        dev = abs(today_real_revenue - today_predicted_revenue) / today_predicted_revenue
        if dev >= settings.risk_revenue_deviation_threshold:
            return [
                RiskAlert(
                    level="CRITICAL",
                    category="TRAFFIC_ANOMALY",
                    message=f"当日变现偏离预测 {dev:.1%}，建议切换保守预算。",
                )
            ]
        return []

    def build_diagnosis(self, context: ClientContext) -> List[ProductDiagnosis]:
        diagnosis: List[ProductDiagnosis] = []
        remaining_days = max(30 - context.date.day, 0)
        for p in context.products:
            # 使用最高m30槽位估算该产品达标D1阈值
            max_m30 = max([slot.m30_multiplier for slot in p.slots], default=1.0)
            target_d1 = context.kpi_roi / max_m30 if max_m30 > 0 else context.kpi_roi
            d1_signal, signal_source = resolve_product_d1_signal(p)
            gap = d1_signal - target_d1
            if gap >= settings.diagnosis_a_gap_threshold:
                level, action = "A", "加量"
            elif gap >= settings.diagnosis_b_gap_threshold:
                level, action = "B", "维持"
            else:
                level, action = "C", "减量或观察"

            diagnosis.append(
                ProductDiagnosis(
                    product_id=p.product_id,
                    product_name=p.product_name,
                    efficiency_level=level,
                    current_d1_signal=round(d1_signal, 4),
                    d1_signal_source=signal_source,
                    target_d1=round(target_d1, 4),
                    gap_to_target=round(gap, 4),
                    action=action,
                    improvement_window_days=remaining_days,
                )
            )
        return diagnosis
