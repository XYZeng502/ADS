from typing import Dict, List

from app.config import settings
from app.schemas import ClientContext, ProductDiagnosis, ProductState, RiskAlert


class RiskService:
    # ---- product-level ----

    def evaluate_product_risks(self, context: ClientContext) -> List[RiskAlert]:
        alerts: List[RiskAlert] = []
        for p in context.products:
            alerts.extend(self._check_d1_drop(p))
            alerts.extend(self._check_roi_decline(p))
            alerts.extend(self._check_cap_proximity(p, context.yesterday_total_spend))
        alerts.extend(self.evaluate_drift())
        return alerts

    def _check_d1_drop(self, p: ProductState) -> List[RiskAlert]:
        if len(p.recent_d1_actual) < settings.risk_d1_drop_days or len(p.recent_d1_predicted) < settings.risk_d1_drop_days:
            return []
        actual = p.recent_d1_actual[-settings.risk_d1_drop_days:]
        pred = p.recent_d1_predicted[-settings.risk_d1_drop_days:]
        drops = sum(1 for a, b in zip(actual, pred) if b > 0 and a < b * settings.risk_d1_drop_ratio)
        if drops >= settings.risk_d1_drop_days:
            return [RiskAlert(
                level="WARN", category="PRODUCT_D1_DROP", product_id=p.product_id,
                message=f"{p.product_name} 连续{settings.risk_d1_drop_days}天D1低于预测90%，建议自动降权。",
            )]
        return []

    def _check_roi_decline(self, p: ProductState) -> List[RiskAlert]:
        """连续N天ROI趋势下行（逐日递减）。"""
        hist = p.recent_d1_actual
        if len(hist) < settings.risk_roi_decline_days:
            return []
        window = hist[-settings.risk_roi_decline_days:]
        declining = all(window[i] >= window[i + 1] for i in range(len(window) - 1)) and any(window[i] > window[i + 1] for i in range(len(window) - 1))
        if declining and window[0] > 0:
            drop_pct = (window[0] - window[-1]) / window[0] if window[0] > 0 else 0
            return [RiskAlert(
                level="WARN", category="ROI_DECLINE_TREND", product_id=p.product_id,
                message=f"{p.product_name} ROI连续{len(window)}天下降（{window[0]:.3f}→{window[-1]:.3f}，跌幅{drop_pct:.0%}），建议关注。",
            )]
        return []

    def _check_cap_proximity(self, p: ProductState, yesterday_total_spend: float) -> List[RiskAlert]:
        """计划预算接近历史cap时预警。"""
        if not p.slots:
            return []
        max_cap = max(s.cap for s in p.slots)
        total_planned = sum(s.yesterday_spend for s in p.slots)
        if max_cap > 0 and total_planned / max_cap >= settings.risk_cap_utilization_threshold:
            return [RiskAlert(
                level="WARN", category="CAP_PROXIMITY", product_id=p.product_id,
                message=f"{p.product_name} 计划预算接近上限（{total_planned:.0f}/{max_cap:.0f}，{total_planned/max_cap:.0%}），建议验证扩量空间。",
            )]
        return []

    # ---- app-level spend anomaly ----

    def evaluate_spend_anomaly(self, app_id: str, current_spend: float,
                               recent_spends: List[float]) -> List[RiskAlert]:
        """检测消耗异常波动（骤降/骤升）。"""
        if len(recent_spends) < 3:
            return []
        avg_spend = sum(recent_spends) / len(recent_spends)
        if avg_spend < settings.risk_spend_anomaly_min_avg_spend:
            return []
        ratio = current_spend / avg_spend
        if ratio <= settings.risk_spend_drop_ratio:
            return [RiskAlert(
                level="CRITICAL", category="SPEND_DROP",
                message=f"应用{app_id} 消耗骤降至近期均值的{ratio:.0%}（{current_spend:.0f} vs 均值{avg_spend:.0f}），请排查投放异常。",
            )]
        if ratio >= settings.risk_spend_spike_ratio:
            return [RiskAlert(
                level="WARN", category="SPEND_SPIKE",
                message=f"应用{app_id} 消耗骤升至近期均值的{ratio:.0%}（{current_spend:.0f} vs 均值{avg_spend:.0f}），请确认是否预期内放量。",
            )]
        return []

    # ---- traffic anomaly ----

    def evaluate_traffic_anomaly(self, today_real_revenue: float, today_predicted_revenue: float) -> List[RiskAlert]:
        if today_predicted_revenue <= 0:
            return []
        dev = abs(today_real_revenue - today_predicted_revenue) / today_predicted_revenue
        if dev >= settings.risk_revenue_deviation_threshold:
            direction = "高于" if today_real_revenue > today_predicted_revenue else "低于"
            return [RiskAlert(
                level="CRITICAL", category="TRAFFIC_ANOMALY",
                message=f"当日变现{direction}预测 {dev:.1%}，建议切换保守预算。",
            )]
        return []

    # ---- drift ----

    def evaluate_drift(self) -> List[RiskAlert]:
        try:
            from app.services.drift import check_prediction_drift
            report = check_prediction_drift()
        except Exception:
            return []

        alerts: List[RiskAlert] = []
        for m in report.models:
            if m.status == "critical":
                alerts.append(RiskAlert(
                    level="CRITICAL",
                    category="PREDICTION_DRIFT",
                    message=(f"{m.model} 模型漂移严重: 近期 MAPE={m.recent_mape:.1f}%, "
                             f"baseline={m.baseline_mape:.1f}%, drift_ratio={m.drift_ratio:.1f}x，"
                             f"建议降级到 EWMA baseline 并触发紧急重训。"),
                ))
            elif m.status == "warning":
                alerts.append(RiskAlert(
                    level="WARN",
                    category="PREDICTION_DRIFT",
                    message=(f"{m.model} 模型漂移预警: 近期 MAPE={m.recent_mape:.1f}%, "
                             f"baseline={m.baseline_mape:.1f}%, drift_ratio={m.drift_ratio:.1f}x，"
                             f"建议关注并考虑重训。"),
                ))
        return alerts

    # ---- diagnosis ----

    def build_diagnosis(self, context: ClientContext) -> List[ProductDiagnosis]:
        diagnosis: List[ProductDiagnosis] = []
        remaining_days = max(30 - context.date.day, 0)
        for p in context.products:
            max_m30 = max([slot.m30_multiplier for slot in p.slots], default=1.0)
            target_d1 = context.kpi_roi / max_m30 if max_m30 > 0 else context.kpi_roi
            d1_signal, signal_source = self._resolve_d1_signal(p)
            gap = d1_signal - target_d1

            if gap >= settings.diagnosis_a_gap_threshold:
                level, action = "A", "加量"
            elif gap >= settings.diagnosis_b_gap_threshold:
                level, action = "B", "维持"
            else:
                level, action = "C", "减量或观察"

            # trend direction
            trend = self._compute_trend(p.recent_d1_actual)

            # cap utilization
            max_cap = max((s.cap for s in p.slots), default=0)
            planned = sum(s.yesterday_spend for s in p.slots)
            cap_util = planned / max_cap if max_cap > 0 else 0

            diagnosis.append(ProductDiagnosis(
                product_id=p.product_id,
                product_name=p.product_name,
                efficiency_level=level,
                current_d1_signal=round(d1_signal, 4),
                d1_signal_source=signal_source,
                target_d1=round(target_d1, 4),
                gap_to_target=round(gap, 4),
                action=action,
                improvement_window_days=remaining_days,
                trend=trend,
                cap_utilization=round(cap_util, 4),
            ))
        return diagnosis

    @staticmethod
    def _resolve_d1_signal(p: ProductState) -> tuple[float, str]:
        """Resolve the best D1 signal for a product."""
        from app.services.signal import resolve_product_d1_signal
        return resolve_product_d1_signal(p)

    @staticmethod
    def _compute_trend(recent: List[float]) -> str:
        """Determine trend direction from recent D1 values."""
        if len(recent) < 3:
            return "stable"
        half = len(recent) // 2
        first_half = sum(recent[:half]) / half if half > 0 else 0
        second_half = sum(recent[-half:]) / half if half > 0 else 0
        if first_half <= 0:
            return "stable"
        change = (second_half - first_half) / first_half
        if change > 0.05:
            return "rising"
        if change < -0.05:
            return "declining"
        return "stable"

    # ---- aggregate risk score ----

    def risk_score(self, alerts: List[RiskAlert], roi_gap: float) -> Dict:
        """综合风险评分：0=安全, 100=最高风险。"""
        score = 0.0
        for a in alerts:
            if a.level == "CRITICAL":
                score += 25
            elif a.level == "WARN":
                score += 10
        if roi_gap < 0:
            score += abs(roi_gap) * 100
        return {
            "score": round(min(score, 100), 1),
            "level": "CRITICAL" if score >= 50 else ("WARN" if score >= 20 else "OK"),
            "alert_count": len(alerts),
        }
