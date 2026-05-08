import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.schemas import ClientContext, DailyPlanResponse, MPCSnapshotResponse


class MPCSnapshotService:
    """
    记录每日 MPC 决策快照，供次日用真实 spend / roi_d1 / revenue 回填校准。

    设计原则：
    - 显式调用才落盘，避免普通预测接口污染训练/校准样本。
    - 快照保存“当时看到的信息”和“当时给出的建议”，方便次日做预测误差归因。
    """

    def build_snapshot(self, context: ClientContext, plan: DailyPlanResponse, snapshot_id: str) -> dict[str, Any]:
        calibration_date = context.date + timedelta(days=1)
        return {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "client": {
                "client_id": context.client_id,
                "client_name": context.client_name,
            },
            "decision_date": context.date.isoformat(),
            "calibration_date": calibration_date.isoformat(),
            "kpi": {
                "kpi_roi": context.kpi_roi,
                "month_spend_so_far": context.month_spend_so_far,
                "month_revenue_so_far": context.month_revenue_so_far,
                "yesterday_total_spend": context.yesterday_total_spend,
            },
            "forecast": {
                "mode": plan.mode,
                "month_end_spend_prediction": plan.month_end_spend_prediction,
                "month_end_roi_prediction": plan.month_end_roi_prediction,
                "month_total_scale_forecast": plan.month_total_scale_forecast,
                "today_total_budget_target": plan.today_total_budget_target,
                "today_d1_revenue_target": plan.today_d1_revenue_target,
                "today_history_revenue_estimate": plan.today_history_revenue_estimate,
                "d1_anchor_raw_mean": plan.d1_anchor_raw_mean,
                "d1_anchor_calibrated_mean": plan.d1_anchor_calibrated_mean,
            },
            "suggestions": [x.model_dump(mode="json") for x in plan.suggestions],
            "diagnosis": [x.model_dump(mode="json") for x in plan.diagnosis],
            "alerts": [x.model_dump(mode="json") for x in plan.alerts],
            "rule_hits": [x.model_dump(mode="json") for x in plan.rule_hits],
            "business_trace": plan.business_trace.model_dump(mode="json") if plan.business_trace else None,
            "calibration_targets": {
                "description": "次日数据回流后回填这些真实值，用于校准 spend / roi_d1 / 月末ROI偏差。",
                "expected_actual_date": calibration_date.isoformat(),
                "fields": [
                    "actual_total_spend",
                    "actual_d1_revenue",
                    "actual_roi_d1",
                    "actual_by_app_or_product",
                    "actual_by_flow_if_available",
                ],
            },
            "raw_plan": plan.model_dump(mode="json"),
        }

    def save(
        self,
        *,
        context: ClientContext,
        plan: DailyPlanResponse,
        output_dir: str | Path = "outputs/mpc_snapshots",
    ) -> MPCSnapshotResponse:
        snapshot_id = f"{context.date.isoformat()}_{context.client_id}_{uuid.uuid4().hex[:8]}"
        root = Path(output_dir)
        client_dir = root / context.client_id
        client_dir.mkdir(parents=True, exist_ok=True)

        snapshot = self.build_snapshot(context=context, plan=plan, snapshot_id=snapshot_id)
        snapshot_path = client_dir / f"{snapshot_id}.json"
        latest_path = client_dir / "latest.json"
        text = json.dumps(snapshot, ensure_ascii=False, indent=2)
        snapshot_path.write_text(text, encoding="utf-8")
        latest_path.write_text(text, encoding="utf-8")

        return MPCSnapshotResponse(
            snapshot_id=snapshot_id,
            snapshot_path=str(snapshot_path),
            latest_path=str(latest_path),
            client_id=context.client_id,
            decision_date=context.date,
            calibration_date=context.date + timedelta(days=1),
            plan=plan,
        )
