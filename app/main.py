from fastapi import FastAPI

from app.config import settings
from app.schemas import (
    AlertEvaluationRequest,
    ClientContext,
    KPIRecomputeRequest,
    KPIRecomputeResponse,
    MPCSnapshotRequest,
    MPCSnapshotResponse,
)
from app.services.mpc_snapshot import MPCSnapshotService
from app.services.orchestrator import DailyOrchestrator
from app.services.risk import RiskService
from app.web import router as web_router

app = FastAPI(title=settings.app_name, version=settings.version)
orchestrator = DailyOrchestrator()
risk_service = RiskService()
mpc_snapshot_service = MPCSnapshotService()
app.include_router(web_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": settings.version}


@app.post("/v1/optimize/daily-plan")
def optimize_daily_plan(context: ClientContext):
    return orchestrator.run(context)


@app.post("/v1/report/daily-execution")
def daily_execution_report(context: ClientContext):
    """
    对齐需求文档第5章：运营日常执行报告（手动调预算场景）。
    当前与daily-plan共享同一计算内核，后续可扩展更多报表字段。
    """
    return orchestrator.run(context)


@app.post("/v1/business/decision-trace")
def business_decision_trace(context: ClientContext):
    """
    业务层级链路：目标/KPI -> 预测 -> 求解 -> 风险 -> 决策解释。
    复用每日计划计算结果，只返回前端/运营最需要的结构化业务解释。
    """
    return orchestrator.run(context).business_trace


@app.post("/v1/mpc/snapshot", response_model=MPCSnapshotResponse)
def record_mpc_snapshot(req: MPCSnapshotRequest):
    """
    显式记录当天 MPC 决策快照，供次日数据回流后做 spend/roi_d1/ROI 偏差校准。
    """
    plan = orchestrator.run(req.context)
    return mpc_snapshot_service.save(context=req.context, plan=plan, output_dir=req.output_dir)


@app.post("/v1/kpi/recompute", response_model=KPIRecomputeResponse)
def recompute_for_kpi(req: KPIRecomputeRequest):
    new_context = req.context.model_copy(update={"kpi_roi": req.new_kpi})
    plan = orchestrator.run(new_context)
    return KPIRecomputeResponse(
        old_kpi=req.context.kpi_roi,
        new_kpi=req.new_kpi,
        delta=req.new_kpi - req.context.kpi_roi,
        plan=plan,
    )


@app.post("/v1/alerts/evaluate")
def evaluate_alerts(req: AlertEvaluationRequest):
    product_alerts = risk_service.evaluate_product_risks(req.context)
    traffic_alerts = risk_service.evaluate_traffic_anomaly(req.today_real_revenue, req.today_predicted_revenue)
    return {"alerts": product_alerts + traffic_alerts}


@app.post("/v1/diagnosis/product-grading")
def product_grading(context: ClientContext):
    return {"diagnosis": risk_service.build_diagnosis(context)}
