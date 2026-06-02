import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 1800  # 30 分钟，数据变化频率低，避免频繁读 548MB CSV


def _cached(key: str, factory):
    now = time.time()
    if key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < _CACHE_TTL:
            return val
    val = factory()
    _CACHE[key] = (now, val)
    return val

from app.config import settings
from app.schemas import (
    AlertEvaluationRequest,
    ClientContext,
    KPIRecomputeRequest,
    KPIRecomputeResponse,
    MPCSnapshotRequest,
    MPCSnapshotResponse,
)
from app.services.data_pipeline import check_data_health
from app.services.retrain_status import load_retrain_status
from app.services.drift import check_prediction_drift
from app.services.calendar_health import check_calendar_health
from app.services.alert_log import AlertLog
from app.services.health_history import HealthHistory
from app.services.mpc_snapshot import MPCSnapshotService
from app.services.orchestrator import DailyOrchestrator
from app.services.risk import RiskService
from app.web import router as web_router

app = FastAPI(title=settings.app_name, version=settings.version)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
orchestrator = DailyOrchestrator()
risk_service = RiskService()
mpc_snapshot_service = MPCSnapshotService()
app.include_router(web_router)


def _warm_multi_horizon():
    from app.web import _warm_multi_horizon as _wmh
    _wmh()

def _warm_watchlist():
    """预热盯盘：预计算并写入排序文件"""
    import logging
    logger = logging.getLogger("startup")
    try:
        from app.web import _build_watchlist_snapshot, _load_wl_snapshot
        snap = _build_watchlist_snapshot()
        if snap:
            logger.info(f"盯盘预热完成: {snap['total']} apps -> {snap.get('_WL_SNAPSHOT_PATH', '')}")
        else:
            _load_wl_snapshot()  # 尝试加载已有文件
            logger.info("盯盘预热: 使用已有快照")
    except Exception as e:
        logger.warning(f"盯盘预热失败: {e}")


@app.on_event("startup")
def _warm_cache():
    """启动时同步预热缓存，确保首个请求无需等待。"""
    import logging
    logger = logging.getLogger("startup")
    try:
        logger.info("预热缓存中...")
        _data_summary()
        _drift_summary()
        _calendar_summary()
        _warm_watchlist()
        # 多步模型改为懒加载，不再启动时预热
        logger.info("缓存预热完成")
    except Exception:
        logger.warning("缓存预热部分失败", exc_info=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "data": _data_summary(),
        "models": _models_summary(),
        "training": _training_summary(),
        "drift": _drift_summary(),
        "calendar_health": _calendar_summary(),
    }


def _data_summary() -> dict:
    def _compute():
        data = check_data_health(Path("daily_merged.csv"))
        return {
            "csv_exists": data.csv_exists,
            "last_date": data.last_date,
            "days_behind": data.days_behind,
            "row_count": data.row_count,
            "app_count": data.app_count,
            "columns_ok": data.columns_ok,
            "missing_cols": data.missing_cols,
            "critical_null_rate": data.critical_null_rate,
            "spend_drift_pct": data.spend_drift_pct,
            "status": data.status,
        }
    return _cached("data", _compute)


def _models_summary() -> dict:
    return {
        "per_app_curves": Path("outputs/per_app_release_curves.json").exists(),
        "daily_revenue_predictions": Path("outputs/daily_revenue_predictions.csv").exists(),
        "spend_t1_predictions": Path("outputs/model_parallel_spend_t1/predictions_XGBoost.csv").exists(),
        "roi_d1_predictions": Path("outputs/model_parallel_roi_d1/predictions_XGBoost.csv").exists(),
        "app_suggestions": Path("outputs/app_level_last_day_suggestions.csv").exists(),
    }


def _training_summary() -> dict:
    ts = load_retrain_status()
    return {
        "last_run": ts.last_run_finished,
        "overall": ts.overall,
        "steps": [
            {"step": s.step, "status": s.status, "duration_s": s.duration_s}
            for s in ts.steps
        ],
    }


def _drift_summary() -> dict:
    def _compute():
        try:
            r = check_prediction_drift()
            return {
                "evaluated_at": r.evaluated_at,
                "overall": r.overall,
                "models": [
                    {
                        "model": m.model,
                        "best_model": m.best_model,
                        "recent_mape": m.recent_mape,
                        "baseline_mape": m.baseline_mape,
                        "drift_ratio": m.drift_ratio,
                        "status": m.status,
                        "sample_days": m.sample_days,
                    }
                    for m in r.models
                ],
            }
        except Exception:
            return {"overall": "error", "models": []}
    return _cached("drift", _compute)


def _calendar_summary() -> dict:
    def _compute():
        try:
            ch = check_calendar_health()
            return {
                "file_exists": ch.file_exists,
                "version": ch.version,
                "holiday_count": ch.holiday_count,
                "first_date": ch.first_date,
                "last_date": ch.last_date,
                "covers_data_dates": ch.covers_data_dates,
                "covers_future_30d": ch.covers_future_30d,
                "status": ch.status,
                "missing_dates": ch.missing_dates,
            }
        except Exception:
            return {"status": "error", "file_exists": False}
    return _cached("calendar", _compute)


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


@app.get("/v1/monitor/alerts")
def monitor_alerts(days: int = 7):
    """告警历史：汇总统计 + 最近告警记录"""
    s = AlertLog.summary(days)
    s["recent"] = AlertLog.query(days)[:50]
    return s


@app.get("/v1/monitor/health-trend")
def monitor_health_trend(days: int = 30):
    """健康趋势：最近 N 天健康快照历史"""
    return {"snapshots": HealthHistory.read(days)}
