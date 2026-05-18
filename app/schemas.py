from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SlotType = Literal["store", "union", "smart"]
DayType = Literal["workday", "weekend", "holiday", "summer_winter"]
BusinessLayerStatus = Literal["OK", "WARN", "CRITICAL"]


class ProductSlotMetrics(BaseModel):
    slot: SlotType
    predicted_d1_roi: float = Field(..., description="当日该版位预测D1 ROI，0-1")
    m30_multiplier: float = Field(..., description="D30 / D1 倍率")
    yesterday_spend: float = 0.0
    cap: float = Field(..., description="该产品该版位单日预算上限")
    flow_name: Optional[str] = Field(
        default=None,
        description="推广流量名称；填入后与离线脚本「应用ID + 推广流量名称」粒度对齐（仅加权展示/审计，不改变槽位加权数学结构）",
    )


class ProductState(BaseModel):
    product_id: str
    product_name: str
    app_id: Optional[str] = Field(
        default=None,
        description="应用ID；可与 flow_name / 离线训练实体键对齐审计",
    )
    priority: int = 0
    slots: List[ProductSlotMetrics]
    recent_d1_actual: List[float] = Field(default_factory=list)
    recent_d1_predicted: List[float] = Field(default_factory=list)


class ClientContext(BaseModel):
    client_id: str
    client_name: str
    date: date
    month_spend_so_far: float
    month_revenue_so_far: float
    kpi_roi: float
    yesterday_total_spend: float
    products: List[ProductState]


class DailyPlanItem(BaseModel):
    product_id: str
    product_name: str
    slot: SlotType
    suggested_budget: float
    expected_d1_revenue: float
    expected_roi: float
    decision_reason: str


class ProductDiagnosis(BaseModel):
    product_id: str
    product_name: str
    efficiency_level: Literal["A", "B", "C"]
    current_d1_signal: float
    d1_signal_source: Literal["BLENDED", "ACTUAL", "PREDICTED_SLOT_WEIGHTED", "PREDICTED_SLOT_FALLBACK"]
    target_d1: float
    gap_to_target: float
    action: Literal["加量", "维持", "减量或观察"]
    improvement_window_days: int
    trend: str = "stable"
    cap_utilization: float = 0.0


class RiskAlert(BaseModel):
    level: Literal["INFO", "WARN", "CRITICAL"]
    category: Literal["PRODUCT_D1_DROP", "TRAFFIC_ANOMALY", "ROI_GUARD", "SPEND_DROP", "SPEND_SPIKE", "ROI_DECLINE_TREND", "CAP_PROXIMITY", "PREDICTION_DRIFT"]
    message: str
    product_id: Optional[str] = None


class SlotBudgetSummary(BaseModel):
    slot: SlotType
    suggested_budget: float


class KPIAdjustmentImpact(BaseModel):
    old_kpi: float
    projected_month_end_roi: float
    roi_gap_vs_kpi: float
    suggested_action: Literal["释放预算", "维持", "缩减预算"]
    suggested_budget_delta: float = Field(
        ..., description="建议释放/缩减金额（正数表示释放，负数表示缩减）"
    )
    message: str


class ROIForecastAttribution(BaseModel):
    spend_factor_score: float = Field(..., description="消耗轨迹不确定性评分，0-1")
    d1_anchor_score: float = Field(..., description="D1锚点不确定性评分，0-1")
    curve_factor_score: float = Field(..., description="曲线模板不确定性评分，0-1")
    dominant_factor: Literal["SPEND", "D1_ANCHOR", "CURVE", "BALANCED"]
    details: str


class RuleHit(BaseModel):
    rule_id: str
    priority: int
    action: str
    reason: str
    metric_snapshot: Dict[str, float] = Field(default_factory=dict)


class BusinessLayerSummary(BaseModel):
    layer_id: Literal["kpi", "forecast", "solver", "risk", "decision"]
    title: str
    status: BusinessLayerStatus
    primary_metrics: Dict[str, float | str] = Field(default_factory=dict)
    key_findings: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class BusinessDecisionTrace(BaseModel):
    kpi_layer: BusinessLayerSummary
    forecast_layer: BusinessLayerSummary
    solver_layer: BusinessLayerSummary
    risk_layer: BusinessLayerSummary
    decision_layer: BusinessLayerSummary


class DailyPlanResponse(BaseModel):
    client_id: str
    date: date
    mode: Literal["GUARD", "SCALE"]
    # 5.1 每日输出 - 月度累计消耗预测
    month_cumulative_spend_forecast: float
    # 5.1 每日输出 - 月度预测总规模（与累计消耗预测口径一致，作为显式展示字段）
    month_total_scale_forecast: float
    # 原有字段兼容
    month_end_spend_prediction: float
    # 5.1 每日输出 - 月度预测ROI
    month_roi_forecast: float
    # 原有字段兼容
    month_end_roi_prediction: float
    # 5.1 每日输出 - 今日D1回收目标（分产品分版位求和）
    today_d1_revenue_target: float
    # 5.1 每日输出 - 今日建议总预算
    today_total_budget_target: float
    # 5.1 每日输出 - 今日分版位预算建议汇总
    today_slot_budget_summary: List[SlotBudgetSummary]
    # 5.1 每日输出 - KPI调整响应（按当前KPI下的偏差与动作建议）
    kpi_adjustment_impact: KPIAdjustmentImpact
    # 月末ROI偏差来源分解（用于定位主因）
    roi_forecast_attribution: ROIForecastAttribution
    # 5.1 每日输出 - 今日历史回收预估
    today_history_revenue_estimate: float
    # 5.1 每日输出 - 今日分产品分版位建议预算
    suggestions: List[DailyPlanItem]
    # 5.1 每日输出 - 产品分级诊断
    diagnosis: List[ProductDiagnosis]
    # 5.1 每日输出 - 风险预警
    alerts: List[RiskAlert]
    # 规则引擎命中详情（用于运营解释与审计）
    rule_hits: List[RuleHit] = Field(default_factory=list)
    d1_anchor_raw_mean: float = 0.0
    d1_anchor_calibrated_mean: float = 0.0
    business_trace: Optional[BusinessDecisionTrace] = None
    notes: List[str]


class KPIRecomputeRequest(BaseModel):
    context: ClientContext
    new_kpi: float


class KPIRecomputeResponse(BaseModel):
    old_kpi: float
    new_kpi: float
    delta: float
    plan: DailyPlanResponse


class MPCSnapshotRequest(BaseModel):
    context: ClientContext
    output_dir: str = "outputs/mpc_snapshots"


class MPCSnapshotResponse(BaseModel):
    snapshot_id: str
    snapshot_path: str
    latest_path: str
    client_id: str
    decision_date: date
    calibration_date: date
    plan: DailyPlanResponse


class AlertEvaluationRequest(BaseModel):
    context: ClientContext
    today_real_revenue: float
    today_predicted_revenue: float


class DayScaleFactor(BaseModel):
    day_type: DayType
    factor: float


class ForecastAssumptions(BaseModel):
    remaining_days: int
    day_type_factors: Dict[DayType, float]
