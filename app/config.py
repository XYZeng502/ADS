from typing import Literal

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Smart Budget Decision System"
    version: str = "0.1.0"
    default_kpi: float = 1.05
    release_surplus_threshold: float = 0.05
    risk_d1_drop_days: int = 3
    risk_d1_drop_ratio: float = 0.9
    risk_revenue_deviation_threshold: float = 0.2
    risk_month_roi_guard_buffer: float = 0.03
    kpi_release_gap_threshold: float = 0.03
    kpi_guard_gap_threshold: float = 0.02
    kpi_release_budget_ratio: float = 0.15
    kpi_guard_budget_ratio: float = 0.12
    diagnosis_a_gap_threshold: float = 0.02
    diagnosis_b_gap_threshold: float = -0.01
    d1_signal_min_points: int = 3
    d1_signal_actual_weight: float = 0.7
    d1_signal_pred_weight: float = 0.3
    d1_signal_calibration_enabled: bool = True
    d1_signal_shrink_to_actual_mean: float = 0.35
    d1_signal_clip_lower_quantile: float = 0.1
    d1_signal_clip_upper_quantile: float = 0.9
    d1_signal_clip_expand_ratio: float = 0.15
    d1_signal_global_min: float = 0.05
    d1_signal_global_max: float = 3.0
    lp_today_floor_ratio_guard: float = 0.92
    lp_today_floor_ratio_scale: float = 1.0
    lp_today_shortfall_penalty: float = 2.0
    lp_time_decay_factor: float = 0.997
    lp_scale_release_extra_cap: float = 0.25
    lp_near_term_days: int = 3
    lp_near_term_floor_ratio: float = 0.95
    lp_near_term_shortfall_penalty: float = 0.8
    lp_smooth_up_factor: float = 1.35
    lp_smooth_down_factor: float = 0.55
    lp_smooth_abs_tolerance: float = 300.0
    lp_roi_surplus_target_gap: float = 0.015
    lp_roi_surplus_penalty: float = 0.0
    lp_month_release_floor_ratio: float = 0.0
    lp_month_release_shortfall_penalty: float = 0.0
    rule_month_end_days: int = 5
    rule_scale_release_ratio: float = 0.12
    rule_guard_reduce_ratio: float = 0.1
    rule_c_level_ratio_threshold: float = 0.4
    # 与离线脚本稳定口径对齐：求解器对产品内各版位/槽位的「参考 D1」用消耗加权，
    # 与 `_weighted_slot_pred_d1`、月末组合锚点逻辑一致。「arithmetic_mean」保留旧算术平均，便于逐项对比误差。
    solver_slot_d1_normalize_mode: Literal["spend_weighted", "arithmetic_mean"] = "spend_weighted"
    # 应用级回放/推荐表主用的「月末 ROI」cohort 口径（与 offline_backtest 三条分支对齐，只改此处即可）。
    app_last_day_canonical_month_end_roi: Literal["stable", "pred_only", "fused"] = "fused"


settings = Settings()
