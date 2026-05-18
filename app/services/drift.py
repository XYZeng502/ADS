"""预测漂移检测：滚动窗口 MAPE 追踪 + baseline 对比。"""
import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from app.prediction_artifacts import resolve_roi_predictions_csv, resolve_spend_predictions_csv

_BASELINE_ROI = Path("outputs/baseline_roi_d1_metrics.json")
_SPEND_DIR = Path("outputs/model_parallel_spend_t1")
_ROI_DIR = Path("outputs/model_parallel_roi_d1")
_DRIFT_OUTPUT = Path("outputs/prediction_drift.json")


@dataclass
class ModelDrift:
    model: str              # "spend_t1" | "roi_d1"
    best_model: str         # "XGBoost" 等
    recent_mape: float      # 最近 N 天 overall MAPE
    baseline_mape: float    # 训练时 baseline MAPE
    drift_ratio: float      # recent_mape / baseline_mape
    status: str             # "ok" | "warning" | "critical"
    sample_days: int
    sample_rows: int
    evaluated_at: str


@dataclass
class DriftReport:
    evaluated_at: str
    models: list = field(default_factory=list)
    overall: str = "ok"


def _to_float(v: str) -> float:
    try:
        return float(v) if v not in ("", None) else 0.0
    except ValueError:
        return 0.0


def _compute_recent_mape(prediction_csv: Path, lookback_days: int) -> tuple[float, int, int]:
    """从 prediction CSV 计算最近 N 天的 overall MAPE。"""
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    app_spend: dict[str, float] = {}
    app_abs_err: dict[str, float] = {}

    with prediction_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = row.get("日期", "")
            if day < cutoff:
                continue
            y_true = _to_float(row.get("y_true", "0"))
            y_pred = _to_float(row.get("y_pred", "0"))
            app_id = row.get("应用ID", "")
            app_spend[app_id] = app_spend.get(app_id, 0.0) + y_true
            app_abs_err[app_id] = app_abs_err.get(app_id, 0.0) + abs(y_true - y_pred)

    total_spend = sum(app_spend.values())
    total_abs_err = sum(app_abs_err.values())
    if total_spend <= 0:
        return 0.0, 0, 0

    wmape = total_abs_err / total_spend * 100  # spend-weighted MAPE
    return round(wmape, 2), len(app_spend), sum(1 for v in app_spend.values() if v > 0)


def _load_baseline_mape(baseline_path: Path, model: str) -> float:
    """从 baseline metrics JSON 读取 overall 最优模型 MAPE。"""
    if not baseline_path.exists():
        # spend_t1 没有 baseline 文件，返回 -1 表示用绝对阈值
        return -1.0
    try:
        raw = json.loads(baseline_path.read_text(encoding="utf-8"))
        overall = raw.get("overall", [])
        best_mape = min((m.get("mape", 999) for m in overall), default=999)
        return round(best_mape * 100, 2)  # JSON 存的是小数（0.123 → 12.3%）
    except (json.JSONDecodeError, KeyError):
        return -1.0


def _resolve_best_model_name(prediction_csv: Optional[Path]) -> str:
    if prediction_csv is None:
        return "unknown"
    name = prediction_csv.stem
    if name.startswith("predictions_"):
        name = name[len("predictions_"):]
    return name


def check_prediction_drift(
    lookback_days: int = 7,
    warn_ratio: float = 2.0,
    critical_ratio: float = 3.0,
    spend_warn_mape: float = 80.0,
    spend_critical_mape: float = 120.0,
) -> DriftReport:
    """
    检查模型预测漂移。
    对于 roi_d1：与 baseline MAPE 比较；对于 spend_t1：用绝对阈值（无 baseline 文件）。
    """
    now = datetime.now(timezone.utc).isoformat()
    report = DriftReport(evaluated_at=now)

    # --- Spend T+1 ---
    spend_csv = resolve_spend_predictions_csv(_SPEND_DIR if _SPEND_DIR.is_dir() else None)
    if spend_csv and spend_csv.exists():
        recent_mape, n_days, n_rows = _compute_recent_mape(spend_csv, lookback_days)
        baseline_mape = _load_baseline_mape(_BASELINE_ROI, "spend_t1")
        if baseline_mape > 0:
            drift_ratio = recent_mape / baseline_mape if baseline_mape > 0 else 0
            status = "ok" if drift_ratio < warn_ratio else ("warning" if drift_ratio < critical_ratio else "critical")
        else:
            # 无 baseline，用绝对阈值
            drift_ratio = 0.0
            status = "ok" if recent_mape <= spend_warn_mape else ("warning" if recent_mape <= spend_critical_mape else "critical")
        report.models.append(ModelDrift(
            model="spend_t1",
            best_model=_resolve_best_model_name(spend_csv),
            recent_mape=recent_mape,
            baseline_mape=baseline_mape,
            drift_ratio=round(drift_ratio, 2),
            status=status,
            sample_days=n_days,
            sample_rows=n_rows,
            evaluated_at=now,
        ))

    # --- ROI D1 ---
    roi_csv = resolve_roi_predictions_csv(_ROI_DIR if _ROI_DIR.is_dir() else None)
    if roi_csv and roi_csv.exists():
        recent_mape, n_days, n_rows = _compute_recent_mape(roi_csv, lookback_days)
        baseline_mape = _load_baseline_mape(_BASELINE_ROI, "roi_d1")
        if baseline_mape > 0:
            drift_ratio = recent_mape / baseline_mape if baseline_mape > 0 else 0
            status = "ok" if drift_ratio < warn_ratio else ("warning" if drift_ratio < critical_ratio else "critical")
        else:
            drift_ratio = 0.0
            status = "ok" if recent_mape <= 30 else ("warning" if recent_mape <= 60 else "critical")
        report.models.append(ModelDrift(
            model="roi_d1",
            best_model=_resolve_best_model_name(roi_csv),
            recent_mape=recent_mape,
            baseline_mape=baseline_mape,
            drift_ratio=round(drift_ratio, 2),
            status=status,
            sample_days=n_days,
            sample_rows=n_rows,
            evaluated_at=now,
        ))

    # overall 取最差
    statuses = [m.status for m in report.models]
    if "critical" in statuses:
        report.overall = "critical"
    elif "warning" in statuses:
        report.overall = "warning"
    elif all(s == "ok" for s in statuses) and statuses:
        report.overall = "ok"
    else:
        report.overall = "ok"

    # 持久化
    _DRIFT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _DRIFT_OUTPUT.write_text(json.dumps({
        "evaluated_at": report.evaluated_at,
        "overall": report.overall,
        "models": [
            {
                "model": m.model,
                "best_model": m.best_model,
                "recent_mape": m.recent_mape,
                "baseline_mape": m.baseline_mape,
                "drift_ratio": m.drift_ratio,
                "status": m.status,
                "sample_days": m.sample_days,
                "sample_rows": m.sample_rows,
            }
            for m in report.models
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return report
