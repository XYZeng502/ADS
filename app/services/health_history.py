"""健康快照历史：JSONL 追加，保留 90 天。"""
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PATH = Path("outputs/health_history.jsonl")
_MAX_DAYS = 90


@dataclass
class HealthSnapshot:
    timestamp: str = ""
    data_status: str = "unknown"
    days_behind: int = -1
    row_count: int = 0
    app_count: int = 0
    drift_overall: str = "unknown"
    drift_spend_mape: float = 0.0
    drift_roi_mape: float = 0.0
    training_overall: str = "never"
    calendar_status: str = "unknown"
    model_files_ok: int = 0
    model_files_total: int = 0


class HealthHistory:
    @classmethod
    def snapshot(cls) -> HealthSnapshot:
        """采集当前系统健康状态并追加到 JSONL 文件。"""
        from app.services.data_pipeline import check_data_health
        from app.services.drift import check_prediction_drift
        from app.services.calendar_health import check_calendar_health
        from app.services.retrain_status import load_retrain_status

        now = datetime.now(timezone.utc).isoformat()

        # 数据健康
        dh = check_data_health(Path("daily_merged.csv"))
        # 漂移
        drift = check_prediction_drift()
        spend_d = next((m for m in drift.models if m.model == "spend_t1"), None)
        roi_d = next((m for m in drift.models if m.model == "roi_d1"), None)
        # 日历
        cal = check_calendar_health()
        # 重训状态
        ts = load_retrain_status()
        # 模型文件计数
        _model_paths = [
            Path("outputs/per_app_release_curves.json"),
            Path("outputs/daily_revenue_predictions.csv"),
            Path("outputs/model_parallel_spend_t1/predictions_XGBoost.csv"),
            Path("outputs/model_parallel_roi_d1/predictions_XGBoost.csv"),
            Path("outputs/app_level_last_day_suggestions.csv"),
        ]
        model_ok = sum(1 for p in _model_paths if p.exists())

        snap = HealthSnapshot(
            timestamp=now,
            data_status=dh.status,
            days_behind=dh.days_behind or -1,
            row_count=dh.row_count,
            app_count=dh.app_count,
            drift_overall=drift.overall,
            drift_spend_mape=spend_d.recent_mape if spend_d else 0.0,
            drift_roi_mape=roi_d.recent_mape if roi_d else 0.0,
            training_overall=ts.overall,
            calendar_status=cal.status,
            model_files_ok=model_ok,
            model_files_total=len(_model_paths),
        )

        _PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(snap), ensure_ascii=False) + "\n")

        cls._prune()
        return snap

    @classmethod
    def read(cls, days: int = 30) -> list[dict]:
        """读取最近 N 天的健康快照。"""
        if not _PATH.exists():
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        snapshots = []
        with _PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snap = json.loads(line)
                    ts = snap.get("timestamp", "")
                    if ts >= cutoff.isoformat():
                        snapshots.append(snap)
                except json.JSONDecodeError:
                    continue
        return snapshots

    @classmethod
    def _prune(cls) -> None:
        """清理超过 MAX_DAYS 的旧记录。"""
        if not _PATH.exists():
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=_MAX_DAYS)
        lines = []
        with _PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snap = json.loads(line)
                    if snap.get("timestamp", "") >= cutoff.isoformat():
                        lines.append(line)
                except json.JSONDecodeError:
                    continue
        _PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
