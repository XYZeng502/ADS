"""训练状态持久化，记录最近一次重训每步状态。"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_STATUS_PATH = Path("outputs/retrain_status.json")


@dataclass
class StepStatus:
    step: str  # "health_check" | "spend_t1" | "roi_d1" | "app_curves" | "daily_revenue"
    status: str = "pending"  # "ok" | "failed" | "running" | "pending"
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    error: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class RetrainStatus:
    last_run_started: Optional[str] = None
    last_run_finished: Optional[str] = None
    overall: str = "never"  # "success" | "partial" | "failed" | "running" | "never"
    steps: list = field(default_factory=list)


def get_status_path() -> Path:
    return _STATUS_PATH


def _default_status() -> RetrainStatus:
    return RetrainStatus(
        steps=[
            StepStatus(step=s)
            for s in ["health_check", "spend_t1", "roi_d1", "app_curves", "daily_revenue"]
        ]
    )


def load_retrain_status() -> RetrainStatus:
    path = _STATUS_PATH
    if not path.exists():
        return _default_status()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        steps = [StepStatus(**s) for s in raw.get("steps", [])]
        return RetrainStatus(
            last_run_started=raw.get("last_run_started"),
            last_run_finished=raw.get("last_run_finished"),
            overall=raw.get("overall", "never"),
            steps=steps,
        )
    except (json.JSONDecodeError, TypeError):
        return _default_status()


def save_retrain_status(status: RetrainStatus) -> None:
    path = _STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "last_run_started": status.last_run_started,
        "last_run_finished": status.last_run_finished,
        "overall": status.overall,
        "steps": [
            {
                "step": s.step,
                "status": s.status,
                "started_at": s.started_at,
                "finished_at": s.finished_at,
                "duration_s": s.duration_s,
                "error": s.error,
                "details": s.details,
            }
            for s in status.steps
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_running() -> RetrainStatus:
    status = _default_status()
    status.last_run_started = datetime.now(timezone.utc).isoformat()
    status.overall = "running"
    for s in status.steps:
        s.status = "running" if s.step == "health_check" else "pending"
    save_retrain_status(status)
    return status
