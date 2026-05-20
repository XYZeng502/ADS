#!/usr/bin/env python
"""每日模型重训练编排器。cron 调用: PYTHONPATH=. python scripts/daily_retrain.py"""
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.data_pipeline import check_data_health
from app.services.retrain_status import (
    RetrainStatus,
    StepStatus,
    load_retrain_status,
    save_retrain_status,
    mark_running,
)

STEPS = [
    {
        "id": "spend_t1",
        "label": "T+1 Spend 重训",
        "cmd": [
            sys.executable, "scripts/run_parallel_models_spend_t1.py",
            "--input", "daily_merged.csv",
            "--output-dir", "outputs/model_parallel_spend_t1",
            "--min-train-days", "20",
            "--use-gpu",
            "--tree-models", "XGBoost",
            "--use-log-target",
            "--enable-cqr",
        ],
    },
    {
        "id": "roi_d1",
        "label": "T+1 ROI_D1 重训",
        "cmd": [
            sys.executable, "scripts/run_parallel_models_roi_d1.py",
            "--input", "daily_merged.csv",
            "--output-dir", "outputs/model_parallel_roi_d1",
            "--min-train-days", "20",
            "--use-gpu",
        ],
    },
    {
        "id": "app_curves",
        "label": "Per-app 释放曲线",
        "cmd": [
            sys.executable, "scripts/offline_backtest.py",
            "--task", "app_last_day",
            "--input", "daily_merged.csv",
        ],
    },
    {
        "id": "daily_revenue",
        "label": "每日收入预测回测",
        "cmd": [sys.executable, "scripts/predict_daily_revenue.py"],
    },
]

_TIMEOUT_PER_STEP = 7200  # 单步最长 2 小时


def _extract_details(step_id: str, stdout: str) -> dict:
    """从脚本输出中提取关键指标"""
    details: dict = {}
    for line in stdout.splitlines():
        if step_id in ("spend_t1", "roi_d1"):
            if "overall" in line.lower() and "mape" in line.lower():
                # 提取 MAPE=xx% 行
                parts = line.split()
                for i, p in enumerate(parts):
                    if "mape" in p.lower() and i + 1 < len(parts):
                        try:
                            details["mape_overall"] = float(parts[i + 1].replace("%", ""))
                        except ValueError:
                            pass
        elif step_id == "daily_revenue":
            if "overall" in line.lower() and "mape" in line.lower():
                parts = line.split()
                try:
                    details["mape_overall"] = float(parts[-1].replace("%", ""))
                except (ValueError, IndexError):
                    pass
    return details


def run_step(step: dict) -> StepStatus:
    started = datetime.now(timezone.utc)
    st = StepStatus(
        step=step["id"],
        status="running",
        started_at=started.isoformat(),
    )

    print(f"\n{'='*60}")
    print(f"[{started.strftime('%H:%M:%S')}] 开始: {step['label']} ({step['id']})")
    print(f"{'='*60}")

    try:
        t0 = time.time()
        result = subprocess.run(
            step["cmd"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_PER_STEP,
            cwd=Path(__file__).resolve().parent.parent,
        )
        elapsed = time.time() - t0

        stdout_tail = "\n".join(result.stdout.splitlines()[-30:]) if result.stdout else ""

        if result.returncode == 0:
            st.status = "ok"
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        else:
            st.status = "failed"
            st.error = f"exit={result.returncode}"
            print(f"[ERROR] exit={result.returncode}")
            print(stdout_tail)
            if result.stderr:
                print(result.stderr[-1000:])

        st.details = _extract_details(step["id"], result.stdout)
        st.duration_s = round(elapsed, 1)
        print(f"[{st.status.upper()}] {step['label']} 耗时 {st.duration_s:.0f}s")

    except subprocess.TimeoutExpired:
        st.status = "failed"
        st.error = f"timeout ({_TIMEOUT_PER_STEP}s)"
        st.duration_s = _TIMEOUT_PER_STEP
        print(f"[TIMEOUT] {step['label']} 超时 {_TIMEOUT_PER_STEP}s")

    st.finished_at = datetime.now(timezone.utc).isoformat()
    return st


def main():
    status = mark_running()
    print(f"每日重训练编排器启动: {status.last_run_started}")

    # 1. 数据健康检查
    print("\n[1/5] 数据健康检查...")
    health = check_data_health(Path("daily_merged.csv"))
    hs = status.steps[0]
    hs.status = "ok" if health.ok else "failed"
    hs.details = {"status": health.status, "days_behind": health.days_behind,
                  "row_count": health.row_count, "app_count": health.app_count}
    if not health.ok:
        hs.error = f"数据不健康: {health.missing_cols}"
        print(f"[ABORT] 数据不健康: {health.missing_cols}")
        status.overall = "failed"
        status.last_run_finished = hs.finished_at
        save_retrain_status(status)
        print("重训中止。")
        return
    print(f"  数据健康: status={health.status}, rows={health.row_count}, apps={health.app_count}, behind={health.days_behind}d")

    # 2. 逐个执行模型训练步骤
    all_ok = True
    any_failure = False
    for i, step_def in enumerate(STEPS):
        st = run_step(step_def)
        status.steps[i + 1] = st  # steps[0] 是 health_check
        save_retrain_status(status)

        if st.status == "failed":
            any_failure = True
            all_ok = False
        elif st.status != "ok":
            all_ok = False

    status.last_run_finished = datetime.now(timezone.utc).isoformat()
    status.overall = "success" if all_ok else ("failed" if not any_failure else "partial")
    save_retrain_status(status)

    # 3. 告警持久化 + 健康快照
    print("\n[附] 采集告警与健康快照...")
    try:
        from app.services.alert_log import AlertLog
        from app.services.risk import RiskService

        risk_svc = RiskService()
        drift_alerts = risk_svc.evaluate_drift()
        AlertLog.append(drift_alerts)
        print(f"  告警已持久化: {len(drift_alerts)} 条")
    except Exception as e:
        print(f"  [WARN] 告警采集失败: {e}")

    try:
        from app.services.health_history import HealthHistory
        snap = HealthHistory.snapshot()
        print(f"  健康快照已保存: drift={snap.drift_overall} spend_mape={snap.drift_spend_mape:.2%} roi_mape={snap.drift_roi_mape:.2%} data={snap.data_status}")
    except Exception as e:
        print(f"  [WARN] 健康快照失败: {e}")

    print(f"\n{'='*60}")
    print(f"重训完成: overall={status.overall}")
    for s in status.steps:
        icon = "OK" if s.status == "ok" else ("FAIL" if s.status == "failed" else s.status.upper())
        print(f"  [{icon}] {s.step} ({s.duration_s:.0f}s)" + (f" err={s.error}" if s.error else ""))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
