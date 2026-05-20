import csv
import html
import json
import threading
from collections import deque
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode

from fastapi import APIRouter, Query
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.core.calendar import CalendarService
from app.prediction_artifacts import resolve_roi_predictions_csv, resolve_spend_predictions_csv
from scripts.offline_backtest import run_app_level_last_day_prediction, run_backtest

router = APIRouter(tags=["web"])
_BACKTEST_JOBS: Dict[str, Dict[str, Any]] = {}
_BACKTEST_JOBS_LOCK = threading.Lock()


def _repo_root() -> Path:
    """项目根目录（含 outputs），避免 uvicorn 从其它 cwd 启动时 Path('outputs') 读不到文件。"""
    return Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_job(job_id: str) -> Dict[str, Any]:
    with _BACKTEST_JOBS_LOCK:
        return dict(_BACKTEST_JOBS.get(job_id, {}))


def _update_job(job_id: str, **kwargs: Any) -> Dict[str, Any]:
    with _BACKTEST_JOBS_LOCK:
        if job_id not in _BACKTEST_JOBS:
            return {}
        _BACKTEST_JOBS[job_id].update(kwargs)
        return dict(_BACKTEST_JOBS[job_id])


def _start_backtest_job(*, input_csv: str, output_dir: str, kpi: float, task: str) -> str:
    job_id = uuid.uuid4().hex
    created_at = _utc_now_iso()
    with _BACKTEST_JOBS_LOCK:
        _BACKTEST_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "任务已入队，等待执行",
            "task": task,
            "input_csv": input_csv,
            "output_dir": output_dir,
            "kpi": kpi,
            "created_at": created_at,
            "started_at": "",
            "finished_at": "",
            "elapsed_sec": 0.0,
            "error": "",
            "progress_pct": 5,
        }

    def _worker() -> None:
        started = time.time()
        _update_job(
            job_id,
            status="running",
            message="回放执行中，请稍候...",
            started_at=_utc_now_iso(),
            progress_pct=35,
        )
        try:
            csv_path = Path(input_csv)
            out_dir = Path(output_dir)
            if task == "app_last_day":
                run_app_level_last_day_prediction(csv_path=csv_path, output_dir=out_dir, kpi=kpi)
                done_msg = "应用层最后一天预测执行完成，结果已刷新。"
            else:
                run_backtest(csv_path=csv_path, output_dir=out_dir, kpi=kpi)
                done_msg = "全量回放执行完成，结果已刷新。"
            elapsed = round(time.time() - started, 2)
            _update_job(
                job_id,
                status="completed",
                message=done_msg,
                finished_at=_utc_now_iso(),
                elapsed_sec=elapsed,
                progress_pct=100,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - started, 2)
            _update_job(
                job_id,
                status="failed",
                message="回放执行失败，请检查输入与日志。",
                error=str(exc),
                finished_at=_utc_now_iso(),
                elapsed_sec=elapsed,
                progress_pct=100,
            )

    thread = threading.Thread(target=_worker, daemon=True, name=f"backtest-job-{job_id[:8]}")
    thread.start()
    return job_id


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, limit: int) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _read_csv_tail(path: Path, limit: int) -> List[Dict[str, str]]:
    """取 CSV 末尾若干行（预测文件多按日期升序，取前 N 行会卡在旧月份如图表只到 4 月下旬）。"""
    if not path.exists() or limit <= 0:
        return []
    buf: deque[Dict[str, str]] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            buf.append(row)
    return list(buf)


def _filter_rows(rows: List[Dict[str, str]], task: str, app_id: str, advertiser_id: str, date: str) -> List[Dict[str, str]]:
    filtered = rows
    if app_id:
        filtered = [r for r in filtered if r.get("app_id") == app_id]
    if advertiser_id and task != "app_last_day":
        filtered = [r for r in filtered if r.get("advertiser_id") == advertiser_id]
    if date:
        filtered = [r for r in filtered if (r.get("date") == date or r.get("target_day") == date)]
    return filtered


def _load_stage_compare() -> Dict[str, Any]:
    """
    读取阶段性成果对比（若文件存在）：
    - 启发式基线
    - 联合求解初版
    - 联合求解调优版
    """
    root = Path("outputs")
    candidates = {
        "heuristic": [
            root / "backtest_heuristic_25d_ref4" / "offline_backtest_summary.json",
            root / "backtest_heuristic_25d_ref3" / "offline_backtest_summary.json",
            root / "backtest_heuristic_25d_ref2" / "offline_backtest_summary.json",
        ],
        "joint_v1": [
            root / "backtest_joint_lp_25d" / "offline_backtest_summary.json",
        ],
        "joint_tuned": [
            root / "backtest_joint_lp_25d_tuned2" / "offline_backtest_summary.json",
            root / "backtest_joint_lp_25d_tuned" / "offline_backtest_summary.json",
        ],
    }

    def _pick(paths: List[Path]) -> Dict[str, Any]:
        for p in paths:
            if p.exists():
                return _read_json(p)
        return {}

    heuristic = _pick(candidates["heuristic"])
    joint_v1 = _pick(candidates["joint_v1"])
    joint_tuned = _pick(candidates["joint_tuned"])
    return {"heuristic": heuristic, "joint_v1": joint_v1, "joint_tuned": joint_tuned}


def _load_payload(output_dir: Path, task: str, app_id: str, advertiser_id: str, date: str, limit: int) -> Dict[str, Any]:
    limit = max(20, min(limit, 1000))
    if task == "app_last_day":
        summary = _read_json(output_dir / "app_level_last_day_prediction.json")
        rows = _read_csv(output_dir / "app_level_last_day_suggestions.csv", limit)
        attribution_rows = _read_csv(output_dir / "app_level_roi_attribution_ranking.csv", limit)
    else:
        summary = _read_json(output_dir / "offline_backtest_summary.json")
        rows = _read_csv(output_dir / "offline_backtest_detail.csv", limit)
        attribution_rows = []
    rows = _filter_rows(rows, task=task, app_id=app_id.strip(), advertiser_id=advertiser_id.strip(), date=date.strip())
    attribution_rows = _filter_rows(
        attribution_rows, task=task, app_id=app_id.strip(), advertiser_id=advertiser_id.strip(), date=date.strip()
    )

    mode_dist = summary.get("mode_distribution", {}) if summary else {}
    if task == "app_last_day" and summary:
        preds = summary.get("predictions", [])
        mode_dist = {
            "GUARD": sum(1 for p in preds if p.get("mode") == "GUARD"),
            "SCALE": sum(1 for p in preds if p.get("mode") == "SCALE"),
        }

    return {
        "task": task,
        "summary": summary,
        "rows": rows,
        "attribution_rows": attribution_rows,
        "mode_distribution": mode_dist,
        "stage_compare": _load_stage_compare(),
    }


def _pick_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _read_csv_rows(path: Path | None, limit: int = 5000, *, from_end: bool = False) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    if from_end:
        return _read_csv_tail(path, limit)
    return _read_csv(path, limit)


def _load_model_dashboard_payload() -> Dict[str, Any]:
    root = _repo_root() / "outputs"
    roi_dir = _pick_existing(
        [
            root / "model_parallel_roi_d1_exp035",
            root / "model_parallel_roi_d1_v9_unified",
            root / "model_parallel_roi_d1_v8_001",
            root / "model_parallel_roi_d1_v7_001",
        ]
    )
    spend_dir = _pick_existing(
        [
            root / "model_parallel_spend_t1_exp035",
            root / "model_parallel_spend_t1_v12_unified",
            root / "model_parallel_spend_t1_v11_001",
        ]
    )
    date_feature_dir = root / "model_parallel_spend_t1_date_features_lgbm"
    old_spend_path = root / "model_parallel_spend_t1_v12_unified" / "predictions_XGBoost_log.csv"
    app_month_roi_path = root / "app_level_last_day_suggestions.csv"
    app_month_roi_summary = _read_json(root / "app_level_last_day_prediction.json")

    roi_metrics_path = roi_dir / "metrics_summary.csv" if roi_dir else None
    roi_pred_path = resolve_roi_predictions_csv(roi_dir) if roi_dir else None
    spend_pred_path = resolve_spend_predictions_csv(spend_dir) if spend_dir else None

    spend_predictions = _read_csv_rows(spend_pred_path, 50000, from_end=True)

    # CQR interval predictions
    cqr_path = spend_dir / "predictions_XGBoost_CQR.csv" if spend_dir else None
    cqr_predictions = None
    cqr_coverage = None
    if cqr_path and cqr_path.exists():
        cqr_raw = _read_csv_rows(cqr_path, 50000, from_end=True)
        cqr_by_date: dict[str, dict] = {}
        for row in cqr_raw:
            day = str(row.get("日期", ""))[:10]
            if not day:
                continue
            if day not in cqr_by_date:
                cqr_by_date[day] = {"y_pred": 0.0, "y_lower": 0.0, "y_upper": 0.0}
            cqr_by_date[day]["y_pred"] += float(row.get("y_pred", 0))
            cqr_by_date[day]["y_lower"] += float(row.get("y_lower", 0))
            cqr_by_date[day]["y_upper"] += float(row.get("y_upper", 0))
        cqr_predictions = cqr_by_date
        report = _read_json(spend_dir / "report.json") if spend_dir else {}
        cqr_report = report.get("cqr", {})
        cqr_coverage = cqr_report.get("coverage")

    calendar = CalendarService()
    spend_calendar = {}
    for row in spend_predictions:
        day_text = str(row.get("日期", ""))[:10]
        if not day_text:
            continue
        day = datetime.fromisoformat(day_text).date()
        target_day = day + timedelta(days=1)
        target_features = calendar.features_for_day(target_day)
        spend_calendar[day_text] = {
            "target_date": target_day.isoformat(),
            "target_day_type": target_features["day_type"],
            "target_holiday_name": target_features["holiday_display_name"],
            "target_is_last_holiday_day": target_features["is_last_holiday_day"],
            "target_is_first_workday_after_holiday": target_features["is_first_workday_after_holiday"],
        }

    # 全量统计（不受显示行数限制）
    roi_stats = _compute_prediction_stats("roi")
    spend_stats = _compute_prediction_stats("spend")

    # 全量日期聚合图表数据（不受显示行数限制）
    roi_chart_data = _compute_chart_data("roi")
    spend_chart_data = _compute_chart_data("spend")

    payload = {
        "roi": {
            "title": "T+1 ROI_D1 预测效果",
            "metrics": _read_csv_rows(roi_metrics_path, 100),
            "predictions": _read_csv_rows(roi_pred_path, 50000, from_end=True),
            "prediction_file": str(roi_pred_path.resolve()) if roi_pred_path and roi_pred_path.exists() else "",
            "source": str(roi_dir) if roi_dir else "",
            "stats": roi_stats["overall"] if roi_stats else None,
            "apps": roi_stats["apps"] if roi_stats else [],
            "chart_data": roi_chart_data or [],
        },
        "spend": {
            "title": "T+1 Spend 预测效果（目标日历 + 假期切换校准）",
            "report": _read_json(spend_dir / "report.json") if spend_dir else {},
            "predictions": spend_predictions,
            "prediction_file": str(spend_pred_path.resolve()) if spend_pred_path and spend_pred_path.exists() else "",
            "baseline_predictions": _read_csv_rows(old_spend_path, 20000, from_end=True),
            "cqr_predictions": cqr_predictions,
            "cqr_coverage": cqr_coverage,
            "calendar": spend_calendar,
            "source": str(spend_dir) if spend_dir else "",
            "stats": spend_stats["overall"] if spend_stats else None,
            "apps": spend_stats["apps"] if spend_stats else [],
            "chart_data": spend_chart_data or [],
        },
        "date_feature_compare": _read_json(date_feature_dir / "date_feature_business_compare.json"),
        "date_feature_source": str(date_feature_dir),
        "app_month_roi": {
            "title": "不同应用月末 ROI 预测",
            "rows": _read_csv_rows(app_month_roi_path, 5000),
            "kpi": app_month_roi_summary.get("kpi", 1.05),
            "summary": app_month_roi_summary,
            "source": str(app_month_roi_path.resolve()) if app_month_roi_path.exists() else "",
        },
    }

    # 合并三档预算推荐数据（CQR 区间）
    tiered_path = root / "tiered_budget_recommendations.csv"
    if tiered_path.exists():
        tiered_rows = _read_csv(tiered_path, 5000)
        tiered_map = {r.get("应用ID", ""): r for r in tiered_rows if r.get("应用ID")}
        for row in payload["app_month_roi"]["rows"]:
            tid = row.get("app_id", "")
            if tid in tiered_map:
                t = tiered_map[tid]
                row["conservative_budget"] = t.get("保守预算(P05)", "")
                row["neutral_budget"] = t.get("中性预算(P50)", "")
                row["aggressive_budget"] = t.get("激进预算(P95)", "")

    return payload


def _render_model_dashboard(payload: Dict[str, Any], initial_view: str = "predict") -> HTMLResponse:
    safe_payload_js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    safe_initial_view_js = json.dumps(initial_view if initial_view in {"predict", "recommend", "daily-revenue", "monitor"} else "predict")
    page = f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>智能预算决策系统 - 模型评估看板</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --line: #e6eaf2;
      --text: #172033;
      --muted: #667085;
      --blue: #2563eb;
      --green: #16a34a;
      --orange: #f59e0b;
      --red: #dc2626;
      --shadow: 0 10px 30px rgba(16, 24, 40, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, "PingFang SC", "Microsoft YaHei", Arial, sans-serif; }}
    .shell {{ max-width: 1680px; margin: 0 auto; padding: 28px; display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 20px; }}
    .hero {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 18px; }}
    .title {{ font-size: 28px; font-weight: 760; letter-spacing: -.02em; }}
    .subtitle {{ margin-top: 8px; color: var(--muted); line-height: 1.6; max-width: 880px; }}
    .badge {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid #bfdbfe; background: #eff6ff; color: #1d4ed8; padding: 7px 10px; border-radius: 999px; font-size: 13px; white-space: nowrap; }}
    .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); padding: 18px; margin: 14px 0; }}
    .hero, .view {{ grid-column: 2; }}
    .topnav {{ grid-column: 1; grid-row: 1 / span 20; position: sticky; top: 24px; align-self: start; display:flex; flex-direction:column; gap:10px; padding: 12px; border: 1px solid var(--line); border-radius: 20px; background: rgba(255,255,255,.78); box-shadow: var(--shadow); }}
    .nav-title {{ font-size: 12px; color: var(--muted); padding: 6px 8px 2px; }}
    .navbtn {{ width: 100%; justify-content: flex-start; background:#fff; color:var(--text); border:1px solid var(--line); box-shadow:0 4px 14px rgba(16,24,40,.05); border-left: 5px solid transparent; }}
    .navbtn.active {{ background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe; border-left-color:var(--blue); }}
    .view {{ display:block; min-width: 0; }}
    .view.hidden {{ display:none; }}
    .controls {{ display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr auto; gap: 12px; align-items: end; }}
    .roi-controls {{ display:grid; grid-template-columns: 1.1fr .75fr .8fr .85fr .7fr .7fr auto; gap:12px; align-items:end; }}
    label {{ display: grid; gap: 6px; font-size: 12px; color: var(--muted); }}
    select, input {{ height: 40px; border: 1px solid #d7ddea; border-radius: 10px; padding: 0 11px; background: #fff; color: var(--text); outline: none; }}
    select:focus, input:focus {{ border-color: #93c5fd; box-shadow: 0 0 0 4px #dbeafe; }}
    button {{ height: 40px; border: 0; border-radius: 10px; padding: 0 16px; color: #fff; background: var(--blue); font-weight: 650; cursor: pointer; }}
    .grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .metric {{ background: linear-gradient(180deg, #fff, #f9fbff); border: 1px solid var(--line); border-radius: 16px; padding: 15px; }}
    .metric .k {{ color: var(--muted); font-size: 12px; }}
    .metric .v {{ font-size: 26px; font-weight: 780; margin-top: 6px; }}
    .metric .d {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
    .section-title {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px; }}
    .section-title h2 {{ font-size: 18px; margin: 0; }}
    .hint {{ color: var(--muted); font-size: 12px; }}
    .charts {{ display: grid; grid-template-columns: 1.25fr .75fr; gap: 14px; }}
    .chart {{ height: 360px; border: 1px solid var(--line); border-radius: 16px; background: #fff; }}
    .wide {{ height: 330px; }}
    .note {{ background: #fffbeb; border: 1px solid #fde68a; color: #92400e; padding: 12px 14px; border-radius: 14px; line-height: 1.6; }}
    .note-section {{ padding: 6px 0; }}
    .note-section + .note-section {{ border-top: 1px solid #fde68a; margin-top: 4px; padding-top: 10px; }}
    .note-label {{ font-weight: 700; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
    .note-body {{ color: #78350f; }}
    .note-list {{ margin: 4px 0 0 18px; padding: 0; }}
    .note-list li {{ margin-bottom: 3px; }}
    .note-list-risk {{ color: #b91c1c; }}
    .trace-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }}
    .trace-card {{ border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: #fff; min-height: 132px; }}
    .trace-card .t {{ display:flex; justify-content:space-between; gap:8px; align-items:center; font-weight:700; }}
    .trace-card .s {{ font-size: 11px; border-radius: 999px; padding: 2px 8px; }}
    .s-ok {{ color:#15803d; background:#dcfce7; }}
    .s-warn {{ color:#a16207; background:#fef3c7; }}
    .s-critical {{ color:#b91c1c; background:#fee2e2; }}
    .s-release {{ color:#7c3aed; background:#ede9fe; }}
    .trace-card ul {{ margin: 10px 0 0 16px; padding:0; color: var(--muted); line-height: 1.5; font-size: 12px; }}
    .layer-panels {{ display: flex; flex-direction: column; gap: 8px; }}
    .layer-panel {{ border: 1px solid var(--line); border-radius: 14px; background: #fff; overflow: hidden; }}
    .layer-panel.open {{ border-color: var(--blue); box-shadow: 0 0 0 3px #dbeafe; }}
    .layer-header {{ display: flex; justify-content: space-between; align-items: center; padding: 14px 16px; cursor: pointer; user-select: none; gap: 12px; }}
    .layer-header:hover {{ background: #f8fafc; }}
    .layer-header .layer-icon {{ font-size: 20px; width: 32px; text-align: center; flex-shrink: 0; }}
    .layer-header .layer-name {{ font-weight: 700; font-size: 14px; flex-shrink: 0; }}
    .layer-header .layer-summary {{ flex: 1; font-size: 12px; color: var(--muted); min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .layer-header .layer-status {{ font-size: 11px; border-radius: 999px; padding: 3px 10px; font-weight: 650; flex-shrink: 0; }}
    .layer-header .layer-arrow {{ font-size: 12px; color: var(--muted); transition: transform 0.2s; flex-shrink: 0; }}
    .layer-panel.open .layer-arrow {{ transform: rotate(180deg); }}
    .layer-body {{ display: none; padding: 0 16px 16px; border-top: 1px solid var(--line); }}
    .layer-panel.open .layer-body {{ display: block; }}
    .layer-body table {{ font-size: 12px; }}
    .layer-body table th {{ background: #f8fafc; font-size: 11px; }}
    .layer-body .layer-stat {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 12px; }}
    .layer-body .layer-stat-item {{ background: #f8fafc; border-radius: 10px; padding: 10px 12px; }}
    .layer-body .layer-stat-item .sv {{ font-size: 20px; font-weight: 780; }}
    .layer-body .layer-stat-item .sk {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 12px 14px; border-bottom: 1px solid var(--line); line-height: 1.5; }}
    th {{ color: var(--muted); font-weight: 650; background: #f8fafc; position: sticky; top: 0; white-space: nowrap; }}
    td {{ vertical-align: middle; }}
    th.sortable {{ cursor:pointer; user-select:none; }}
    th.sortable:hover {{ color: var(--blue); background:#eef5ff; }}
    .table-wrap {{ max-height: 360px; overflow: auto; border: 1px solid var(--line); border-radius: 14px; }}
    .table-wrap table {{ font-size: 13px; }}
    .table-wrap th {{ top: 0; z-index: 1; }}
    .good {{ color: var(--green); }}
    .bad {{ color: var(--red); }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .pill {{ display:inline-flex; align-items:center; padding:3px 8px; border-radius:999px; font-size:12px; border:1px solid var(--line); background:#f8fafc; white-space:nowrap; }}
    .pill.good {{ background:#ecfdf3; border-color:#bbf7d0; }}
    .pill.bad {{ background:#fef2f2; border-color:#fecaca; }}
    .basis {{ color: var(--muted); line-height: 1.45; min-width: 260px; }}
    .plain-note {{ background:#f8fafc; border:1px solid var(--line); color:var(--muted); padding:12px 14px; border-radius:14px; line-height:1.6; }}
    .recommend-hero {{ background: linear-gradient(135deg, #eff6ff, #ffffff 55%, #f0fdf4); border: 1px solid #bfdbfe; }}
    .quick-filters {{ display:flex; flex-wrap:wrap; gap:8px; margin: 12px 0 10px; }}
    .chip {{ height:34px; border-radius:999px; border:1px solid var(--line); background:#fff; color:var(--text); padding:0 12px; box-shadow:none; font-weight:650; }}
    .chip.active {{ background:#2563eb; color:#fff; border-color:#2563eb; }}
    .filter-row {{ display:flex; gap:10px; align-items:end; flex-wrap:wrap; margin-top: 10px; }}
    .filter-row label {{ min-width: 220px; }}
    .advanced-filters {{ display:none; margin-top:10px; }}
    .advanced-filters.open {{ display:block; }}
    .secondary-btn {{ background:#fff; color:var(--blue); border:1px solid #bfdbfe; box-shadow:none; }}
    .compact-metrics {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:10px; margin: 10px 0 12px; }}
    .compact-metrics .metric {{ padding:10px 12px; border-radius:12px; }}
    .compact-metrics .metric .v {{ font-size:20px; }}
    .recommend-summary-title {{ margin:18px 0 8px; font-size:16px; font-weight:760; }}
    .app-popup {{ position: fixed; z-index: 9999; background: #fff; border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 16px 48px rgba(16,24,40,.18); padding: 16px; min-width: 540px; max-width: 620px; pointer-events: auto; }}
    .app-popup-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .app-popup-header b {{ font-size: 16px; color: var(--blue); }}
    .app-popup-header .close-hint {{ font-size: 11px; color: var(--muted); }}
    .app-popup-info {{ font-size: 12px; line-height: 1.55; color: var(--text); margin-bottom: 10px; }}
    .app-popup-info .lbl {{ color: var(--muted); }}
    .app-popup-charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .app-popup-chart {{ height: 200px; border: 1px solid var(--line); border-radius: 10px; }}
    .popup-app-id {{ cursor: pointer; color: var(--blue); text-decoration: underline; text-underline-offset: 2px; }}
    .popup-app-id:hover {{ color: #1d4ed8; }}
    @media (max-width: 1100px) {{ .shell {{ display:block; }} .topnav {{ position:static; flex-direction:row; margin-bottom:14px; }} .controls, .roi-controls, .grid4, .grid2, .charts, .trace-grid {{ grid-template-columns: 1fr; }} .hero {{ flex-direction: column; }} }}
  </style>
</head>
<body>
<div class="shell">
  <div class="hero">
    <div>
      <div class="title">智能预算决策系统</div>
    </div>
  </div>

  <div class="topnav">
    <div class="nav-title">业务模块</div>
    <button class="navbtn active" id="predictNav" data-view="predict">预测模块</button>
    <button class="navbtn" id="recommendNav" data-view="recommend">推荐模块</button>
    <button class="navbtn" id="dailyRevenueNav" data-view="daily-revenue">每日收入验证</button>
    <button class="navbtn" id="monitorNav" data-view="monitor">系统监控</button>
  </div>

  <div id="dataStatusBar" style="display:flex;gap:16px;flex-wrap:wrap;padding:8px 16px;background:var(--bg);border:1px solid var(--line);border-radius:12px;margin-bottom:6px;font-size:12px;align-items:center;">
    <span class="hint">数据状态：</span><span id="dsLoading">加载中...</span>
  </div>
  <div id="retrainStatusBar" style="display:flex;gap:12px;flex-wrap:wrap;padding:6px 16px;background:var(--bg);border:1px solid var(--line);border-radius:12px;margin-bottom:12px;font-size:12px;align-items:center;">
    <span class="hint">重训状态：</span><span id="rsLoading">加载中...</span>
    <button id="retrainTriggerBtn" onclick="triggerRetrain()" style="margin-left:auto;padding:4px 14px;font-size:11px;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;">触发重训</button>
  </div>

  <div class="view" id="view-predict">
  <div class="panel">
    <div class="controls">
      <label>评估目标
        <select id="targetSel">
          <option value="roi">T+1 ROI_D1 预测</option>
          <option value="spend">T+1 Spend 预测</option>
        </select>
      </label>
      <label>开始日期
        <input type="date" id="startDate" />
      </label>
      <label>结束日期
        <input type="date" id="endDate" />
      </label>
      <button id="resetBtn">重置筛选</button>
    </div>
    <div id="appSelectHint" style="margin-top:10px; display:none; padding:8px 12px; background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; font-size:13px;">
      <span id="appSelectText"></span>
      <button id="clearAppSelBtn" style="height:26px; padding:0 10px; font-size:12px; margin-left:12px;">返回全量</button>
    </div>
  </div>

  <div class="grid4" id="metricCards"></div>

  <div class="panel">
    <div class="section-title">
      <h2 id="lineTitle">预测值 vs 实际值</h2>
    </div>
    <div class="charts">
      <div id="lineChart" class="chart"></div>
      <div id="scatterChart" class="chart"></div>
    </div>
  </div>

  <div class="panel">
    <div class="section-title">
      <h2>每日误差走势</h2>
    </div>
    <div id="errorChart" class="chart wide"></div>
  </div>

  <div class="panel" id="modelPanel">
    <div class="section-title">
      <h2>模型指标与业务结论</h2>
      <div class="hint" id="sourceHint"></div>
    </div>
    <div id="modelSummary" class="note"></div>
  </div>

  <div class="panel">
    <div class="section-title">
      <h2>应用总览表</h2>
    </div>
    <div style="margin-bottom:8px;"><input type="text" id="appTableSearch" placeholder="搜索应用ID…" /></div>
    <div class="table-wrap" style="max-height:320px;">
      <table>
        <thead id="appSummaryHead"></thead>
        <tbody id="appSummaryBody"></tbody>
      </table>
    </div>
  </div>

  <div class="panel" id="detailPanel">
    <div class="section-title">
      <h2>预测明细 <span id="detailTitleSuffix" style="font-weight:400;color:var(--muted);font-size:14px;"></span></h2>
      <div class="hint" id="detailPageInfo"></div>
    </div>
    <div style="margin-bottom:8px; display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <input type="text" id="detailAppSearch" placeholder="搜索应用ID…" style="width:160px;" />
      <button id="detailPrevBtn" disabled>上一页</button>
      <span id="detailPageLabel">-</span>
      <button id="detailNextBtn" disabled>下一页</button>
    </div>
    <div class="table-wrap" style="max-height:520px;">
      <table>
        <thead id="detailHead"></thead>
        <tbody id="detailBody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="view hidden" id="view-recommend">
  <div class="panel recommend-hero" id="businessPanel">
    <div class="section-title">
      <h2>应用推荐模块</h2>
      <div class="hint" id="mpcSourceHint"></div>
    </div>
    <div class=”panel” style=”margin-top:0;”>
      <div class=”section-title”>
        <h2>应用月末 ROI 预测表</h2>
      </div>
      <div class="filter-row">
        <label>应用ID搜索
          <input id="appRoiSearch" placeholder="例如 31440303" />
        </label>
        <button class="secondary-btn" id="appRoiAdvancedToggle" type="button">高级筛选</button>
        <button id="appRoiReset" type="button">重置表格</button>
      </div>
      <div class="quick-filters" id="appRoiQuickFilters">
        <button class="chip active" type="button" data-filter="all">全部</button>
        <button class="chip" type="button" data-filter="below_kpi">低于 KPI</button>
        <button class="chip" type="button" data-filter="scale">建议放量</button>
        <button class="chip" type="button" data-filter="control">建议控量</button>
        <button class="chip" type="button" data-filter="alert">有告警</button>
        <button class="chip" type="button" data-filter="curve">长尾主导</button>
        <button class="chip" type="button" data-filter="d1">D1主导</button>
        <button class="chip" type="button" data-filter="spend">消耗主导</button>
      </div>
      <div class="advanced-filters" id="appRoiAdvancedFilters">
        <div class="roi-controls">
          <label>KPI状态
          <select id="appRoiKpiStatus">
            <option value="">全部</option>
            <option value="above">达标</option>
            <option value="below">未达标</option>
          </select>
          </label>
          <label>推荐动作
          <select id="appRoiAction"><option value="">全部动作</option></select>
          </label>
          <label>风险
          <select id="appRoiRisk">
            <option value="">全部</option>
            <option value="has_alert">有告警</option>
            <option value="no_alert">无告警</option>
          </select>
          </label>
          <label>最低月末ROI
          <input id="appRoiMin" type="number" step="0.01" placeholder="如 1.05" />
          </label>
          <label>最高月末ROI
          <input id="appRoiMax" type="number" step="0.01" placeholder="可选" />
          </label>
        </div>
      </div>
      <div class="hint" id="appRoiSourceHint" style="margin-top:10px;"></div>
      <div class="table-wrap" style="margin-top:14px; max-height:420px;">
        <table>
          <thead id="appRoiHead"></thead>
          <tbody id="appRoiBody"></tbody>
        </table>
      </div>
      <div class="compact-metrics" id="appRoiCards"></div>
    </div>
    <div class="recommend-summary-title">推荐总览</div>
    <div class="grid4" id="businessCards"></div>
    <div class="grid2" style="margin-top:14px;">
      <div id="budgetChart" class="chart"></div>
      <div id="productChart" class="chart"></div>
    </div>
    <div class="grid2" style="margin-top:14px;">
      <div id="layerChart" class="chart"></div>
      <div class="note" id="decisionNote"></div>
    </div>
    <div class="section-title" style="margin-top:16px;">
      <h2>六层决策面板</h2>
    </div>
    <div class="layer-panels" id="layerPanels"></div>
  </div>
</div>

<div class="view hidden" id="view-daily-revenue">
  <div class="panel">
    <div class="section-title">
      <h2>每日买量收入验证</h2>
    </div>
    <div style="margin-bottom:8px;">
      <input type="text" id="drAppSearch" placeholder="搜索应用ID…" style="width:160px;" />
      <button onclick="loadDailyRevenueChart()" type="button">查询</button>
    </div>
    <div id="dailyRevenueChart" style="width:100%;height:320px;border:1px solid var(--line);border-radius:16px;background:#fff;"></div>
    <div id="dailyRevenueCarryoverChart" style="width:100%;height:280px;border:1px solid var(--line);border-radius:16px;background:#fff;margin-top:8px;"></div>
    <div id="dailyRevenueStats" style="margin-top:8px;"></div>
  </div>

  <div class="panel" style="margin-top:16px;">
    <div class="section-title"><h2>应用曲线诊断</h2><span style="font-size:11px;color:var(--muted);">Carryover MAPE ↓ = 释放曲线越可靠 → 月末ROI推算越可信</span></div>
    <div id="dailyRevenueAppSummary" class="table-wrap" style="max-height:360px;">
      <div class="plain-note">加载中...</div>
    </div>
  </div>

  <div class="panel" style="margin-top:16px;">
    <div class="section-title"><h2>预测数据表</h2></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
      <input type="text" id="drTableSearch" placeholder="搜索应用ID…" style="width:140px;" />
      <input type="date" id="drTableDateFrom" style="width:130px;" title="起始日期" />
      <input type="date" id="drTableDateTo" style="width:130px;" title="结束日期" />
      <select id="drTableDays" style="width:120px;">
        <option value="0">全部天数</option>
        <option value="30">days ≥ 30</option>
        <option value="60">days ≥ 60</option>
      </select>
      <button onclick="loadDailyRevenueTable()" type="button">查询</button>
      <span id="drTableInfo" style="font-size:12px;color:var(--muted);line-height:2;"></span>
    </div>
    <div id="dailyRevenueTableWrap" class="table-wrap" style="max-height:420px;">
      <div class="plain-note">点击「查询」加载数据</div>
    </div>
  </div>

  <div class="panel" style="margin-top:16px;">
    <div class="section-title">
      <h2>在线预测</h2>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin-bottom:8px;">
      <label>目标日期 <input type="date" id="onlineTargetDate" value="2026-05-17" style="width:140px;" /></label>
      <label>模式
        <select id="onlineInputMode" onchange="toggleOnlineInputMode()" style="width:110px;">
          <option value="json">JSON</option>
          <option value="csv">CSV 批量</option>
        </select>
      </label>
      <button onclick="runOnlinePredict()" type="button" style="background:var(--accent);color:#fff;padding:6px 16px;border-radius:8px;">预测</button>
    </div>
    <div id="onlineJsonInput" style="margin-bottom:8px;">
      <textarea id="onlineInputJson" rows="6" style="width:100%;font-family:monospace;font-size:12px;" placeholder='[{{"应用ID": "30262609", "spend": 5000, "d1_revenue": 750}}]'></textarea>
    </div>
    <div id="onlineCsvInput" style="display:none;margin-bottom:8px;">
      <textarea id="onlineInputCsv" rows="8" style="width:100%;font-family:monospace;font-size:12px;" placeholder="应用ID,spend,d1_revenue&#10;30262609,5000,750&#10;30744533,12000,1800"></textarea>
    </div>
    <div id="onlinePredictResult" style="margin-top:8px;"></div>
  </div>
</div>

<div class="view hidden" id="view-monitor">
  <div class="panel">
    <div class="section-title"><h2>告警与健康</h2></div>
    <div class="grid4">
      <div class="metric"><div class="k">告警总数 (7天)</div><div class="v" id="monitorAlertTotal">--</div><div class="d" id="monitorAlertBreakdown"></div></div>
      <div class="metric"><div class="k">漂移状态</div><div class="v" id="monitorDriftStatus">--</div><div class="d" id="monitorDriftDetail"></div></div>
      <div class="metric"><div class="k">数据状态</div><div class="v" id="monitorDataStatus">--</div><div class="d" id="monitorDataDetail"></div></div>
      <div class="metric"><div class="k">训练 & 日历</div><div class="v" id="monitorTrainStatus">--</div><div class="d" id="monitorCalStatus"></div></div>
    </div>
  </div>
  <div class="panel">
    <div class="section-title"><h2>健康趋势 (30天)</h2></div>
    <div id="healthTrendChart" style="width:100%;height:340px;border:1px solid var(--line);border-radius:16px;background:#fff;"></div>
  </div>
  <div class="panel">
    <div class="section-title"><h2>告警历史</h2></div>
    <div class="table-wrap" style="max-height:320px;" id="monitorAlertTable"></div>
  </div>
</div>

</div>

<div id="appHoverPopup" class="app-popup" style="display:none;">
  <div class="app-popup-header">
    <b id="popupAppId"></b>
    <span class="close-hint">点击应用ID跳转详情 | 移开隐藏</span>
  </div>
  <div class="app-popup-info" id="popupInfo"></div>
  <div class="app-popup-charts">
    <div id="popupRoiChart" class="app-popup-chart"></div>
    <div id="popupSpendChart" class="app-popup-chart"></div>
  </div>
</div>

<script>
const payload = {safe_payload_js};
const initialView = {safe_initial_view_js};
const charts = {{
  budget: echarts.init(document.getElementById('budgetChart')),
  product: echarts.init(document.getElementById('productChart')),
  layer: echarts.init(document.getElementById('layerChart')),
  line: echarts.init(document.getElementById('lineChart')),
  scatter: echarts.init(document.getElementById('scatterChart')),
  error: echarts.init(document.getElementById('errorChart')),
}};

const _popupCharts = {{ roi: null, spend: null }};
let _popupAppId = '';
let _popupTimer = null;
let _popupHideTimer = null;

const fmt = (v, digits=2) => Number.isFinite(v) ? v.toFixed(digits) : '-';
const pct = (v, digits=2) => Number.isFinite(v) ? (v * 100).toFixed(digits) + '%' : '-';
const num = (v) => {{
  const x = Number(v);
  return Number.isFinite(x) ? x : 0;
}};
const dateOnly = (s) => String(s || '').slice(0, 10);
const inputTrim = (id) => {{
  const el = document.getElementById(id);
  return el ? String(el.value || '').trim() : '';
}};
const parseRoiBound = (id) => {{
  const t = inputTrim(id);
  if (t === '') return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}};
const oldSpendMap = new Map((payload.spend.baseline_predictions || []).map(r => {{
  const key = `${{dateOnly(r['日期'])}}|${{String(r['应用ID'] || '')}}`;
  return [key, {{
    pred: num(r.y_pred_fused ?? r.y_pred),
    ape: num(r.y_true) > 1e-8 ? Math.abs((num(r.y_true) - num(r.y_pred_fused ?? r.y_pred)) / num(r.y_true)) : null,
  }}];
}}));

function rowsForTarget(target) {{
  if (target === 'roi') {{
    return (payload.roi.predictions || []).map(r => ({{
      date: dateOnly(r['日期']),
      app: String(r['应用ID'] || ''),
      actual: num(r.y_true),
      pred: num(r.y_pred),
      pred2: null,
      model: r.model || 'ExtraTrees_hard_tuned_log',
    }}));
  }}
  return (payload.spend.predictions || []).map(r => ({{
    date: dateOnly(r['日期']),
    app: String(r['应用ID'] || ''),
    actual: num(r.y_true),
    pred: num(r.y_pred_fused ?? r.y_pred),
    pred2: Number.isFinite(Number(r.y_pred_app)) ? num(r.y_pred_app) : null,
    splitAgg: num(r.y_pred_split_agg),
    splitCnt: num(r.split_entity_cnt),
    model: r.model || (r.y_pred_fused !== undefined ? 'online_fused' : 'target_calendar_calibrated'),
  }})).map(r => {{
    const old = oldSpendMap.get(`${{r.date}}|${{r.app}}`) || {{}};
    const cal = (payload.spend.calendar || {{}})[r.date] || {{}};
    const newApe = r.actual > 1e-8 ? Math.abs((r.actual - r.pred) / r.actual) : null;
    const oldApe = Number.isFinite(old.ape) ? old.ape : null;
    const flags = [];
    if (cal.target_holiday_name) flags.push(cal.target_holiday_name);
    if (Number(cal.target_is_last_holiday_day) === 1) flags.push('假期最后一天');
    if (Number(cal.target_is_first_workday_after_holiday) === 1) flags.push('节后首个工作日');
    return {{
      ...r,
      oldPred: Number.isFinite(old.pred) ? old.pred : null,
      oldApe,
      newApe,
      improvement: Number.isFinite(oldApe) && Number.isFinite(newApe) ? oldApe - newApe : null,
      targetDate: cal.target_date || '',
      calendarLabel: flags.join(' / ') || (cal.target_day_type || '-'),
    }};
  }});
}}

function currentRows() {{
  const target = document.getElementById('targetSel').value;
  const start = document.getElementById('startDate').value;
  const end = document.getElementById('endDate').value;
  return rowsForTarget(target).filter(r => (!start || r.date >= start) && (!end || r.date <= end));
}}

let appRoiSort = {{ key: 'priorityScore', dir: 'desc' }};
let appRoiQuickFilter = 'all';
let detailSort = {{ key: 'date', dir: 'asc' }};

function compareRows(a, b, sortState) {{
  const av = a[sortState.key];
  const bv = b[sortState.key];
  const cmp = typeof av === 'string' ? String(av).localeCompare(String(bv)) : Number(av) - Number(bv);
  return sortState.dir === 'asc' ? cmp : -cmp;
}}

function appRoiCanonicalLabel(s) {{
  if (s === 'stable') return '稳定锚点';
  if (s === 'pred_only') return '纯预测';
  if (s === 'fused') return '融合';
  return String(s || '-');
}}

function appRoiDecision(r) {{
  const gap = r.roiGap;
  const alertCount = r.alertCount;
  const kpiT = num(r.kpiTarget || 1.05);
  const budgetChange = r.actualSpend > 1e-8 ? (r.plannedBudget - r.actualSpend) / r.actualSpend : 0;
  const mode = String(r.mode || '').toUpperCase();
  const scaleMode = mode === 'SCALE';
  const guardMode = mode === 'GUARD';

  let action = '维持观察';
  const hardBad = gap < -0.11 || alertCount >= 3 || (gap < -0.08 && alertCount >= 2);
  const midBad = gap < -0.08 || (alertCount >= 2 && gap < 0.1);
  const softBad = (gap < -0.05 && alertCount >= 1) || (gap < -0.06 && alertCount === 0);
  const strongOk = gap >= 0.14 && alertCount === 0;
  const okScale = gap >= 0.06 && alertCount <= 1 && (scaleMode || gap >= 0.1);
  const okTight = gap >= 0.06 && alertCount === 0;

  if (hardBad || (midBad && !scaleMode)) {{
    action = '收紧并降低出价';
  }} else if (midBad && scaleMode) {{
    action = '保守控量';
  }} else if (strongOk) {{
    action = '放量并小幅调高出价';
  }} else if (okScale || okTight) {{
    action = '小幅放量';
  }} else if (softBad) {{
    action = '保守控量';
  }} else if (alertCount === 1 && gap >= -0.04) {{
    action = '维持观察';
  }} else if (guardMode && gap < -0.03 && gap >= -0.08) {{
    action = '保守控量';
  }} else if (gap < -0.05) {{
    action = '保守控量';
  }}

  const actionTexts = {{
    '正常放量': '放量',
    '小幅放量': '小幅放量',
    '维持现状': '维持',
    '保守控量': '控量',
    '暂停投放': '暂停',
  }};
  const basis = [
    `月末ROI ${{fmt(r.monthEndRoi,4)}} vs KPI ${{fmt(kpiT,4)}}（${{pct(gap)}}）`,
    `预算 ${{fmt(r.plannedBudget,2)}}（变化${{pct(budgetChange)}}）| 昨日消耗 ${{fmt(r.actualSpend,2)}}`,
    `模式 ${{mode || '-'}} | 告警 ${{alertCount}} | ${{r.dominantFactor}}`,
  ].join('<br>');
  const actionLabel = actionTexts[action] || action;
  return {{ action: actionLabel, basis, budgetChange }};
}}

function appMonthRoiRows() {{
  return (payload.app_month_roi.rows || []).map(r => {{
    const monthEndRoi = num(r.month_end_roi_prediction);
    const kpi = num(payload.app_month_roi.kpi || 1.05);
    const alertCount = num(r.alert_count);
    const srcRaw = r.canonical_month_end_roi_source
      || (payload.app_month_roi.summary || {{}}).canonical_month_end_roi_source
      || 'fused';
    const row = {{
      targetDay: r.target_day || '',
      app: String(r.app_id || ''),
      mode: r.mode || '-',
      kpiTarget: kpi,
      canonicalSource: String(srcRaw),
      canonicalLabel: appRoiCanonicalLabel(srcRaw),
      actualSpend: num(r.actual_spend_last_day),
      actualRevenue: num(r.actual_revenue_last_day),
      plannedBudget: num(r.planned_total_budget),
      plannedDailyStable: num(r.planned_daily_spend_stable_anchor),
      plannedDailyPredOnly: num(r.planned_daily_spend_pred_only),
      plannedDailyFused: num(r.planned_daily_spend_fused),
      monthEndRoi,
      monthEndRoiStable: num(r.month_end_roi_stable_anchor),
      monthEndRoiPredOnly: num(r.month_end_roi_pred_only),
      monthEndRoiFused: num(r.month_end_roi_fused),
      monthEndSpend: num(r.month_end_spend_prediction),
      roiA: num(r.month_end_roi_prediction_a),
      roiB: num(r.month_end_roi_prediction_b),
      roiDelta: num(r.ab_roi_delta_b_minus_a),
      roi30d: num(r.month_end_roi_30d_total),
      carryover: num(r.cross_month_carryover_revenue),
      alertCount,
      dominantFactor: r.roi_dominant_factor || '-',
      d1Raw: num(r.d1_anchor_raw_mean),
      d1Calibrated: num(r.d1_anchor_calibrated_mean),
      spendT1Calibrated: num(r.spend_t1_pred_calibrated),
      roiD1T1Calibrated: num(r.roi_d1_t1_pred_calibrated),
      topActions: r.top_actions || '',
      topSuggestionsRaw: r.top_suggestions || '',
      alertsRaw: r.alerts || '',
      trainDays: num(r.train_days),
      conservativeBudget: num(r.conservative_budget),
      neutralBudget: num(r.neutral_budget),
      aggressiveBudget: num(r.aggressive_budget),
      roiGap: monthEndRoi - kpi,
    }};
    const decision = appRoiDecision(row);
    const absBudgetDelta = Math.abs(row.plannedBudget - row.actualSpend);
    const spendWeight = Math.log1p(row.actualSpend);
    const trainDays = row.trainDays || 0;
    const confidenceW = Math.min(1.0, Math.max(0.5, (trainDays - 30) / 30));
    const priorityScore = (
      Math.max(0, -row.roiGap) * 100 +
      row.alertCount * 8 +
      absBudgetDelta * 0.05 +
      spendWeight * 3 +
      (row.dominantFactor === 'BALANCED' ? 1 : 2)
    ) * confidenceW;
    return {{ ...row, ...decision, priorityScore }};
  }});
}}

function matchAppRoiQuickFilter(r) {{
  if (appRoiQuickFilter === 'below_kpi') return r.roiGap < 0;
  if (appRoiQuickFilter === 'scale') return r.action.includes('放量');
  if (appRoiQuickFilter === 'control') return r.action.includes('收紧') || r.action.includes('控量');
  if (appRoiQuickFilter === 'alert') return r.alertCount > 0;
  if (appRoiQuickFilter === 'curve') return r.dominantFactor === 'CURVE';
  if (appRoiQuickFilter === 'd1') return r.dominantFactor === 'D1_ANCHOR';
  if (appRoiQuickFilter === 'spend') return r.dominantFactor === 'SPEND';
  return true;
}}

function currentAppRoiRows() {{
  const search = inputTrim('appRoiSearch');
  const kpiEl = document.getElementById('appRoiKpiStatus');
  const actionEl = document.getElementById('appRoiAction');
  const riskEl = document.getElementById('appRoiRisk');
  const kpiStatus = kpiEl ? kpiEl.value : '';
  const action = actionEl ? actionEl.value : '';
  const risk = riskEl ? riskEl.value : '';
  const minRoi = parseRoiBound('appRoiMin');
  const maxRoi = parseRoiBound('appRoiMax');
  return appMonthRoiRows().filter(r =>
    (!search || r.app.includes(search)) &&
    matchAppRoiQuickFilter(r) &&
    (!kpiStatus || (kpiStatus === 'above' ? r.roiGap >= 0 : r.roiGap < 0)) &&
    (!action || r.action === action) &&
    (!risk || (risk === 'has_alert' ? r.alertCount > 0 : r.alertCount <= 0)) &&
    (minRoi === null || r.monthEndRoi >= minRoi) &&
    (maxRoi === null || r.monthEndRoi <= maxRoi)
  ).sort((a,b) => {{
    return compareRows(a, b, appRoiSort);
  }});
}}

function refreshAppRoiActions() {{
  const sel = document.getElementById('appRoiAction');
  if (!sel) return;
  const current = sel.value;
  const actions = [...new Set(appMonthRoiRows().map(r => r.action).filter(Boolean))].sort();
  sel.innerHTML = '<option value="">全部动作</option>' + actions.map(a => `<option value="${{a}}">${{a}}</option>`).join('');
  if (actions.includes(current)) sel.value = current;
}}

function renderAppMonthRoiTable() {{
  refreshAppRoiActions();
  const hintEl = document.getElementById('appRoiSourceHint');
  const sum = payload.app_month_roi.summary || {{}};
  const cs = sum.canonical_month_end_roi_source || '';
  const canonHint = cs ? ` · 主决策口径「${{appRoiCanonicalLabel(cs)}}」` : '';
  if (hintEl) hintEl.textContent = `数据来源：${{payload.app_month_roi.source || '暂无'}} · 原始行数 ${{ (payload.app_month_roi.rows || []).length }}${{canonHint}}`;
  const allRows = appMonthRoiRows();
  const visibleRows = currentAppRoiRows();
  const aboveKpi = visibleRows.filter(r => r.roiGap >= 0).length;
  const avgRoi = visibleRows.reduce((s,r)=>s+r.monthEndRoi,0) / Math.max(visibleRows.length, 1);
  const totalSpend = visibleRows.reduce((s,r)=>s+r.monthEndSpend,0);
  const highRisk = visibleRows.filter(r => r.alertCount > 0 || r.roiGap < 0).length;
  const cardsEl = document.getElementById('appRoiCards');
  if (cardsEl) cardsEl.innerHTML = [
    ['应用数', `${{visibleRows.length}} / ${{allRows.length}}`, '当前筛选 / 全部应用'],
    ['平均预测月末ROI', fmt(avgRoi,4), '筛选后应用均值'],
    ['达标应用数', String(aboveKpi), '预测月末ROI >= KPI'],
    ['风险应用数', String(highRisk), '有告警或低于KPI'],
  ].map(c => `<div class="metric"><div class="k">${{c[0]}}</div><div class="v">${{c[1]}}</div><div class="d">${{c[2]}}</div></div>`).join('');
  const headers = [
    ['app', '应用ID'],
    ['mode', '模式'],
    ['targetDay', '预测日期'],
    ['canonicalLabel', '主口径'],
    ['monthEndRoi', '月末ROI(主)'],
    ['monthEndRoiStable', '稳定锚点'],
    ['monthEndRoiPredOnly', '纯预测'],
    ['monthEndRoiFused', '融合'],
    ['roiGap', '较KPI差距'],
    ['action', '建议动作'],
    ['monthEndSpend', '预测月末消耗'],
    ['plannedBudget', '主用计划预算'],
    ['budgetChange', '预算变化'],
    ['roi30d', '30日总ROI'],
    ['roiA', 'Raw锚点ROI'],
    ['roiB', '校准锚点ROI'],
    ['roiDelta', '校准差异'],
    ['alertCount', '告警数'],
    ['trainDays', '训练天数'],
    ['dominantFactor', '偏差主因'],
    ['d1Raw', 'D1 Raw'],
    ['d1Calibrated', 'D1校准'],
    ['spendT1Calibrated', 'T+1 Spend(校准)'],
    ['roiD1T1Calibrated', 'T+1 ROI_D1(校准)'],
    ['priorityScore', '处理优先级'],
    ['conservativeBudget', '保守预算(P05)'],
    ['neutralBudget', '中性预算(P50)'],
    ['aggressiveBudget', '激进预算(P95)'],
  ];
  const headEl = document.getElementById('appRoiHead');
  const bodyEl = document.getElementById('appRoiBody');
  if (!headEl || !bodyEl) return;
  headEl.innerHTML = '<tr>' + headers.map(([key, name]) => {{
    const mark = appRoiSort.key === key ? (appRoiSort.dir === 'asc' ? ' ↑' : ' ↓') : ' ↕';
    return `<th class="sortable" data-sort="${{key}}">${{name}}${{mark}}</th>`;
  }}).join('') + '</tr>';
  const colspan = headers.length;
  if (!allRows.length) {{
    bodyEl.innerHTML = `<tr><td colspan="${{colspan}}" style="padding:16px;color:var(--muted);">未加载到应用列表。请在项目根目录启动服务，并确认存在 <code>outputs/app_level_last_day_suggestions.csv</code>（可运行 <code>PYTHONPATH=. python scripts/offline_backtest.py --task app_last_day</code> 生成）。</td></tr>`;
    return;
  }}
  if (!visibleRows.length) {{
    bodyEl.innerHTML = `<tr><td colspan="${{colspan}}" style="padding:16px;color:var(--muted);">当前筛选条件下无匹配应用。请点击「重置表格」或选择顶部「全部」筛选。</td></tr>`;
    return;
  }}
  bodyEl.innerHTML = visibleRows.map(r => {{
    const roiCls = r.roiGap >= 0 ? 'good' : 'bad';
    const actionCls = r.action.includes('收紧') || r.action.includes('控量') ? 'bad' : 'good';
    return `<tr>
      <td class="popup-app-id" data-app="${{r.app}}">${{r.app}}</td>
      <td>${{r.mode}}</td>
      <td>${{r.targetDay}}</td>
      <td>${{r.canonicalLabel}}</td>
      <td class="${{roiCls}}">${{fmt(r.monthEndRoi,4)}}</td>
      <td>${{fmt(r.monthEndRoiStable,4)}}</td>
      <td>${{fmt(r.monthEndRoiPredOnly,4)}}</td>
      <td>${{fmt(r.monthEndRoiFused,4)}}</td>
      <td class="${{roiCls}}">${{pct(r.roiGap)}}</td>
      <td><span class="pill ${{actionCls}}">${{r.action}}</span></td>
      <td>${{fmt(r.monthEndSpend,2)}}</td>
      <td>${{fmt(r.plannedBudget,2)}}</td>
      <td>${{pct(r.budgetChange)}}</td>
      <td>${{fmt(r.roi30d,4)}}</td>
      <td>${{fmt(r.roiA,4)}}</td>
      <td>${{fmt(r.roiB,4)}}</td>
      <td>${{fmt(r.roiDelta,4)}}</td>
      <td class="${{r.alertCount > 0 ? 'bad' : 'good'}}">${{r.alertCount}}</td>
      <td>${{r.trainDays || '-'}}</td>
      <td>${{r.dominantFactor === 'CURVE' ? '释放曲线' : r.dominantFactor === 'SPEND' ? '消耗规模' : r.dominantFactor === 'D1_ANCHOR' ? 'D1水平' : r.dominantFactor}}</td>
      <td>${{fmt(r.d1Raw,4)}}</td>
      <td>${{fmt(r.d1Calibrated,4)}}</td>
      <td>${{fmt(r.spendT1Calibrated,2)}}</td>
      <td>${{fmt(r.roiD1T1Calibrated,4)}}</td>
      <td>${{fmt(r.priorityScore,2)}}</td>
      <td>${{r.conservativeBudget ? fmt(r.conservativeBudget,2) : '-'}}</td>
      <td>${{r.neutralBudget ? fmt(r.neutralBudget,2) : '-'}}</td>
      <td>${{r.aggressiveBudget ? fmt(r.aggressiveBudget,2) : '-'}}</td>
    </tr>`;
  }}).join('');

  // event delegation for app hover popup + click navigation
  bodyEl.querySelectorAll('td.popup-app-id').forEach(td => {{
    const appId = td.dataset.app;
    td.addEventListener('mouseenter', (evt) => {{
      const row = appMonthRoiRows().find(r => r.app === appId);
      if (row) showAppPopup(appId, row, evt);
    }});
    td.addEventListener('mouseleave', () => hideAppPopup());
    td.addEventListener('click', () => {{
      hideAppPopup(true);
      selectedAppId = appId;
      _chartAppId = '';
      detailState = {{ appId: appId, page: 0, limit: 200, total: 0 }};
      detailSort = {{ key: 'date', dir: 'asc' }};
      document.getElementById('targetSel').value = 'spend';
      setActiveView('predict');
      document.getElementById('appSelectHint').style.display = '';
      document.getElementById('appSelectText').textContent = '已选中应用 ' + appId + ' — 图表和明细均仅显示该应用数据';
      render();
    }});
  }});
}}

let detailState = {{ appId: '', page: 0, limit: 200, total: 0 }};
let selectedAppId = '';
let appSummarySort = {{ key: 'mape_pct', dir: 'asc' }};

function renderAppSummary() {{
  const target = document.getElementById('targetSel').value;
  const apps = (target === 'roi' ? payload.roi.apps : payload.spend.apps) || [];
  const search = inputTrim('appTableSearch').toLowerCase();
  let filtered = search ? apps.filter(a => String(a.app_id).toLowerCase().includes(search)) : apps.slice();
  filtered = [...filtered].sort((a, b) => {{
    const av = a[appSummarySort.key]; const bv = b[appSummarySort.key];
    const cmp = typeof av === 'string' ? String(av).localeCompare(String(bv)) : Number(av) - Number(bv);
    return appSummarySort.dir === 'asc' ? cmp : -cmp;
  }});
  document.getElementById('appSummaryHead').innerHTML = '<tr>' + [
    ['app_id','应用ID'],['mae','MAE'],['rmse','RMSE'],['mape_pct','MAPE%'],['samples','样本数']
  ].map(([key,name]) => {{
    const mark = appSummarySort.key === key ? (appSummarySort.dir === 'asc' ? ' ↑' : ' ↓') : ' ↕';
    return `<th class="sortable" data-asort="${{key}}">${{name}}${{mark}}</th>`;
  }}).join('') + '</tr>';
  document.getElementById('appSummaryBody').innerHTML = filtered.map(a => {{
    const cls = a.mape_pct <= 50 ? 'good' : (a.mape_pct <= 100 ? '' : 'bad');
    const sel = a.app_id === selectedAppId ? ' style="background:#eff6ff;"' : '';
    return `<tr class="app-summary-row" data-app="${{a.app_id}}"${{sel}}>
      <td><b>${{a.app_id}}</b></td><td>${{fmt(a.mae,4)}}</td><td>${{fmt(a.rmse,4)}}</td>
      <td class="${{cls}}">${{pct(a.mape)}}</td><td>${{a.samples}}</td></tr>`;
  }}).join('');
  document.querySelectorAll('#appSummaryHead th[data-asort]').forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.dataset.asort;
      if (appSummarySort.key === key) appSummarySort.dir = appSummarySort.dir === 'asc' ? 'desc' : 'asc';
      else appSummarySort = {{ key, dir: key === 'app_id' ? 'asc' : 'desc' }};
      renderAppSummary();
    }});
  }});
  document.querySelectorAll('.app-summary-row').forEach(row => {{
    row.addEventListener('click', () => selectApp(row.dataset.app));
  }});
}}

function selectApp(appId) {{
  if (selectedAppId === appId) {{
    selectedAppId = '';
    document.getElementById('appSelectHint').style.display = 'none';
  }} else {{
    selectedAppId = appId;
    document.getElementById('appSelectHint').style.display = '';
    document.getElementById('appSelectText').textContent = `已选中应用 ${{appId}} — 图表和明细均仅显示该应用数据`;
  }}
  detailState.appId = selectedAppId;
  detailState.page = 0;
  detailState.total = 0;
  detailSort = {{ key: 'date', dir: 'asc' }};
  render();
}}

function loadAppDetail() {{
  const target = document.getElementById('targetSel').value;
  const {{ appId, page, limit }} = detailState;
  if (!appId) {{ renderTable(currentRows()); return; }}
  const offset = page * limit;
  document.getElementById('detailBody').innerHTML = '<tr><td colspan="10" style="text-align:center;padding:20px;">加载中…</td></tr>';
  fetch(`/web/data/${{target}}/app/${{encodeURIComponent(appId)}}?offset=${{offset}}&limit=${{limit}}`)
    .then(r => r.json())
    .then(data => {{
      detailState.total = data.total;
      renderDetailTable(target, data.rows, data.total, data.offset, data.limit);
    }})
    .catch(() => {{
      document.getElementById('detailBody').innerHTML = '<tr><td colspan="10" style="text-align:center;color:var(--red);padding:20px;">加载失败，请重试</td></tr>';
    }});
}}

function renderDetailTable(target, rows, total, offset, limit) {{
  const digits = target === 'roi' ? 4 : 2;
  const headers = target === 'roi'
    ? [['日期','日期'],['应用ID','应用ID'],['y_true','实际 ROI_D1'],['y_pred','预测 ROI_D1'],['ape','APE'],['model','模型']]
    : [['日期','日期'],['target_date','T+1日期'],['calendar_label','日历标签'],['应用ID','应用ID'],['y_true','实际 Spend'],
       ['y_pred','新版预测'],['y_pred_old','旧版预测'],['ape','APE'],['model','模型']];
  document.getElementById('detailHead').innerHTML = '<tr>' + headers.map(([key, name]) => {{
    return `<th>${{name}}</th>`;
  }}).join('') + '</tr>';
  document.getElementById('detailBody').innerHTML = rows.map(r => {{
    const y = Number(r.y_true); const p = Number(r.y_pred);
    const ape = y > 1e-8 ? Math.abs((y - p) / y) : 0;
    const cls = ape <= .3 ? 'good' : 'bad';
    const date = String(r['日期'] || '').slice(0,10);
    const app = String(r['应用ID'] || '');
    if (target === 'roi') {{
      return `<tr><td>${{date}}</td><td>${{app}}</td><td>${{fmt(y,digits)}}</td><td>${{fmt(p,digits)}}</td><td class="${{cls}}">${{pct(ape)}}</td><td>${{r.model || '-'}}</td></tr>`;
    }}
    const oldPred = Number(r.y_pred_old);
    const oldApe = y > 1e-8 && Number.isFinite(oldPred) ? Math.abs((y - oldPred) / y) : null;
    const improvement = Number.isFinite(oldApe) && ape !== null ? oldApe - ape : null;
    const impCls = Number.isFinite(improvement) && improvement >= 0 ? 'good' : 'bad';
    return `<tr><td>${{date}}</td><td>${{r.target_date || '-'}}</td><td>${{r.calendar_label || '-'}}</td><td>${{app}}</td><td>${{fmt(y,digits)}}</td><td>${{fmt(p,digits)}}</td><td>${{fmt(oldPred,digits)}}</td><td class="${{cls}}">${{pct(ape)}}</td><td>${{r.model || '-'}}</td></tr>`;
  }}).join('');
  const currentPage = Math.floor(offset / Math.max(limit, 1));
  const totalPages = Math.max(1, Math.ceil(total / limit));
  document.getElementById('detailPrevBtn').disabled = currentPage <= 0;
  document.getElementById('detailNextBtn').disabled = currentPage >= totalPages - 1;
  document.getElementById('detailPageLabel').textContent = `第 ${{currentPage + 1}}/${{totalPages}} 页 (共 ${{total}} 条)`;
  document.getElementById('detailTitleSuffix').textContent = selectedAppId ? `应用 ${{selectedAppId}}` : '';
  document.getElementById('detailPageInfo').textContent = selectedAppId ? `共 ${{total}} 条记录` : '';
}}

function calcMetrics(rows) {{
  const y = rows.map(r => r.actual);
  const p = rows.map(r => r.pred);
  const abs = rows.map((r, i) => Math.abs(y[i] - p[i]));
  const ape = rows.map((r, i) => y[i] > 1e-8 ? Math.abs((y[i] - p[i]) / y[i]) : 0);
  const sumY = y.reduce((a,b)=>a+b,0) || 1;
  const mae = abs.reduce((a,b)=>a+b,0) / Math.max(rows.length, 1);
  const mape = ape.reduce((a,b)=>a+b,0) / Math.max(rows.length, 1);
  const wmape = abs.reduce((a,b)=>a+b,0) / sumY;
  const bias = (p.reduce((a,b)=>a+b,0) - y.reduce((a,b)=>a+b,0)) / sumY;
  const hit30 = ape.filter(x => x <= .3).length / Math.max(rows.length, 1);
  return {{ mae, mape, wmape, bias, hit30, n: rows.length }};
}}

function renderCards(rows) {{
  const target = document.getElementById('targetSel').value;
  const stats = target === 'roi' ? payload.roi.stats : payload.spend.stats;
  const m = calcMetrics(rows);
  const cards = [
    ['全量样本数', stats ? String(stats.samples) : String(m.n), target === 'roi' ? '全量应用-日期 ROI 样本' : '全量应用-日期 spend 样本'],
    ['全量MAPE', stats ? pct(stats.mape) : pct(m.mape), '全量平均百分比误差'],
    ['全量MAE', stats ? fmt(stats.mae, 4) : fmt(m.mae, 4), '全量平均绝对误差'],
    ['全量应用数', stats ? String(stats.apps) : '-', '覆盖的应用数量'],
  ];
  document.getElementById('metricCards').innerHTML = cards.map(c => `<div class="metric"><div class="k">${{c[0]}}</div><div class="v">${{c[1]}}</div><div class="d">${{c[2]}}</div></div>`).join('');
}}

function aggregateByDate(rows) {{
  const map = new Map();
  rows.forEach(r => {{
    if (!map.has(r.date)) map.set(r.date, {{date:r.date, actual:0, pred:0, pred2:0, n:0, apeSum:0}});
    const x = map.get(r.date);
    x.actual += r.actual;
    x.pred += r.pred;
    x.pred2 += Number.isFinite(r.pred2) ? r.pred2 : 0;
    x.n += 1;
  }});
  return [...map.values()].sort((a,b)=>a.date.localeCompare(b.date)).map(x => {{
    x.ape = x.actual > 1e-8 ? Math.abs((x.actual - x.pred) / x.actual) : 0;
    return x;
  }});
}}

function money(v) {{
  const x = Number(v);
  if (!Number.isFinite(x)) return '-';
  if (Math.abs(x) >= 10000) return (x / 10000).toFixed(2) + '万';
  return x.toFixed(0);
}}

function businessLayers(snapshot) {{
  const t = snapshot.business_trace || {{}};
  return [t.kpi_layer, t.forecast_layer, t.solver_layer, t.risk_layer, t.decision_layer].filter(Boolean);
}}

function renderBusinessSnapshot() {{
  const rows = appMonthRoiRows();
  document.getElementById('mpcSourceHint').textContent = `推荐来源：${{payload.app_month_roi.source || '暂无'}}`;
  if (!rows.length) {{
    document.getElementById('businessCards').innerHTML = '<div class="metric"><div class="k">暂无推荐数据</div><div class="v">-</div><div class="d">请先生成应用层月末 ROI 预测结果</div></div>';
    return;
  }}
  const kpi = num(payload.app_month_roi.kpi || 1.05);
  const avgRoi = rows.reduce((s,r)=>s+r.monthEndRoi,0) / rows.length;
  const totalPlan = rows.reduce((s,r)=>s+r.plannedBudget,0);
  const totalMonthSpend = rows.reduce((s,r)=>s+r.monthEndSpend,0);
  const actionCounts = new Map();
  rows.forEach(r => actionCounts.set(r.action, (actionCounts.get(r.action) || 0) + 1));
  const topAction = [...actionCounts.entries()].sort((a,b)=>b[1]-a[1])[0] || ['-', 0];
  const riskRows = rows.filter(r => (r.alertCount > 0 || r.roiGap < 0) && r.plannedBudget > 0 && r.actualSpend > 0);
  const cards = [
    ['应用数', String(rows.length), `KPI：${{pct(kpi)}}`],
    ['平均预测月末 ROI', fmt(avgRoi,4), `相对 KPI：${{pct(avgRoi - kpi)}}`],
    ['今日计划预算', money(totalPlan), `预测月末消耗：${{money(totalMonthSpend)}}`],
    ['主要推荐动作', topAction[0], `覆盖应用数：${{topAction[1]}}`],
  ];
  document.getElementById('businessCards').innerHTML = cards.map(c => `<div class="metric"><div class="k">${{c[0]}}</div><div class="v">${{c[1]}}</div><div class="d">${{c[2]}}</div></div>`).join('');

  const actionData = [...actionCounts.entries()].map(([name, value]) => ({{name, value}}));
  charts.budget.setOption({{
    color: ['#2563eb', '#16a34a', '#f59e0b', '#7c3aed'],
    tooltip: {{ trigger:'item', formatter: p => `${{p.name}}<br/>应用数：${{p.value}} (${{fmt(p.percent,1)}}%)` }},
    legend: {{ bottom: 8 }},
    title: {{ text:'推荐动作分布', subtext:'按应用数汇总', left:16, top:12, textStyle:{{fontSize:16}} }},
    series: [{{ type:'pie', radius:['42%','68%'], center:['50%','54%'], data: actionData }}]
  }});

  const topRows = [...rows].sort((a,b)=>b.priorityScore-a.priorityScore).slice(0, 12);
  charts.product.setOption({{
    color: ['#2563eb', '#16a34a'],
    tooltip: {{ trigger:'axis' }},
    legend: {{ top: 8 }},
    grid: {{ left: 72, right: 24, top: 56, bottom: 74 }},
    xAxis: {{ type:'category', axisLabel:{{rotate:25}}, data:topRows.map(r=>r.app) }},
    yAxis: [
      {{ type:'value' }},
      {{ type:'value' }}
    ],
    series: [
      {{ name:'今日计划预算', type:'bar', data:topRows.map(r=>r.plannedBudget) }},
      {{ name:'预测月末ROI', type:'line', yAxisIndex:1, data:topRows.map(r=>r.monthEndRoi) }},
    ]
  }});

  const factorCounts = new Map();
  rows.forEach(r => factorCounts.set(r.dominantFactor, (factorCounts.get(r.dominantFactor) || 0) + 1));
  charts.layer.setOption({{
    color: ['#2563eb'],
    tooltip: {{ trigger:'axis' }},
    grid: {{ left: 72, right: 24, top: 46, bottom: 54 }},
    xAxis: {{ type:'category', data:[...factorCounts.keys()] }},
    yAxis: {{ type:'value' }},
    series: [{{ name:'应用数', type:'bar', data:[...factorCounts.values()] }}]
  }});

  const topRecommend = [...rows].sort((a,b)=>b.priorityScore-a.priorityScore).slice(0, 5)
    .map(r => `<li><b>${{r.app}}</b>：${{r.action}}；${{r.basis}}</li>`).join('');
  const riskList = [...riskRows].sort((a,b)=>b.priorityScore-a.priorityScore).slice(0, 5).map(r => `<li>${{r.app}}：ROI=${{fmt(r.monthEndRoi,4)}}，告警${{r.alertCount}}次，建议${{r.action}}</li>`).join('');

  const kpiOk = avgRoi >= kpi;
  const kpiStatusCls = kpiOk ? 's-ok' : 's-warn';
  const kpiStatusText = kpiOk ? '达标' : '未达标';
  const kpiGapText = kpiOk ? `超出 ${{pct(avgRoi - kpi)}}` : `差距 ${{pct(kpi - avgRoi)}}`;

  document.getElementById('decisionNote').innerHTML = `
    <div class="note-section">
      <div class="note-label">KPI 状态</div>
      <div class="note-body">
        平均月末 ROI = <b>${{fmt(avgRoi,4)}}</b>，KPI = <b>${{fmt(kpi,4)}}</b>，
        <span class="s ${{kpiStatusCls}}">${{kpiStatusText}}</span>
        （${{kpiGapText}}）
      </div>
    </div>
    <div class="note-section">
      <div class="note-label">重点推荐</div>
      <ol class="note-list">${{topRecommend || '<li>暂无推荐</li>'}}</ol>
    </div>
    <div class="note-section">
      <div class="note-label">风险提示</div>
      ${{riskRows.length
        ? `<ol class="note-list note-list-risk">${{riskList}}</ol>`
        : '<div class="note-body" style="color:#15803d;">暂无风险应用</div>'
      }}
    </div>
  `;

  // ---- 六层交互面板 ----
  const aboveKpiCount = rows.filter(r => r.monthEndRoi >= kpi).length;
  const belowKpiCount = rows.length - aboveKpiCount;
  const factorEntries = [...factorCounts.entries()];

  const layers = [
    {{
      id: 'kpi', icon: '🎯', name: 'KPI 层 — 做什么',
      status: kpiOk ? 'OK' : 'WARN',
      summary: `月末ROI均值 ${{fmt(avgRoi,4)}}` + (kpiOk ? ' ≥' : ' <') + ` KPI ${{fmt(kpi,4)}}，${{aboveKpiCount}}/${{rows.length}} 应用达标`,
      body: `
        <div class="layer-stat">
          <div class="layer-stat-item"><div class="sv">${{fmt(avgRoi,4)}}</div><div class="sk">平均预测月末ROI</div></div>
          <div class="layer-stat-item"><div class="sv">${{aboveKpiCount}}</div><div class="sk">达标应用数（ROI≥${{fmt(kpi,4)}}）</div></div>
          <div class="layer-stat-item"><div class="sv" style="color:${{belowKpiCount > 0 ? 'var(--red)' : 'var(--green)'}}">${{belowKpiCount}}</div><div class="sk">未达标应用数</div></div>
          <div class="layer-stat-item"><div class="sv">${{pct(avgRoi - kpi)}}</div><div class="sk">相对KPI差距</div></div>
        </div>
        <table><thead><tr><th>主因</th><th>应用数</th><th>占比</th><th>含义</th></tr></thead><tbody>
          ${{factorEntries.sort((a,b)=>b[1]-a[1]).map(([k,v]) => {{
            const meaning = {{SPEND:'消耗主导',D1_ANCHOR:'D1锚点主导',CURVE:'长尾曲线主导',BALANCED:'多因子均衡'}}[k] || k;
            return `<tr><td><b>${{k}}</b></td><td>${{v}}</td><td>${{fmt(v/rows.length*100,1)}}%</td><td>${{meaning}}</td></tr>`;
          }}).join('')}}
        </tbody></table>
      `
    }},
    {{
      id: 'time', icon: '📅', name: '时间层 — 什么时候冲',
      status: 'INFO',
      summary: '日历驱动：周末/节假日/调休日识别 + scale因子调节预算节奏',
      body: `
        <table><thead><tr><th>日期类型</th><th>scale 因子</th><th>策略</th></tr></thead><tbody>
          <tr><td>工作日</td><td>1.00</td><td>基准节奏，均匀消耗</td></tr>
          <tr><td>周末</td><td>1.08</td><td>流量高约8%，适度多投</td></tr>
          <tr><td>节假日</td><td>1.12</td><td>流量高且持续多天，提前蓄量</td></tr>
          <tr><td>寒暑假</td><td>1.10</td><td>持续时间长，长期预算规划窗口</td></tr>
        </tbody></table>
      `
    }},
    {{
      id: 'alloc', icon: '📊', name: '配置层 — 钱分给谁',
      status: 'OK',
      summary: `主要动作 <b>${{topAction[0]}}</b>（${{topAction[1]}}个应用）；全局ROI池化，A级补贴C级`,
      body: `
        <div class="layer-stat">
          <div class="layer-stat-item"><div class="sv">${{totalPlan.toFixed(0)}}</div><div class="sk">今日计划预算总额</div></div>
          <div class="layer-stat-item"><div class="sv">${{topAction[0]}}</div><div class="sk">主要推荐动作</div></div>
          <div class="layer-stat-item"><div class="sv">${{topAction[1]}}</div><div class="sk">覆盖应用数（${{fmt(topAction[1]/rows.length*100,1)}}%）</div></div>
          <div class="layer-stat-item"><div class="sv">${{actionCounts.size}}</div><div class="sk">动作种类数</div></div>
        </div>
        <table><thead><tr><th>动作</th><th>应用数</th><th>占比</th></tr></thead><tbody>
          ${{[...actionCounts.entries()].sort((a,b)=>b[1]-a[1]).map(([k,v]) => `<tr><td>${{k}}</td><td>${{v}}</td><td>${{fmt(v/rows.length*100,1)}}%</td></tr>`).join('')}}
        </tbody></table>
      `
    }},
    {{
      id: 'rhythm', icon: '🔄', name: '节奏层 — 月内每天怎么排',
      status: avgRoi >= kpi + 0.05 ? 'RELEASE' : 'OK',
      summary: avgRoi >= kpi + 0.05
        ? `盈余充足（ROI盈余 ${{pct(avgRoi - kpi - 0.05)}}），可切换规模最大化模式`
        : '盈余不足，维持达标优先节奏；月末根据累计ROI动态切换',
      body: `
        <div class="layer-stat">
          <div class="layer-stat-item"><div class="sv">${{pct(avgRoi - kpi)}}</div><div class="sk">ROI盈余（>5pp可释放）</div></div>
          <div class="layer-stat-item"><div class="sv">${{totalMonthSpend.toFixed(0)}}</div><div class="sk">预测月末总消耗</div></div>
          <div class="layer-stat-item"><div class="sv">${{rows.filter(r => (r.mode||'').toUpperCase() === 'SCALE').length}}</div><div class="sk">放量模式应用数</div></div>
          <div class="layer-stat-item"><div class="sv">${{rows.filter(r => (r.mode||'').toUpperCase() === 'GUARD').length}}</div><div class="sk">保守模式应用数</div></div>
        </div>
        <div class="plain-note">U型节奏：月初冲量 → 月中平稳 → 月末在盈余充足时释放跨月蓄力。</div>
      `
    }},
    {{
      id: 'risk', icon: '🛡️', name: '风控层 — 出了偏差怎么办',
      status: riskRows.length ? 'WARN' : 'OK',
      summary: riskRows.length
        ? `${{riskRows.length}} 个风险应用（${{riskRows.filter(r=>r.roiGap<0).length}} 个低于KPI，${{riskRows.filter(r=>r.alertCount>0).length}} 个有告警）`
        : '全部应用达标且无告警，暂无风险',
      body: riskRows.length ? `
        <table><thead><tr><th>应用ID</th><th>月末ROI</th><th>ROI差距</th><th>告警数</th><th>建议动作</th></tr></thead><tbody>
          ${{[...riskRows].sort((a,b)=>b.priorityScore-a.priorityScore).slice(0, 15).map(r => `<tr>
            <td><b>${{r.app}}</b></td>
            <td class="${{r.roiGap >= 0 ? 'good' : 'bad'}}">${{fmt(r.monthEndRoi,4)}}</td>
            <td class="${{r.roiGap >= 0 ? 'good' : 'bad'}}">${{pct(r.roiGap)}}</td>
            <td>${{r.alertCount}}</td>
            <td><span class="pill ${{r.action.includes('控量')||r.action.includes('收紧') ? 'bad' : 'good'}}">${{r.action}}</span></td>
          </tr>`).join('')}}
        </tbody></table>
        ${{riskRows.length > 15 ? `<div class="hint" style="margin-top:8px;">仅展示前 15 个，共 ${{riskRows.length}} 个风险应用</div>` : ''}}
      ` : '<div class="note-body" style="color:#15803d;padding: 8px 0;">✅ 当前全部应用达标且无告警</div>'
    }},
    {{
      id: 'decision', icon: '🧭', name: '决策排序层 — 先定什么后定什么',
      status: 'OK',
      summary: `按ROI差距排序，前5优先：${{topRecommend ? '已列出' : '暂无'}}；产品×时间联合求解`,
      body: `
        <div class="plain-note" style="margin-bottom:12px;">按产品比例分配后依节假日节奏微调，优先处理ROI差距最大的应用。</div>
        <table><thead><tr><th>优先级</th><th>应用ID</th><th>月末ROI</th><th>ROI差距</th><th>建议动作</th></tr></thead><tbody>
          ${{[...rows].sort((a,b)=>b.priorityScore-a.priorityScore).slice(0, 10).map((r, i) => `<tr>
            <td>${{i + 1}}</td>
            <td><b>${{r.app}}</b></td>
            <td class="${{r.roiGap >= 0 ? 'good' : 'bad'}}">${{fmt(r.monthEndRoi,4)}}</td>
            <td class="${{r.roiGap >= 0 ? 'good' : 'bad'}}">${{pct(r.roiGap)}}</td>
            <td><span class="pill ${{r.action.includes('控量')||r.action.includes('收紧') ? 'bad' : 'good'}}">${{r.action}}</span></td>
          </tr>`).join('')}}
        </tbody></table>
      `
    }},
  ];

  document.getElementById('layerPanels').innerHTML = layers.map(l => {{
    const cls = l.status === 'CRITICAL' ? 's-critical' : (l.status === 'WARN' ? 's-warn' : (l.status === 'RELEASE' ? 's-release' : 's-ok'));
    return `<div class="layer-panel" id="lp-${{l.id}}">
      <div class="layer-header" data-layer="${{l.id}}">
        <span class="layer-icon">${{l.icon}}</span>
        <span class="layer-name">${{l.name}}</span>
        <span class="layer-summary">${{l.summary}}</span>
        <span class="layer-status s ${{cls}}">${{l.status}}</span>
        <span class="layer-arrow">▼</span>
      </div>
      <div class="layer-body">${{l.body}}</div>
    </div>`;
  }}).join('');

  // toggle expand/collapse
  document.querySelectorAll('.layer-header').forEach(h => {{
    h.addEventListener('click', () => {{
      const panel = h.parentElement;
      const wasOpen = panel.classList.contains('open');
      // close all
      document.querySelectorAll('.layer-panel.open').forEach(p => p.classList.remove('open'));
      // open clicked (unless it was open)
      if (!wasOpen) panel.classList.add('open');
    }});
  }});
}}

function renderCharts(rows) {{
  const target = document.getElementById('targetSel').value;
  const unit = target === 'roi' ? 'ROI_D1' : '消耗金额';
  // All-apps: use pre-computed full chart_data; single-app: fetch per-app chart data
  if (selectedAppId) {{
    renderChartsForApp(target, unit);
    return;
  }}
  const chartData = (target === 'roi' ? payload.roi.chart_data : payload.spend.chart_data) || [];
  const data = chartData.length > 0 ? chartData : aggregateByDate(rows);
  renderChartSeries(data, unit, target);
}}

let _chartAppId = '';
function renderChartsForApp(target, unit) {{
  if (_chartAppId === selectedAppId) return;  // already loading/loaded
  _chartAppId = selectedAppId;
  fetch(`/web/data/${{target}}/app/${{encodeURIComponent(selectedAppId)}}/chart`)
    .then(r => r.json())
    .then(data => {{
      if (selectedAppId !== _chartAppId) return;  // selection changed during fetch
      renderChartSeries(data, unit, target);
    }});
}}

function renderChartSeries(data, unit, target) {{
  const dates = data.map(x => x.date);
  const actualName = target === 'roi' ? '实际 T+1 ROI_D1' : '实际 T+1 Spend';
  const predName = target === 'roi' ? '预测 T+1 ROI_D1' : '目标日历校准预测 T+1 Spend';
  const titleSuffix = selectedAppId ? ` — 应用 ${{selectedAppId}}` : '';
  const series = [
    {{ name: actualName, type: 'line', smooth: true, symbolSize: 6, data: data.map(x => x.actual) }},
    {{ name: predName, type: 'line', smooth: true, symbolSize: 6, data: data.map(x => x.pred) }},
  ];
  // CQR prediction interval bands for spend chart (all-apps only, CQR data is aggregated)
  if (target === 'spend' && !selectedAppId && payload.spend.cqr_predictions) {{
    const cqrData = payload.spend.cqr_predictions;
    const cqrLower = dates.map(d => cqrData[d] ? cqrData[d].y_lower : null);
    const cqrUpper = dates.map(d => cqrData[d] ? cqrData[d].y_upper : null);
    const cqrPred  = dates.map(d => cqrData[d] ? cqrData[d].y_pred  : null);
    // remove the predName series so we can re-order
    const predSeries = series.pop();
    series.push({{
      name: '区间下界', type: 'line', data: cqrLower,
      lineStyle: {{opacity: 0}}, stack: 'cqr-band', symbol: 'none',
      silent: true, emphasis: {{disabled: true}},
    }});
    series.push({{
      name: '预测区间(P05-P95)', type: 'line', data: cqrUpper,
      lineStyle: {{opacity: 0}},
      areaStyle: {{color: 'rgba(66,133,244,0.15)'}},
      stack: 'cqr-band', symbol: 'none', silent: true,
    }});
    series.push({{
      name: '预测值(CQR-P50)', type: 'line', data: cqrPred,
      lineStyle: {{width: 2, type: 'dashed', color: '#4285f4'}},
      itemStyle: {{color: '#4285f4'}}, symbol: 'none',
    }});
    series.push(predSeries);
  }}
  charts.line.setOption({{
    color: ['#16a34a', '#2563eb', '#f59e0b', '#8ab4f8'],
    tooltip: {{
      trigger: 'axis',
      formatter: function(params) {{
        let html = '<b>' + params[0].axisValue + '</b><br/>';
        let cqrVal = null;
        for (let i = 0; i < params.length; i++) {{
          const p = params[i];
          if (p.seriesName === '区间下界' || p.seriesName === '预测区间(P05-P95)') continue;
          if (p.seriesName === '预测值(CQR-P50)') {{ cqrVal = p.value; continue; }}
          html += p.marker + p.seriesName + ': ' + fmt(p.value, target === 'roi' ? 4 : 2) + '<br/>';
        }}
        if (cqrVal !== null && target === 'spend' && payload.spend.cqr_predictions) {{
          const c = payload.spend.cqr_predictions[params[0].axisValue];
          if (c) {{
            html += '<span style="color:#888;font-size:11px;">预测区间: ' + Math.round(c.y_lower) + ' ~ ' + Math.round(c.y_upper) + '</span><br/>';
          }}
        }}
        return html;
      }}
    }},
    legend: {{ top: 8 }},
    title: {{ text: (target === 'roi' ? 'T+1 ROI_D1' : 'T+1 Spend') + titleSuffix, left: 16, top: 6, textStyle: {{fontSize:14}} }},
    grid: {{ left: 64, right: 28, top: 58, bottom: 55 }},
    xAxis: {{ type: 'category', data: dates }},
    yAxis: {{ type: 'value' }},
    dataZoom: [{{type:'inside'}}, {{type:'slider', height: 18, bottom: 12}}],
    series
  }});
  charts.scatter.setOption({{
    color: ['#2563eb'],
    tooltip: {{ formatter: p => `实际：${{fmt(p.value[0], target==='roi'?4:2)}}<br/>预测：${{fmt(p.value[1], target==='roi'?4:2)}}` }},
    grid: {{ left: 62, right: 24, top: 36, bottom: 54 }},
    xAxis: {{ type:'value' }},
    yAxis: {{ type:'value' }},
    series: [{{ name:'每日预测校准', type:'scatter', symbolSize: 9, data: data.map(x => [x.actual, x.pred]) }}]
  }});
  charts.error.setOption({{
    color: ['#dc2626'],
    tooltip: {{ trigger:'axis', valueFormatter: v => fmt(v, 2) + '%' }},
    grid: {{ left: 64, right: 28, top: 36, bottom: 55 }},
    xAxis: {{ type:'category', data: dates }},
    yAxis: {{ type:'value' }},
    dataZoom: [{{type:'inside'}}, {{type:'slider', height: 18, bottom: 12}}],
    series: [{{ name:'每日汇总绝对百分比误差', type:'bar', data: data.map(x => x.ape * 100) }}]
  }});
}}

function showAppPopup(appId, rowData, evt) {{
  clearTimeout(_popupHideTimer);
  const popup = document.getElementById('appHoverPopup');
  popup.style.display = 'block';
  positionPopup(evt);
  if (_popupAppId === appId) return;
  _popupAppId = appId;
  document.getElementById('popupAppId').innerHTML = '<span class="popup-app-id" id="popupAppIdLink">' + appId + '</span>';
  document.getElementById('popupAppIdLink').addEventListener('click', (e) => {{
    e.stopPropagation();
    hideAppPopup(true);
    selectedAppId = appId;
    _chartAppId = '';
    detailState = {{ appId: appId, page: 0, limit: 200, total: 0 }};
    detailSort = {{ key: 'date', dir: 'asc' }};
    document.getElementById('targetSel').value = 'spend';
    setActiveView('predict');
    document.getElementById('appSelectHint').style.display = '';
    document.getElementById('appSelectText').textContent = '已选中应用 ' + appId + ' — 图表和明细均仅显示该应用数据';
    render();
  }});
  // 解析 Python 风格字符串 → 提取基本信息
  const parsePythonList = (s) => {{
    if (!s || s === '-') return [];
    try {{ return JSON.parse(s.replace(/'/g, '"').replace(/None/g, 'null').replace(/True/g, 'true').replace(/False/g, 'false')); }} catch(e) {{ return []; }}
  }};
  const alertsList = parsePythonList(rowData.alertsRaw || '');
  const suggestionsList = parsePythonList(rowData.topSuggestionsRaw || '');

  let alertsHtml = '';
  if (alertsList.length > 0) {{
    alertsHtml = '<div style="margin-top:4px;font-size:11px;color:#dc2626;">';
    alertsList.forEach(a => {{
      alertsHtml += '<div>' + (a.level||'') + ': ' + (a.message||'') + '</div>';
    }});
    alertsHtml += '</div>';
  }}

  let suggestionsHtml = '';
  if (suggestionsList.length > 0) {{
    suggestionsHtml = '<div style="margin-top:4px;font-size:11px;line-height:1.5;">';
    suggestionsList.slice(0, 3).forEach(s => {{
      suggestionsHtml += '<div>' + (s.slot||'') + ' | 预算' + (s.suggested_budget||'-') + ' | ROI ' + (s.expected_roi||'-') + '</div>';
    }});
    suggestionsHtml += '</div>';
  }}

  const actionLabel = rowData.action || '-';
  document.getElementById('popupInfo').innerHTML =
    '<div style="font-size:13px;font-weight:600;margin-bottom:4px;">决策: <span style="color:' + (actionLabel.includes('控')||actionLabel.includes('停') ? '#dc2626' : '#16a34a') + '">' + actionLabel + '</span></div>' +
    '<div style="font-size:11px;line-height:1.55;color:var(--muted);">' + (rowData.basis || '-') + '</div>' +
    alertsHtml +
    '<div style="margin-top:4px;font-size:11px;color:var(--muted);">分场景推荐:</div>' +
    suggestionsHtml;

  // init or re-use charts
  if (!_popupCharts.roi) _popupCharts.roi = echarts.init(document.getElementById('popupRoiChart'));
  if (!_popupCharts.spend) _popupCharts.spend = echarts.init(document.getElementById('popupSpendChart'));
  _popupCharts.roi.setOption({{ title: {{ text: 'T+1 ROI_D1', left: 8, top: 4, textStyle: {{fontSize:12}} }}, tooltip: {{ trigger:'axis' }}, grid: {{ left: 52, right: 20, top: 32, bottom: 34 }}, xAxis: {{ type:'category', data: [], axisLabel:{{fontSize:10, rotate:20}} }}, yAxis: {{ type:'value', axisLabel:{{fontSize:10}} }}, series: [] }});
  _popupCharts.spend.setOption({{ title: {{ text: 'T+1 Spend', left: 8, top: 4, textStyle: {{fontSize:12}} }}, tooltip: {{ trigger:'axis' }}, grid: {{ left: 52, right: 20, top: 32, bottom: 34 }}, xAxis: {{ type:'category', data: [], axisLabel:{{fontSize:10, rotate:20}} }}, yAxis: {{ type:'value', axisLabel:{{fontSize:10}} }}, series: [] }});

  _popupCharts.roi.showLoading();
  _popupCharts.spend.showLoading();
  Promise.all([
    fetch('/web/data/roi/app/' + encodeURIComponent(appId) + '/chart').then(r => r.json()).catch(() => []),
    fetch('/web/data/spend/app/' + encodeURIComponent(appId) + '/chart').then(r => r.json()).catch(() => []),
  ]).then(([roiData, spendData]) => {{
    if (_popupAppId !== appId) return;
    _renderPopupChart(_popupCharts.roi, roiData, 'ROI', false);
    _renderPopupChart(_popupCharts.spend, spendData, '消耗', true);
  }});
}}

function _renderPopupChart(chart, data, unit, isSpend) {{
  chart.hideLoading();
  const dates = data.map(x => x.date);
  const actualName = isSpend ? '实际Spend' : '实际ROI';
  const predName = isSpend ? '预测Spend' : '预测ROI';
  chart.setOption({{
    xAxis: {{ data: dates }},
    series: [
      {{ name: actualName, type: 'line', smooth: true, symbolSize: 3, data: data.map(x => x.actual) }},
      {{ name: predName, type: 'line', smooth: true, symbolSize: 3, data: data.map(x => x.pred) }},
    ],
    color: ['#16a34a', '#2563eb'],
    legend: {{ bottom: 6, textStyle: {{fontSize:10}} }},
  }});
}}

function positionPopup(evt) {{
  const popup = document.getElementById('appHoverPopup');
  const pw = popup.offsetWidth;
  const ph = popup.offsetHeight;
  let left = evt.clientX + 14;
  let top = evt.clientY - 12;
  if (left + pw > window.innerWidth - 12) left = window.innerWidth - pw - 12;
  if (top + ph > window.innerHeight - 12) top = window.innerHeight - ph - 12;
  if (left < 4) left = 4;
  if (top < 4) top = 4;
  popup.style.left = left + 'px';
  popup.style.top = top + 'px';
}}

function hideAppPopup(immediate) {{
  const doHide = () => {{
    document.getElementById('appHoverPopup').style.display = 'none';
    _popupAppId = '';
    if (_popupCharts.roi) {{ _popupCharts.roi.dispose(); _popupCharts.roi = null; }}
    if (_popupCharts.spend) {{ _popupCharts.spend.dispose(); _popupCharts.spend = null; }}
  }};
  if (immediate) {{ doHide(); return; }}
  _popupHideTimer = setTimeout(doHide, 200);
}}

document.getElementById('appHoverPopup').addEventListener('mouseenter', () => clearTimeout(_popupHideTimer));
document.getElementById('appHoverPopup').addEventListener('mouseleave', hideAppPopup);

function renderSummary(rows) {{
  const target = document.getElementById('targetSel').value;
  const m = calcMetrics(rows);
  const hint = target === 'roi' ? payload.roi.source : payload.spend.source;
  document.getElementById('sourceHint').textContent = `数据来源：${{hint || '暂无'}}`;
  if (target === 'roi') {{
    const best = (payload.roi.metrics || [])[0] || {{}};
    document.getElementById('modelSummary').innerHTML = `当前 ROI_D1 最优结果来自 <b>${{best.model || '模型输出'}}</b>：MAE=${{fmt(num(best.mae),4)}}，RMSE=${{fmt(num(best.rmse),4)}}，MAPE=${{pct(num(best.mape))}}。本页图表展示“下一天 ROI_D1 预测值”和“下一天实际 ROI_D1”，用于判断模型是否贴近真实投放结果。`;
  }} else {{
    const r = payload.spend.report || {{}};
    const holidayRows = rows.filter(x => x.calendarLabel.includes('清明') || x.calendarLabel.includes('假期') || x.calendarLabel.includes('节后'));
    const avgImprove = holidayRows.length ? holidayRows.reduce((s,x)=>s+(Number.isFinite(x.improvement)?x.improvement:0),0)/holidayRows.length : NaN;
    document.getElementById('modelSummary').innerHTML = `Spend 当前使用 <b>目标日历特征 + 假期切换校准</b>：最优模型=${{r.best_model_by_mape || '模型输出'}}，MAPE=${{fmt(num(r.best_mape_pct),2)}}%。表格已增加“旧预测/旧APE/新APE/改善幅度/T+1日历标签”，清明与节后样本平均改善=${{pct(avgImprove)}}。${{payload.spend.cqr_coverage != null ? ` CQR预测区间覆盖率=${{pct(payload.spend.cqr_coverage)}}。` : ''}}`;
  }}
}}

function renderTable(rows) {{
  const target = document.getElementById('targetSel').value;
  // Single-app mode: AJAX detail
  if (selectedAppId) {{
    if (detailState.appId !== selectedAppId) {{ detailState.appId = selectedAppId; detailState.page = 0; detailState.total = 0; }}
    loadAppDetail();
    return;
  }}
  // All-apps mode: in-memory pagination
  detailState.appId = '';
  const digits = target === 'roi' ? 4 : 2;
  const appSearch = (document.getElementById('detailAppSearch')?.value || '').trim().toLowerCase();
  let enriched = rows.map(r => ({{
    ...r,
    ape: r.actual > 1e-8 ? Math.abs((r.actual-r.pred)/r.actual) : 0,
  }}));
  if (appSearch) {{
    enriched = enriched.filter(r => String(r.app).toLowerCase().includes(appSearch));
  }}
  const headers = target === 'roi'
    ? [['date','日期'], ['app','应用ID'], ['actual','实际 ROI_D1'], ['pred','预测 ROI_D1'], ['ape','APE'], ['model','模型']]
    : [['date','日期'], ['targetDate','T+1日期'], ['calendarLabel','日历标签'], ['app','应用ID'], ['actual','实际 Spend'],
      ['pred','新版预测'], ['oldPred','旧版预测'], ['newApe','新版APE'], ['oldApe','旧版APE'], ['improvement','改善'], ['model','模型']];
  document.getElementById('detailHead').innerHTML = '<tr>' + headers.map(([key, name]) => {{
    const mark = detailSort.key === key ? (detailSort.dir === 'asc' ? ' ↑' : ' ↓') : ' ↕';
    return `<th class="sortable" data-detail-sort="${{key}}">${{name}}${{mark}}</th>`;
  }}).join('') + '</tr>';
  document.querySelectorAll('#detailHead th[data-detail-sort]').forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.dataset.detailSort;
      if (detailSort.key === key) detailSort.dir = detailSort.dir === 'asc' ? 'desc' : 'asc';
      else detailSort = {{ key, dir: ['date','app','model','calendarLabel','targetDate'].includes(key) ? 'asc' : 'desc' }};
      detailState.page = 0;
      renderTable(currentRows());
    }});
  }});
  const sorted = [...enriched].sort((a,b) => compareRows(a, b, detailSort));
  const limit = detailState.limit;
  const total = sorted.length;
  const totalPages = Math.max(1, Math.ceil(total / limit));
  if (detailState.page >= totalPages) detailState.page = totalPages - 1;
  if (detailState.page < 0) detailState.page = 0;
  const offset = detailState.page * limit;
  const pageRows = sorted.slice(offset, offset + limit);
  document.getElementById('detailBody').innerHTML = pageRows.map(r => {{
    const ape = r.ape;
    const cls = ape <= .3 ? 'good' : 'bad';
    if (target === 'roi') return `<tr><td>${{r.date}}</td><td>${{r.app}}</td><td>${{fmt(r.actual,digits)}}</td><td>${{fmt(r.pred,digits)}}</td><td class="${{cls}}">${{pct(ape)}}</td><td>${{r.model}}</td></tr>`;
    const impCls = Number.isFinite(r.improvement) && r.improvement >= 0 ? 'good' : 'bad';
    return `<tr><td>${{r.date}}</td><td>${{r.targetDate}}</td><td>${{r.calendarLabel}}</td><td>${{r.app}}</td><td>${{fmt(r.actual,digits)}}</td><td>${{fmt(r.pred,digits)}}</td><td>${{fmt(r.oldPred,digits)}}</td><td class="${{cls}}">${{pct(r.newApe)}}</td><td>${{pct(r.oldApe)}}</td><td class="${{impCls}}">${{pct(r.improvement)}}</td><td>${{r.model}}</td></tr>`;
  }}).join('');
  document.getElementById('detailPrevBtn').disabled = detailState.page <= 0;
  document.getElementById('detailNextBtn').disabled = detailState.page >= totalPages - 1;
  document.getElementById('detailPageLabel').textContent = `第 ${{detailState.page + 1}}/${{totalPages}} 页 (共 ${{total}} 条)`;
  document.getElementById('detailTitleSuffix').textContent = '';
  document.getElementById('detailPageInfo').textContent = '';
}}

function render() {{
  renderAppSummary();
  const rows = currentRows();
  const target = document.getElementById('targetSel').value;
  document.getElementById('lineTitle').textContent = target === 'roi' ? 'T+1 ROI_D1：预测值 vs 实际值' : 'T+1 Spend：预测值 vs 实际值';
  renderBusinessSnapshot();
  renderCards(rows);
  renderCharts(rows);
  renderAppMonthRoiTable();
  renderSummary(rows);
  renderTable(rows);
}}

['targetSel','startDate','endDate'].forEach(id => {{
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', () => {{ selectedAppId = ''; _chartAppId = ''; document.getElementById('appSelectHint').style.display = 'none'; detailState = {{ appId: '', page: 0, limit: 200, total: 0 }}; render(); }});
}});
const appTableSearchEl = document.getElementById('appTableSearch');
if (appTableSearchEl) appTableSearchEl.addEventListener('input', renderAppSummary);
const detailAppSearchEl = document.getElementById('detailAppSearch');
if (detailAppSearchEl) detailAppSearchEl.addEventListener('input', () => {{ detailState.page = 0; renderTable(currentRows()); }});
const clearAppSelBtn = document.getElementById('clearAppSelBtn');
if (clearAppSelBtn) clearAppSelBtn.addEventListener('click', () => {{
  selectedAppId = '';
  document.getElementById('appSelectHint').style.display = 'none';
  detailState = {{ appId: '', page: 0, limit: 200, total: 0 }};
  detailSort = {{ key: 'date', dir: 'asc' }};
  render();
}});
const detailPrevBtn = document.getElementById('detailPrevBtn');
if (detailPrevBtn) detailPrevBtn.addEventListener('click', () => {{
  if (detailState.page > 0) {{ detailState.page--; }}
  if (selectedAppId) {{ loadAppDetail(); }} else {{ renderTable(currentRows()); }}
}});
const detailNextBtn = document.getElementById('detailNextBtn');
if (detailNextBtn) detailNextBtn.addEventListener('click', () => {{
  detailState.page++;
  if (selectedAppId) {{ loadAppDetail(); }} else {{ renderTable(currentRows()); }}
}});
['appRoiSearch','appRoiKpiStatus','appRoiAction','appRoiRisk','appRoiMin','appRoiMax'].forEach(id => {{
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', renderAppMonthRoiTable);
}});
document.querySelectorAll('#appRoiQuickFilters .chip').forEach(btn => {{
  btn.addEventListener('click', () => {{
    appRoiQuickFilter = btn.dataset.filter || 'all';
    document.querySelectorAll('#appRoiQuickFilters .chip').forEach(x => x.classList.toggle('active', x === btn));
    renderAppMonthRoiTable();
  }});
}});
const advToggle = document.getElementById('appRoiAdvancedToggle');
if (advToggle) advToggle.addEventListener('click', () => {{
  const pane = document.getElementById('appRoiAdvancedFilters');
  if (pane) pane.classList.toggle('open');
}});
const appRoiResetBtn = document.getElementById('appRoiReset');
if (appRoiResetBtn) appRoiResetBtn.addEventListener('click', () => {{
  ['appRoiSearch','appRoiMin','appRoiMax'].forEach(id => {{ const el = document.getElementById(id); if (el) el.value = ''; }});
  const ks = document.getElementById('appRoiKpiStatus'); if (ks) ks.value = '';
  const ac = document.getElementById('appRoiAction'); if (ac) ac.value = '';
  const rk = document.getElementById('appRoiRisk'); if (rk) rk.value = '';
  appRoiQuickFilter = 'all';
  document.querySelectorAll('#appRoiQuickFilters .chip').forEach(x => x.classList.toggle('active', x.dataset.filter === 'all'));
  appRoiSort = {{ key: 'priorityScore', dir: 'desc' }};
  renderAppMonthRoiTable();
}});
const viewRecommend = document.getElementById('view-recommend');
if (viewRecommend) {{
  viewRecommend.addEventListener('click', (e) => {{
    const th = e.target.closest && e.target.closest('#appRoiHead th[data-sort]');
    if (!th) return;
    const key = th.dataset.sort;
    if (!key) return;
    if (appRoiSort.key === key) appRoiSort.dir = appRoiSort.dir === 'asc' ? 'desc' : 'asc';
    else appRoiSort = {{
      key,
      dir: key === 'monthEndRoi' || key === 'roiGap' || key === 'monthEndRoiStable' || key === 'monthEndRoiPredOnly'
        || key === 'monthEndRoiFused' || key === 'canonicalLabel'
        ? 'asc'
        : 'desc',
    }};
    renderAppMonthRoiTable();
  }});
}}
function setActiveView(view) {{
  document.querySelectorAll('.navbtn').forEach(x => x.classList.toggle('active', x.dataset.view === view));
  document.getElementById('view-predict').classList.toggle('hidden', view !== 'predict');
  document.getElementById('view-recommend').classList.toggle('hidden', view !== 'recommend');
  document.getElementById('view-daily-revenue').classList.toggle('hidden', view !== 'daily-revenue');
  document.getElementById('view-monitor').classList.toggle('hidden', view !== 'monitor');
  hideAppPopup(true);
  setTimeout(() => {{ Object.values(charts).forEach(c => c.resize()); if (view === 'daily-revenue') {{ loadDailyRevenueChart(); loadDailyRevenueAppSummary(); }} if (view === 'monitor') loadMonitorView(); }}, 0);
}}
document.querySelectorAll('.navbtn').forEach(btn => btn.addEventListener('click', () => setActiveView(btn.dataset.view)));

async function loadDailyRevenueChart() {{
  const appId = document.getElementById('drAppSearch').value.trim();
  const params = appId ? `?app_id=${{encodeURIComponent(appId)}}` : '';
  const resp = await fetch(`/web/predictions/daily_revenue${{params}}`);
  const data = await resp.json();

  const chartDom = document.getElementById('dailyRevenueChart');
  const coChartDom = document.getElementById('dailyRevenueCarryoverChart');
  const statsDiv = document.getElementById('dailyRevenueStats');

  if (!data.rows || data.rows.length === 0) {{
    chartDom.innerHTML = '<div class="plain-note">暂无预测数据，请先执行每日预测回测。</div>';
    coChartDom.innerHTML = '';
    statsDiv.innerHTML = '';
    return;
  }}

  // Aggregate by date for both combined and carryover
  const dateMap = {{}};
  data.rows.forEach(r => {{
    const d = r['日期'];
    if (!dateMap[d]) dateMap[d] = {{ y_true: 0, y_pred: 0, co_true: 0, co_pred: 0 }};
    dateMap[d].y_true += parseFloat(r.y_true || 0);
    dateMap[d].y_pred += parseFloat(r.y_pred || 0);
    dateMap[d].co_true += parseFloat(r.y_true_carryover || 0);
    dateMap[d].co_pred += parseFloat(r.y_pred_carryover || 0);
  }});

  const dates = Object.keys(dateMap).sort();
  const yTrue = dates.map(d => dateMap[d].y_true);
  const yPred = dates.map(d => dateMap[d].y_pred);
  const coTrue = dates.map(d => dateMap[d].co_true);
  const coPred = dates.map(d => dateMap[d].co_pred);

  // 双口径 MAPE (days >= 30 filter)
  let combinedSum = 0, combinedCount = 0, carryoverSum = 0, carryoverCount = 0;
  data.rows.forEach(r => {{
    if (parseInt(r.days_since_start) >= 30 && parseFloat(r.y_true) > 0) {{
      combinedSum += Math.abs(parseFloat(r.y_true) - parseFloat(r.y_pred)) / parseFloat(r.y_true);
      combinedCount++;
    }}
    if (parseInt(r.days_since_start) >= 30 && parseFloat(r.y_true_carryover) > 0) {{
      carryoverSum += Math.abs(parseFloat(r.y_true_carryover) - parseFloat(r.y_pred_carryover)) / parseFloat(r.y_true_carryover);
      carryoverCount++;
    }}
  }});
  const combinedMape = combinedCount > 0 ? (combinedSum / combinedCount * 100).toFixed(1) : 'N/A';
  const carryoverMape = carryoverCount > 0 ? (carryoverSum / carryoverCount * 100).toFixed(1) : 'N/A';

  // D1 占比
  let d1Total = 0, buyTotal = 0;
  data.rows.forEach(r => {{ d1Total += parseFloat(r.y_true_d1||0); buyTotal += parseFloat(r.y_true||0); }});
  const d1Ratio = buyTotal > 0 ? (d1Total / buyTotal * 100).toFixed(1) : '0';

  statsDiv.innerHTML = `<b>${{data.total}}</b> 行 | <b>${{new Set(data.rows.map(r=>r['应用ID'])).size}}</b> 个应用`
    + ` | D1占比 <b>${{d1Ratio}}%</b>`
    + ` | 组合MAPE(D1已知) <b>${{combinedMape}}%</b>`
    + ` | CarryoverMAPE(曲线) <b>${{carryoverMape}}%</b>`;

  // ---- 上图：组合收入（D1 + Carryover） ----
  echarts.dispose(chartDom);
  const chart = echarts.init(chartDom);
  chart.setOption({{
    color: ['#16a34a', '#2563eb'],
    title: {{ text: '每日买量收入（D1 + Carryover）', left: 16, top: 4, textStyle: {{ fontSize: 13 }} }},
    tooltip: {{ trigger: 'axis' }},
    legend: {{ data: ['实际总收入', '预测总收入'], top: 4 }},
    grid: {{ left: 64, right: 28, top: 48, bottom: 48 }},
    xAxis: {{ type: 'category', data: dates, axisLabel: {{ fontSize: 10 }} }},
    yAxis: {{ type: 'value', axisLabel: {{ fontSize: 10 }} }},
    dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', height: 16, bottom: 8 }}],
    series: [
      {{ name: '实际总收入', type: 'line', data: yTrue, smooth: true,
        lineStyle: {{ width: 2 }}, symbol: 'none' }},
      {{ name: '预测总收入', type: 'line', data: yPred, smooth: true,
        lineStyle: {{ width: 2, type: 'dashed' }}, symbol: 'none' }},
    ],
  }});

  // ---- 下图：Carryover 尾量 ----
  echarts.dispose(coChartDom);
  const coChart = echarts.init(coChartDom);
  coChart.setOption({{
    color: ['#dc2626', '#f59e0b'],
    title: {{ text: 'Carryover 尾量（仅释放曲线预测部分）', left: 16, top: 4, textStyle: {{ fontSize: 13 }} }},
    tooltip: {{ trigger: 'axis' }},
    legend: {{ data: ['实际Carryover', '预测Carryover'], top: 4 }},
    grid: {{ left: 64, right: 28, top: 48, bottom: 48 }},
    xAxis: {{ type: 'category', data: dates, axisLabel: {{ fontSize: 10 }} }},
    yAxis: {{ type: 'value', axisLabel: {{ fontSize: 10 }} }},
    dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', height: 16, bottom: 8 }}],
    series: [
      {{ name: '实际Carryover', type: 'line', data: coTrue, smooth: true,
        lineStyle: {{ width: 2 }}, symbol: 'none' }},
      {{ name: '预测Carryover', type: 'line', data: coPred, smooth: true,
        lineStyle: {{ width: 2, type: 'dashed' }}, symbol: 'none' }},
    ],
  }});

  window.addEventListener('resize', () => {{ chart.resize(); coChart.resize(); }});
}}

async function loadDailyRevenueAppSummary() {{
  const wrap = document.getElementById('dailyRevenueAppSummary');
  wrap.innerHTML = '<div class="plain-note">加载中...</div>';
  const resp = await fetch('/web/predictions/daily_revenue');
  const data = await resp.json();
  if (!data.rows || data.rows.length === 0) {{
    wrap.innerHTML = '<div class="plain-note">暂无数据</div>';
    return;
  }}

  // Per-app aggregation (only days >= 30 for stable metrics)
  const appMap = {{}};
  data.rows.forEach(r => {{
    const app = r['应用ID'];
    if (!appMap[app]) appMap[app] = {{ co_sum:0, co_count:0, comb_sum:0, comb_count:0, total_buy:0, total_d1:0, n:0 }};
    const a = appMap[app];
    const yt = parseFloat(r.y_true||0), yp = parseFloat(r.y_pred||0);
    const coT = parseFloat(r.y_true_carryover||0), coP = parseFloat(r.y_pred_carryover||0);
    const d1 = parseFloat(r.y_true_d1||0);
    if (parseInt(r.days_since_start) >= 30) {{
      if (yt > 0) {{ a.comb_sum += Math.abs(yt-yp)/yt; a.comb_count++; }}
      if (coT > 0) {{ a.co_sum += Math.abs(coT-coP)/coT; a.co_count++; }}
    }}
    a.total_buy += yt;
    a.total_d1 += d1;
    a.n++;
  }});

  const apps = Object.entries(appMap)
    .map(([id, a]) => ({{
      id,
      n: a.n,
      co_mape: a.co_count > 0 ? a.co_sum / a.co_count * 100 : null,
      comb_mape: a.comb_count > 0 ? a.comb_sum / a.comb_count * 100 : null,
      total_buy: a.total_buy,
      d1_ratio: a.total_buy > 0 ? a.total_d1 / a.total_buy * 100 : 0,
    }}))
    .filter(a => a.co_mape !== null)
    .sort((a, b) => a.co_mape - b.co_mape);

  if (apps.length === 0) {{
    wrap.innerHTML = '<div class="plain-note">无足够样本（需 days≥30）</div>';
    return;
  }}

  const totalBuy = apps.reduce((s,a) => s + a.total_buy, 0);
  let html = '<table><thead><tr><th>#</th><th>应用ID</th><th class="num">样本</th><th class="num">Carryover MAPE</th><th class="num">组合 MAPE</th><th class="num">总买量收入</th><th class="num">收入占比</th><th class="num">D1占比</th></tr></thead><tbody>';
  apps.forEach((a, i) => {{
    const coCls = a.co_mape < 50 ? 's-ok' : a.co_mape < 100 ? '' : 'bad';
    const share = (a.total_buy / totalBuy * 100).toFixed(1);
    html += '<tr>'
      + '<td>' + (i+1) + '</td>'
      + '<td><a href="#" onclick="document.getElementById(&#39;drAppSearch&#39;).value=&#39;' + a.id + '&#39;;document.getElementById(&#39;drTableSearch&#39;).value=&#39;' + a.id + '&#39;;loadDailyRevenueChart();loadDailyRevenueTable();return false">' + a.id + '</a></td>'
      + '<td class="num">' + a.n + '</td>'
      + '<td class="num ' + coCls + '"><b>' + a.co_mape.toFixed(1) + '%</b></td>'
      + '<td class="num">' + (a.comb_mape !== null ? a.comb_mape.toFixed(1) + '%' : '-') + '</td>'
      + '<td class="num">' + a.total_buy.toLocaleString() + '</td>'
      + '<td class="num">' + share + '%</td>'
      + '<td class="num">' + a.d1_ratio.toFixed(1) + '%</td>'
      + '</tr>';
  }});
  html += '</tbody></table>';
  wrap.innerHTML = html;
}}

async function loadDailyRevenueTable() {{
  const appId = document.getElementById('drTableSearch').value.trim();
  const from = document.getElementById('drTableDateFrom').value;
  const to = document.getElementById('drTableDateTo').value;
  const minDays = parseInt(document.getElementById('drTableDays').value) || 0;
  const infoDiv = document.getElementById('drTableInfo');
  const wrap = document.getElementById('dailyRevenueTableWrap');

  const params = appId ? `?app_id=${{encodeURIComponent(appId)}}` : '';
  infoDiv.textContent = '加载中...';
  const resp = await fetch(`/web/predictions/daily_revenue${{params}}`);
  const data = await resp.json();

  if (!data.rows || data.rows.length === 0) {{
    wrap.innerHTML = '<div class="plain-note">暂无数据</div>';
    infoDiv.textContent = '';
    return;
  }}

  let rows = data.rows;
  if (from) rows = rows.filter(r => r['日期'] >= from);
  if (to) rows = rows.filter(r => r['日期'] <= to);
  if (minDays > 0) rows = rows.filter(r => parseInt(r.days_since_start) >= minDays);

  if (rows.length === 0) {{
    wrap.innerHTML = '<div class="plain-note">筛选后无数据</div>';
    infoDiv.textContent = '';
    return;
  }}

  const totalYTrue = rows.reduce((s,r) => s + parseFloat(r.y_true||0), 0);
  const totalYPred = rows.reduce((s,r) => s + parseFloat(r.y_pred||0), 0);
  const totalD1 = rows.reduce((s,r) => s + parseFloat(r.y_true_d1||0), 0);
  const totalCOTrue = rows.reduce((s,r) => s + parseFloat(r.y_true_carryover||0), 0);
  const totalCOPred = rows.reduce((s,r) => s + parseFloat(r.y_pred_carryover||0), 0);
  const mapeCombined = rows.filter(r => parseFloat(r.y_true) > 0).reduce((s,r) => s + Math.abs(parseFloat(r.y_true)-parseFloat(r.y_pred))/parseFloat(r.y_true), 0)
    / Math.max(1, rows.filter(r => parseFloat(r.y_true) > 0).length) * 100;
  const mapeCO = rows.filter(r => parseFloat(r.y_true_carryover) > 0).reduce((s,r) => s + Math.abs(parseFloat(r.y_true_carryover)-parseFloat(r.y_pred_carryover))/parseFloat(r.y_true_carryover), 0)
    / Math.max(1, rows.filter(r => parseFloat(r.y_true_carryover) > 0).length) * 100;
  const d1Ratio = totalYTrue > 0 ? (totalD1/totalYTrue*100).toFixed(1) : '0';
  infoDiv.innerHTML = `${{rows.length}} 行 | D1占比 ${{d1Ratio}}% | 组合MAPE=${{mapeCombined.toFixed(1)}}% | CarryoverMAPE=${{mapeCO.toFixed(1)}}%`;

  let html = '<table><thead><tr>';
  html += '<th>日期</th><th>应用ID</th><th class="num">实际收入</th><th class="num">预测收入</th><th class="num">误差%</th><th class="num">D1收入</th><th class="num">尾量实际</th><th class="num">尾量预测</th><th class="num">尾量误差%</th><th>累计天</th>';
  html += '</tr></thead><tbody>';

  const display = rows.slice(0, 2000);
  display.forEach(r => {{
    const yt = parseFloat(r.y_true||0), yp = parseFloat(r.y_pred||0);
    const err = yt > 0 ? ((Math.abs(yt-yp)/yt)*100).toFixed(1) : '-';
    const errCls = err!=='-' && parseFloat(err)>50 ? 'bad' : '';
    const d1 = parseFloat(r.y_true_d1||0);
    const coTrue = parseFloat(r.y_true_carryover||0);
    const coPred = parseFloat(r.y_pred_carryover||0);
    const coErr = coTrue > 0 ? ((Math.abs(coTrue-coPred)/coTrue)*100).toFixed(1) : '-';
    const coErrCls = coErr!=='-' && parseFloat(coErr)>100 ? 'bad' : '';
    const appId = r['应用ID'];
    html += '<tr><td>' + r['日期'] + '</td><td><a href="#" onclick="selectAppForDrTable(&#39;' + appId + '&#39;)">' + appId + '</a></td><td class="num">' + yt.toLocaleString() + '</td><td class="num">' + yp.toLocaleString() + '</td><td class="num ' + errCls + '">' + err + '</td><td class="num">' + d1.toLocaleString() + '</td><td class="num">' + coTrue.toLocaleString() + '</td><td class="num">' + coPred.toLocaleString() + '</td><td class="num ' + coErrCls + '">' + coErr + '</td><td class="num">' + r.days_since_start + '</td></tr>';
  }});
  html += '</tbody></table>';
  if (rows.length > 2000) html += '<div class="hint" style="margin-top:4px;">显示前 2000 行，共 ' + rows.length + ' 行</div>';
  wrap.innerHTML = html;
}}

function selectAppForDrTable(appId) {{
  document.getElementById('drTableSearch').value = appId;
  document.getElementById('drAppSearch').value = appId;
  loadDailyRevenueTable();
}}

async function loadDataStatus() {{
  try {{
    const resp = await fetch('/health');
    const h = await resp.json();
    const bar = document.getElementById('dataStatusBar');
    if (!bar) return;
    const d = h.data || {{}};
    const m = h.models || {{}};
    const dsDays = d.days_behind;
    let daysStyle = dsDays <= 1 ? 'var(--green)' : dsDays <= 3 ? '#e6a817' : 'var(--red)';
    let daysLabel = dsDays === null ? '未知' : dsDays === 0 ? '今日' : dsDays + '天前';
    const modelOk = Object.values(m).filter(Boolean).length;
    const modelTotal = Object.values(m).length;
    bar.innerHTML = `<span class="hint">数据:</span>`
      + `<b style="color:${{daysStyle}};">${{d.last_date || 'N/A'}}</b>`
      + `<span style="color:var(--muted);">(${{daysLabel}})</span>`
      + `<span>|</span>`
      + `<span>${{(d.row_count/1e4).toFixed(0)}}万行</span>`
      + `<span>|</span>`
      + `<span>${{d.app_count}} apps</span>`
      + `<span>|</span>`
      + `<span style="color:var(--muted);">空值: 消耗${{((d.critical_null_rate||{{}})['消耗金额']*100||0).toFixed(1)}}% D1${{((d.critical_null_rate||{{}})['首日广告收入']*100||0).toFixed(1)}}%</span>`
      + `<span>|</span>`
      + `<span>模型: <b>${{modelOk}}/${{modelTotal}}</b></span>`
      + (dsDays > 3 ? '<span style="color:var(--red);margin-left:4px;">⚠ 数据滞后，建议更新 daily_merged.csv</span>' : '');
    const drift = h.drift || {{}};
    const cal = h.calendar_health || {{}};
    if (drift.overall) {{
      const dColors = {{'ok':'var(--green)','warning':'#e6a817','critical':'var(--red)','error':'var(--muted)'}};
      const dLabels = {{'ok':'正常','warning':'预警','critical':'异常','error':'--'}};
      bar.innerHTML += '<span>|</span><span>漂移: <b style="color:' + (dColors[drift.overall]||'var(--muted)') + ';">' + (dLabels[drift.overall]||drift.overall) + '</b></span>';
    }}
    if (cal.status) {{
      const cColors = {{'ok':'var(--green)','warning':'#e6a817','error':'var(--red)','unknown':'var(--muted)'}};
      const cLabels = {{'ok':'正常','warning':'预警','error':'异常','unknown':'--'}};
      bar.innerHTML += '<span>|</span><span>日历: <b style="color:' + (cColors[cal.status]||'var(--muted)') + ';">' + (cLabels[cal.status]||cal.status) + '</b>' + (cal.status==='warning'?' (未来30d缺)':'') + '</span>';
    }}
  }} catch(e) {{
    console.error('loadDataStatus:', e);
    const el = document.getElementById('dsLoading');
    if (el) el.textContent = '获取失败';
  }}
}}

async function loadRetrainStatus() {{
  try {{
    const resp = await fetch('/health');
    const h = await resp.json();
    const bar = document.getElementById('retrainStatusBar');
    if (!bar || !h.training) return;
    const t = h.training;
    const btn = document.getElementById('retrainTriggerBtn');
    const statusColors = {{'success': 'var(--green)', 'partial': '#e6a817', 'failed': 'var(--red)', 'running': 'var(--accent)', 'never': 'var(--muted)'}};
    const color = statusColors[t.overall] || 'var(--muted)';
    const labels = {{'success': '全部通过', 'partial': '部分失败', 'failed': '失败', 'running': '运行中', 'never': '无记录'}};
    const stepLabels = {{'health_check': '数据', 'spend_t1': 'Spend', 'roi_d1': 'ROI', 'app_curves': '曲线', 'daily_revenue': '收入'}};
    let stepsHtml = (t.steps || []).map(s => {{
      const sColor = s.status === 'ok' ? 'var(--green)' : s.status === 'failed' ? 'var(--red)' : s.status === 'running' ? 'var(--accent)' : 'var(--muted)';
      return '<span style="color:' + sColor + ';">' + (stepLabels[s.step] || s.step) + '</span>';
    }}).join(' <span style="color:var(--muted);">·</span> ');
    const lastRun = t.last_run ? new Date(t.last_run).toLocaleString('zh-CN') : '--';
    bar.innerHTML = '<span class="hint">重训:</span>'
      + '<b style="color:' + color + ';">' + (labels[t.overall] || t.overall) + '</b>'
      + '<span style="color:var(--muted);">(' + lastRun + ')</span>'
      + (stepsHtml ? '<span>|</span>' + stepsHtml : '');
    if (btn) {{
      btn.disabled = t.overall === 'running';
      btn.style.opacity = t.overall === 'running' ? '0.5' : '1';
    }}
  }} catch(e) {{
    const el = document.getElementById('rsLoading');
    if (el) el.textContent = '获取失败';
  }}
}}

async function triggerRetrain() {{
  const btn = document.getElementById('retrainTriggerBtn');
  if (!btn) return;
  btn.disabled = true;
  btn.textContent = '启动中...';
  try {{
    const resp = await fetch('/web/retrain/trigger', {{method: 'POST'}});
    const data = await resp.json();
    btn.textContent = data.status === 'started' ? '已触发' : '失败';
    if (data.status === 'started') {{
      setTimeout(loadRetrainStatus, 3000);
      setTimeout(loadRetrainStatus, 10000);
      setTimeout(loadRetrainStatus, 30000);
    }}
  }} catch(e) {{
    btn.textContent = '触发失败';
    btn.disabled = false;
  }} finally {{
    setTimeout(() => {{ btn.disabled = false; btn.textContent = '触发重训'; }}, 5000);
  }}
}}

let healthTrendChart = null;

async function loadMonitorView() {{
  loadMonitorAlerts();
  loadHealthTrend();
}}

async function loadMonitorAlerts() {{
  try {{
    const resp = await fetch('/v1/monitor/alerts?days=7');
    const data = await resp.json();
    document.getElementById('monitorAlertTotal').textContent = data.total || 0;
    const byLvl = data.by_level || {{}};
    const byCat = data.by_category || {{}};
    document.getElementById('monitorAlertBreakdown').innerHTML =
      'CRITICAL:' + (byLvl.CRITICAL || 0) + ' WARN:' + (byLvl.WARN || 0) +
      ' | 漂移:' + (byCat.PREDICTION_DRIFT || 0) + ' D1:' + (byCat.PRODUCT_D1_DROP || 0) + ' Spend:' + (byCat.SPEND_DROP || 0);

    // 告警历史表格
    const recent = data.recent || [];
    const tabDiv = document.getElementById('monitorAlertTable');
    if (recent.length === 0) {{
      tabDiv.innerHTML = '<div style="text-align:center;color:var(--muted);padding:40px 0;font-size:13px;">近7天无告警<br/><span style="font-size:11px;">系统运行正常</span></div>';
    }} else {{
      let thtml = '<table><thead><tr><th>时间</th><th>级别</th><th>类别</th><th>详情</th></tr></thead><tbody>';
      recent.forEach(a => {{
        const levelClr = a.level === 'CRITICAL' ? 'var(--red)' : '#f59e0b';
        const catLabel = {{PREDICTION_DRIFT:'预测漂移',PRODUCT_D1_DROP:'D1下滑',TRAFFIC_ANOMALY:'流量异常',ROI_GUARD:'ROI警戒',SPEND_DROP:'消耗骤降',SPEND_SPIKE:'消耗骤升',ROI_DECLINE_TREND:'ROI趋势下行',CAP_PROXIMITY:'Cap逼近'}}[a.category] || a.category;
        thtml += '<tr><td>' + (a.date||'') + '</td><td style="color:' + levelClr + ';font-weight:600;">' + a.level + '</td><td>' + catLabel + '</td><td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="' + (a.message||'') + '">' + (a.message||'') + '</td></tr>';
      }});
      thtml += '</tbody></table>';
      tabDiv.innerHTML = thtml;
    }}
  }} catch(e) {{}}
  try {{
    const r = await fetch('/health');
    const h = await r.json();
    const drift = h.drift || {{}};
    const cal = h.calendar_health || {{}};
    const training = h.training || {{}};
    const dataStatus = (h.data||{{}}).status || '--';
    const driftOverall = drift.overall || '--';
    const trainOverall = training.overall || '--';
    const calStatus = cal.status || '--';
    document.getElementById('monitorDriftStatus').textContent = driftOverall === 'ok' ? '正常' : driftOverall;
    document.getElementById('monitorDriftDetail').innerHTML = 'Spend MAPE: ' + ((drift.spend_mape||0)*100).toFixed(1) + '%';
    document.getElementById('monitorDataStatus').textContent = dataStatus === 'healthy' ? '正常' : dataStatus;
    document.getElementById('monitorDataDetail').textContent = '滞后 ' + ((h.data||{{}}).days_behind||'--') + ' 天';
    document.getElementById('monitorTrainStatus').textContent = trainOverall === 'success' ? '正常' : trainOverall;
    document.getElementById('monitorCalStatus').textContent = '日历: ' + calStatus;
  }} catch(e) {{}}
}}

async function loadHealthTrend() {{
  try {{
    const resp = await fetch('/v1/monitor/health-trend?days=30');
    const data = await resp.json();
    const snaps = data.snapshots || [];
    if (snaps.length === 0) {{
      const el = document.getElementById('healthTrendChart');
      if (el) el.innerHTML = '<div style="text-align:center;color:var(--muted);padding-top:100px;">暂无健康快照数据<br/>等待首次重训后自动记录</div>';
      return;
    }}
    const dates = snaps.map(s => s.timestamp ? s.timestamp.slice(0,10) : '');
    const behind = snaps.map(s => s.days_behind || 0);
    const driftSpend = snaps.map(s => s.drift_spend_mape || 0);
    const driftRoi = snaps.map(s => s.drift_roi_mape || 0);

    if (!healthTrendChart) {{
      const el = document.getElementById('healthTrendChart');
      if (!el) return;
      healthTrendChart = echarts.init(el);
    }}
    healthTrendChart.setOption({{
      color: ['#94a3b8', '#f59e0b', '#10b981'],
      tooltip: {{ trigger: 'axis' }},
      legend: {{ data: ['数据滞后(天)', 'Spend MAPE%', 'ROI MAPE%'], top: 8 }},
      grid: {{ left: 64, right: 28, top: 58, bottom: 55 }},
      xAxis: {{ type: 'category', data: dates }},
      yAxis: [
        {{ type: 'value', min: 0 }},
        {{ type: 'value', min: 0 }},
      ],
      dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', height: 18, bottom: 12 }}],
      series: [
        {{ name: '数据滞后(天)', type: 'bar', data: behind }},
        {{ name: 'Spend MAPE%', type: 'line', yAxisIndex: 1, data: driftSpend }},
        {{ name: 'ROI MAPE%', type: 'line', yAxisIndex: 1, data: driftRoi }},
      ],
    }});
  }} catch(e) {{}}
}}

function toggleOnlineInputMode() {{
  const mode = document.getElementById('onlineInputMode').value;
  document.getElementById('onlineJsonInput').style.display = mode === 'json' ? '' : 'none';
  document.getElementById('onlineCsvInput').style.display = mode === 'csv' ? '' : 'none';
}}

async function runOnlinePredict() {{
  const targetDate = document.getElementById('onlineTargetDate').value;
  const mode = document.getElementById('onlineInputMode').value;
  const resultDiv = document.getElementById('onlinePredictResult');

  let apps = [];
  try {{
    if (mode === 'json') {{
      apps = JSON.parse(document.getElementById('onlineInputJson').value);
    }} else {{
      const lines = document.getElementById('onlineInputCsv').value.trim().split('\\n');
      const header = lines[0].replace(/\\r/g, '');
      const cols = header.split(',');
      const idxApp = cols.findIndex(c => c.trim() === '应用ID');
      const idxSpend = cols.findIndex(c => c.trim().toLowerCase() === 'spend');
      const idxD1 = cols.findIndex(c => c.trim().toLowerCase() === 'd1_revenue');
      for (let i = 1; i < lines.length; i++) {{
        const vals = lines[i].replace(/\\r/g, '').split(',');
        if (vals.length >= 3) {{
          apps.push({{
            '应用ID': vals[idxApp].trim(),
            spend: parseFloat(vals[idxSpend]),
            d1_revenue: parseFloat(vals[idxD1]),
          }});
        }}
      }}
    }}
  }} catch (e) {{
    resultDiv.innerHTML = '<div class="risk-alert critical">输入格式错误: ' + e.message + '</div>';
    return;
  }}

  if (!targetDate || apps.length === 0) {{
    resultDiv.innerHTML = '<div class="risk-alert warn">请输入目标日期和至少一个应用</div>';
    return;
  }}

  resultDiv.innerHTML = '<span style="color:var(--accent);">计算中...</span>';
  try {{
    const resp = await fetch('/web/revenue/daily-predict', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ target_date: targetDate, apps: apps }}),
    }});
    const data = await resp.json();
    if (data.error) {{
      resultDiv.innerHTML = '<div class="risk-alert critical">' + data.error + '</div>';
      return;
    }}

    const top10 = data.rows.sort((a,b) => b.y_pred - a.y_pred).slice(0, 10);
    let html = '<b>预测完成:</b> ' + data.predicted_apps + ' apps | 总收入预测: <b>' + data.total_predicted_revenue.toLocaleString() + '</b>';
    html += '<table style="margin-top:8px;"><tr><th>应用ID</th><th class="num">预测收入</th><th class="num">消耗</th><th class="num">D1_ROI</th><th class="num">累计天</th><th class="num">Cohort数</th></tr>';
    top10.forEach(r => {{
      html += '<tr><td>' + r['应用ID'] + '</td><td class="num">' + r.y_pred.toLocaleString() + '</td><td class="num">' + r.spend.toLocaleString() + '</td><td class="num">' + r.d1_roi + '</td><td class="num">' + r.days_since_start + '</td><td class="num">' + r.cohort_count + '</td></tr>';
    }});
    html += '</table><div class="hint">显示 top 10 by y_pred | ' + (data.rows.length > 10 ? '共 ' + data.rows.length + ' apps' : '') + '</div>';
    resultDiv.innerHTML = html;
  }} catch (e) {{
    resultDiv.innerHTML = '<div class="risk-alert critical">请求失败: ' + e.message + '</div>';
  }}
}}

const rb = document.getElementById('resetBtn');
if (rb) rb.addEventListener('click', () => {{
  selectedAppId = '';
  _chartAppId = '';
  document.getElementById('appSelectHint').style.display = 'none';
  detailState = {{ appId: '', page: 0, limit: 200, total: 0 }};
  detailSort = {{ key: 'date', dir: 'asc' }};
  document.getElementById('startDate').value = '';
  document.getElementById('endDate').value = '';
  const das = document.getElementById('detailAppSearch');
  if (das) das.value = '';
  render();
}});
window.addEventListener('resize', () => {{ Object.values(charts).forEach(c => c.resize()); if (_popupCharts.roi) _popupCharts.roi.resize(); if (_popupCharts.spend) _popupCharts.spend.resize(); }});
render();
setActiveView(initialView);
loadDataStatus();
loadRetrainStatus();
</script>
</body>
</html>
"""
    return HTMLResponse(page, headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/web", response_class=HTMLResponse)
def web_dashboard(
    input_csv: str = "daily_20260421_120112.csv",
    output_dir: str = "outputs",
    kpi: float = 1.05,
    task: str = "app_last_day",
    app_id: str = "",
    advertiser_id: str = "",
    date: str = "",
    limit: int = 300,
    message: str = "",
    job_id: str = "",
):
    return _render_model_dashboard(_load_model_dashboard_payload())


@router.get("/web/predict", response_class=HTMLResponse)
def web_prediction_dashboard():
    return _render_model_dashboard(_load_model_dashboard_payload(), initial_view="predict")


@router.get("/web/recommend", response_class=HTMLResponse)
def web_recommendation_dashboard():
    return _render_model_dashboard(_load_model_dashboard_payload(), initial_view="recommend")

    out_dir = Path(output_dir)
    payload = _load_payload(out_dir, task, app_id, advertiser_id, date, limit)
    job_status = _snapshot_job(job_id.strip()) if job_id.strip() else {}
    safe_payload_js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    safe_message = html.escape(message)
    safe_job_status_js = json.dumps(job_status, ensure_ascii=False).replace("</", "<\\/")
    safe_job_id_js = json.dumps(job_id.strip(), ensure_ascii=False)

    page = f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>智能预算决策系统 - 阶段成果看板</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121a2f;
      --muted: #8fa2c7;
      --text: #e8eefc;
      --accent: #4f8cff;
      --ok: #1fbf75;
      --warn: #f5b73b;
      --danger: #f87171;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, "PingFang SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 16px 18px 28px; }}
    .head {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }}
    .title {{ font-size: 24px; font-weight: 700; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .msg {{ background: rgba(79, 140, 255, 0.18); border: 1px solid rgba(79, 140, 255, 0.5); padding: 8px 10px; border-radius: 8px; }}
    .panel {{ background: var(--panel); border: 1px solid rgba(143, 162, 199, 0.15); border-radius: 12px; padding: 12px; margin-top: 12px; }}
    .form-grid {{ display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 8px; }}
    label {{ font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 5px; }}
    input, select {{ background: #0d1428; color: var(--text); border: 1px solid #22345f; border-radius: 8px; padding: 8px; }}
    .btns {{ display: flex; gap: 8px; align-items: end; }}
    button, .btn {{
      background: linear-gradient(135deg, #4f8cff, #6f63ff);
      border: none;
      color: #fff;
      border-radius: 8px;
      padding: 9px 12px;
      cursor: pointer;
      font-weight: 600;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
    }}
    .cards {{ display: grid; grid-template-columns: repeat(6, minmax(160px, 1fr)); gap: 10px; margin-top: 10px; }}
    .card {{ background: #0d1428; border: 1px solid #22345f; border-radius: 10px; padding: 10px; }}
    .card .k {{ color: var(--muted); font-size: 12px; }}
    .card .v {{ font-size: 20px; font-weight: 700; margin-top: 6px; }}
    .layout {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 12px; margin-top: 12px; }}
    .chart {{ height: 280px; background: #0d1428; border: 1px solid #22345f; border-radius: 10px; }}
    .table-wrap {{ margin-top: 12px; overflow: auto; max-height: 520px; border: 1px solid #22345f; border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid #22345f; white-space: nowrap; text-align: left; }}
    th {{ position: sticky; top: 0; background: #121d39; z-index: 2; }}
    tr:hover td {{ background: rgba(79, 140, 255, 0.08); }}
    .tabs {{ display: flex; gap: 8px; margin-top: 12px; }}
    .tab {{ padding: 6px 10px; border-radius: 8px; border: 1px solid #2a3d6b; color: var(--muted); cursor: pointer; }}
    .tab.active {{ color: #fff; border-color: var(--accent); background: rgba(79, 140, 255, 0.15); }}
    .stage-grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 10px; margin-top: 10px; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; }}
    .b-ok {{ background: rgba(31,191,117,.2); color: #54dd9a; }}
    .b-warn {{ background: rgba(245,183,59,.2); color: #ffd27f; }}
    .b-danger {{ background: rgba(248,113,113,.18); color: #ff9a9a; }}
    .job-status {{
      margin-top: 10px;
      background: #0d1428;
      border: 1px solid #22345f;
      border-radius: 10px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .job-row {{ display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }}
    .progress {{ width: 100%; height: 10px; border-radius: 999px; background: #1a2648; overflow: hidden; }}
    .progress > i {{ display: block; height: 100%; width: 0%; background: linear-gradient(90deg, #4f8cff, #1fbf75); transition: width .2s ease; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div>
        <div class="title">智能预算决策系统 · 阶段成果强交互看板</div>
        <div class="sub">聚焦：月末ROI达标前提下最大化消耗规模（文档 v2.3）</div>
      </div>
      {"<div class='msg'>" + safe_message + "</div>" if safe_message else ""}
    </div>

    <div class="panel">
      <form id="runForm" action="/web/run" method="get">
        <div class="form-grid">
          <label>CSV路径<input name="input_csv" value="{html.escape(input_csv)}" /></label>
          <label>输出目录<input name="output_dir" value="{html.escape(output_dir)}" /></label>
          <label>KPI<input name="kpi" value="{kpi}" /></label>
          <label>任务
            <select name="task" id="taskSel">
              <option value="app_last_day" {"selected" if task == "app_last_day" else ""}>应用层最后一天预测</option>
              <option value="backtest" {"selected" if task == "backtest" else ""}>全量回放</option>
            </select>
          </label>
          <label>应用ID筛选<input id="fApp" value="{html.escape(app_id)}" /></label>
          <label>广告主ID筛选<input id="fAdv" value="{html.escape(advertiser_id)}" /></label>
          <label>日期筛选<input id="fDate" type="date" value="{html.escape(date)}" /></label>
          <label>展示条数上限<input id="fLimit" type="number" min="20" max="1000" value="{max(20, min(limit, 1000))}" /></label>
          <div class="btns">
            <button id="runBtn" type="submit">运行离线回放</button>
            <a class="btn" id="refreshBtn" href="#">刷新筛选</a>
          </div>
        </div>
      </form>
      <div id="jobStatus" class="job-status" style="display:none;">
        <div class="job-row">
          <span id="jobText">任务状态</span>
          <span id="jobMeta"></span>
        </div>
        <div class="progress"><i id="jobBar"></i></div>
      </div>
    </div>

    <div class="tabs">
      <div class="tab active" data-pane="predict">预测模块</div>
      <div class="tab" data-pane="solver">求解模块</div>
      <div class="tab" data-pane="risk">风控模块</div>
      <div class="tab" data-pane="rule">规则模块</div>
      <div class="tab" data-pane="stage">阶段成果对比</div>
      <div class="tab" data-pane="detail">明细数据</div>
    </div>

    <div class="pane panel" id="pane-predict">
      <div class="sub">预测模块：月末ROI与锚点校准表现</div>
      <div class="cards" id="summaryCards"></div>
      <div class="layout">
        <div id="modeChart" class="chart"></div>
        <div id="roiChart" class="chart"></div>
      </div>
    </div>

    <div class="pane panel" id="pane-solver" style="display:none;">
      <div class="sub">求解模块：建议预算与实际消耗对照</div>
      <div id="solverChart" class="chart" style="height:320px;"></div>
    </div>

    <div class="pane panel" id="pane-risk" style="display:none;">
      <div class="sub">风控模块：告警强度与分布</div>
      <div id="riskChart" class="chart" style="height:320px;"></div>
    </div>

    <div class="pane panel" id="pane-rule" style="display:none;">
      <div class="sub">规则模块：ROI偏差主因统计</div>
      <div id="attrChart" class="chart" style="height:300px;"></div>
    </div>

    <div class="pane panel" id="pane-stage" style="display:none;">
      <div class="stage-grid" id="stageGrid"></div>
      <div id="stageCompareChart" class="chart" style="margin-top:12px;"></div>
    </div>

    <div class="pane panel" id="pane-detail" style="display:none;">
      <div class="table-wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
    </div>
  </div>

  <script>
    let payload = {safe_payload_js};
    const initialJobStatus = {safe_job_status_js};
    let currentJobId = {safe_job_id_js};
    let pollTimer = null;
    let task = payload.task;
    let rowsRaw = payload.rows || [];
    let attributionRowsRaw = payload.attribution_rows || [];
    let summary = payload.summary || {{}};
    let modeDist = payload.mode_distribution || {{}};

    const toNum = (v) => Number(v || 0);
    const fmt = (v, d=2) => isFinite(Number(v)) ? Number(v).toFixed(d) : '-';
    const fmtPct = (v) => isFinite(Number(v)) ? (Number(v) * 100).toFixed(2) + '%' : '-';

    function renderJobStatus(status) {{
      const box = document.getElementById('jobStatus');
      const text = document.getElementById('jobText');
      const meta = document.getElementById('jobMeta');
      const bar = document.getElementById('jobBar');
      if (!status || (!status.job_id && !status.message)) {{
        box.style.display = 'none';
        return;
      }}
      box.style.display = 'flex';
      const state = status.status || 'queued';
      const stateMap = {{
        queued: '已入队',
        running: '执行中',
        completed: '已完成',
        failed: '失败',
        expired: '状态失效'
      }};
      text.textContent = `任务状态：${{stateMap[state] || state}} - ${{status.message || ''}}`;
      const elapsed = Number(status.elapsed_sec || 0);
      const err = status.error ? ` | error: ${{status.error}}` : '';
      meta.textContent = `job_id=${{status.job_id}} | 用时=${{elapsed.toFixed(2)}}s${{err}}`;
      bar.style.width = `${{Math.max(0, Math.min(100, Number(status.progress_pct || 0)))}}%`;
      if (state === 'failed') {{
        box.style.borderColor = 'rgba(248,113,113,.6)';
      }} else if (state === 'completed') {{
        box.style.borderColor = 'rgba(31,191,117,.6)';
      }} else {{
        box.style.borderColor = '#22345f';
      }}
    }}

    async function pollJobStatus() {{
      if (!currentJobId) return;
      try {{
        const r = await fetch(`/web/run/status?job_id=${{encodeURIComponent(currentJobId)}}`);
        if (!r.ok) return;
        const status = await r.json();
        renderJobStatus(status);
        if (status.status === 'completed' || status.status === 'failed' || status.status === 'expired') {{
          if (pollTimer) clearInterval(pollTimer);
          if (status.status === 'completed') {{
            await applyServerFilter();
          }}
        }}
      }} catch (_e) {{
      }}
    }}

    function syncPayload(newPayload) {{
      payload = newPayload || {{}};
      task = payload.task || document.getElementById('taskSel').value || 'app_last_day';
      rowsRaw = payload.rows || [];
      attributionRowsRaw = payload.attribution_rows || [];
      summary = payload.summary || {{}};
      modeDist = payload.mode_distribution || {{}};
    }}

    async function applyServerFilter() {{
      const outputDir = document.querySelector('input[name=\"output_dir\"]').value.trim();
      const f = getFilters();
      const selTask = document.getElementById('taskSel').value;
      const params = new URLSearchParams();
      params.set('output_dir', outputDir || 'outputs');
      params.set('task', selTask);
      params.set('app_id', f.app);
      params.set('advertiser_id', f.adv);
      params.set('date', f.date);
      params.set('limit', String(f.limit));
      const r = await fetch(`/web/data?${{params.toString()}}`);
      if (!r.ok) return;
      const data = await r.json();
      syncPayload(data);
      refreshAll();
    }}

    function getFilters() {{
      return {{
        app: document.getElementById('fApp').value.trim(),
        adv: document.getElementById('fAdv').value.trim(),
        date: document.getElementById('fDate').value.trim(),
        limit: Number(document.getElementById('fLimit').value || 300),
      }};
    }}

    function filterRows(rows) {{
      const f = getFilters();
      let out = rows.slice();
      if (f.app) out = out.filter(r => (r.app_id || '') === f.app);
      if (f.adv && task !== 'app_last_day') out = out.filter(r => (r.advertiser_id || '') === f.adv);
      if (f.date) out = out.filter(r => (r.date || r.target_day || '') === f.date);
      return out.slice(0, Math.max(20, Math.min(1000, f.limit)));
    }}

    function renderCards(rows) {{
      const cards = document.getElementById('summaryCards');
      const modeGuard = rows.filter(r => (r.mode || '') === 'GUARD').length;
      const modeScale = rows.filter(r => (r.mode || '') === 'SCALE').length;
      const avgPredROI = rows.length
        ? rows.reduce((s, r) => s + toNum(r.month_end_roi_prediction || r.predicted_month_end_roi), 0) / rows.length
        : toNum(summary.avg_predicted_month_end_roi || 0);
      const avgActualROI = rows.length
        ? rows.reduce((s, r) => s + toNum(r.actual_roi || 0), 0) / rows.length
        : toNum(summary.avg_actual_roi || 0);
      const avgPlanSpend = rows.length
        ? rows.reduce((s, r) => s + toNum(r.planned_spend || r.planned_total_budget), 0) / rows.length
        : 0;
      const avgAnchorRaw = rows.length
        ? rows.reduce((s, r) => s + toNum(r.d1_anchor_raw_mean), 0) / rows.length
        : toNum(summary.avg_d1_anchor_raw_mean || 0);
      const avgAnchorCal = rows.length
        ? rows.reduce((s, r) => s + toNum(r.d1_anchor_calibrated_mean), 0) / rows.length
        : toNum(summary.avg_d1_anchor_calibrated_mean || 0);
      const avgRoiA = rows.length
        ? rows.reduce((s, r) => s + toNum(r.month_end_roi_prediction_a), 0) / rows.length
        : toNum((summary.ab_compare || {{}}).A_raw_anchor_avg_month_end_roi || 0);
      const avgRoiB = rows.length
        ? rows.reduce((s, r) => s + toNum(r.month_end_roi_prediction_b || r.month_end_roi_prediction), 0) / rows.length
        : toNum((summary.ab_compare || {{}}).B_calibrated_anchor_avg_month_end_roi || summary.avg_predicted_month_end_roi || 0);

      const data = [
        ['样本行数', rows.length],
        ['平均预测月末ROI', fmt(avgPredROI, 4)],
        ['A组ROI(原始锚点)', fmt(avgRoiA, 4)],
        ['B组ROI(校准锚点)', fmt(avgRoiB, 4)],
        ['B-A ROI差值', fmt(avgRoiB - avgRoiA, 4)],
        ['平均实际ROI', fmt(avgActualROI, 4)],
        ['平均建议日预算', fmt(avgPlanSpend, 2)],
        ['D1锚点(原始均值)', fmt(avgAnchorRaw, 4)],
        ['D1锚点(校准均值)', fmt(avgAnchorCal, 4)],
        ['锚点校准差值', fmt(avgAnchorRaw - avgAnchorCal, 4)],
        ['模式分布', `GUARD ${{modeGuard}} / SCALE ${{modeScale}}`],
        ['告警总数', rows.reduce((s, r) => s + toNum(r.alert_count), 0)],
      ];
      cards.innerHTML = data.map(([k, v]) => `<div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join('');
    }}

    function renderModeChart(rows) {{
      const chart = echarts.init(document.getElementById('modeChart'));
      const guard = rows.filter(r => (r.mode || '') === 'GUARD').length;
      const scale = rows.filter(r => (r.mode || '') === 'SCALE').length;
      chart.setOption({{
        title: {{ text: '模式分布', left: 'center', textStyle: {{ color: '#e8eefc', fontSize: 14 }} }},
        tooltip: {{ trigger: 'item' }},
        series: [{{
          type: 'pie',
          radius: ['45%', '72%'],
          data: [{{ name: 'GUARD', value: guard }}, {{ name: 'SCALE', value: scale }}],
          label: {{ color: '#e8eefc' }}
        }}],
        backgroundColor: 'transparent'
      }});
    }}

    function renderROIChart(rows) {{
      const chart = echarts.init(document.getElementById('roiChart'));
      const x = rows.map((r, i) => r.date || r.target_day || String(i + 1));
      const pred = rows.map(r => toNum(r.month_end_roi_prediction || r.predicted_month_end_roi));
      const predA = rows.map(r => toNum(r.month_end_roi_prediction_a));
      const predB = rows.map(r => toNum(r.month_end_roi_prediction_b || r.month_end_roi_prediction || r.predicted_month_end_roi));
      const actual = rows.map(r => toNum(r.actual_roi || 0));
      chart.setOption({{
        title: {{ text: 'ROI对比（筛选样本）', left: 'center', textStyle: {{ color: '#e8eefc', fontSize: 14 }} }},
        tooltip: {{ trigger: 'axis' }},
        legend: {{ top: 24, textStyle: {{ color: '#c9d7f5' }} }},
        xAxis: {{ type: 'category', data: x, axisLabel: {{ color: '#a9bddf' }} }},
        yAxis: {{ type: 'value', axisLabel: {{ color: '#a9bddf' }} }},
        series: [
          {{ name: '预测月末ROI', type: 'line', smooth: true, data: pred }},
          {{ name: 'A组ROI(原始锚点)', type: 'line', smooth: true, data: predA }},
          {{ name: 'B组ROI(校准锚点)', type: 'line', smooth: true, data: predB }},
          {{ name: '实际ROI', type: 'line', smooth: true, data: actual }},
        ]
      }});
    }}

    function renderAttributionChart(rows) {{
      const chart = echarts.init(document.getElementById('attrChart'));
      const count = {{ SPEND: 0, D1_ANCHOR: 0, CURVE: 0, BALANCED: 0 }};
      rows.forEach(r => {{
        const k = String(r.roi_dominant_factor || '').toUpperCase();
        if (count[k] != null) count[k] += 1;
      }});
      const data = Object.entries(count).map(([k, v]) => ({{ name: k, value: v }}));
      chart.setOption({{
        title: {{ text: 'ROI偏差主因分布', left: 'center', textStyle: {{ color: '#e8eefc', fontSize: 14 }} }},
        tooltip: {{ trigger: 'item' }},
        xAxis: {{ type: 'category', data: data.map(d => d.name), axisLabel: {{ color: '#a9bddf' }} }},
        yAxis: {{ type: 'value', axisLabel: {{ color: '#a9bddf' }} }},
        series: [{{ type: 'bar', data: data.map(d => d.value), itemStyle: {{ color: '#4f8cff' }} }}]
      }});
    }}

    function renderSolverChart(rows) {{
      const chart = echarts.init(document.getElementById('solverChart'));
      const x = rows.map((r, i) => r.app_id || r.advertiser_id || String(i + 1));
      const planned = rows.map(r => toNum(r.planned_total_budget || r.planned_spend));
      const actual = rows.map(r => toNum(r.actual_spend_last_day || r.actual_spend));
      chart.setOption({{
        title: {{ text: '建议预算 vs 实际消耗', left: 'center', textStyle: {{ color: '#e8eefc', fontSize: 14 }} }},
        tooltip: {{ trigger: 'axis' }},
        legend: {{ top: 24, textStyle: {{ color: '#c9d7f5' }} }},
        xAxis: {{ type: 'category', data: x, axisLabel: {{ color: '#a9bddf', interval: 0, rotate: 30 }} }},
        yAxis: {{ type: 'value', axisLabel: {{ color: '#a9bddf' }} }},
        series: [
          {{ name: '建议预算', type: 'bar', data: planned }},
          {{ name: '实际消耗', type: 'bar', data: actual }},
        ],
      }});
    }}

    function renderRiskChart(rows) {{
      const chart = echarts.init(document.getElementById('riskChart'));
      const x = rows.map((r, i) => r.app_id || r.advertiser_id || String(i + 1));
      const alerts = rows.map(r => toNum(r.alert_count));
      chart.setOption({{
        title: {{ text: '告警数量分布', left: 'center', textStyle: {{ color: '#e8eefc', fontSize: 14 }} }},
        tooltip: {{ trigger: 'axis' }},
        xAxis: {{ type: 'category', data: x, axisLabel: {{ color: '#a9bddf', interval: 0, rotate: 30 }} }},
        yAxis: {{ type: 'value', axisLabel: {{ color: '#a9bddf' }} }},
        series: [{{ type: 'line', smooth: true, data: alerts, itemStyle: {{ color: '#f5b73b' }} }}],
      }});
    }}

    function renderStagePanel() {{
      const stage = payload.stage_compare || {{}};
      const nodes = [
        ['启发式基线', stage.heuristic || {{}}, 'b-warn'],
        ['联合求解初版', stage.joint_v1 || {{}}, 'b-danger'],
        ['联合求解调优版', stage.joint_tuned || {{}}, 'b-ok'],
      ];
      const grid = document.getElementById('stageGrid');
      grid.innerHTML = nodes.map(([name, s, badge]) => {{
        if (!s || Object.keys(s).length === 0) return `<div class="card"><div class="k">${{name}}</div><div class="v">暂无数据</div></div>`;
        return `
          <div class="card">
            <div class="k">${{name}} <span class="badge ${{badge}}">阶段</span></div>
            <div class="sub">avg_pred_roi: ${{fmt(s.avg_predicted_month_end_roi || 0, 4)}}</div>
            <div class="sub">avg_plan-actual_spend: ${{fmt(s.avg_planned_minus_actual_spend || 0, 2)}}</div>
            <div class="sub">roi_warning_count: ${{s.roi_warning_count ?? '-'}}</div>
          </div>
        `;
      }}).join('');

      const chart = echarts.init(document.getElementById('stageCompareChart'));
      const labels = ['启发式', '联合v1', '联合调优'];
      const roi = [
        toNum(stage.heuristic?.avg_predicted_month_end_roi),
        toNum(stage.joint_v1?.avg_predicted_month_end_roi),
        toNum(stage.joint_tuned?.avg_predicted_month_end_roi),
      ];
      const spendGap = [
        toNum(stage.heuristic?.avg_planned_minus_actual_spend),
        toNum(stage.joint_v1?.avg_planned_minus_actual_spend),
        toNum(stage.joint_tuned?.avg_planned_minus_actual_spend),
      ];
      chart.setOption({{
        title: {{ text: '阶段成果对比（25天子集）', left: 'center', textStyle: {{ color: '#e8eefc', fontSize: 14 }} }},
        tooltip: {{ trigger: 'axis' }},
        legend: {{ top: 24, textStyle: {{ color: '#c9d7f5' }} }},
        xAxis: {{ type: 'category', data: labels, axisLabel: {{ color: '#a9bddf' }} }},
        yAxis: [
          {{ type: 'value', name: '预测月末ROI', axisLabel: {{ color: '#a9bddf' }} }},
          {{ type: 'value', name: 'planned-actual', axisLabel: {{ color: '#a9bddf' }} }},
        ],
        series: [
          {{ name: 'avg_pred_month_roi', type: 'line', data: roi, smooth: true }},
          {{ name: 'avg_planned_minus_actual_spend', type: 'bar', yAxisIndex: 1, data: spendGap }},
        ],
      }});
    }}

    function renderTable(rows) {{
      const appHeaders = [
        'target_day','app_id','mode','canonical_month_end_roi_source','actual_spend_last_day','actual_revenue_last_day',
        'planned_total_budget','planned_daily_spend_stable_anchor','planned_daily_spend_pred_only','planned_daily_spend_fused',
        'month_end_roi_prediction','month_end_roi_stable_anchor','month_end_roi_pred_only','month_end_roi_fused',
        'month_end_spend_prediction','alert_count','roi_dominant_factor',
        'month_end_roi_prediction_a','month_end_roi_prediction_b','ab_roi_delta_b_minus_a',
        'd1_anchor_raw_mean','d1_anchor_calibrated_mean','top_actions'
      ];
      const backtestHeaders = [
        'date','advertiser_id','mode','products','actual_spend','actual_revenue','actual_roi','planned_spend','predicted_month_end_roi','alert_count'
      ];
      const headers = task === 'app_last_day' ? appHeaders : backtestHeaders;
      document.getElementById('thead').innerHTML = '<tr>' + headers.map(h => `<th>${{h}}</th>`).join('') + '</tr>';
      document.getElementById('tbody').innerHTML = rows.map(r => '<tr>' + headers.map(h => `<td>${{r[h] ?? ''}}</td>`).join('') + '</tr>').join('');
    }}

    function refreshAll() {{
      const rows = filterRows(rowsRaw);
      const attrs = filterRows(attributionRowsRaw);
      renderCards(rows);
      renderModeChart(rows);
      renderROIChart(rows);
      renderSolverChart(rows);
      renderRiskChart(rows);
      renderAttributionChart(attrs);
      renderTable(rows);
      renderStagePanel();
    }}

    document.getElementById('refreshBtn').addEventListener('click', (e) => {{
      e.preventDefault();
      applyServerFilter();
    }});
    document.getElementById('taskSel').addEventListener('change', () => {{
      applyServerFilter();
    }});

    const runBtn = document.getElementById('runBtn');
    const runForm = document.getElementById('runForm');
    runForm.addEventListener('submit', async (e) => {{
      e.preventDefault();
      runBtn.disabled = true;
      runBtn.textContent = '任务提交中...';
      try {{
        const fd = new FormData(runForm);
        const params = new URLSearchParams();
        for (const [k, v] of fd.entries()) {{
          params.set(k, String(v));
        }}
        const r = await fetch(`/web/run/start?${{params.toString()}}`);
        const data = await r.json();
        if (!r.ok || !data.ok) {{
          renderJobStatus({{
            job_id: '',
            status: 'failed',
            message: data.error || '任务提交失败',
            elapsed_sec: 0,
            progress_pct: 100,
            error: data.error || '',
          }});
          runBtn.disabled = false;
          runBtn.textContent = '运行离线回放';
          return;
        }}
        currentJobId = data.job_id;
        renderJobStatus({{
          job_id: data.job_id,
          status: 'queued',
          message: data.message || '任务已提交',
          elapsed_sec: 0,
          progress_pct: 5,
          error: '',
        }});
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(pollJobStatus, 2000);
        await pollJobStatus();
      }} catch (err) {{
        renderJobStatus({{
          job_id: '',
          status: 'failed',
          message: '请求失败，请检查服务是否可用',
          elapsed_sec: 0,
          progress_pct: 100,
          error: String(err || ''),
        }});
      }} finally {{
        runBtn.disabled = false;
        runBtn.textContent = '运行离线回放';
      }}
    }});

    document.querySelectorAll('.tab').forEach(el => {{
      el.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        el.classList.add('active');
        const pane = el.getAttribute('data-pane');
        document.querySelectorAll('.pane').forEach(p => p.style.display = 'none');
        document.getElementById(`pane-${{pane}}`).style.display = 'block';
      }});
    }});

    const dateInput = document.getElementById('fDate');
    if (dateInput) {{
      const openPicker = () => {{
        if (typeof dateInput.showPicker === 'function') {{
          dateInput.showPicker();
        }}
      }};
      dateInput.addEventListener('click', openPicker);
      dateInput.addEventListener('focus', openPicker);
    }}

    refreshAll();
    renderJobStatus(initialJobStatus);
    if (initialJobStatus && initialJobStatus.job_id) {{
      currentJobId = initialJobStatus.job_id;
    }}
    if (currentJobId) {{
      pollTimer = setInterval(pollJobStatus, 2000);
      pollJobStatus();
    }}
  </script>
</body>
</html>
"""
    return HTMLResponse(content=page)


@router.get("/web/data")
def web_data(
    output_dir: str = "outputs",
    task: str = "app_last_day",
    app_id: str = "",
    advertiser_id: str = "",
    date: str = "",
    limit: int = 300,
):
    payload = _load_payload(Path(output_dir), task, app_id, advertiser_id, date, limit)
    return JSONResponse(content=payload)


@router.get("/web/run")
def web_run(
    input_csv: str = "daily_20260421_120112.csv",
    output_dir: str = "outputs",
    kpi: float = 1.05,
    task: str = "app_last_day",
):
    csv_path = Path(input_csv)
    if not csv_path.exists():
        params = urlencode(
            {
                "input_csv": input_csv,
                "output_dir": output_dir,
                "kpi": kpi,
                "task": task,
                "message": f"文件不存在: {input_csv}",
            }
        )
        return RedirectResponse(url=f"/web?{params}", status_code=302)

    job_id = _start_backtest_job(input_csv=input_csv, output_dir=output_dir, kpi=kpi, task=task)
    params = urlencode(
        {
            "input_csv": input_csv,
            "output_dir": output_dir,
            "kpi": kpi,
            "task": task,
            "job_id": job_id,
            "message": "任务已提交，正在后台执行。",
        }
    )
    return RedirectResponse(url=f"/web?{params}", status_code=302)


@router.get("/web/run/start")
def web_run_start(
    input_csv: str = "daily_20260421_120112.csv",
    output_dir: str = "outputs",
    kpi: float = 1.05,
    task: str = "app_last_day",
):
    csv_path = Path(input_csv)
    if not csv_path.exists():
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"文件不存在: {input_csv}"},
        )
    job_id = _start_backtest_job(input_csv=input_csv, output_dir=output_dir, kpi=kpi, task=task)
    return JSONResponse(
        content={
            "ok": True,
            "job_id": job_id,
            "message": "任务已提交，正在后台执行。",
            "task": task,
            "input_csv": input_csv,
            "output_dir": output_dir,
            "kpi": kpi,
        }
    )


@router.get("/web/run/status")
def web_run_status(job_id: str):
    data = _snapshot_job(job_id.strip())
    if not data:
        return JSONResponse(
            content={
                "job_id": job_id,
                "status": "expired",
                "message": "任务状态已失效（服务重启或代码热更新后内存任务会丢失），请重新提交回放任务。",
                "elapsed_sec": 0.0,
                "progress_pct": 100,
                "error": "",
            }
        )
    return JSONResponse(content=data)


# ── 按应用搜索 + 分页 API ───────────────────────────────────────────

def _resolve_prediction_path(target: str) -> Path | None:
    root = _repo_root() / "outputs"
    if target == "roi":
        roi_dir = _pick_existing(
            [
                root / "model_parallel_roi_d1_exp035",
                root / "model_parallel_roi_d1_v9_unified",
                root / "model_parallel_roi_d1_v8_001",
                root / "model_parallel_roi_d1_v7_001",
            ]
        )
        if roi_dir is None:
            return None
        return resolve_roi_predictions_csv(roi_dir)
    spend_dir = _pick_existing(
        [
            root / "model_parallel_spend_t1_exp035",
            root / "model_parallel_spend_t1_v12_unified",
            root / "model_parallel_spend_t1_v11_001",
        ]
    )
    if spend_dir is None:
        return None
    return resolve_spend_predictions_csv(spend_dir)


def _compute_chart_data(target: str) -> list[dict] | None:
    """全量CSV按日期聚合，供前端图表使用（不受行数限制）。"""
    path = _resolve_prediction_path(target)
    if path is None or not path.exists():
        return None
    date_map: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            day = str(r.get("日期", ""))[:10]
            if not day:
                continue
            y = float(r.get("y_true", 0) or 0)
            p = float(r.get("y_pred_fused", 0) or 0) or float(r.get("y_pred", 0) or 0)
            if day not in date_map:
                date_map[day] = {"date": day, "actual": 0.0, "pred": 0.0, "n": 0}
            date_map[day]["actual"] += y
            date_map[day]["pred"] += p
            date_map[day]["n"] += 1
    result = []
    for day in sorted(date_map):
        d = date_map[day]
        ape = abs((d["actual"] - d["pred"]) / d["actual"]) if d["actual"] > 1e-8 else 0.0
        result.append({
            "date": day,
            "actual": round(d["actual"], 4),
            "pred": round(d["pred"], 4),
            "n": d["n"],
            "ape": round(ape, 6),
        })
    return result


def _compute_prediction_stats(target: str) -> dict | None:
    """读取全量预测CSV，计算整体+分应用统计（不受5000行限制）。"""
    path = _resolve_prediction_path(target)
    if path is None or not path.exists():
        return None
    import numpy as np
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return None
    app_stats: dict[str, dict] = {}
    all_y, all_p = [], []
    for r in rows:
        y = float(r.get("y_true", 0) or 0)
        p = float(r.get("y_pred_fused", 0) or 0) or float(r.get("y_pred", 0) or 0)
        all_y.append(y)
        all_p.append(p)
        app_id = str(r.get("应用ID", "")).strip()
        if not app_id:
            continue
        if app_id not in app_stats:
            app_stats[app_id] = {"y": [], "p": [], "app_id": app_id}
        app_stats[app_id]["y"].append(y)
        app_stats[app_id]["p"].append(p)

    y_arr = np.array(all_y)
    p_arr = np.array(all_p)
    mae = float(np.mean(np.abs(y_arr - p_arr)))
    rmse = float(np.sqrt(np.mean((y_arr - p_arr) ** 2)))
    mask = y_arr > 1e-8
    mape = float(np.mean(np.abs((y_arr[mask] - p_arr[mask]) / y_arr[mask]))) if mask.any() else 0.0

    app_list = []
    for app_id, s in app_stats.items():
        ya = np.array(s["y"])
        pa = np.array(s["p"])
        amae = float(np.mean(np.abs(ya - pa)))
        amask = ya > 1e-8
        amape = float(np.mean(np.abs((ya[amask] - pa[amask]) / ya[amask]))) if amask.any() else 0.0
        arms = float(np.sqrt(np.mean((ya - pa) ** 2)))
        app_list.append({
            "app_id": app_id,
            "mae": round(amae, 4),
            "rmse": round(arms, 4),
            "mape": round(amape, 4),
            "mape_pct": round(amape * 100, 2),
            "samples": len(s["y"]),
        })
    app_list.sort(key=lambda x: x["mape"])
    return {
        "overall": {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4), "mape_pct": round(mape * 100, 2), "samples": len(rows), "apps": len(app_list)},
        "apps": app_list,
    }


@router.get("/web/data/{target}/stats")
def web_prediction_stats(target: str):
    stats = _compute_prediction_stats(target)
    if stats is None:
        return JSONResponse(status_code=404, content={"error": f"未找到 {target} 预测文件"})
    return JSONResponse(content=stats["overall"])


@router.get("/web/data/{target}/apps")
def web_prediction_apps(target: str):
    stats = _compute_prediction_stats(target)
    if stats is None:
        return JSONResponse(status_code=404, content={"error": f"未找到 {target} 预测文件"})
    return JSONResponse(content=stats["apps"])


@router.get("/web/data/{target}/app/{app_id}")
def web_prediction_app_detail(target: str, app_id: str, offset: int = 0, limit: int = 200):
    path = _resolve_prediction_path(target)
    if path is None or not path.exists():
        return JSONResponse(status_code=404, content={"error": f"未找到 {target} 预测文件"})
    # spend 模式：预加载 baseline 对照数据
    baseline_map = {}
    calendar_map = {}
    if target == "spend":
        root = _repo_root() / "outputs"
        old_spend_path = root / "model_parallel_spend_t1_split_online_fusion_script_verified" / "predictions_reconciled.csv"
        if old_spend_path.exists():
            with old_spend_path.open("r", encoding="utf-8-sig", newline="") as bf:
                for br in csv.DictReader(bf):
                    key = f"{str(br.get('日期', ''))[:10]}|{str(br.get('应用ID', '')).strip()}"
                    baseline_map[key] = {
                        "pred": float(br.get("y_pred_fused", 0) or 0) or float(br.get("y_pred", 0) or 0),
                    }
        calendar = CalendarService()
        calendar_map = {}  # date_str -> {target_date, day_type, ...}
    app_rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("应用ID", "")).strip() == app_id:
                y = float(r.get("y_true", 0) or 0)
                p = float(r.get("y_pred_fused", 0) or 0) or float(r.get("y_pred", 0) or 0)
                day_text = str(r.get("日期", ""))[:10]
                row = {
                    "日期": day_text,
                    "应用ID": app_id,
                    "y_true": round(y, 4),
                    "y_pred": round(p, 4),
                    "model": r.get("model", ""),
                }
                if target == "spend":
                    bk = baseline_map.get(f"{day_text}|{app_id}", {})
                    old_pred = bk.get("pred")
                    row["y_pred_old"] = round(old_pred, 2) if old_pred is not None else None
                    day = datetime.fromisoformat(day_text).date()
                    target_day = day + timedelta(days=1)
                    tfeat = calendar.features_for_day(target_day)
                    flags = []
                    if tfeat.get("holiday_display_name"):
                        flags.append(tfeat["holiday_display_name"])
                    if tfeat.get("is_last_holiday_day"):
                        flags.append("假期最后一天")
                    if tfeat.get("is_first_workday_after_holiday"):
                        flags.append("节后首个工作日")
                    row["target_date"] = target_day.isoformat()
                    row["calendar_label"] = " / ".join(flags) or tfeat.get("day_type", "-")
                app_rows.append(row)
    app_rows.sort(key=lambda x: x["日期"])
    total = len(app_rows)
    page = app_rows[offset : offset + limit]
    return JSONResponse(content={"rows": page, "total": total, "offset": offset, "limit": limit})


@router.get("/web/data/{target}/app/{app_id}/chart")
def web_prediction_app_chart(target: str, app_id: str):
    """单个应用全量日期聚合，供前端图表使用。"""
    path = _resolve_prediction_path(target)
    if path is None or not path.exists():
        return JSONResponse(status_code=404, content={"error": f"未找到 {target} 预测文件"})
    date_map: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("应用ID", "")).strip() != app_id:
                continue
            day = str(r.get("日期", ""))[:10]
            if not day:
                continue
            y = float(r.get("y_true", 0) or 0)
            p = float(r.get("y_pred_fused", 0) or 0) or float(r.get("y_pred", 0) or 0)
            if day not in date_map:
                date_map[day] = {"date": day, "actual": 0.0, "pred": 0.0, "n": 0}
            date_map[day]["actual"] += y
            date_map[day]["pred"] += p
            date_map[day]["n"] += 1
    result = []
    for day in sorted(date_map):
        d = date_map[day]
        ape = abs((d["actual"] - d["pred"]) / d["actual"]) if d["actual"] > 1e-8 else 0.0
        result.append({
            "date": day,
            "actual": round(d["actual"], 4),
            "pred": round(d["pred"], 4),
            "n": d["n"],
            "ape": round(ape, 6),
        })
    return JSONResponse(content=result)


@router.get("/web/predictions/daily_revenue")
async def web_daily_revenue_predictions(
    request: Request,
    app_id: str = Query(""),
):
    """每日买量收入预测验证数据"""
    csv_path = _repo_root() / "outputs" / "daily_revenue_predictions.csv"
    if not csv_path.exists():
        return JSONResponse(content={"rows": [], "message": "预测数据尚未生成，请先运行 scripts/predict_daily_revenue.py"})

    rows = _read_csv_rows(csv_path, limit=0)
    if app_id:
        rows = [r for r in rows if r.get("应用ID") == app_id]
    return JSONResponse(content={"rows": rows, "total": len(rows)})


@router.post("/web/revenue/daily-predict")
async def web_revenue_daily_predict(request: Request):
    """在线预测：给定目标日期 + per-app spend/D1，返回当日收入预测"""
    from scripts.predict_daily_revenue import (
        load_app_daily,
        load_curves,
        build_cohorts_until,
        predict_single_day,
    )

    body = await request.json()
    target_date_str = body.get("target_date", "")
    apps_input = body.get("apps", [])

    if not target_date_str or not apps_input:
        return JSONResponse(
            content={"error": "缺少 target_date 或 apps"},
            status_code=400,
        )

    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    csv_path = _repo_root() / "daily_merged.csv"
    data = load_app_daily(csv_path)

    curves_path = _repo_root() / "outputs" / "per_app_release_curves.json"
    curves = load_curves(curves_path)

    cohorts_map = build_cohorts_until(data, target_date)
    rows = predict_single_day(target_date, cohorts_map, apps_input, curves)

    total_predicted = sum(r["y_pred"] for r in rows)
    return JSONResponse(content={
        "target_date": target_date_str,
        "predicted_apps": len(rows),
        "total_predicted_revenue": round(total_predicted, 2),
        "rows": rows,
    })


@router.post("/web/retrain/trigger")
async def web_trigger_retrain():
    """手动触发每日重训（异步执行，不阻塞响应）"""
    from scripts.daily_retrain import main as retrain_main

    def _worker():
        try:
            retrain_main()
        except Exception:
            pass

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return JSONResponse({"status": "started"})


@router.get("/web/monitor", response_class=HTMLResponse)
def web_monitor(request: Request):
    """监控面板：告警历史 + 健康趋势"""
    return _render_model_dashboard({}, initial_view="monitor")

