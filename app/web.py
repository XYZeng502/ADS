import csv
import html
import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.core.calendar import CalendarService
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
            if len(rows) >= limit:
                break
    return rows


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


def _read_csv_rows(path: Path | None, limit: int = 5000) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    return _read_csv(path, limit)


def _load_model_dashboard_payload() -> Dict[str, Any]:
    root = _repo_root() / "outputs"
    roi_dir = _pick_existing(
        [
            root / "model_parallel_roi_d1_default_weekly_v3",
            root / "model_parallel_roi_d1_default_weekly_v2",
            root / "model_parallel_roi_d1_default_weekly",
            root / "model_parallel_roi_d1",
        ]
    )
    spend_dir = _pick_existing(
        [
            root / "model_parallel_spend_t1_target_calendar_calibrated_app",
            root / "model_parallel_spend_t1_split_online_fusion_script_verified",
        ]
    )
    date_feature_dir = root / "model_parallel_spend_t1_date_features_lgbm"
    old_spend_path = root / "model_parallel_spend_t1_split_online_fusion_script_verified" / "predictions_reconciled.csv"
    app_month_roi_path = root / "app_level_last_day_suggestions.csv"
    app_month_roi_summary = _read_json(root / "app_level_last_day_prediction.json")

    roi_metrics_path = roi_dir / "metrics_summary.csv" if roi_dir else None
    roi_pred_path = roi_dir / "predictions_ExtraTrees_hard_tuned_log.csv" if roi_dir else None
    if roi_pred_path and not roi_pred_path.exists() and roi_dir:
        roi_pred_path = _pick_existing(list(roi_dir.glob("predictions_*.csv")))

    spend_predictions = _read_csv_rows(
        _pick_existing(
            [
                spend_dir / "predictions_GBDT_log.csv" if spend_dir else Path("__missing__"),
                spend_dir / "predictions_LightGBM_log.csv" if spend_dir else Path("__missing__"),
                spend_dir / "predictions_reconciled.csv" if spend_dir else Path("__missing__"),
            ]
        ),
        5000,
    )
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

    payload = {
        "roi": {
            "title": "T+1 ROI_D1 预测效果",
            "metrics": _read_csv_rows(roi_metrics_path, 100),
            "predictions": _read_csv_rows(roi_pred_path, 5000),
            "source": str(roi_dir) if roi_dir else "",
        },
        "spend": {
            "title": "T+1 Spend 预测效果（目标日历 + 假期切换校准）",
            "report": _read_json(spend_dir / "report.json") if spend_dir else {},
            "predictions": spend_predictions,
            "baseline_predictions": _read_csv_rows(old_spend_path, 5000),
            "calendar": spend_calendar,
            "source": str(spend_dir) if spend_dir else "",
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
    return payload


def _render_model_dashboard(payload: Dict[str, Any], initial_view: str = "predict") -> HTMLResponse:
    safe_payload_js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    safe_initial_view_js = json.dumps(initial_view if initial_view in {"predict", "recommend"} else "predict")
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
    .trace-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }}
    .trace-card {{ border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: #fff; min-height: 132px; }}
    .trace-card .t {{ display:flex; justify-content:space-between; gap:8px; align-items:center; font-weight:700; }}
    .trace-card .s {{ font-size: 11px; border-radius: 999px; padding: 2px 8px; }}
    .s-ok {{ color:#15803d; background:#dcfce7; }}
    .s-warn {{ color:#a16207; background:#fef3c7; }}
    .s-critical {{ color:#b91c1c; background:#fee2e2; }}
    .trace-card ul {{ margin: 10px 0 0 16px; padding:0; color: var(--muted); line-height: 1.5; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); }}
    th {{ color: var(--muted); font-weight: 650; background: #f8fafc; position: sticky; top: 0; }}
    th.sortable {{ cursor:pointer; user-select:none; }}
    th.sortable:hover {{ color: var(--blue); background:#eef5ff; }}
    .table-wrap {{ max-height: 360px; overflow: auto; border: 1px solid var(--line); border-radius: 14px; }}
    .good {{ color: var(--green); }}
    .bad {{ color: var(--red); }}
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
    @media (max-width: 1100px) {{ .shell {{ display:block; }} .topnav {{ position:static; flex-direction:row; margin-bottom:14px; }} .controls, .roi-controls, .grid4, .grid2, .charts, .trace-grid {{ grid-template-columns: 1fr; }} .hero {{ flex-direction: column; }} }}
  </style>
</head>
<body>
<div class="shell">
  <div class="hero">
    <div>
      <div class="title">智能预算决策系统 · 模型评估看板</div>
      <div class="subtitle">这个页面只展示已有模型输出支撑的内容：预测值 vs 实际值、误差走势、散点校准和明细。月末 ROI 暂不作为主展示，因为当前偏差还不能支撑强业务判断。</div>
    </div>
    <div class="badge">白色简洁版 · 强交互 · 真实结果</div>
  </div>

  <div class="topnav">
    <div class="nav-title">业务模块</div>
    <button class="navbtn active" id="predictNav" data-view="predict">预测模块</button>
    <button class="navbtn" id="recommendNav" data-view="recommend">推荐模块</button>
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
      <label>应用ID
        <select id="appSel"><option value="">全部应用</option></select>
      </label>
      <label>开始日期
        <input type="date" id="startDate" />
      </label>
      <label>结束日期
        <input type="date" id="endDate" />
      </label>
      <button id="resetBtn">重置筛选</button>
    </div>
  </div>

  <div class="grid4" id="metricCards"></div>

  <div class="panel">
    <div class="section-title">
      <h2 id="lineTitle">预测值 vs 实际值</h2>
      <div class="hint">横轴：日期；纵轴：当前目标数值；图例说明每条线含义</div>
    </div>
    <div class="charts">
      <div id="lineChart" class="chart"></div>
      <div id="scatterChart" class="chart"></div>
    </div>
  </div>

  <div class="panel">
    <div class="section-title">
      <h2>每日误差走势</h2>
      <div class="hint">横轴：日期；纵轴：绝对百分比误差 APE（越低越好）</div>
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
      <h2>预测明细</h2>
      <div class="hint">每一行是某应用在某天的预测和实际结果，可用于追查具体偏差</div>
    </div>
    <div class="table-wrap">
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
    <div class="plain-note" style="margin-bottom:14px;">
      推荐模块现在直接以“应用月末 ROI 预测表”为主入口：先筛出未达标、放量、控量或有告警的应用，再看建议动作和推荐依据。图表只作为总览辅助，不再让你先看一堆不明确的图。
    </div>
    <div class="panel" style="margin-top:14px;">
      <div class="section-title">
        <h2>应用月末 ROI 预测表</h2>
        <div class="hint">推荐模块核心表；每一行对应一个应用，点击任意表头可排序</div>
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
      <h2>重点应用推荐依据</h2>
      <div class="hint">按风险和动作优先级排序</div>
    </div>
    <div class="trace-grid" id="traceCards"></div>
  </div>
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
  const app = document.getElementById('appSel').value;
  const start = document.getElementById('startDate').value;
  const end = document.getElementById('endDate').value;
  return rowsForTarget(target).filter(r => (!app || r.app === app) && (!start || r.date >= start) && (!end || r.date <= end));
}}

let appRoiSort = {{ key: 'app', dir: 'asc' }};
let appRoiQuickFilter = 'all';
let detailSort = {{ key: 'date', dir: 'asc' }};

function compareRows(a, b, sortState) {{
  const av = a[sortState.key];
  const bv = b[sortState.key];
  const cmp = typeof av === 'string' ? String(av).localeCompare(String(bv)) : Number(av) - Number(bv);
  return sortState.dir === 'asc' ? cmp : -cmp;
}}

function appRoiDecision(r) {{
  const gap = r.roiGap;
  const alertCount = r.alertCount;
  const budgetChange = r.actualSpend > 1e-8 ? (r.plannedBudget - r.actualSpend) / r.actualSpend : 0;
  let action = '维持观察';
  if (gap >= 0.15 && alertCount === 0) action = '放量并小幅调高出价';
  else if (gap >= 0.05 && alertCount <= 1) action = '小幅放量';
  else if (gap < -0.05 || alertCount >= 2) action = '收紧并降低出价';
  else if (gap < 0.02 || alertCount > 0) action = '保守控量';

  const basis = [
    `预测月末ROI=${{fmt(r.monthEndRoi,4)}}，KPI差距=${{pct(gap)}}`,
    `昨日实际消耗=${{fmt(r.actualSpend,2)}}，计划预算=${{fmt(r.plannedBudget,2)}}，预算变化=${{pct(budgetChange)}}`,
    `风险告警=${{alertCount}}，偏差主因=${{r.dominantFactor}}`,
    `D1锚点 raw=${{fmt(r.d1Raw,4)}} → calibrated=${{fmt(r.d1Calibrated,4)}}`,
  ].join('；');
  return {{ action, basis, budgetChange }};
}}

function appMonthRoiRows() {{
  return (payload.app_month_roi.rows || []).map(r => {{
    const monthEndRoi = num(r.month_end_roi_prediction);
    const kpi = num(payload.app_month_roi.kpi || 1.05);
    const alertCount = num(r.alert_count);
    const row = {{
      targetDay: r.target_day || '',
      app: String(r.app_id || ''),
      mode: r.mode || '-',
      actualSpend: num(r.actual_spend_last_day),
      actualRevenue: num(r.actual_revenue_last_day),
      plannedBudget: num(r.planned_total_budget),
      monthEndRoi,
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
      topActions: r.top_actions || '',
      roiGap: monthEndRoi - kpi,
    }};
    const decision = appRoiDecision(row);
    const priorityScore =
      Math.max(0, -row.roiGap) * 100 +
      row.alertCount * 8 +
      Math.abs(decision.budgetChange) * 12 +
      (row.dominantFactor === 'BALANCED' ? 1 : 2);
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
  if (hintEl) hintEl.textContent = `数据来源：${{payload.app_month_roi.source || '暂无'}} · 原始行数 ${{ (payload.app_month_roi.rows || []).length }}`;
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
    ['targetDay', '预测日期'],
    ['monthEndRoi', '预测月末ROI'],
    ['roiGap', '较KPI差距'],
    ['action', '建议动作'],
    ['monthEndSpend', '预测月末消耗'],
    ['plannedBudget', '今日计划预算'],
    ['budgetChange', '预算变化'],
    ['roi30d', '30日总ROI'],
    ['roiA', 'Raw锚点ROI'],
    ['roiB', '校准锚点ROI'],
    ['roiDelta', '校准差异'],
    ['alertCount', '告警数'],
    ['dominantFactor', '偏差主因'],
    ['d1Raw', 'D1 Raw'],
    ['d1Calibrated', 'D1校准'],
    ['priorityScore', '处理优先级'],
    ['basis', '推荐依据'],
    ['topActions', 'Top计划动作'],
  ];
  const headEl = document.getElementById('appRoiHead');
  const bodyEl = document.getElementById('appRoiBody');
  if (!headEl || !bodyEl) return;
  headEl.innerHTML = '<tr>' + headers.map(([key, name]) => {{
    const mark = appRoiSort.key === key ? (appRoiSort.dir === 'asc' ? ' ↑' : ' ↓') : '';
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
      <td>${{r.app}}</td>
      <td>${{r.targetDay}}</td>
      <td class="${{roiCls}}">${{fmt(r.monthEndRoi,4)}}</td>
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
      <td>${{r.dominantFactor}}</td>
      <td>${{fmt(r.d1Raw,4)}}</td>
      <td>${{fmt(r.d1Calibrated,4)}}</td>
      <td>${{fmt(r.priorityScore,2)}}</td>
      <td class="basis">${{r.basis}}</td>
      <td class="basis">${{r.topActions}}</td>
    </tr>`;
  }}).join('');
}}

function refreshApps() {{
  const target = document.getElementById('targetSel').value;
  const current = document.getElementById('appSel').value;
  const apps = [...new Set(rowsForTarget(target).map(r => r.app).filter(Boolean))].sort();
  document.getElementById('appSel').innerHTML = '<option value="">全部应用</option>' + apps.map(a => `<option value="${{a}}">${{a}}</option>`).join('');
  if (apps.includes(current)) document.getElementById('appSel').value = current;
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
  const m = calcMetrics(rows);
  const cards = [
    ['样本数', String(m.n), target === 'roi' ? '应用-日期 ROI 样本' : '应用-日期 spend 样本'],
    ['MAPE', pct(m.mape), '平均百分比误差，越低越好'],
    ['WMAPE', pct(m.wmape), '按实际规模加权，更接近业务影响'],
    ['30%内命中率', pct(m.hit30), '预测偏差在 30% 以内的样本占比'],
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
  const riskRows = rows.filter(r => r.alertCount > 0 || r.roiGap < 0);
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

  const topRows = [...rows].sort((a,b)=>Math.abs(b.roiGap)-Math.abs(a.roiGap)).slice(0, 12);
  charts.product.setOption({{
    color: ['#2563eb', '#16a34a'],
    tooltip: {{ trigger:'axis' }},
    legend: {{ top: 8 }},
    grid: {{ left: 72, right: 24, top: 56, bottom: 74 }},
    xAxis: {{ type:'category', name:'应用ID', nameLocation:'middle', nameGap:50, axisLabel:{{rotate:25}}, data:topRows.map(r=>r.app) }},
    yAxis: [
      {{ type:'value', name:'今日计划预算' }},
      {{ type:'value', name:'预测月末ROI' }}
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
    xAxis: {{ type:'category', name:'偏差主因', nameLocation:'middle', nameGap:34, data:[...factorCounts.keys()] }},
    yAxis: {{ type:'value', name:'应用数' }},
    series: [{{ name:'应用数', type:'bar', data:[...factorCounts.values()] }}]
  }});

  const topRecommend = [...rows].sort((a,b)=>Math.abs(b.roiGap)-Math.abs(a.roiGap)).slice(0, 5)
    .map(r => `<li><b>应用 ${{r.app}}</b>：${{r.action}}；${{r.basis}}</li>`).join('');
  const riskList = riskRows.slice(0, 5).map(r => `<li>应用 ${{r.app}}：月末ROI=${{fmt(r.monthEndRoi,4)}}，告警=${{r.alertCount}}，建议=${{r.action}}</li>`).join('');
  document.getElementById('decisionNote').innerHTML = `
    <b>业务结论：</b> 基于历史 CSV 的应用层月末 ROI 预测，平均预测月末 ROI=${{fmt(avgRoi,4)}}，相对 KPI=${{pct(avgRoi-kpi)}}。<br/>
    <b>重点推荐：</b><ul>${{topRecommend}}</ul>
    <b>风险应用：</b><ul>${{riskList || '<li>暂无风险应用</li>'}}</ul>
  `;

  const traceCards = [
    ['KPI层', avgRoi >= kpi ? 'OK' : 'WARN', [`平均预测月末ROI=${{fmt(avgRoi,4)}}，KPI=${{fmt(kpi,4)}}。`]],
    ['预测层', 'OK', [`覆盖 ${{rows.length}} 个应用；偏差主因分布来自应用层 attribution。`]],
    ['推荐层', 'OK', [`主要动作=${{topAction[0]}}，覆盖 ${{topAction[1]}} 个应用。`]],
    ['风险层', riskRows.length ? 'WARN' : 'OK', [`风险应用数=${{riskRows.length}}，判定依据为告警或低于KPI。`]],
    ['解释层', 'OK', ['每个应用的推荐依据已在预测模块表格中按行展开。']],
  ];
  document.getElementById('traceCards').innerHTML = traceCards.map(([title, status, findings]) => {{
    const cls = status === 'CRITICAL' ? 's-critical' : (status === 'WARN' ? 's-warn' : 's-ok');
    return `<div class="trace-card"><div class="t"><span>${{title}}</span><span class="s ${{cls}}">${{status}}</span></div><ul>${{findings.map(x=>`<li>${{x}}</li>`).join('')}}</ul></div>`;
  }}).join('');
}}

function renderCharts(rows) {{
  const target = document.getElementById('targetSel').value;
  const unit = target === 'roi' ? 'ROI_D1' : '消耗金额';
  const data = aggregateByDate(rows);
  const dates = data.map(x => x.date);
  const actualName = target === 'roi' ? '实际 T+1 ROI_D1' : '实际 T+1 Spend';
  const predName = target === 'roi' ? '预测 T+1 ROI_D1' : '目标日历校准预测 T+1 Spend';
  const series = [
    {{ name: actualName, type: 'line', smooth: true, symbolSize: 6, data: data.map(x => x.actual) }},
    {{ name: predName, type: 'line', smooth: true, symbolSize: 6, data: data.map(x => x.pred) }},
  ];
  if (target === 'spend' && data.some(x => Number.isFinite(x.pred2) && x.pred2 > 0)) series.push({{ name: '应用基线预测 T+1 Spend', type: 'line', smooth: true, symbolSize: 5, data: data.map(x => x.pred2) }});
  charts.line.setOption({{
    color: ['#16a34a', '#2563eb', '#f59e0b'],
    tooltip: {{ trigger: 'axis' }},
    legend: {{ top: 8 }},
    grid: {{ left: 64, right: 28, top: 58, bottom: 55 }},
    xAxis: {{ type: 'category', name: '日期', nameLocation: 'middle', nameGap: 34, data: dates }},
    yAxis: {{ type: 'value', name: unit, nameGap: 48 }},
    dataZoom: [{{type:'inside'}}, {{type:'slider', height: 18, bottom: 12}}],
    series
  }});
  charts.scatter.setOption({{
    color: ['#2563eb'],
    tooltip: {{ formatter: p => `实际：${{fmt(p.value[0], target==='roi'?4:2)}}<br/>预测：${{fmt(p.value[1], target==='roi'?4:2)}}` }},
    grid: {{ left: 62, right: 24, top: 36, bottom: 54 }},
    xAxis: {{ type:'value', name:`实际${{unit}}`, nameLocation:'middle', nameGap:34 }},
    yAxis: {{ type:'value', name:`预测${{unit}}`, nameGap:44 }},
    series: [{{ name:'单样本预测校准', type:'scatter', symbolSize: 7, data: rows.map(r => [r.actual, r.pred]) }}]
  }});
  charts.error.setOption({{
    color: ['#dc2626'],
    tooltip: {{ trigger:'axis', valueFormatter: v => fmt(v, 2) + '%' }},
    grid: {{ left: 64, right: 28, top: 36, bottom: 55 }},
    xAxis: {{ type:'category', name:'日期', nameLocation:'middle', nameGap:34, data: dates }},
    yAxis: {{ type:'value', name:'APE (%)', nameGap:46 }},
    dataZoom: [{{type:'inside'}}, {{type:'slider', height: 18, bottom: 12}}],
    series: [{{ name:'每日汇总绝对百分比误差', type:'bar', data: data.map(x => x.ape * 100) }}]
  }});
}}

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
    document.getElementById('modelSummary').innerHTML = `Spend 当前使用 <b>目标日历特征 + 假期切换校准</b>：最优模型=${{r.best_model_by_mape || '模型输出'}}，MAPE=${{fmt(num(r.best_mape_pct),2)}}%。表格已增加“旧预测/旧APE/新APE/改善幅度/T+1日历标签”，清明与节后样本平均改善=${{pct(avgImprove)}}。`;
  }}
}}

function renderTable(rows) {{
  const target = document.getElementById('targetSel').value;
  const digits = target === 'roi' ? 4 : 2;
  const enriched = rows.map(r => ({{
    ...r,
    ape: r.actual > 1e-8 ? Math.abs((r.actual-r.pred)/r.actual) : 0,
  }}));
  const headers = target === 'roi'
    ? [
      ['date','日期'], ['app','应用ID'], ['actual','实际 ROI_D1'], ['pred','预测 ROI_D1'], ['ape','APE'], ['model','模型']
    ]
    : [
      ['date','日期'], ['targetDate','T+1日期'], ['calendarLabel','日历标签'], ['app','应用ID'], ['actual','实际 Spend'],
      ['pred','新版预测'], ['oldPred','旧版预测'], ['newApe','新版APE'], ['oldApe','旧版APE'], ['improvement','改善'], ['model','模型']
    ];
  document.getElementById('detailHead').innerHTML = '<tr>' + headers.map(([key, name]) => {{
    const mark = detailSort.key === key ? (detailSort.dir === 'asc' ? ' ↑' : ' ↓') : ' ↕';
    return `<th class="sortable" data-detail-sort="${{key}}">${{name}}${{mark}}</th>`;
  }}).join('') + '</tr>';
  document.querySelectorAll('#detailHead th[data-detail-sort]').forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.dataset.detailSort;
      if (detailSort.key === key) detailSort.dir = detailSort.dir === 'asc' ? 'desc' : 'asc';
      else detailSort = {{ key, dir: ['date','app','model','calendarLabel','targetDate'].includes(key) ? 'asc' : 'desc' }};
      renderTable(currentRows());
    }});
  }});
  const tableRows = [...enriched].sort((a,b) => compareRows(a, b, detailSort));
  document.getElementById('detailBody').innerHTML = tableRows.slice(0, 300).map(r => {{
    const ape = r.ape;
    const cls = ape <= .3 ? 'good' : 'bad';
    if (target === 'roi') return `<tr><td>${{r.date}}</td><td>${{r.app}}</td><td>${{fmt(r.actual,digits)}}</td><td>${{fmt(r.pred,digits)}}</td><td class="${{cls}}">${{pct(ape)}}</td><td>${{r.model}}</td></tr>`;
    const impCls = Number.isFinite(r.improvement) && r.improvement >= 0 ? 'good' : 'bad';
    return `<tr><td>${{r.date}}</td><td>${{r.targetDate}}</td><td>${{r.calendarLabel}}</td><td>${{r.app}}</td><td>${{fmt(r.actual,digits)}}</td><td>${{fmt(r.pred,digits)}}</td><td>${{fmt(r.oldPred,digits)}}</td><td class="${{cls}}">${{pct(r.newApe)}}</td><td>${{pct(r.oldApe)}}</td><td class="${{impCls}}">${{pct(r.improvement)}}</td><td>${{r.model}}</td></tr>`;
  }}).join('');
}}

function render() {{
  refreshApps();
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

['targetSel','appSel','startDate','endDate'].forEach(id => {{
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', render);
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
  appRoiSort = {{ key: 'app', dir: 'asc' }};
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
    else appRoiSort = {{ key, dir: key === 'monthEndRoi' || key === 'roiGap' ? 'asc' : 'desc' }};
    renderAppMonthRoiTable();
  }});
}}
function setActiveView(view) {{
  document.querySelectorAll('.navbtn').forEach(x => x.classList.toggle('active', x.dataset.view === view));
  document.getElementById('view-predict').classList.toggle('hidden', view !== 'predict');
  document.getElementById('view-recommend').classList.toggle('hidden', view !== 'recommend');
  setTimeout(() => Object.values(charts).forEach(c => c.resize()), 0);
}}
document.querySelectorAll('.navbtn').forEach(btn => btn.addEventListener('click', () => setActiveView(btn.dataset.view)));
const rb = document.getElementById('resetBtn');
if (rb) rb.addEventListener('click', () => {{
  document.getElementById('appSel').value = '';
  document.getElementById('startDate').value = '';
  document.getElementById('endDate').value = '';
  render();
}});
window.addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));
render();
setActiveView(initialView);
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
        'target_day','app_id','mode','actual_spend_last_day','actual_revenue_last_day','planned_total_budget',
        'month_end_roi_prediction','month_end_spend_prediction','alert_count','roi_dominant_factor',
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

