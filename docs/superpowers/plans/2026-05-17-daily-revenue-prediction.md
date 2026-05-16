# 每日买量收入预测验证表 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用释放曲线模型预测每日买量收入，输出 y_true/y_pred 对比 CSV，并在 Web 面板展示验证图表。

**Architecture:** 新建独立脚本 `scripts/predict_daily_revenue.py` 读取 CSV + 加载 per-app 曲线 → 按 app 按天遍历所有 cohort 计算预测收入 → 输出 `outputs/daily_revenue_predictions.csv`。Web 面板新增区域加载该 CSV 展示 ECharts 对比图。

**Tech Stack:** Python, ECharts (已有), per-app release curves

---

### Task 1: 新建 `scripts/predict_daily_revenue.py` — 数据加载 + 聚合

**Files:**
- Create: `scripts/predict_daily_revenue.py`

- [ ] **Step 1: 创建脚本骨架 + 数据加载函数**

```python
#!/usr/bin/env python
"""每日买量收入预测：用释放曲线模型预测每日总收入，对比实际买量广告收入。"""
import csv
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

_CSV_PATH = Path("data/daily_merged.csv")
_OUTPUT_DIR = Path("outputs")
_CURVE_PATH = Path("outputs/per_app_release_curves.json")

# 默认倍率节点
_NODE_DAYS = [1, 3, 7, 30]
_NODE_MULT = [1.0, 1.1942, 1.3107, 1.4666]


def to_float(v: str) -> float:
    try:
        return float(v) if v not in ("", None) else 0.0
    except ValueError:
        return 0.0


def load_app_daily(csv_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    读取 CSV，按 (app_id, day) 聚合 spend / D1 / buy_revenue。
    返回: {app_id: {day_str: {"spend": ..., "d1_revenue": ..., "buy_revenue": ...}}}
    """
    app_daily: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"spend": 0.0, "d1_revenue": 0.0, "buy_revenue": 0.0})
    )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            app_id = row.get("应用ID", "")
            day = row.get("日期", "")
            if not app_id or not day:
                continue
            d = app_daily[app_id][day]
            d["spend"] += to_float(row.get("消耗金额", "0"))
            d["d1_revenue"] += to_float(row.get("首日广告收入", "0"))
            d["buy_revenue"] += to_float(row.get("买量广告收入", "0"))
    return app_daily
```

- [ ] **Step 2: 验证数据加载**

```bash
python -c "
from scripts.predict_daily_revenue import load_app_daily
from pathlib import Path
data = load_app_daily(Path('data/daily_merged.csv'))
apps = list(data.keys())
print(f'{len(apps)} apps loaded')
app0 = apps[0]
days = list(data[app0].keys())[:3]
for d in days:
    print(f'  {app0} {d}: spend={data[app0][d][\"spend\"]:.0f}, d1={data[app0][d][\"d1_revenue\"]:.0f}, buy={data[app0][d][\"buy_revenue\"]:.0f}')
"
```

Expected: prints N apps and sample daily data

- [ ] **Step 3: Commit**

```bash
git add scripts/predict_daily_revenue.py
git commit -m "feat: 每日买量收入预测 — 数据加载 + 聚合"
```

---

### Task 2: 加载释放曲线 + 日收入预测核心逻辑

**Files:**
- Modify: `scripts/predict_daily_revenue.py`

- [ ] **Step 1: 曲线加载函数 + 插值**

```python
def _build_default_curve() -> Dict[int, float]:
    curve: Dict[int, float] = {}
    for d in range(1, 31):
        if d <= _NODE_DAYS[0]:
            curve[d] = _NODE_MULT[0]
        elif d >= _NODE_DAYS[-1]:
            curve[d] = _NODE_MULT[-1]
        else:
            for i in range(len(_NODE_DAYS) - 1):
                d0, d1 = _NODE_DAYS[i], _NODE_DAYS[i + 1]
                if d0 <= d <= d1:
                    y0, y1 = _NODE_MULT[i], _NODE_MULT[i + 1]
                    curve[d] = y0 + (y1 - y0) * (d - d0) / (d1 - d0)
                    break
    return curve


def load_curves(curve_path: Path) -> Dict[str, Dict[int, float]]:
    """加载 per-app 释放曲线。文件缺失返回空 dict。"""
    if not curve_path.exists():
        print(f"  [WARN] 曲线文件缺失: {curve_path}，全部使用默认曲线")
        return {}
    raw = json.loads(curve_path.read_text(encoding="utf-8"))
    return {
        app_id: {int(d): m for d, m in days.items()}
        for app_id, days in raw.items()
    }


def _curve_value(curve: Dict[int, float], age_day: int) -> float:
    if age_day <= 0:
        return 0.0
    max_day = max(curve.keys())
    age = min(age_day, max_day)
    return curve.get(age, curve[max_day])
```

- [ ] **Step 2: 日收入预测函数**

```python
def predict_daily_revenues(
    app_daily: Dict[str, Dict[str, Dict[str, float]]],
    curves: Dict[str, Dict[int, float]],
) -> List[Dict]:
    """
    遍历每个 app 的每天，用释放曲线预测当日总收入。
    返回 list of dict，每行：日期, 应用ID, y_true, y_pred, days_since_start
    """
    default_curve = _build_default_curve()
    rows: List[Dict] = []

    for app_id, day_data in app_daily.items():
        curve = curves.get(app_id, default_curve)
        sorted_days = sorted(day_data.keys())
        app_start_day = sorted_days[0]

        # 累积 cohort 列表: [(spend_day, spend, d1_roi)]
        cohorts: List[Tuple[date, float, float]] = []

        for day_str in sorted_days:
            current_day = datetime.strptime(day_str, "%Y-%m-%d").date()
            info = day_data[day_str]
            spend = info["spend"]
            d1_rev = info["d1_revenue"]
            buy_rev = info["buy_revenue"]
            d1_roi = d1_rev / spend if spend > 0 else 0.0
            days_since_start = (current_day - datetime.strptime(app_start_day, "%Y-%m-%d").date()).days + 1

            # 当日新 cohort 加入
            if spend > 0 and d1_roi > 0:
                cohorts.append((current_day, spend, d1_roi))

            # 预测当日收入 = Σ cohort × d1_roi × [F(age_today) - F(age_yesterday)]
            predicted = 0.0
            for cday, cspend, cd1 in cohorts:
                age_today = (current_day - cday).days + 1
                age_yesterday = age_today - 1
                f_today = _curve_value(curve, age_today)
                f_yesterday = _curve_value(curve, age_yesterday)
                delta = f_today - f_yesterday
                if delta > 0:
                    predicted += cspend * cd1 * delta

            rows.append({
                "日期": day_str,
                "应用ID": app_id,
                "y_true": round(buy_rev, 2),
                "y_pred": round(predicted, 2),
                "days_since_start": days_since_start,
            })

    return rows
```

- [ ] **Step 3: 验证逻辑正确性**

```bash
python -c "
from scripts.predict_daily_revenue import load_app_daily, load_curves, predict_daily_revenues
from pathlib import Path
data = load_app_daily(Path('data/daily_merged.csv'))
curves = load_curves(Path('outputs/per_app_release_curves.json'))
rows = predict_daily_revenues(data, curves)
apps = set(r['应用ID'] for r in rows)
print(f'{len(rows)} rows, {len(apps)} apps')
# 检查首日：y_pred 应接近 d1_revenue（age=1 时倍率=1.0）
sample = [r for r in rows if r['days_since_start'] == 1][:3]
for r in sample:
    print(f'  {r[\"日期\"]} app={r[\"应用ID\"]} y_true={r[\"y_true\"]:.0f} y_pred={r[\"y_pred\"]:.0f} (day1)')
"
```

Expected: prints row count, app count, sample day-1 predictions (y_pred ≈ actual D1 revenue)

- [ ] **Step 4: Commit**

```bash
git add scripts/predict_daily_revenue.py
git commit -m "feat: 每日买量收入预测 — 曲线加载 + 日收入预测核心逻辑"
```

---

### Task 3: 主流程 + CSV 输出 + MAPE 评估

**Files:**
- Modify: `scripts/predict_daily_revenue.py`

- [ ] **Step 1: 保存函数 + 评估函数**

```python
def save_predictions(rows: List[Dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["日期", "应用ID", "y_true", "y_pred", "days_since_start"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"预测结果已保存: {output_path} ({len(rows)} rows)")
    return output_path


def compute_mape(rows: List[Dict], min_days: int = 30) -> Dict[str, float]:
    """按 app 计算 MAPE，仅统计 days_since_start >= min_days 的行。"""
    app_errors: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["days_since_start"] < min_days:
            continue
        if r["y_true"] > 0:
            ape = abs(r["y_true"] - r["y_pred"]) / r["y_true"]
            app_errors[r["应用ID"]].append(ape)

    per_app = {}
    for app_id, apes in sorted(app_errors.items()):
        if apes:
            per_app[app_id] = round(sum(apes) / len(apes) * 100, 2)

    all_apes = [a for apes in app_errors.values() for a in apes]
    overall = round(sum(all_apes) / len(all_apes) * 100, 2) if all_apes else 0.0

    print(f"\nMAPE (days>={min_days}): overall={overall}%")
    print(f"Per-app MAPE (top 5 by sample count):")
    sorted_apps = sorted(app_errors.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for app_id, apes in sorted_apps:
        print(f"  {app_id}: {per_app[app_id]}% ({len(apes)} samples)")
    return {"overall": overall, "per_app": per_app}


def main():
    csv_path = _CSV_PATH
    output_path = _OUTPUT_DIR / "daily_revenue_predictions.csv"

    print("加载数据...")
    data = load_app_daily(csv_path)
    print(f"  {len(data)} apps")

    print("加载释放曲线...")
    curves = load_curves(_CURVE_PATH)
    print(f"  {len(curves)} per-app curves")

    print("预测每日收入...")
    rows = predict_daily_revenues(data, curves)

    save_predictions(rows, output_path)
    compute_mape(rows, min_days=30)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行脚本验证端到端**

```bash
python scripts/predict_daily_revenue.py
```

Expected: prints app count, curve count, saves CSV, prints MAPE

- [ ] **Step 3: Commit**

```bash
git add scripts/predict_daily_revenue.py
git commit -m "feat: 每日买量收入预测 — 主流程 + CSV 输出 + MAPE"
```

---

### Task 4: Web 面板 — 每日收入验证区域

**Files:**
- Modify: `app/web.py`

- [ ] **Step 1: 新增数据加载端点**

在 `app/web.py` 中找到其他 `/web/` 数据端点，新增：

```python
@app.get("/web/predictions/daily_revenue")
async def web_daily_revenue_predictions(
    request: Request,
    app_id: str = Query(""),
):
    """每日买量收入预测验证数据"""
    csv_path = Path("outputs/daily_revenue_predictions.csv")
    if not csv_path.exists():
        return JSONResponse(content={"rows": [], "message": "预测数据尚未生成，请先运行 scripts/predict_daily_revenue.py"})

    rows = _read_csv_rows(csv_path)
    if app_id:
        rows = [r for r in rows if r.get("应用ID") == app_id]
    return JSONResponse(content={"rows": rows, "total": len(rows)})
```

- [ ] **Step 2: 前端 HTML — 新增 tab 页签**

在决策面板 tab 区域（搜索现有 `tab-btn` 或 `tabButton`）新增一个 tab：

```html
<button class="tab-btn" data-tab="daily-revenue">每日收入验证</button>
```

- [ ] **Step 3: 前端 JS — ECharts 对比图**

新增 tab 内容和图表渲染函数：

```javascript
async function renderDailyRevenueChart(appId = '') {
  const params = appId ? `?app_id=${appId}` : '';
  const resp = await fetch(`/web/predictions/daily_revenue${params}`);
  const data = await resp.json();

  if (!data.rows || data.rows.length === 0) {
    document.getElementById('dailyRevenueChart').innerHTML =
      '<div class="plain-note">暂无数据。运行 scripts/predict_daily_revenue.py 生成预测。</div>';
    return;
  }

  const dates = [...new Set(data.rows.map(r => r['日期']))].sort();
  const yTrue = dates.map(d => {
    const r = data.rows.find(rr => rr['日期'] === d);
    return r ? parseFloat(r.y_true) : 0;
  });
  const yPred = dates.map(d => {
    const r = data.rows.find(rr => rr['日期'] === d);
    return r ? parseFloat(r.y_pred) : 0;
  });

  const chart = echarts.init(document.getElementById('dailyRevenueChart'));
  chart.setOption({
    title: { text: '每日买量收入：预测 vs 实际' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['实际 (y_true)', '预测 (y_pred)'] },
    xAxis: { type: 'category', data: dates, axisLabel: { rotate: 45 } },
    yAxis: { type: 'value', name: '收入' },
    series: [
      { name: '实际 (y_true)', type: 'line', data: yTrue, smooth: true },
      { name: '预测 (y_pred)', type: 'line', data: yPred, smooth: true,
        lineStyle: { type: 'dashed' } },
    ],
  });
}
```

App 筛选下拉框复用现有模式（搜索 `appTableSearch` 类似的输入框）。

- [ ] **Step 4: 验证 Web 面板**

```bash
# 先确保预测 CSV 已生成
python scripts/predict_daily_revenue.py
# 启动 Web 服务
bash scripts/run_web_8000.sh &
sleep 2
# 检查端点
curl -s http://localhost:8000/web/predictions/daily_revenue | python -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"total\"]} rows')"
```

Expected: prints N rows

- [ ] **Step 5: Commit**

```bash
git add app/web.py
git commit -m "feat: Web 面板新增每日买量收入预测验证区域"
```
