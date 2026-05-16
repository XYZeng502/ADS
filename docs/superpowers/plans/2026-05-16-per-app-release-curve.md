# Per-App 真实释放曲线 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用每个 app 历史数据中观测到的真实 D3/D1、D7/D1、D30/D1 比率构建 per-app 释放曲线，替换全局固定 archetype 曲线。

**Architecture:** 在 `offline_backtest.py` 训练循环中累积 per-app 的 D1/D3/D7/D30 总量 → 构建 per-app 倍率曲线 → 存入 JSON 文件。`predictor.py` 启动时加载该文件，`predict_month_end_roi` 按 `app_id` 查找对应曲线。Fallback 链：per-app 观测曲线 > 竞品模板 > 默认曲线。

**Tech Stack:** Python, no new dependencies.

---

### Task 1: SlotAgg 增加 D3/D7 字段 + 数据加载读取 D3/D7

**Files:**
- Modify: `scripts/offline_backtest.py:32-36` (SlotAgg)
- Modify: `scripts/offline_backtest.py:454-472` (load_aggregates_app_level)
- Modify: `scripts/offline_backtest.py:432-451` (load_aggregates, 保持一致性)

- [ ] **Step 1: SlotAgg 加字段**

```python
# scripts/offline_backtest.py line 32-36
@dataclass
class SlotAgg:
    spend: float = 0.0
    d1_revenue: float = 0.0
    d3_revenue: float = 0.0
    d7_revenue: float = 0.0
    d30_revenue: float = 0.0
```

- [ ] **Step 2: load_aggregates_app_level 读取 D3/D7 列**

```python
# scripts/offline_backtest.py line 468-471, 替换 agg 字段赋值
agg: SlotAgg = nested[day][app][adv][product][slot]
agg.spend += to_float(row.get("消耗金额", "0"))
agg.d1_revenue += to_float(row.get("首日广告收入", "0"))
agg.d3_revenue += to_float(row.get("3日累计变现金额", "0"))
agg.d7_revenue += to_float(row.get("7日累计变现金额", "0"))
agg.d30_revenue += to_float(row.get("30日累计变现金额", "0"))
```

- [ ] **Step 3: load_aggregates 同步更新（非 app 级版本，保持一致性）**

```python
# scripts/offline_backtest.py line 446-449
agg: SlotAgg = nested[day][adv][product][slot]
agg.spend += to_float(row.get("消耗金额", "0"))
agg.d1_revenue += to_float(row.get("首日广告收入", "0"))
agg.d3_revenue += to_float(row.get("3日累计变现金额", "0"))
agg.d7_revenue += to_float(row.get("7日累计变现金额", "0"))
agg.d30_revenue += to_float(row.get("30日累计变现金额", "0"))
```

- [ ] **Step 4: 验证导入**

```bash
python -c "from scripts.offline_backtest import SlotAgg; a=SlotAgg(); print(a)"
```

Expected: `SlotAgg(spend=0.0, d1_revenue=0.0, d3_revenue=0.0, d7_revenue=0.0, d30_revenue=0.0)`

---

### Task 2: Per-app 曲线构建函数

**Files:**
- Modify: `scripts/offline_backtest.py` (在 `_load_release_multiplier_curve` 附近新增函数)

- [ ] **Step 1: 新增 `_build_per_app_curve` 函数**

```python
# scripts/offline_backtest.py, 在 _load_release_multiplier_curve 之后
def _build_per_app_curve(
    app_id: str,
    total_d1: float,
    total_d3: float,
    total_d7: float,
    total_d30: float,
    train_days: int,
    min_train_days: int,
) -> Dict[int, float] | None:
    """
    用 app 历史累积的 D1/D3/D7/D30 总量构建释放倍率曲线。
    返回 day->multiplier 的 dict，或 None 表示回退到全局曲线。
    """
    if train_days < min_train_days:
        return None
    if total_d1 <= 0:
        return None

    raw_nodes = {
        1: 1.0,
        3: total_d3 / total_d1,
        7: total_d7 / total_d1,
        30: total_d30 / total_d1,
    }

    # 异常值检查：任一节点比率 > 5.0 则回退
    for d, ratio in raw_nodes.items():
        if ratio <= 0 or ratio > 5.0:
            return None

    # clamp 到 [1.0, 3.0]
    nodes = {d: max(1.0, min(ratio, 3.0)) for d, ratio in raw_nodes.items()}

    # 节点间线性插值，day 1-30
    node_days = [1, 3, 7, 30]
    node_mult = [nodes[d] for d in node_days]
    curve: Dict[int, float] = {}
    for d in range(1, 31):
        if d <= node_days[0]:
            curve[d] = node_mult[0]
            continue
        if d >= node_days[-1]:
            curve[d] = node_mult[-1]
            continue
        for i in range(len(node_days) - 1):
            d0, d1 = node_days[i], node_days[i + 1]
            if d0 <= d <= d1:
                y0, y1 = node_mult[i], node_mult[i + 1]
                ratio = (d - d0) / (d1 - d0)
                curve[d] = y0 + (y1 - y0) * ratio
                break
    return curve
```

- [ ] **Step 2: 新增 `_build_all_per_app_curves` 函数**

```python
# scripts/offline_backtest.py
def _build_all_per_app_curves(
    app_totals: Dict[str, Dict[str, float]],
    app_train_days: Dict[str, int],
    min_train_days: int,
) -> Dict[str, Dict[int, float]]:
    """
    遍历所有 app 的累积总量，构建 per-app 曲线。
    app_totals: {app_id: {"d1": float, "d3": float, "d7": float, "d30": float}}
    返回: {app_id: {day: multiplier}}，仅包含成功构建的 app
    """
    curves: Dict[str, Dict[int, float]] = {}
    for app_id, totals in app_totals.items():
        curve = _build_per_app_curve(
            app_id=app_id,
            total_d1=totals["d1"],
            total_d3=totals["d3"],
            total_d7=totals["d7"],
            total_d30=totals["d30"],
            train_days=app_train_days.get(app_id, 0),
            min_train_days=min_train_days,
        )
        if curve is not None:
            curves[app_id] = curve
    return curves
```

- [ ] **Step 3: 验证函数可导入**

```bash
python -c "
from scripts.offline_backtest import _build_per_app_curve
# 正常情况
c = _build_per_app_curve('test', 1000, 1200, 1350, 1500, 30, 30)
print('curve d3:', c[3], 'd30:', c[30])
# 数据不足
c2 = _build_per_app_curve('test2', 1000, 1200, 1350, 1500, 10, 30)
print('insufficient data:', c2)
# 异常比率
c3 = _build_per_app_curve('test3', 1000, 6000, 7000, 8000, 30, 30)
print('anomaly:', c3)
"
```

Expected: 正常返回曲线，不足返回 None，异常返回 None

---

### Task 3: 训练循环中累积 per-app 总量 + 替换曲线调用

**Files:**
- Modify: `scripts/offline_backtest.py:720-765` (训练循环)
- Modify: `scripts/offline_backtest.py:867-950` (预测阶段，曲线调用)

- [ ] **Step 1: 训练循环中新增 per-app 总量累积**

在 `run_app_level_last_day_prediction` 函数中，训练循环开始前新增：

```python
# 在 slot_stats = defaultdict(...) 之后，训练循环之前
app_d1_total = defaultdict(float)
app_d3_total = defaultdict(float)
app_d7_total = defaultdict(float)
app_d30_total = defaultdict(float)
```

训练循环内部（在 `for slot, agg in slot_map.items():` 块中，读取 agg 数据处），累积每个 slot 的数据后追加：

```python
# 在 p_spend/p_d1 累积之后，slot_stats 更新之前
app_d1_total[app_id] += agg.d1_revenue
app_d3_total[app_id] += agg.d3_revenue
app_d7_total[app_id] += agg.d7_revenue
app_d30_total[app_id] += agg.d30_revenue
```

完整上下文中，训练循环内的核心部分变为：

```python
for slot, agg in slot_map.items():
    p_spend += agg.spend
    p_d1 += agg.d1_revenue
    app_day_spend += agg.spend
    app_day_rev += agg.d1_revenue

    # per-app 曲线累积
    app_d1_total[app_id] += agg.d1_revenue
    app_d3_total[app_id] += agg.d3_revenue
    app_d7_total[app_id] += agg.d7_revenue
    app_d30_total[app_id] += agg.d30_revenue

    key = (app_id, product_uid, slot)
    slot_stats[key]["spend"] += agg.spend
    slot_stats[key]["d1"] += agg.d1_revenue
    slot_stats[key]["d30"] += agg.d30_revenue
    slot_stats[key]["max_spend"] = max(slot_stats[key]["max_spend"], agg.spend)
    slot_stats[key]["days"] += 1
```

- [ ] **Step 2: 训练完成后构建 per-app curves**

在训练循环结束后（`# 上一日消耗...` 之前），新增：

```python
# 构建 per-app 释放曲线
app_totals = {
    app_id: {
        "d1": app_d1_total[app_id],
        "d3": app_d3_total[app_id],
        "d7": app_d7_total[app_id],
        "d30": app_d30_total[app_id],
    }
    for app_id in app_d1_total
}
per_app_curves = _build_all_per_app_curves(
    app_totals=app_totals,
    app_train_days=app_train_days,
    min_train_days=min_train_days,
)
print(f"  Per-app 曲线: {len(per_app_curves)}/{len(app_totals)} 个 app 构建成功")
```

- [ ] **Step 3: 将 per-app curve 传入 `_month_end_incremental_roi_for_app`**

当前调用在 `run_app_level_last_day_prediction` 的预测阶段（A/B 分支内部）。将 `release_curve` 替换为 per-app 曲线：

```python
# 获取该 app 的曲线，fallback 到全局曲线
app_curve = per_app_curves.get(app_id, release_curve)

# 调用 _month_end_incremental_roi_for_app 时传入 app_curve
(
    month_end_spend_a,
    inc_rev_a,
    month_end_roi_a,
    roi_30d_total_a,
    cross_carryover_a,
) = _month_end_incremental_roi_for_app(
    app_id=app_id,
    month_start=month_start_dt,
    month_end=month_end_dt,
    target_day=target_day_dt,
    month_spend_before_target=month_spend_so_far[app_id],
    historical_daily=app_daily_history,
    planned_daily_spend=planned_daily_spend_a,  # 自变量，随 A/B 分支
    planned_d1_roi=app_future_d1_roi_a,         # 自变量，随 A/B 分支
    curve=app_curve,  # <-- 改为 per-app 曲线
)
```

A/B 两个分支都需要替换。找到两处 `_month_end_incremental_roi_for_app` 调用，将 `curve=release_curve` 改为 `curve=app_curve`。

---

### Task 4: 存储 per-app 曲线到文件

**Files:**
- Modify: `scripts/offline_backtest.py` (新增保存函数 + 在 `run_app_level_last_day_prediction` 末尾调用)

- [ ] **Step 1: 新增 `_save_per_app_curves` 函数**

```python
# scripts/offline_backtest.py
def _save_per_app_curves(curves: Dict[str, Dict[int, float]], output_dir: Path) -> Path:
    """
    保存 per-app 释放曲线到 JSON 文件，供在线 predictor 使用。
    """
    output_path = Path("outputs/per_app_release_curves.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 将 int key 转为 str 以符合 JSON 规范
    serializable = {
        app_id: {str(d): m for d, m in curve.items()}
        for app_id, curve in curves.items()
    }
    output_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Per-app 曲线已保存: {output_path} ({len(curves)} apps)")
    return output_path
```

- [ ] **Step 2: 在 `run_app_level_last_day_prediction` 末尾调用保存**

在 `return summary` 之前：

```python
_save_per_app_curves(per_app_curves, output_dir)
```

---

### Task 5: 在线 predictor 加载 per-app 曲线 + 使用

**Files:**
- Modify: `app/services/predictor.py`

- [ ] **Step 1: 新增 `_per_app_curves` 类变量 + 加载函数**

```python
# app/services/predictor.py, PredictorService 类变量区
_per_app_curves: Dict[str, Dict[int, float]] | None = None

@classmethod
def _load_per_app_curves(cls) -> Dict[str, Dict[int, float]]:
    """
    加载 per-app 释放曲线文件。
    Fallback 链：per-app 曲线 -> 竞品模板 -> 默认曲线
    在 _release_multiplier_for_app 中按 app_id 查找。
    """
    if cls._per_app_curves is not None:
        return cls._per_app_curves

    curve_path = Path("outputs/per_app_release_curves.json")
    if not curve_path.exists():
        cls._per_app_curves = {}
        return cls._per_app_curves

    raw = json.loads(curve_path.read_text(encoding="utf-8"))
    # JSON keys 是字符串，转回 int
    cls._per_app_curves = {
        app_id: {int(d): m for d, m in days.items()}
        for app_id, days in raw.items()
    }
    return cls._per_app_curves
```

- [ ] **Step 2: 修改 `_release_multiplier` 支持 per-app 查找**

将现有的 `_release_multiplier` 改为内部逻辑不变，新增 `_release_multiplier_for_app`：

```python
@classmethod
def _release_multiplier_for_app(cls, day_window: int, app_id: str | None) -> float:
    """按 app_id 查找 per-app 曲线，找不到则 fallback 全局曲线。"""
    if app_id:
        per_app = cls._load_per_app_curves()
        if app_id in per_app:
            curve = per_app[app_id]
            max_day = max(curve.keys())
            if day_window <= 1:
                return curve.get(1, 1.0)
            if day_window >= max_day:
                return curve.get(max_day, max(curve.values()))
            return curve.get(day_window, curve.get(day_window - 1, 1.0))
    # fallback 到现有全局曲线
    return cls._release_multiplier(day_window)
```

`_release_multiplier` 保持不变。

- [ ] **Step 3: `predict_month_end_roi` 使用 `_release_multiplier_for_app`**

```python
# app/services/predictor.py, predict_month_end_roi 方法
# 将 line 363:
#   multiplier = PredictorService._release_multiplier(release_window)
# 改为:
multiplier = PredictorService._release_multiplier_for_app(
    release_window, context.client_id
)
```

- [ ] **Step 4: 验证导入和调用**

```bash
python -c "
from datetime import date
from app.services.predictor import PredictorService

# 无文件时 fallback 到默认曲线
m = PredictorService._release_multiplier_for_app(7, 'nonexistent')
print('fallback d7:', m)
"
```

Expected: `fallback d7: 1.3107` （默认曲线 D7 节点值）

---

### Task 6: 端到端验证

- [ ] **Step 1: 运行离线回测，确认 per-app 曲线生成**

```bash
python -c "
from pathlib import Path
from scripts.offline_backtest import run_app_level_last_day_prediction
result = run_app_level_last_day_prediction(
    Path('data/daily_merged.csv'),
    Path('outputs/backtest_per_app_curve_test'),
    kpi=0.065,
)
print('summary keys:', list(result.keys()))
"
```

- [ ] **Step 2: 确认曲线文件已生成**

```bash
python -c "
import json
from pathlib import Path
p = Path('outputs/per_app_release_curves.json')
if p.exists():
    data = json.loads(p.read_text())
    print(f'{len(data)} apps in curve file')
    for app_id, curve in list(data.items())[:2]:
        print(f'  {app_id}: D3={curve.get(\"3\")}, D7={curve.get(\"7\")}, D30={curve.get(\"30\")}')
else:
    print('FILE NOT FOUND')
"
```

- [ ] **Step 3: 确认在线 predictor 能使用 per-app 曲线**

```bash
python -c "
from app.services.predictor import PredictorService
PredictorService._per_app_curves = None  # 强制重新加载
curves = PredictorService._load_per_app_curves()
print(f'Loaded {len(curves)} per-app curves')
if curves:
    app_id = next(iter(curves))
    m = PredictorService._release_multiplier_for_app(7, app_id)
    print(f'{app_id} D7 multiplier: {m}')
"
```
