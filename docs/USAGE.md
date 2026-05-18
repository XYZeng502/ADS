# 智能预算决策系统 — 使用手册

## 目录

1. [系统概述](#1-系统概述)
2. [快速开始](#2-快速开始)
3. [Web 看板](#3-web-看板)
4. [数据管理](#4-数据管理)
5. [模型训练与评估](#5-模型训练与评估)
6. [每日重训练](#6-每日重训练)
7. [收入预测验证](#7-收入预测验证)
8. [API 接口](#8-api-接口)
9. [配置参数](#9-配置参数)
10. [监控与告警](#10-监控与告警)
11. [常见问题](#11-常见问题)

---

## 1. 系统概述

智能预算决策系统为广告投放提供 **T+1 消耗/ROI 预测 + 每日预算优化推荐**。核心流程：

```
daily_merged.csv → 数据健康检查 → 模型重训(Spend/ROI/曲线) → 收入验证 → Web看板
```

**目标用户**：广告投放运营团队。系统根据日历（工作日/周末/节假日/寒暑假）自动调整预算节奏，目标群体为小学生。

### 核心模块

| 模块 | 功能 |
|------|------|
| **预测模块** | T+1 消耗预测 + T+1 ROI_D1 预测，多模型并行训练 |
| **推荐模块** | 应用月末 ROI 预测 + 每日预算建议（放量/控量/维稳） |
| **收入验证** | 释放曲线模型预测 vs 实际买量收入对比 |
| **系统监控** | 告警历史 + 健康趋势 + 数据新鲜度 + 漂移检测 |

### 模块依赖关系

```
daily_merged.csv ───────────┬────────────────────────────────────┐
   (唯一数据入口)            │                                    │
   ┌─────────────────────────┼──────────────────────┐             │
   │                         │                      │             │
   ▼                         ▼                      ▼             ▼
T+1 Spend/ROI 预测      释放曲线构建           推荐模块        收入验证
(run_parallel_*)    (per_app_release_    (offline_backtest)  (predict_daily
                     curves.json)                              _revenue.py)
   │                         │                      │             │
   │ 预测值 ─────────────────┼───→ 求解器输入       │             │
   │                         │    (spend/roi_d1     │             │
   │                         │     预测校准)         │             │
   │                         │                      │             │
   │                         ├──→ 月末ROI推算 ←─────┤             │
   │                         │    (释放曲线模型)     │             │
   │                         │                      │             │
   │                         │       预算建议 + 告警              │
   │                         │       (app_level_last_day_         │
   │                         │        suggestions.csv)            │
   │                         │                                    │
   └──────── 无直接依赖 ──────┴──────────────────────┤             │
                                                     │             │
                                         释放曲线验证 ←────────────┘
                                         (daily_revenue_predictions.csv)
```

**关键关系：**

- **推荐模块** 和 **收入验证模块** 是两条独立管道，均以 `daily_merged.csv` 为数据源
- 推荐模块使用 T+1 预测模块的输出（spend/roi_d1 预测值）作为求解器校准输入
- 收入验证模块使用与推荐模块相同的释放曲线模型，但独立运行，用于交叉验证曲线模型的准确性
- 两条管道的共同依赖：释放曲线 (`per_app_release_curves.json`) + 业务日历 (`CalendarService`)
- **推荐模块不直接使用收入验证的输出** (`daily_revenue_predictions.csv`)，避免循环依赖

---

## 2. 快速开始

### 环境要求

- Python 3.11+
- 依赖见 `requirements.txt`

### 启动 Web 服务

```bash
# 启动服务（端口 8000）
bash scripts/run_web_8000.sh

# 或手动启动
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动阶段会自动预热缓存（约 40 秒读取数据），之后所有请求瞬时响应。

### 访问看板

| 页面 | URL |
|------|-----|
| 预测模块 | `http://localhost:8000/web` |
| 推荐模块 | 点击导航栏「推荐模块」 |
| 收入验证 | 点击导航栏「每日收入验证」 |
| 系统监控 | 点击导航栏「系统监控」 |

---

## 3. Web 看板

### 3.1 预测模块

查看 T+1 预测模型的实际效果。

- **评估目标切换**：ROI_D1 或 Spend，下拉选择
- **单应用筛选**：搜索框输入应用 ID，图表切换到单应用视角
- **三个图表**：
  - 预测值 vs 实际值：折线对比图 + 散点校准图
  - 每日误差走势：APE 柱状图
  - 模型指标面板：MAPE/RMSE 汇总
- **应用总览表**：所有应用的聚合指标，点击表头排序，点击行加载明细
- **预测明细表**：分页浏览单应用每日预测记录

### 3.2 推荐模块

每日预算决策的核心入口。

- **应用月末 ROI 预测表**：每行一个应用
  - 列：应用ID、月末ROI预测、ROI差距、推荐动作、模式、告警数、建议预算、主因
  - 快速筛选：全部 / 低于KPI / 建议放量 / 建议控量 / 有告警 / 长尾主导 / D1主导 / 消耗主导
  - 高级筛选：KPI状态、推荐动作、风险状态、ROI 范围
  - 点击表头排序，点击应用ID查看详情
- **推荐总览**：KPI 状态、KPI 看板卡片、动作分布饼图、预算 vs ROI 双轴图
- **六层决策面板**：点击展开
  - KPI 层 — 目标完成情况
  - 时间层 — 日历驱动的 scale 因子表
  - 配置层 — 预算分配动作分布
  - 节奏层 — U 型月内节奏
  - 风控层 — 风险应用列表
  - 决策排序层 — 按优先级排序的处理建议

### 3.3 每日收入验证

验证释放曲线模型对买量收入的预测准确性。

- **图表**：按日期汇总的实际 vs 预测收入折线图，鼠标滚轮缩放
- **搜索**：输入应用 ID 查看单个应用
- **数据表**：per-app per-day 明细，支持日期范围筛选和天数过滤
- **在线预测**：输入目标日期 + per-app spend/D1，实时预测当日收入
  - JSON 模式：粘贴 JSON 数组
  - CSV 模式：粘贴 CSV（应用ID,spend,d1_revenue）

### 3.4 系统监控

运维视图，查看系统运行状态。

- **告警统计卡**：近 7 天告警总数 + 按级别/类别细分
- **健康快照卡**：漂移状态、数据新鲜度、训练状态、日历状态
- **健康趋势图**：30 天数据滞后天数 + Spend/ROI MAPE 变化趋势
- **告警历史表**：最近告警记录列表

---

## 4. 数据管理

### 主数据文件

`daily_merged.csv` 是系统唯一数据入口，约 548MB，包含以下列：

| 列名 | 说明 |
|------|------|
| 应用ID | 应用唯一标识 |
| 日期 | YYYY-MM-DD 格式 |
| 消耗金额 | 当日广告消耗 |
| 首日广告收入 | 当日新增用户的 D1 广告收入 |
| 买量广告收入 | 当日全部买量渠道广告收入 |

数据按 (应用ID, 日期) 聚合，覆盖约 1800 个应用、70+ 天历史。

### 数据合并

```bash
# 增量合并每日下载
PYTHONPATH=. python scripts/merge_daily_downloads.py \
  --legacy daily_merged.csv \
  --snapshot-date 2026-05-18 \
  --output daily_merged.csv

# 按日期范围合并
PYTHONPATH=. python scripts/merge_daily_downloads.py \
  --legacy daily_merged.csv \
  --start 2026-05-12 --end 2026-05-18 \
  --output daily_merged.csv
```

### 数据健康检查

```bash
PYTHONPATH=. python -c "
from app.services.data_pipeline import check_data_health
from pathlib import Path
h = check_data_health(Path('daily_merged.csv'))
print(f'状态: {h.status} | 最后日期: {h.last_date} | 滞后: {h.days_behind}天')
print(f'行数: {h.row_count} | 应用数: {h.app_count}')
print(f'列完整: {h.columns_ok} | 缺失列: {h.missing_cols}')
"
```

---

## 5. 模型训练与评估

### T+1 Spend 预测

```bash
PYTHONPATH=. python scripts/run_parallel_models_spend_t1.py \
  --input daily_merged.csv \
  --output-dir outputs/model_parallel_spend_t1 \
  --min-train-days 20 \
  --use-gpu
```

输出到 `outputs/model_parallel_spend_t1/`：
- `predictions_XGBoost.csv` — 最优模型预测结果
- `metrics_summary.csv` — 各模型指标汇总
- `report.json` — 训练报告

### T+1 ROI_D1 预测

```bash
PYTHONPATH=. python scripts/run_parallel_models_roi_d1.py \
  --input daily_merged.csv \
  --output-dir outputs/model_parallel_roi_d1 \
  --min-train-days 20 \
  --use-gpu
```

输出同上结构。

### 离线回测

全量历史回放，评估整个决策链路的长期表现。

```bash
# 完整回放
PYTHONPATH=. python scripts/offline_backtest.py \
  --task backtest \
  --input daily_merged.csv \
  --kpi 1.05

# 应用层月末推荐
PYTHONPATH=. python scripts/offline_backtest.py \
  --task app_last_day \
  --input daily_merged.csv \
  --kpi 1.05
```

输出到 `outputs/backtest_*25d*/`：
- 每日预算建议
- 月末 ROI 预测 vs 实际
- 偏差归因分析
- 应用级建议 CSV

---

## 6. 每日重训练

### 自动重训

通过 cron 定时触发，依次执行：数据检查 → Spend 重训 → ROI 重训 → 释放曲线 → 收入验证。

```bash
# 添加到 crontab（每天早上 6:00）
0 6 * * * cd /home/lsh/ad_ml && PYTHONPATH=. python scripts/daily_retrain.py >> logs/retrain.log 2>&1
```

### 手动触发

```bash
PYTHONPATH=. python scripts/daily_retrain.py
```

### Web 触发

在 Web 看板顶部的「重训状态」栏点击「触发重训」按钮。

### 重训步骤

| 步骤 | 脚本 | 超时 |
|------|------|------|
| 数据检查 | `check_data_health()` | - |
| Spend T+1 | `run_parallel_models_spend_t1.py` | 2h |
| ROI D1 | `run_parallel_models_roi_d1.py` | 2h |
| 释放曲线 | `offline_backtest.py --task app_last_day` | 2h |
| 收入验证 | `predict_daily_revenue.py` | 2h |

重训结束后自动记录告警和健康快照。

---

## 7. 收入预测验证

### 批量回测

```bash
PYTHONPATH=. python scripts/predict_daily_revenue.py
```

输出 `outputs/daily_revenue_predictions.csv`，包含：
- 日期、应用ID、实际收入 (y_true)、预测收入 (y_pred)、累计天数
- MAPE 汇总（仅统计 days_since_start ≥30 的样本）

### 在线预测

在 Web 看板的「在线预测」面板使用，或通过 API：

```bash
curl -X POST http://localhost:8000/web/revenue/daily-predict \
  -H "Content-Type: application/json" \
  -d '{
    "target_date": "2026-05-18",
    "apps": [
      {"应用ID": "30262609", "spend": 5000, "d1_revenue": 750}
    ]
  }'
```

### 休息日校准

系统自动识别休息日（周末/节假日/寒暑假）并应用 scale 因子提升预测精度：
- 工作日：×1.00
- 周末：×1.08
- 节假日：×1.12
- 寒暑假：×1.10

---

## 8. API 接口

### 核心 API（`/v1/`）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 系统健康检查 |
| POST | `/v1/optimize/daily-plan` | 执行每日预算优化 |
| POST | `/v1/report/daily-execution` | 每日执行报告 |
| POST | `/v1/business/decision-trace` | 业务决策链路 |
| POST | `/v1/kpi/recompute` | KPI 变更重算 |
| POST | `/v1/alerts/evaluate` | 告警评估 |
| POST | `/v1/diagnosis/product-grading` | 产品分级诊断 |
| GET | `/v1/monitor/alerts?days=7` | 告警历史 |
| GET | `/v1/monitor/health-trend?days=30` | 健康趋势 |

### 每日优化请求示例

```bash
curl -X POST http://localhost:8000/v1/optimize/daily-plan \
  -H "Content-Type: application/json" \
  -d '{
    "target_date": "2026-05-18",
    "kpi_roi": 1.05,
    "total_budget": 500000,
    "products": [...]
  }'
```

### Web 数据 API（`/web/`）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/web/data/{target}/stats` | 预测统计 (target=roi/spend) |
| GET | `/web/data/{target}/apps` | 应用列表 |
| GET | `/web/data/{target}/app/{app_id}` | 单应用明细（分页） |
| GET | `/web/data/{target}/app/{app_id}/chart` | 单应用图表数据 |
| GET | `/web/predictions/daily_revenue?app_id=` | 收入验证数据 |
| POST | `/web/revenue/daily-predict` | 在线收入预测 |
| POST | `/web/retrain/trigger` | 触发重训 |

---

## 9. 配置参数

所有配置集中在 `app/config.py` 的 `Settings` 类中。

### KPI 与预算

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `default_kpi` | 1.05 | 默认目标 ROI |
| `release_surplus_threshold` | 0.05 | 转为放量模式的 ROI 盈余 |
| `kpi_release_budget_ratio` | 0.15 | 放量预算比例 |
| `kpi_guard_budget_ratio` | 0.12 | 控量预算比例 |

### 风控阈值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `risk_d1_drop_days` | 3 | D1 连续下降触发天数 |
| `risk_d1_drop_ratio` | 0.9 | D1 低于预测比例 |
| `risk_revenue_deviation_threshold` | 0.2 | 流量异常偏差 |
| `risk_spend_drop_ratio` | 0.5 | 消耗骤降比例 |
| `risk_spend_spike_ratio` | 2.5 | 消耗骤升倍率 |
| `risk_roi_decline_days` | 5 | ROI 连续下降天数 |
| `risk_cap_utilization_threshold` | 0.85 | Cap 利用率预警 |

### 求解器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `solver_slot_d1_normalize_mode` | spend_weighted | 槽位 D1 聚合方式 |
| `app_last_day_canonical_month_end_roi` | fused | 月末 ROI 口径 |
| `app_last_day_min_train_days` | 30 | 应用最小训练天数 |

---

## 10. 监控与告警

### 告警类型

| 类型 | 级别 | 触发条件 |
|------|------|---------|
| PRODUCT_D1_DROP | WARN | 连续 3 天 D1 低于预测 90% |
| TRAFFIC_ANOMALY | CRITICAL | 实际收入偏离预测 ≥20% |
| ROI_GUARD | WARN | 月末 ROI 预测低于 KPI buffer |
| SPEND_DROP | CRITICAL | 消耗骤降至均值 50% 以下 |
| SPEND_SPIKE | WARN | 消耗骤升至均值 2.5x 以上 |
| ROI_DECLINE_TREND | WARN | 连续 5 天 ROI 趋势下行 |
| CAP_PROXIMITY | WARN | 预算达 cap 的 85% |
| PREDICTION_DRIFT | WARN | 预测模型漂移比超阈值 |

### 告警数据

告警持久化到 `outputs/alert_history.jsonl`，保留 30 天。

```bash
# 手动查看告警汇总
PYTHONPATH=. python -c "
from app.services.alert_log import AlertLog
print(AlertLog.summary(7))
"
```

### 健康快照

健康快照持久化到 `outputs/health_history.jsonl`，保留 90 天。每次重训自动记录。

---

## 11. 常见问题

### 服务启动慢

首次启动约 40 秒（同步读取 548MB CSV 预热缓存）。后续重启也会同样耗时。这是正常的。

### 数据状态显示「加载中」不消失

通常是浏览器缓存了旧版 JS。按 Ctrl+Shift+R 硬刷新。如果仍不行，检查 `daily_merged.csv` 是否存在且路径正确。

### 模型重训失败

1. 检查 GPU 是否可用：`python -c "import xgboost; print(xgboost.__version__)"`
2. 若 GPU 不可用，去掉 `--use-gpu` 参数
3. 检查数据是否健康：查看日志中的 `DataHealth` 输出
4. 检查磁盘空间：`outputs/` 目录需要足够空间

### 买量收入图表无数据

需要先运行：
```bash
PYTHONPATH=. python scripts/offline_backtest.py --task app_last_day --input daily_merged.csv
PYTHONPATH=. python scripts/predict_daily_revenue.py
```

### 推荐模块显示「未加载到应用列表」

运行：
```bash
PYTHONPATH=. python scripts/offline_backtest.py --task app_last_day --input daily_merged.csv --kpi 1.05
```

### 日历相关问题

业务日历文件：`data/business_calendar_2026.json`。如需更新：
1. 编辑 JSON 文件添加节假日窗口
2. 验证：`python -c "from app.services.calendar_health import check_calendar_health; print(check_calendar_health().status)"`
3. 重启服务

---

## 项目文件速查

| 文件/目录 | 用途 |
|------|------|
| `app/main.py` | FastAPI 入口 + 全部 API 端点 |
| `app/web.py` | Web 看板（路由 + ECharts 前端） |
| `app/config.py` | 集中配置 |
| `app/schemas.py` | 数据模型 |
| `app/core/calendar.py` | 业务日历服务 |
| `app/services/orchestrator.py` | 每日决策编排 |
| `app/services/predictor.py` | D1/月末 ROI 预测 |
| `app/services/solver.py` | 预算求解器 |
| `app/services/risk.py` | 风控 + 告警 |
| `app/services/rule_engine.py` | P0-3 规则引擎 |
| `scripts/daily_retrain.py` | 每日重训练编排器 |
| `scripts/predict_daily_revenue.py` | 收入预测验证 |
| `scripts/offline_backtest.py` | 离线回测 |
| `daily_merged.csv` | 主数据文件 |
| `outputs/` | 所有预测/回放/监控产物 |
| `logs/` | 应用日志 |
| `data/business_calendar_2026.json` | 业务日历 |
