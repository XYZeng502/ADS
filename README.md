# 智能预算决策系统

面向广告投放的 T+1 Spend/ROI 预测 + 每日预算优化推荐系统。

## 系统概览

```
daily_merged.csv ──→ 每日重训练 ──→ 模型产出 ──→ Web 看板
                     │                │              │
                     ├─ Spend T+1    ├─ 点预测      ├─ 预测模块
                     ├─ ROI T+1      ├─ CQR区间     ├─ 推荐模块
                     ├─ 释放曲线     ├─ 三档预算    ├─ 收入验证
                     └─ 收入验证     └─ 告警检测    └─ 系统监控
```

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Web 服务
PYTHONPATH=. bash scripts/run_web_8000.sh

# 访问
# 看板: http://localhost:8000/web
# 预测: http://localhost:8000/web/predict
# 推荐: http://localhost:8000/web/recommend
# API文档: http://localhost:8000/docs
```

## 配置

所有配置项集中在 `app/config.py` 的 `Settings` 类中，支持通过 `APP_` 前缀的环境变量覆盖默认值：

```bash
# 环境变量覆盖示例
export APP_DEFAULT_KPI=1.08
export APP_API_KEY="your-secret-key"
export APP_RETRAIN_TIMEOUT_SECONDS=10800
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_DEFAULT_KPI` | 1.05 | 全局 ROI KPI 目标 |
| `APP_API_KEY` | (空) | API 鉴权密钥，为空则不启用鉴权 |
| `APP_RETRAIN_TIMEOUT_SECONDS` | 7200 | 单步重训练超时（秒） |
| `APP_RISK_D1_DROP_DAYS` | 3 | D1 连续低于预测触发告警的天数 |
| `APP_RISK_SPEND_DROP_RATIO` | 0.5 | 消耗骤降告警阈值 |
| `APP_RISK_SPEND_SPIKE_RATIO` | 2.5 | 消耗骤升告警阈值 |
| `APP_RISK_ROI_DECLINE_DAYS` | 5 | ROI 连续下行触发告警的天数 |

若配置了 `api_key`，以下写操作接口需要 `Authorization: Bearer <key>` 或 `X-API-Key: <key>` 请求头：
`/web/run/start`、`/web/retrain/trigger`、`/web/revenue/daily-predict`。

## 核心模块

### 预测模块

| 模型 | 说明 | MAPE | 版本 |
|------|------|------|------|
| Spend T+1 | 预测下一天消耗金额 | ~58% | v12 |
| ROI T+1 | 预测下一天首日回收/消耗 | ~35% | v9 |
| CQR 区间 | Spend T+1 的分桶校准预测区间 (P05/P50/P95) | 覆盖率 ~75% | v1 |
| 释放曲线 | 30 天日级 ROI 乘数曲线 | per-app | - |

模型输出位于 `outputs/model_parallel_spend_t1_v12_unified/` 和 `outputs/model_parallel_roi_d1_v9_unified/`。

### 风控与告警

| 告警类型 | 级别 | 触发条件 |
|---------|------|---------|
| `PRODUCT_D1_DROP` | WARN | 产品连续 3 天 D1 低于预测 90% |
| `TRAFFIC_ANOMALY` | CRITICAL | 当日变现偏离预测 ≥20% |
| `ROI_GUARD` | WARN | 月末 ROI 预测低于 KPI buffer |
| `SPEND_DROP` | CRITICAL | 消耗骤降至近期均值 50% 以下 |
| `SPEND_SPIKE` | WARN | 消耗骤升至近期均值 2.5x 以上 |
| `ROI_DECLINE_TREND` | WARN | 连续 5 天 ROI 趋势下行 |
| `CAP_PROXIMITY` | WARN | 计划预算达到 cap 的 85% |
| `RETRAIN_FAILURE` | CRITICAL | 每日重训练步骤执行失败 |
| `PREDICTION_DRIFT` | CRITICAL/WARN | 模型预测漂移检测 |

告警持久化到 `outputs/alert_history.jsonl`（保留 30 天），Web 监控面板可查看。

### Web 看板

| 页签 | 功能 |
|------|------|
| 预测模块 | T+1 Spend/ROI 预测效果、CQR 区间带、分应用查询、CSV 导出 |
| 推荐模块 | 月末 ROI 预测表、三档预算推荐（保守/中性/激进）、六层决策面板、CSV 导出 |
| 每日收入验证 | 释放曲线反向预测 vs 实际收入对比、Carryover 尾量分解 |
| 系统监控 | 模型管理、告警历史、健康趋势、重训触发 |

### API 接口

| 路径 | 方法 | 鉴权 | 说明 |
|------|------|------|------|
| `/health` | GET | - | 系统健康状态（数据、模型、训练、漂移） |
| `/web` | GET | - | Web 看板主页 |
| `/web/predict` | GET | - | 预测模块页面 |
| `/web/recommend` | GET | - | 推荐模块页面 |
| `/web/data/{target}` | GET | - | 预测统计数据（spend/roi） |
| `/web/data/{target}/app/{app_id}` | GET | - | 分应用预测明细（支持分页） |
| `/web/data/{target}/app/{app_id}/chart` | GET | - | 分应用图表数据 |
| `/web/predictions/daily_revenue` | GET | - | 每日收入验证数据 |
| `/web/export/predictions/{target}` | GET | - | 导出预测明细 CSV |
| `/web/export/recommendations` | GET | - | 导出推荐结果 CSV |
| `/v1/monitor/alerts` | GET | - | 告警列表（近 N 天） |
| `/v1/monitor/health-trend` | GET | - | 健康趋势快照历史 |
| `/web/retrain/trigger` | POST | 需要 | 手动触发每日重训练 |
| `/web/run/start` | GET | 需要 | 触发离线回放任务 |
| `/web/revenue/daily-predict` | POST | 需要 | 在线收入预测 |

## 关键脚本

```bash
# 每日重训练编排（cron 调用）
PYTHONPATH=. python scripts/daily_retrain.py

# T+1 Spend 预测训练
PYTHONPATH=. python scripts/run_parallel_models_spend_t1.py \
  --input daily_merged.csv --output-dir outputs/model_parallel_spend_t1_v12_unified \
  --tree-models XGBoost --use-log-target --enable-cqr

# T+1 ROI 预测训练
PYTHONPATH=. python scripts/run_parallel_models_roi_d1.py \
  --input daily_merged.csv --output-dir outputs/model_parallel_roi_d1_v9_unified

# 离线回放 + 应用层月末推荐
PYTHONPATH=. python scripts/offline_backtest.py \
  --task app_last_day --input daily_merged.csv --output-dir outputs --kpi 1.05

# 释放曲线预测
PYTHONPATH=. python scripts/predict_daily_revenue.py

# 同类竞品曲线预测
PYTHONPATH=. python scripts/predict_competitor_by_template.py \
  --input daily_merged.csv --name demo --d1-roi 0.65
```

## 数据要求

输入文件 `daily_merged.csv` 需包含以下关键列：

| 列名 | 说明 |
|------|------|
| 日期 | YYYY-MM-DD 格式 |
| 应用ID | 广告应用标识 |
| 消耗金额 | 当日广告消耗 |
| 首日广告收入 | 当日新增用户的 D1 广告收入 |
| 买量广告收入 | 当日全部买量广告收入 |

日历数据文件 `data/business_calendar_2026.json` 定义节假日窗口和调休工作日，`app/core/calendar.py` 基于 chinese_calendar + JSON 日历提供统一的休息日判定、假期窗口特征和 scale 因子。

健康数据应满足：
- 覆盖至少 30 天历史
- 应用数 ≥ 300
- 每日行数 ≥ 500

## 部署

### Cron 定时重训练

```bash
# 每天凌晨 3 点执行重训练（crontab -e）
0 3 * * * cd /path/to/ad_ml && PYTHONPATH=. python scripts/daily_retrain.py >> logs/retrain.log 2>&1
```

### 服务管理（systemd）

```ini
# /etc/systemd/system/ad-ml.service
[Unit]
Description=Smart Budget Decision System
After=network.target

[Service]
Type=simple
User=app
WorkingDirectory=/path/to/ad_ml
Environment=PYTHONPATH=/path/to/ad_ml
Environment=APP_API_KEY=your-secret-key
ExecStart=/path/to/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ad-ml
```

### 环境变量

建议部署时至少设置：
- `APP_API_KEY` — 保护写操作接口
- `APP_DEFAULT_KPI` — 业务 KPI 目标

## 项目结构

```
├── app/                          # FastAPI 应用
│   ├── main.py                   # 入口 + 全局路由
│   ├── web.py                    # Web 看板 + 前端
│   ├── config.py                 # 全局配置（支持 APP_* 环境变量）
│   ├── schemas.py                # 数据模型
│   ├── core/calendar.py          # 节假日日历服务
│   ├── experiments/core.py       # 模型训练核心 + CQR
│   ├── services/                 # 业务服务
│   │   ├── predictor.py          # 释放曲线预测
│   │   ├── solver.py             # 预算分配求解
│   │   ├── risk.py               # 风控告警（8 种告警类型）
│   │   ├── orchestrator.py       # 每日编排
│   │   ├── alert_log.py          # 告警持久化
│   │   ├── retrain_status.py     # 重训练状态追踪
│   │   └── health_history.py     # 健康快照历史
│   └── unified_daily.py          # 数据管线
├── scripts/                      # 训练和执行脚本
│   ├── run_parallel_models_spend_t1.py
│   ├── run_parallel_models_roi_d1.py
│   ├── daily_retrain.py          # 每日重训练编排
│   ├── offline_backtest.py       # 离线回放 + 应用层推荐
│   ├── predict_daily_revenue.py  # 收入预测
│   └── experiments/              # 实验脚本
├── data/
│   └── business_calendar_2026.json  # 节假日日历
├── outputs/                      # 模型产出与告警
│   ├── model_parallel_spend_t1_v12_unified/
│   ├── model_parallel_roi_d1_v9_unified/
│   ├── alert_history.jsonl
│   └── ...
└── docs/                         # 文档
```

## 技术栈

- **后端**: Python 3.13, FastAPI, Uvicorn
- **模型**: XGBoost, CatBoost, LightGBM
- **区间估计**: MAPIE (Conformalized Quantile Regression)
- **前端**: ECharts 5, 原生 JavaScript
- **数据处理**: Pandas, NumPy, scikit-learn
- **日历**: chinese_calendar

## 故障排查

| 问题 | 检查项 |
|------|--------|
| 看板无数据 | `outputs/` 下模型目录和 CSV 是否存在；`PYTHONPATH=.` 是否正确 |
| 重训失败 | `outputs/retrain_status.json` 查看步骤错误信息；监控面板可见告警 |
| 推荐面板不可见 | 确认浏览器刷新缓存（Ctrl+Shift+R） |
| 导出 404 | 先确认模型已训练完成，对应 CSV 文件存在 |
| 页面样式异常 | ECharts CDN 是否可访问；Python 版本 ≥ 3.10 |

## 文档索引

- **`docs/项目交付概述.md`** — 交付总览：项目背景、能力总览、架构、部署、配置（推荐首先阅读）
- `README.md` — 本文档：快速启动、核心模块、API 接口表、项目结构、故障排查
- `docs/system_architecture.md` — 系统架构详解：数据流、模型架构、CQR 区间、决策六层
- `docs/USAGE.md` — 使用手册：Web 看板操作、数据管理、训练命令、API 示例、配置参数全表、常见问题
- `docs/modules/` — 各模块详细文档
