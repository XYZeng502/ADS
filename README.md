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
# API文档: http://localhost:8000/docs
```

## 核心模块

### 预测模块

| 模型 | 说明 | MAPE | 版本 |
|------|------|------|------|
| Spend T+1 | 预测下一天消耗金额 | ~58% | v12 |
| ROI T+1 | 预测下一天首日回收/消耗 | ~35% | v9 |
| CQR 区间 | Spend T+1 的分桶校准预测区间 (P05/P50/P95) | 覆盖率 ~75% | v1 |
| 释放曲线 | 30 天日级 ROI 乘数曲线 | per-app | - |

模型输出位于 `outputs/model_parallel_spend_t1_v12_unified/` 和 `outputs/model_parallel_roi_d1_v9_unified/`。

### Web 看板

| 页签 | 功能 |
|------|------|
| 预测模块 | T+1 Spend/ROI 预测效果、CQR 区间带、分应用查询 |
| 推荐模块 | 月末 ROI 预测表、三档预算推荐（保守/中性/激进）、六层决策面板 |
| 每日收入验证 | 释放曲线反向预测 vs 实际收入对比、Carryover 尾量分解 |
| 系统监控 | 模型管理、告警历史、健康趋势、重训触发 |

### API 接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 系统健康状态（数据、模型、训练、漂移） |
| `/web` | GET | Web 看板主页 |
| `/web/predict` | GET | 预测模块页面 |
| `/web/recommend` | GET | 推荐模块页面 |
| `/web/retrain/trigger` | POST | 手动触发每日重训练 |
| `/web/data/{target}` | GET | 预测统计数据（spend/roi） |
| `/web/data/{target}/app/{app_id}/chart` | GET | 分应用图表数据 |
| `/v1/monitor/alerts` | GET | 告警列表（近N天） |
| `/v1/monitor/health-trend` | GET | 健康趋势快照 |
| `/web/predictions/daily_revenue` | GET | 每日收入验证数据 |

## 关键脚本

```bash
# 每日重训练编排（cron 调用）
PYTHONPATH=. python scripts/daily_retrain.py

# T+1 Spend 预测训练
PYTHONPATH=. python scripts/run_parallel_models_spend_t1.py \
  --input daily_merged.csv --output-dir outputs/model_parallel_spend_t1 \
  --tree-models XGBoost --use-log-target --enable-cqr

# T+1 ROI 预测训练
PYTHONPATH=. python scripts/run_parallel_models_roi_d1.py \
  --input daily_merged.csv --output-dir outputs/model_parallel_roi_d1

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

健康数据应满足：
- 覆盖至少 30 天历史
- 应用数 ≥ 300
- 每日行数 ≥ 500

## 项目结构

```
├── app/                          # FastAPI 应用
│   ├── main.py                   # 入口 + 全局路由
│   ├── web.py                    # Web 看板 + 前端
│   ├── config.py                 # 全局配置
│   ├── schemas.py                # 数据模型
│   ├── core/calendar.py          # 节假日日历
│   ├── experiments/core.py       # 模型训练核心 + CQR
│   ├── services/                 # 业务服务
│   │   ├── predictor.py          # 释放曲线预测
│   │   ├── solver.py             # 预算分配求解
│   │   ├── risk.py               # 风控告警
│   │   ├── orchestrator.py       # 每日编排
│   │   └── ...
│   └── unified_daily.py          # 数据管线
├── scripts/                      # 训练和执行脚本
│   ├── run_parallel_models_spend_t1.py
│   ├── run_parallel_models_roi_d1.py
│   ├── daily_retrain.py          # 每日重训练编排
│   ├── offline_backtest.py       # 离线回放
│   ├── predict_daily_revenue.py  # 收入预测
│   └── experiments/              # 实验脚本
├── outputs/                      # 模型产出
│   ├── model_parallel_spend_t1_v12_unified/
│   ├── model_parallel_roi_d1_v9_unified/
│   └── ...
└── docs/                         # 文档
```

## 技术栈

- **后端**: Python 3.13, FastAPI, Uvicorn
- **模型**: XGBoost, CatBoost, LightGBM
- **区间估计**: MAPIE (Conformalized Quantile Regression)
- **前端**: ECharts 5, 原生 JavaScript
- **数据**: Pandas, NumPy, scikit-learn

## 文档索引

- `CLAUDE.md` — 开发者上下文
- `docs/system_architecture.md` — 系统架构
- `docs/USAGE.md` — 使用指南
- `docs/modules/` — 模块详细文档
- `docs/superpowers/specs/` — 设计文档
