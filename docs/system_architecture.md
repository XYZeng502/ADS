# 系统架构

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Web 看板 (ECharts)                      │
│  预测模块 │ 推荐模块 │ 收入验证 │ 系统监控                    │
├─────────────────────────────────────────────────────────────┤
│                   FastAPI 应用层                             │
│  /health │ /web │ /web/predict │ /web/recommend │ /web/data  │
├─────────────────────────────────────────────────────────────┤
│                     业务服务层                               │
│  predictor │ solver │ risk │ orchestrator │ rule_engine      │
├─────────────────────────────────────────────────────────────┤
│                     模型训练层                               │
│  Spend T+1 │ ROI T+1 │ CQR 区间 │ 释放曲线 │ 收入预测        │
├─────────────────────────────────────────────────────────────┤
│                     数据管线                                 │
│  daily_merged.csv → unified_daily → 特征工程 → 训练/预测     │
└─────────────────────────────────────────────────────────────┘
```

## 数据流

```
daily_merged.csv (548MB, 3/1~5/11, 72天)
       │
       ▼
  build_unified_daily()          ← 数据清洗 + 特征工程
       │
       ├──→ run_parallel_models_spend_t1.py  → Spend T+1 预测
       ├──→ run_parallel_models_roi_d1.py    → ROI T+1 预测
       ├──→ predict_daily_revenue.py         → 每日收入验证
       └──→ offline_backtest.py              → 月末预算推荐
                    │
                    ▼
              outputs/ 目录 (CSV + JSON)
                    │
                    ▼
              Web 看板加载 CSV → ECharts 渲染 → 交互查询
```

## 预测模型架构

### Spend T+1 (消耗预测)

- **模型**: XGBoost (log1p 目标)
- **特征**: 74 维（消耗/ROI 滞后值、日历特征、假期交互特征、组合特征）
- **样本权重**: spend^0.35
- **评估方式**: 40 天 walk-forward expanding window
- **MAPE**: ~58%
- **CQR 区间**: 分桶校准 (Q1-Q4)，覆盖率 ~75%

### ROI T+1 (首日回收率预测)

- **模型**: XGBoost
- **特征**: 含 act_per_spend / rev_per_act_d1 分解
- **过滤**: min_spend_train=2.0
- **MAPE**: ~35%

### 释放曲线 (30 天日级 ROI 倍率)

- **输入**: 成熟 cohort (age≥30) 的历史 ROI
- **输出**: per-app 日级倍率曲线 F(age)
- **用途**: 月末 ROI 推算 + 每日收入反向验证

### CQR 区间预测

```
XGBoost (P05) ─┐
XGBoost (P50) ─┼─→ 分桶校准 ─→ 最终区间 [P05_cal, P50, P95_cal]
XGBoost (P95) ─┘    (Q1/Q2/Q3/Q4 各独立校准)
```

目的：为每个 app 的 spend 预测提供校准后的 80% 置信区间，支持保守/中性/激进三档预算决策。

## 决策六层

| 层 | 内容 | 数据来源 |
|----|------|---------|
| KPI 层 | 月末 ROI vs 目标差距 | offline_backtest |
| 时间层 | 距月末天数、可释放窗口 | CalendarService |
| 配置层 | 预算上限、KPI buffer | config.py Settings |
| 节奏层 | 月末加速/月初保守判断 | scale_factors |
| 风控层 | spend 异常、ROI 趋势、cap 逼近 | RiskService |
| 决策排序层 | 优先级评分 → 行动建议 | rule_engine |

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 训练框架 | Walk-forward expanding window | 无数据泄漏 |
| 目标编码 | Expanding-window cumulative mean | 避免未来信息泄漏 |
| 节假日处理 | 目标日特征 + holiday calibration | 小学生用户群体，休息日模式显著 |
| 区间估计 | CQR 分桶校准 | 处理 spend 异方差 |
| 样本权重 | spend^0.35 | 提升高 spend 样本的业务权重 |

## 接口总览

| 路径 | 说明 |
|------|------|
| `/health` | 系统健康（数据、模型、训练、漂移） |
| `/web` | Web 看板主页 |
| `/web/predict` | 预测模块（T+1 Spend/ROI 对比） |
| `/web/recommend` | 推荐模块（月末预算建议） |
| `/web/retrain/trigger` | 触发每日重训练 |
| `/web/data/{target}` | 预测统计查询 |
| `/v1/monitor/alerts` | 告警列表 |
| `/v1/monitor/health-trend` | 健康趋势快照 |
