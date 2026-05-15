# 智能预算决策系统 (Smart Budget Decision System)

## 项目概览

FastAPI 应用，为广告投放提供 T+1 Spend/ROI 预测 + 每日预算优化推荐。
目标业务群体：**小学生**（休息日模式对预测至关重要）。

## 启动 Web 服务

```bash
bash scripts/run_web_8000.sh
# 看板: http://localhost:8000/web
# 预测: http://localhost:8000/web/predict
# 推荐: http://localhost:8000/web/recommend
```

## 关键文件

| 文件 | 作用 |
|---|---|
| `app/main.py` | FastAPI 入口 |
| `app/web.py` | Web 看板路由 + 前端 ECharts |
| `app/config.py` | 全局配置（Settings） |
| `scripts/run_parallel_models_spend_t1.py` | T+1 Spend 预测训练与评估 |
| `scripts/run_parallel_models_roi_d1.py` | T+1 ROI_D1 预测训练与评估 |
| `scripts/offline_backtest.py` | 离线回放 + 应用层月末推荐 |
| `app/prediction_artifacts.py` | 预测产物路径解析 |
| `app/core/calendar.py` | 节假日日历服务（chinese_calendar） |
| `app/services/risk.py` | 风控服务：多维度告警 + 风险评分 |
| `app/schemas.py` | Pydantic 数据模型（ProductDiagnosis, RiskAlert 等） |

## 数据

- 最新输入: `daily_merged.csv` (548MB, 3/1~5/11, 72天)

## 预测模块架构

两个独立模型共享 walk-forward 框架，已统一口径（CalendarService、30天最小历史、20天评估窗口、is_rest_day、Duan's smearing）：

1. **ROI (T+1 ROI_D1)**: 预测「下一天首日回收/消耗」比率。ROI=首日收入/消耗，分母小导致波动极大：低消耗段 CV>1.0、零 ROI 率 9-19%；`min_spend_train=2` 可过滤最噪声段（零 ROI 率 18.8%→2.7%，仅丢 5% 样本）。零 ROI = 花了钱但零回收，非数据缺失。ROI 独有特征：act_per_spend/rev_per_act_d1 分解、roi 波动率、pre/post_holiday 单天、summer_winter_break。
2. **Spend (T+1 Spend)**: 预测「下一天消耗金额」。`min_spend_train=0.01`。Spend 独有特征：MTD/月末节奏、spend 滚动统计、is_pre/post_holiday_3d 窗口、week_of_month。

两个脚本默认 XGBoost only + GPU，`skip_baseline_two_stage=True`，`sample_weight = np.log1p(spend)`。

## 推荐模块

`run_app_level_last_day_prediction` 基于历史数据为每个应用生成每日预算建议。
- `app_last_day_min_train_days=30`, `app_last_day_canonical_month_end_roi="fused"`

## 风控系统

`RiskService` 提供多维度告警检测 + 综合风险评分（0-100）：

| 告警类型 | 级别 | 触发条件 |
|---------|------|---------|
| `PRODUCT_D1_DROP` | WARN | 产品连续 N 天 D1 低于预测 90% |
| `TRAFFIC_ANOMALY` | CRITICAL | 当日变现偏离预测 ≥20% |
| `ROI_GUARD` | WARN | 月末 ROI 预测低于 KPI buffer |
| `SPEND_DROP` | CRITICAL | 消耗骤降至近期均值 50% 以下（avg≥30 才触发） |
| `SPEND_SPIKE` | WARN | 消耗骤升至近期均值 2.5x 以上（avg≥30 才触发） |
| `ROI_DECLINE_TREND` | WARN | 连续 5 天 ROI 趋势下行 |
| `CAP_PROXIMITY` | WARN | 计划预算达到 cap 的 85% |

配置项均集中在 `app/config.py` 的 `Settings` 中。

## 最新模型版本

| 版本 | 目录 | MAPE | 关键改动 |
|------|------|------|---------|
| ROI v9 | `model_parallel_roi_d1_v9_unified` | 29.15% | 统一底表 + target_is_rest_day + 交互特征 + XGBoost GPU |
| Spend v12 | `model_parallel_spend_t1_v12_unified` | 59.26% | 统一底表 + target_is_rest_day + 交互特征 + XGBoost GPU |

Web fallback: ROI v9→v8→v7, Spend v12→v11。

## 未提交改动

1. `app/config.py` — 新增风控参数（spend anomaly/ROI decline/cap proximity）+ scale 因子微调
2. `app/core/calendar.py` — scale 因子基于实证校准：weekend 1.08→1.10, holiday 1.12→1.15
3. `app/schemas.py` — ProductDiagnosis 加 trend/cap_utilization；RiskAlert 加 4 个新 category
4. `app/services/risk.py` — 风控重构：拆分子方法 + 新增 spend_anomaly/roi_decline/cap_proximity/趋势判断/风险评分
5. `app/web.py` — 六层决策面板（KPI/时间/配置/节奏/风控/决策排序）+ 优先级排序优化
6. `scripts/offline_backtest.py` — 回测中接入 spend anomaly 检测
