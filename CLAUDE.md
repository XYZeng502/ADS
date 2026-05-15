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

## 最新模型版本

| 版本 | 目录 | MAPE | 关键改动 |
|------|------|------|---------|
| ROI v9 | `model_parallel_roi_d1_v9_unified` | 29.15% | 统一底表 + target_is_rest_day + 交互特征 + XGBoost GPU |
| Spend v12 | `model_parallel_spend_t1_v12_unified` | 59.26% | 统一底表 + target_is_rest_day + 交互特征 + XGBoost GPU |

Web fallback: ROI v9→v8→v7, Spend v12→v11。

## 未提交改动

1. `app/config.py` — `app_last_day_min_train_days=30`
2. `app/core/calendar.py` — 集成 chinese_calendar; is_rest_day
3. `app/web.py` — 悬浮弹窗; 路径更新 v12/v9; 推荐模块可读性优化; 50000 行限制; f-string 转义修复
4. `app/unified_daily.py` — 新增统一数据管线 + target_is_rest_day/target_is_adjusted_workday 特征
5. `scripts/offline_backtest.py` — 30天过滤; 路径更新 v12/v9
6. `scripts/run_parallel_models_spend_t1.py` — 统一底表 + target_is_rest_day + 校准函数调休修复 + XGBoost GPU
7. `scripts/run_parallel_models_roi_d1.py` — 统一底表 + target_is_rest_day 同口径 + XGBoost GPU
