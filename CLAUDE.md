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
- 旧输入: `daily_20260421_120112.csv`

## 预测模块架构

两个独立模型分别预测，共享同一个 walk-forward 框架：

1. **ROI (T+1 ROI_D1)**: ROI 预测「下一天首日回收/消耗」比率
2. **Spend (T+1 Spend)**: Spend 预测「下一天消耗金额」

### 统一口径（2026-05-14 对齐）

两个脚本现已统一以下口径：

| 项目 | ROI | Spend | 说明 |
|------|-----|-------|------|
| 日历源 | CalendarService (chinese_calendar) | 同 | 自动识别调休/节假日 |
| 应用过滤 | 30 天最小历史 | 同（settings 强制） | |
| eval_recent_days | 20 | 20 | 统一评估窗口 |
| 子组构成特征 | 12 列 (traffic/scene/creative/billing) | 同 | |
| T+1 目标日特征 | 14 列 target_* | 14 列 target_* | 目标日节假日/周末/假期窗口 |
| is_rest_day | ✓ | ✓ | 休息日（排除调休上班） |
| Duan's smearing | ✓ (log 模型) | ✓ (log 模型) | 对数还原偏差校正 |
| min_spend_train | **2.0** (默认) | **0.01** (默认) | 过滤低消耗噪声；阈值不同因数据分布不同 |
| 进度打印 | ✓ | ✓ | walk-forward 逐 fold 输出 |

**特征差异（有意的领域差异）：**

| ROI 独有 | Spend 独有 |
|----------|-----------|
| act_per_spend / rev_per_act_d1 分解 | MTD/月末节奏 (month_progress 等) |
| roi_d1_std_7 / iqr / range 波动率 | spend_roll_mean/std 滚动统计 |
| roi_d1_lag_* | roi_lag_* |
| pre_holiday / post_holiday (单天) | is_pre_holiday_3d / is_post_holiday_3d (窗口) |
| is_weekend (当天) | week_of_month / days_to_month_end |
| spend_ratio_3d | spend 滚动统计 |
| summer_winter_break | is_summer_winter_break |

### ROI 数据特征（重要）

ROI = 首日广告收入 / 消耗金额。**分母小导致 ROI 波动极大**：

| 当前消耗 | 样本占比 | T+1 ROI CV | T+1 ROI=0 | 日间波动 |
|---------|---------|-----------|----------|---------|
| 0-1 元 | ~3% | 1.0-2.5 | 9-19% | 0.5-1.6 |
| 1-2 元 | 3.9% | 1.01 | 6.7% | 0.42 |
| 2-5 元 | 11.6% | 0.76 | 2.7% | 0.33 |
| 5-10 元 | 11.8% | 0.56 | 1.2% | 0.28 |
| 10-50 元 | 23.0% | 0.59 | 0.6% | 0.20 |
| 100-500 元 | 23.7% | 0.25 | 0.1% | 0.10 |
| 500+ 元 | 10.5% | 0.19 | 0.0% | 0.07 |

- T+1 ROI=0 100% 是「花了钱但零回收」，非数据缺失
- 零 ROI 在 T+1 是休息日时占比略高（26% vs 整体 30%）
- `min_spend_train=2` 可过滤最噪声段（零 ROI 率 18.8%→2.7%，仅丢 5% 样本）

### 零 ROI 根因分析

359 个零 ROI 样本（1.5%）全部分解为：花了钱但第二天首日广告收入=0。
日期上有聚集：4/4(清明)20个、3/11 17个、5/10 14个——这些天的第二天回收差。

## 推荐模块

`run_app_level_last_day_prediction` 基于历史数据为每个应用生成每日预算建议。
- 配置: `app_last_day_min_train_days=30`
- 配置: `app_last_day_canonical_month_end_roi="fused"`

## 最新模型版本

| 版本 | 目录 | MAPE | 关键改动 |
|------|------|------|---------|
| Spend v9 | `model_parallel_spend_t1_v9_composition` | 117.16% | +子组构成特征（与 v8 持平） |
| Spend v8 | `model_parallel_spend_t1_v8_filtered` | 117.9% | 30天过滤 + weekday校准 |
| ROI v3 | `model_parallel_roi_d1_default_weekly_v3` | - | 旧口径，319 应用 |

Web 当前指向：Spend v9 → v8 fallback, ROI v7 → v3 fallback。

## 前端功能

- **预测模块**：ROI/Spend 切换，折线+散点+误差图，应用总览表+明细分页
- **推荐模块**：月度 ROI 预测表（搜索+筛选+排序），悬浮弹窗——鼠标移到应用 ID 显示推荐依据+分场景推荐+T+1 ROI/Spend 双预测曲线，点击跳转预测详情
- 行限制已从 2000 提升到 50000

## 待跑训练

以下改动已编码但未跑训练验证：

| 版本 | 关键改动 | 状态 |
|------|---------|------|
| ROI v7 | T+1 目标日特征 + min_spend_train=2 + 进度打印 | 未跑 |
| Spend v10 | min_spend_train=0.01 + is_rest_day + Duan's smearing | 未跑 |

## 未提交改动（约 6 个文件）

1. `app/config.py` — `app_last_day_min_train_days=30`
2. `app/core/calendar.py` — 集成 chinese_calendar; is_rest_day
3. `app/web.py` — 悬浮弹窗; 路径更新 v9/v7; 50000 行限制
4. `scripts/offline_backtest.py` — 30天过滤
5. `scripts/run_parallel_models_spend_t1.py` — 统一口径 (target_*特征/is_rest_day/Duan/min_spend_train/组成特征)
6. `scripts/run_parallel_models_roi_d1.py` — 统一口径 (CalendarService/target_*特征/min_spend_train/组成特征/进度打印)
