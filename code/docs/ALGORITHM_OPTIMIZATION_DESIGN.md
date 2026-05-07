# 智能预算决策系统 - 最优算法设计方案

本文档基于当前项目代码、`README.md` 与《智能预算决策系统_业务场景与问题定义》整理，目标是把当前“规则 + 简单线性规划”的版本升级为更接近业务最优的版本。

## 1. 当前版本的主要瓶颈

当前代码已经具备完整流程，但核心算法仍偏简化：

1. 决策粒度偏粗
   - 当前优化主要按 `应用ID` 分配预算。
   - 需求文档要求输出“产品 × 版位/投放维度 × 日期”的预算。
   - 数据中已经存在 `推广流量`、`流量场景`、`创意规格`、`转化类型`，应直接作为优化单元。

2. 使用平均 ROI，而不是边际 ROI
   - 当前用历史 D1、LTV 倍率、产品等级约束来分配预算。
   - 真正的预算分配应看“多投 1 元的边际回收”，不是历史平均回收。
   - 如果没有边际递减曲线，优化器容易把预算推给历史 ROI 高但已经接近流量上限的单元。

3. LTV 曲线对成熟度和截尾处理不足
   - 近期 cohort 不一定已经完整释放 D7/D30。
   - 当前部分逻辑会用已有字段直接汇总，容易把未成熟 cohort 当成完整 cohort。
   - 应该显式建模 cohort age 与观测窗口，处理右截尾。

4. 节假日 scale 是人工常数
   - 当前节假日 scale 因子写死。
   - 更优做法是从历史数据学习“日期类型对 D1、D3、D7、ECPM、流量上限”的影响。

5. 风险只是诊断，不是约束
   - 当前产品能力下滑、流量异常主要用于报告。
   - 最优版本应把风险转成预算降权、鲁棒 ROI 约束、CVaR 或 P20 约束。

6. 优化器不是完整联合求解
   - 当前先产品分配，再按时间 scale 分摊。
   - 推荐改为 `产品/投放单元 × 日期 × 预算段` 联合求解。

## 2. 最优版本总体架构

推荐升级为“预测模型 + 边际响应曲线 + 鲁棒联合优化 + MPC 滚动校准”。

每日流程：

```text
数据清洗与 cohort 状态更新
  -> D1 / LTV / 历史长尾预测
  -> 边际 ROI 与 cap 预测
  -> 风险与不确定性估计
  -> 产品 × 投放维度 × 日期联合优化
  -> 输出今日建议预算、D1 目标、ROI 约束余量、风险原因
  -> 次日用真实结果校准模型
```

核心决策单元：

```text
unit = (
  应用ID,
  推广流量,
  流量场景,
  创意规格,
  转化类型
)
```

核心决策变量：

```text
x[u, t] = 投放单元 u 在日期 t 的建议预算
```

如果要表达边际递减，使用分段预算变量：

```text
x[u, t, s] = 投放单元 u 在日期 t 的第 s 个预算段
```

## 3. 预测层设计

### 3.1 D1 预测：从 LightGBM 升级为校准的时序监督模型

当前 `ml_predictor.py` 已有 LightGBM 雏形，建议保留方向，但做三点升级：

1. 训练目标从单点均值升级为多分位数
   - 预测 `P20 / P50 / P80 D1`。
   - ROI 硬约束使用 P20 或 P30，而不是均值。
   - 规模目标使用 P50，风险报告展示 P20-P80 区间。

2. 增加严格时间切分回测
   - 不随机切分。
   - 用 `2026-03-01 ~ 2026-04-14` 训练，`2026-04-15 ~ 2026-04-21` 验证。
   - 每日滚动评估 MAE、MAPE、Pinball Loss、校准曲线。

3. 加入层级特征与交叉特征
   - 产品历史 D1、投放单元历史 D1、广告主历史 D1。
   - `应用ID × 推广流量`、`应用ID × 流量场景` 等交叉统计。
   - 最近 3/7/14 天趋势、消耗变化率、点击率、激活率、注册率。

推荐模型顺序：

```text
P0: LightGBM quantile 回归
P1: CatBoost，原生处理类别特征，适合高基数投放维度
P2: 层级贝叶斯校准，用于小样本产品/新投放单元
```

### 3.2 LTV 曲线：从固定倍率升级为 cohort 生存/增长曲线

推荐对每个产品或投放单元拟合单调增长曲线：

```text
M_u(a) = D_a / D1
```

其中 `a` 是 cohort age。可选模型：

1. 单调样条 / Isotonic Regression
   - 保证 `D1 <= D3 <= D7 <= D14 <= D30`。
   - 实现简单，适合 P0。

2. 参数化饱和曲线
   - 例如 `M(a) = 1 + alpha * (1 - exp(-beta * log(a)))`。
   - 天然表达“前期增长快、后期趋平”。

3. 层级收缩
   - 单元级数据少时回退到产品级。
   - 产品级数据少时回退到客户/全局标准曲线。

LTV 曲线训练时必须处理右截尾：

```text
如果 cohort 只存活 3 天，只能用于 D1/D3 训练，不能当作 D7/D30 为 0。
```

### 3.3 历史长尾回收预测

对已经发生的每一批 cohort，按存活天数计算到月末还能释放多少：

```text
tail_revenue(c, month_end)
  = D1_revenue(c) * [M(age_at_month_end) - M(age_today)]
```

这部分应进入 ROI 约束分子，而不应该只作为报告字段。

### 3.4 边际响应曲线与 cap

这是提升效果的核心。

对每个投放单元学习：

```text
expected_d1_roi = f_u(spend, date_features, recent_state)
```

更推荐学习“分段边际 ROI”：

```text
第 0~100 元预算：边际 ROI = 0.82
第 100~300 元预算：边际 ROI = 0.75
第 300~600 元预算：边际 ROI = 0.68
第 600 元以上：边际 ROI = 0.58
```

实现方式：

1. P0：历史分位 cap
   - `cap[u,t] = max(P90历史日耗, 最近7日均耗 * holiday_scale * ramp_factor)`。
   - 快速可用，但不是最优。

2. P1：分段边际 ROI
   - 按历史消耗分桶，估计每个桶的 D1 和置信区间。
   - 强制边际 ROI 单调不增。

3. P2：因果/实验校准
   - 通过扩量实验识别真实 ROI 递减。
   - 使用 Doubly Robust / Causal Forest / 分层实验估计边际响应。

## 4. 优化层设计

### 4.1 数学目标

目标是最大化本月剩余消耗：

```text
maximize sum_{u,t,s} x[u,t,s]
```

硬约束是月度 ROI 达标：

```text
actual_revenue
+ historical_tail_revenue
+ sum_{u,t,s} x[u,t,s] * p20_roi[u,t,s]
>=
KPI * (
  actual_spend
  + sum_{u,t,s} x[u,t,s]
)
```

这里建议 ROI 约束使用 `p20_roi` 或保守校准后的 ROI，而不是均值。

### 4.2 分段线性化

为了让 PuLP/CBC 也能求解，可以把边际递减曲线拆成预算段：

```text
segment 1: 0   ~ 100, roi = 0.85
segment 2: 100 ~ 300, roi = 0.78
segment 3: 300 ~ 600, roi = 0.70
segment 4: 600 ~ cap, roi = 0.60
```

变量：

```text
0 <= x[u,t,s] <= segment_width[u,t,s]
```

由于边际 ROI 已经按从高到低排序，在线性目标最大化下，优化器会优先吃高 ROI 预算段。

### 4.3 必要约束

1. 总预算约束

```text
sum x[u,t,s] <= remaining_cash_budget
```

如果客户没有现金预算上限，而是只追求 ROI 达标下规模最大化，则 `remaining_cash_budget` 应设为全局可投上限之和，而不是用启发式函数提前算一个小预算。

2. 单元 cap 约束

```text
sum_s x[u,t,s] <= cap[u,t]
```

3. 弹性爬坡约束

```text
x[u,t] <= last_spend[u] * scale[t] / scale[t-1] * ramp_factor[u]
```

4. 风险降权约束

```text
x[u,t] <= cap[u,t] * risk_multiplier[u]
```

其中 `risk_multiplier` 可取：

```text
正常: 1.0
轻微异常: 0.7
高风险: 0.3
严重异常: 0
```

5. 版位/流量类型约束

```text
sum_{u in 联盟流量} x[u,t] <= alliance_cap[t]
sum_{u in 自有流量} x[u,t] <= owned_cap[t]
```

6. 人工锁定约束

运营可以指定：

```text
x[u,t] = fixed_budget
x[u,t] >= min_budget
x[u,t] <= max_budget
```

### 4.4 优化器选择

P0 阶段继续用 PuLP/CBC：

```text
适合几十个产品、几百个投放单元、30天以内 horizon、3~5个预算段。
```

P1/P2 推荐迁移到：

```text
OR-Tools / Pyomo + HiGHS
```

如果商业环境允许：

```text
Gurobi / CPLEX
```

## 5. MPC 滚动控制

当前 `budget_system_main.py` 已有简单 MPC 校准因子，建议升级为三层校准：

1. D1 校准

```text
calibrated_d1 = model_d1 * EWMA(actual_d1 / predicted_d1)
```

2. LTV 校准

```text
calibrated_multiplier[a] = base_multiplier[a] * cohort_error_factor[a]
```

3. 预算执行校准

```text
actual_spend / suggested_budget
```

如果媒体侧执行不稳定，优化器下一天要根据执行率调整 cap 和 ramp。

MPC 每天只执行今天的预算，未来预算只是轨迹：

```text
每天重新解剩余日期，避免一次性锁死整月。
```

## 6. 输出设计

最终报告不应只给“建议预算”，还要解释为什么：

每个投放单元输出：

```text
应用ID
推广流量 / 流量场景 / 创意规格 / 转化类型
建议日预算
预测 D1 P20/P50/P80
本月内回收倍率
边际 ROI
ROI 约束贡献
cap 使用率
风险状态
约束绑定原因
```

月度汇总输出：

```text
当前累计消耗
当前累计回收
历史长尾预测
未来新增回收预测
预测月末 ROI P20/P50
ROI 安全垫
最大可释放预算
如果 KPI 调整到 102/105/108 时的敏感性分析
```

## 7. 与当前代码的改造映射

### data_loader.py

建议新增：

```text
get_unit_level_data()
```

输出粒度固定为：

```text
日期 + 应用ID + 推广流量 + 流量场景 + 创意规格 + 转化类型
```

并新增 cohort maturity 标记：

```text
has_d3_label
has_d7_label
has_d30_label
age_days
```

### ltv_predictor.py

建议替换为：

```text
LTVCurveModel
```

核心能力：

```text
fit(unit/product/global)
predict_multiplier(unit, age)
predict_tail_revenue(cohort, target_date)
```

### ml_predictor.py

建议替换为：

```text
D1QuantilePredictor
```

输出：

```text
P20 / P50 / P80 D1
```

并保留产品/单元层级兜底。

### product_diagnosis.py

从 A/B/C 静态分级升级为：

```text
unit_health_score
risk_multiplier
cap_adjustment
```

分级仍可作为报告字段，但不应作为主要优化约束。

### roi_planner.py

建议只负责状态计算：

```text
actual_spend
actual_revenue
historical_tail_revenue
roi_safety_buffer
```

不要提前用启发式 `_calculate_remaining_budget` 截断可投预算。最大可投规模应由优化器在 ROI 硬约束下求出来。

### portfolio_optimizer.py

重写为联合优化器：

```text
JointBudgetOptimizer
```

输入：

```text
units
future_dates
d1_quantiles
ltv_multipliers
segment_response_curves
caps
risk_multipliers
actual_month_state
kpi
```

输出：

```text
unit × date 的预算表
今日执行预算
月度轨迹
约束解释
```

### budget_system_main.py

保持编排职责，但流程改为：

```text
load -> feature/state -> predict -> optimize -> report -> persist_mpc_state
```

## 8. 推荐落地路线

### P0：一周内能显著提升的版本

1. 决策粒度改到 `应用ID + 投放维度`
2. 优化器改成 `unit × date` 联合 LP
3. 用 P20 D1 替代均值 D1 做 ROI 约束
4. 用历史 P90/最近7天 × scale 做 cap
5. 去掉 `_calculate_remaining_budget` 对预算的提前截断，让优化器直接求最大可投
6. 修复仍按 `广告主ID` 汇总的报告/统计逻辑

### P1：两到三周内的效果版

1. LightGBM/CatBoost 分位数预测
2. 单调 LTV 曲线 + cohort 截尾处理
3. 分段边际 ROI 曲线
4. 风险 multiplier 进入优化约束
5. 时间序列回测框架

### P2：生产级最优版

1. 扩量实验验证真实边际 ROI 与 cap
2. 鲁棒优化或 CVaR 约束
3. 跨月价值进入目标函数
4. 客户/KPI 敏感性分析
5. 自动执行 API 与人工锁定规则

## 9. 最推荐的最终算法

最终推荐方案：

```text
层级分位数预测 + 单调 LTV cohort 模型 + 分段边际 ROI + 鲁棒联合线性规划 + MPC 滚动校准
```

一句话总结：

```text
用模型预测每个投放单元在每一天、每一段预算的保守边际回收，
再用联合优化器在月度 ROI 硬约束下吃掉所有仍然安全的预算段，
每天根据真实 D1 和执行偏差重新校准。
```

这比当前版本更优的原因：

1. 直接对齐业务目标：ROI 达标前提下最大化消耗。
2. 直接输出运营需要的产品 × 版位预算。
3. 避免平均 ROI 导致的过量投放。
4. 用 P20/鲁棒约束降低月底 ROI 翻车概率。
5. 能自然利用节假日窗口，而不是后处理式分摊。
6. 每天滚动修正，适合投放数据强波动场景。
