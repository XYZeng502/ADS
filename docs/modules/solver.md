# 求解模块（BudgetSolver）

## 位置

- `app/services/solver.py`

## 目标

在月末 ROI 达标约束下，最大化可释放消耗规模（产品 × 时间联合求解）。

## 模型要点

- 决策变量：`x[slot, t]`（某槽位在第 `t` 天的预算）
- 目标函数：最大化剩余期消耗（含时序偏好）
- 关键约束：
  - 今日预算执行上限
  - 槽位日预算上限（cap + scale）
  - 月末 ROI 硬约束（组合级）
  - GUARD 模式低效槽位抑制
  - 时序平滑约束（避免跳变）

## 输入

- `ClientContext`
- `day_scale`
- `current_date`

## 输出

- `SolveResult`
  - mode
  - suggestions（分产品分版位）
  - expected_today_spend
  - expected_today_revenue

## 当前状态

- 已接入 `PuLP` 可运行联合优化。
- 求解失败时自动回退启发式，保证可用性。

## 产品内 D1 对齐（与离线训练口径）

- 将产品级 `resolve_product_d1_signal` 锚点映射到各槽位 `predicted_d1_roi` 时，**参考 D1** 默认按 `yesterday_spend` 加权汇总，与 `signal._weighted_slot_pred_d1`、`PredictorService._portfolio_d1_roi_anchor` 一致，并与离线脚本「应用 + 推广流量」维度的消耗加权思想对齐。
- 配置项 `solver_slot_d1_normalize_mode`：`spend_weighted`（默认）或 `arithmetic_mean`（与早期「槽位算术平均」逐向一致，便于对比误差）。

## 输入扩展

- 槽位可选 `flow_name`（推广流量名称）、产品可选 `app_id`：用于审计与和离线实体键对齐，不改变加权公式（仍按槽位消耗加权）。

## 已知问题

- 在离线 25 天样本上，相比旧启发式仍有“偏保守”现象。
- 参数敏感，放量与告警之间存在权衡。

## 待优化方向

- 约束分层：工作日/节假日分层 floor。
- 引入“执行弹性预算池”减少欠投放。
- 增加分客户参数模板（按历史表现自动选参）。
