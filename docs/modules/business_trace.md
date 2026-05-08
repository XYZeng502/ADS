# 业务链路模块（BusinessTraceService）

## 位置

- `app/services/business_trace.py`

## 目标

把每日计划中的预测、求解、风控和规则结果整理成运营可读的五层业务链路：

- 目标/KPI层
- 预测层
- 求解层
- 风险层
- 决策解释层

## 输入

- `ClientContext`
- `SolveResult`
- 月末消耗/ROI预测
- KPI调整影响
- ROI偏差归因
- 分版位预算汇总
- 产品诊断、风险告警、规则命中

## 输出

- `BusinessDecisionTrace`
  - `kpi_layer`
  - `forecast_layer`
  - `solver_layer`
  - `risk_layer`
  - `decision_layer`

每层统一包含：

- `status`: `OK` / `WARN` / `CRITICAL`
- `primary_metrics`
- `key_findings`
- `recommended_actions`

## 接口

- `POST /v1/business/decision-trace`

该接口复用每日计划计算，只返回 `business_trace`，便于前端在不解析完整日报的情况下渲染业务链路。

## 设计原则

- 不改变预测、求解、风控的计算结果。
- 不新增会影响预算分配的规则。
- 所有状态和解释均从现有结果派生，优先保证兼容性与可审计性。
