# 系统架构说明（对应六层决策）

## 一、分层映射

- KPI层：`/v1/kpi/recompute` + `ClientContext.kpi_roi`
- 时间层：`CalendarService.classify_day()` 与 `get_scale_factors()`
- 配置层：`BudgetSolver.solve_daily()` 做分产品分版位预算建议
- 节奏层：`mode=GUARD/SCALE` + 月末 ROI 约束判定
- 风控层：`RiskService`（产品能力下滑、流量异常、ROI守护）
- 决策排序层：`DailyOrchestrator.run()` 串联每日滚动重算

## 二、每日执行链路

1. 读取当日上下文（累计消耗/累计回收/KPI/产品状态）
2. 识别日期类型（工作日、周末、电商节等）
3. 执行预算求解（当前启发式，后续可替换联合求解）
4. 预测当日历史回收和月末 ROI/规模
5. 输出预警和分级诊断
6. 形成运营可执行建议（手动调预算）

## 三、接口边界（当前）

- `POST /v1/optimize/daily-plan`：核心每日建议
- `POST /v1/report/daily-execution`：运营执行日报（对齐第5章输出）
- `POST /v1/business/decision-trace`：业务链路解释（目标/KPI -> 预测 -> 求解 -> 风险 -> 决策解释）
- `POST /v1/kpi/recompute`：KPI 调整后即时重算
- `POST /v1/alerts/evaluate`：独立评估告警
- `POST /v1/diagnosis/product-grading`：独立输出产品分级

## 四、业务层级输出

`DailyPlanResponse.business_trace` 将原本分散在预测、求解、风控和规则中的信息，整理为前端可直接渲染的五层业务链路：

1. 目标/KPI层：KPI、当前累计ROI、预测月末ROI、KPI差距和运行模式。
2. 预测层：月末消耗预测、月末ROI预测、D1锚点校准和偏差主因。
3. 求解层：今日预算目标、分版位预算、建议数量和执行动作。
4. 风险层：风险告警、A/C档产品数量和风控提示。
5. 决策解释层：命中规则、优先级和最终运营动作。

该层级只增强解释，不改变 `PredictorService`、`BudgetSolver`、`RiskService` 的计算行为。

## 五、算法替换点（后续）

- 预测替换点：`PredictorService`
- 求解替换点：`BudgetSolver`
- 风险策略替换点：`RiskService`
- 节假日来源替换点：`CalendarService`

通过以上边界，系统可在不改接口的情况下持续升级算法能力。