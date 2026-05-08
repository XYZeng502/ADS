# 模块化详细文档索引

本目录用于沉淀“智能预算决策系统”各模块的职责、输入输出、关键参数、已落地能力与待优化项。

## 文档清单

- `orchestrator.md`：每日执行编排主流程
- `solver.md`：联合求解器（产品 × 时间）
- `predictor.md`：月末 ROI 预测与归因
- `risk.md`：风控与诊断模块
- `rule_engine.md`：可配置规则引擎
- `calendar.md`：时间层/节假日识别
- `web_dashboard.md`：强交互可视化看板

## 使用建议

1. 先看 `orchestrator.md` 了解端到端流程。
2. 再看 `solver.md` 与 `predictor.md`，理解“如何在 KPI 约束下放量”。
3. 最后看 `risk.md`、`rule_engine.md`、`web_dashboard.md` 对应运营落地。
