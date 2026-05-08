# 规则引擎模块（RuleEngine）

## 位置

- `app/services/rule_engine.py`
- 配置：`app/config.py`
- 输出模型：`app/schemas.py` 中 `RuleHit`

## 职责

- 将“策略经验”显式化为可配置规则
- 输出命中原因与快照，支持运营解释与审计

## 当前规则（P0）

- `R1_SURPLUS_RELEASE`：ROI 高于 KPI 阈值，建议放量
- `R2_KPI_GUARD`：ROI 低于警戒阈值，建议守护
- `R3_MONTH_END_RELEASE_WINDOW`：月末窗口且有盈余，建议可控释放
- `R4_LOW_EFFICIENCY_CLUSTER`：C档占比过高，建议收缩低效产品
- `R5_TIME_WINDOW_BONUS`：高价值时间窗，建议提高优质槽位上限

## 输出格式

- `rule_id`
- `priority`
- `action`
- `reason`
- `metric_snapshot`

## 已落地价值

- 决策“可解释”，不再是黑箱输出。
- 支持后续产品化配置中心迁移。

## 待优化方向

- 规则冲突治理（同日放量与收缩冲突时的仲裁）
- 规则 AB 实验（记录 hit -> 实际效果）
