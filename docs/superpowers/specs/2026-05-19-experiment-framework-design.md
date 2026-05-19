# 实验框架：可回退的模型改进验证体系

**日期**: 2026-05-19
**状态**: draft

## 问题

Spend T+1 MAPE 62%，ROI T+1 MAPE 35.5%。已尝试 spend^0.35 权重，但预测准确度仍不理想。需要系统性地尝试改进方法，但每次尝试必须是可回退、可对比的。

## 方案总览

采用**独立实验脚本 + 共享核心模块**架构（B 方案）。

- 共享模块：抽取现有训练脚本的核心函数到 `app/experiments/core.py`
- 每个实验：独立脚本 `scripts/experiments/exp_NN_*.py`，改最小逻辑，输出到独立目录
- 回退：删除实验脚本 + 实验输出目录即可，原管线不受影响

### 实验路线图

| 序号 | 实验 | 工作量 | 依赖 |
|------|------|--------|------|
| exp_01 | 分位数回归（P10/P50/P90）| 1-2 天 | 无 |
| exp_02 | 分层预测调和（MinTrace） | 1-2 周 | hierarchicalforecast |

exp_01 和 exp_02 独立评估。两个都 inconclusive 时再评估 DeepAR 等高投入方案。

---

## 1. 共享模块抽取

新建 `app/experiments/__init__.py` 和 `app/experiments/core.py`。

从 `scripts/run_parallel_models_spend_t1.py` 提取以下函数，原封不动：

| 函数 | 作用 |
|------|------|
| `_tree_predict` | 单模型训练+预测 |
| `_evaluate` | 计算 MAE/RMSE/MAPE |
| `_run` | walk-forward 主循环 |
| `_feature_cols` | 特征列名 |
| `_safe_model_filename` | 文件名安全化 |

`run_parallel_models_spend_t1.py` 改为 `from app.experiments.core import ...`，行为不变（纯重构）。

新建目录 `scripts/experiments/` 放置实验脚本。

---

## 2. 实验脚本模板

每个实验脚本 ~80-120 行，统一结构：

```
scripts/experiments/
├── exp_01_quantile.py      # 分位数回归
└── exp_02_hierarchical.py  # 分层调和
```

固定模式：
- **输入**：复用 `build_unified_daily()` 管线，不改数据
- **改动点**：只改模型参数/后处理，其余继承 core.py
- **输出**：`outputs/experiments/<exp_name>/`，含 `predictions_*.csv` + `metrics_summary.csv` + `experiment_report.json`
- **对比**：报告内置 baseline 对比逻辑

回退方式：删除 `outputs/experiments/<exp_name>/` + 对应脚本。

---

## 3. 实验一：分位数回归

### 改动点

在 `_tree_predict` 中增加分位数 variant，训练 3 个独立 XGBoost 模型：

```python
# P10: 悲观估计（只有 10% 的情况实际值低于此）
xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.1)
# P50: 中位估计
xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5)
# P90: 乐观估计（只有 10% 的情况实际值高于此）
xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.9)
```

### 评估指标

不使用 MAPE（分位数的目标是分位数，不是均值）：

| 指标 | 含义 | 理想值 |
|------|------|--------|
| Pinball Loss | 分位数专用损失 | 越低越好 |
| Coverage | 实际值落在 [P10, P90] 区间比例 | ~80% |
| Interval Width | (P90-P10)/P50，标准化宽度 | 尽量窄 |

### 验证假设

1. **区间校准**：Coverage ≈ 80%？在全量和高 spend 桶分别看
2. **决策价值**：在高 spend app 上，P10 是否显著低于 P50？告诉决策者何时应保守

### 高 spend 桶专项分析

按 spend 量级分桶（Q1/Q2/Q3/Q4），看每桶 coverage/interval width。

---

## 4. 实验二：分层预测调和

### 层级结构

```
Level 1: 全部 app 汇总 spend
Level 2: Q1(低) / Q2(中低) / Q3(中高) / Q4(高) 四桶
Level 3: 548 个单 app
```

约束关系：底层预测求和 = 上层预测。

### 流程

1. 同一份 walk-forward 数据，三层分别训练 XGBoost
2. 用 `MinTrace(method='mint_shrink')` 对三层预测做调和（纯后处理，不改模型）
3. 输出调和后 app 级预测，与 baseline 做 MAPE 对比

### 核心假设

Q4 桶的汇总 spend 比单个高 spend app 稳定得多（CV 更低），调和约束能把稳定性传导到 app 级预测。

### 对比逻辑

- 按 spend 桶拆分 MAPE diff
- 重点关注 Q4 桶改善幅度
- Q4 改善而 Q1 退化，可接受（业务价值优先）

### 依赖

```
pip install hierarchicalforecast
```

---

## 5. 实验对比框架

### experiment_report.json 结构

```json
{
  "experiment": "exp_01_quantile",
  "baseline": "spend_v12_unified",
  "timestamp": "2026-05-19T...",
  "metrics": {
    "pinball_loss": null,
    "coverage": null,
    "interval_width_pct": null
  },
  "by_spend_bucket": {
    "Q4_high": {},
    "Q3": {},
    "Q2": {},
    "Q1_low": {}
  },
  "baseline_comparison": {
    "baseline_mape_pct": 62.0,
    "experiment_mape_pct": null,
    "mape_diff_pp": null
  },
  "verdict": "pending"
}
```

### 判定标准

| 判定 | 条件 | 行动 |
|------|------|------|
| Promising | 指标明显优于 baseline，或业务视角确认有用 | 保留脚本，考虑设为默认 |
| Inconclusive | 指标接近 baseline，无明显改善 | 保留产物，标记已探索 |
| Regress | 指标明显退化 | 删除脚本和输出目录 |

### 停止规则

分位数和分层调和独立。都 inconclusive 时，再评估 DeepAR 等高投入方案。
