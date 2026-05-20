# CQR 区间接入生产管线

**日期**: 2026-05-20
**状态**: draft

## 问题

现有 Spend T+1 预测只输出点估计（MSE XGBoost），预算推荐基于单一值，缺乏风险视角。CQR 实验已验证分桶校准可在 80% 覆盖率下产出校准良好的预测区间。

## 方案

将 CQR 分桶校准区间集成到：训练脚本 → Web 看板 → 预算推荐。

---

## 1. CQR 训练集成

**文件**：`scripts/run_parallel_models_spend_t1.py`

- 新增 `--enable-cqr` 和 `--q4-confidence` 参数
- `--enable-cqr` 时在 `_run()` 的 `extra_model_variants` 中增加 3 个量化变体（XGBoost_q5/q50/q95）
- Walk-forward 结束后，调用 `_build_cqr_prediction()` 做分桶 CQR 后处理校准
- 输出 `predictions_XGBoost_CQR.csv`：日期, 应用ID, y_true, y_pred, y_lower, y_upper, model
- report.json 新增 `cqr` 字段

**分桶校准逻辑**（从 exp_01c_cqr 提取到 `app/experiments/core.py`）：
- 按 app 历史均值 spend 分配 Q1-Q4 静态桶
- 对每桶独立做 CQR 校准（`ConformalizedQuantileRegressor` with `prefit=True`）
- Q4 的 confidence_level 可配置（默认 0.85）

---

## 2. Web 看板区间展示

**文件**：`app/web.py`

**数据加载**：`_load_model_dashboard_payload()` 读取 `predictions_XGBoost_CQR.csv`。

**图表**：Spend 预测图叠加半透明阴影带（P05-P95），折线为 P50。

```javascript
series: [
  {name: '实际值', type: 'line', data: y_true},
  {name: '预测区间', type: 'line', data: y_lower,
   lineStyle: {opacity: 0}, stack: 'band', symbol: 'none'},
  {name: '区间上界', type: 'line', data: y_upper,
   lineStyle: {opacity: 0},
   areaStyle: {color: 'rgba(66,133,244,0.15)'},
   stack: 'band', symbol: 'none'},
  {name: '预测值(P50)', type: 'line', data: y_pred},
]
```

**工具提示**：hover 显示 `预测: 1234 (区间: 890-1670)`。

**表格**：加 `预测下限` / `预测上限` 两列。

---

## 3. 预算推荐三档

**文件**：`scripts/offline_backtest.py` 和 `app/web.py`

**计算**：spend 用 CQR 区间的 P05/P50/P95，ROI 统一用 P50。

```
保守 = P05_spend × P50_roi × month_end_factor
中性 = P50_spend × P50_roi × month_end_factor
激进 = P95_spend × P50_roi × month_end_factor
```

**Web 展示**：决策面板新增三档切换，每档显示推荐预算和预期回收。表格每 app 三行。

---

## 改动文件

| 文件 | 改动 |
|------|------|
| `app/experiments/core.py` | 新增 `build_cqr_prediction()` 分桶 CQR 校准函数 |
| `scripts/run_parallel_models_spend_t1.py` | 新增 `--enable-cqr`/`--q4-confidence` 参数 + CQR 后处理 |
| `scripts/daily_retrain.py` | spend 步骤加 `--enable-cqr` |
| `app/web.py` | 读取 CQR CSV，图表加区间带，表格加区间列，预算三档 |
| `scripts/offline_backtest.py` | 预算推荐接口支持三档输出 |

## 回退方式

- 不加 `--enable-cqr` 时行为完全不变
- CQR CSV 不存在时 Web 图表回退到无区间样式
