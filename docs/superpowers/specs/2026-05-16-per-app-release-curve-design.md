# Per-App 真实释放曲线 — 月末 ROI 预测改造

**日期**: 2026-05-16
**状态**: draft

## 问题

月末 ROI 预测当前使用 D1 锚点 × **固定释放曲线**（所有 app 共用 D1=1.0, D3=1.19, D7=1.31, D30=1.47）。CSV 中实际有 `3日累计变现金额` 和 `7日累计变现金额` 列，但从未被月末预测使用。不同 app 的回收节奏差异被忽略。

## 方案

用每个 app 历史数据中观测到的真实 D3/D1、D7/D1、D30/D1 比率，构建 per-app 释放曲线，替换全局固定曲线。

### 曲线构建规则

训练期遍历所有历史天数据时，累积每个 app 的总量：

```
app_total_d1  = Σ(首日广告收入)
app_total_d3  = Σ(3日累计变现金额)
app_total_d7  = Σ(7日累计变现金额)
app_total_d30 = Σ(30日累计变现金额)
```

倍率节点：`{1: 1.0, 3: d3/d1, 7: d7/d1, 30: d30/d1}`，节点间线性插值。
每个节点比率 clamp 到 `[1.0, 3.0]` 防止极端值污染。

曲线在训练数据累积完成后一次性构建（非增量更新）。

**回退条件**（任一项触发则走 fallback 链）：
- app 训练天数 < `app_last_day_min_train_days`（默认 30）
- app 总 D1 收入 <= 0
- 任一节点比率 <= 0 或 > 5.0（异常值剔除）

### 改动文件

| 文件 | 改动 |
|------|------|
| `scripts/offline_backtest.py` | `SlotAgg` +d3/d7；`load_aggregates_app_level` 读取 D3/D7 列；训练循环中累积 per-app 曲线；`_month_end_incremental_roi_for_app` 接受 per-app curve；存储曲线到文件 |
| `app/services/predictor.py` | 支持按 `app_id` 查找 per-app 曲线；`predict_month_end_roi` 使用 per-app 曲线；服务启动时加载曲线文件 |
| `app/data_cleaning.py` | 确认 D3/D7 列名 |

### 曲线文件格式

存储到 `outputs/per_app_release_curves.json`：

```json
{
  "app_001": {"1": 1.0, "2": 1.12, "3": 1.23, ..., "30": 1.52},
  "app_002": {"1": 1.0, "2": 1.06, "3": 1.10, ..., "30": 1.35}
}
```

在线 predictor 启动时加载此文件，按 `context.client_id` 查找。

### 兼容性

- 无 per-app 曲线时（文件缺失 / app 不在文件中），回退到现有逻辑（竞品模板 → 默认曲线）
- 现有 API 接口签名不变，`app_id` 已存在于 `ClientContext.client_id`
- **Fallback 链优先级**: per-app 观测曲线 > 竞品模板曲线 > 硬编码默认曲线
