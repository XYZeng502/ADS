# 多维预测模块

## 位置

- `app/services/predictor.py`: 释放曲线 + 月末 ROI 推算
- `app/experiments/core.py`: T+1 模型训练核心 + CQR 区间校准
- `scripts/run_parallel_models_spend_t1.py`: Spend T+1 训练
- `scripts/run_parallel_models_roi_d1.py`: ROI T+1 训练

## 预测能力

| 预测目标 | 模型 | 指标 | 输出 |
|---------|------|------|------|
| T+1 Spend | XGBoost log1p | MAPE ~58% | 点预测 |
| T+1 Spend 区间 | CQR 分桶校准 | 覆盖率 75% | P05/P50/P95 |
| T+1 ROI D1 | XGBoost | MAPE ~35% | 首日回收率 |
| 30 天释放曲线 | 倍率模板 | per-app | 日级乘数 |
| 每日买量收入 | 释放曲线反向 | - | 预测 vs 实际 |

## CQR 区间原理

1. 训练 3 个分位数 XGBoost (P05/P50/P95)
2. 按 app 历史 spend 分 Q1-Q4 四桶
3. 每桶独立 Conformalized Quantile Regression 校准
4. 输出 [y_lower, y_pred, y_upper]

业务用途：保守(P05) / 中性(P50) / 激进(P95) 三档预算决策。

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| min_train_days | 20 | 最少历史天数 |
| min_spend_train | 2.0 | 最低消耗过滤 |
| weight_exponent | 0.35 | 权重 spend^exponent |
| eval_recent_days | 40 | 评估窗口 |
| q4_confidence | 0.85 | CQR Q4 覆盖率 |
