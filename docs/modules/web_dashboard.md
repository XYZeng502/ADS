# Web 看板

## 入口

`http://localhost:8000/web`

## 页签功能

### 预测模块
- T+1 Spend/ROI 预测效果对比
- CQR 预测区间带（P05-P95 阴影 + P50 虚线）
- 分应用查询：点击应用查看其独立预测曲线
- 预测明细表：含 CQR 区间列（下限 P05 / P50 / 上限 P95）
- 日期范围筛选、分页

### 推荐模块
- 应用月末 ROI 预测表（搜索、筛选、排序）
- 三档预算推荐：保守(P05)、中性(P50)、激进(P95)
- 六层决策面板：KPI→时间→配置→节奏→风控→决策排序
- 推荐动作分布图、优先级评分

### 每日收入验证
- 释放曲线反向预测 vs 实际买量收入对比
- Carryover 尾量分解（仅释放曲线部分）
- 分应用曲线诊断表（Carryover MAPE）
- 日期范围筛选

### 系统监控
- 模型管理面板：数据状态、模型文件、重训触发
- 告警历史（近7天）
- 健康趋势图（30天快照）

## 数据来源

看板直接从 `outputs/` 目录读取模型预测 CSV：

| 页签 | 数据文件 |
|------|---------|
| 预测模块 | `model_parallel_spend_t1_v12_unified/predictions_*.csv` |
| | `model_parallel_roi_d1_v9_unified/predictions_*.csv` |
| CQR 区间 | `model_parallel_spend_t1_v12_unified/predictions_XGBoost_CQR.csv` |
| 推荐模块 | `app_level_last_day_suggestions.csv` |
| | `tiered_budget_recommendations.csv` |
| 收入验证 | `daily_revenue_predictions.csv` |
