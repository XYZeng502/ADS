# 每日买量收入预测验证表

**日期**: 2026-05-17
**状态**: draft

## 问题

月末 ROI 预测依赖释放曲线模型，但缺少中间验证环节。释放曲线是否正确，应该能在每日粒度上接受检验：如果曲线准确，它应该能解释观测到的每日买量总收入。

## 方案

用释放曲线模型反向预测每日买量收入，输出预测 vs 实际对比表，与 T+1 预测格式一致。

### 核心公式

```
predicted_revenue(day) = Σ spend(s) × D1_ROI(s) × [F(age_today) - F(age_yesterday)]
                         s ∈ 所有 spend_day(s) ≤ day 的 cohort
```

- `spend(s)`: cohort s 的日消耗
- `D1_ROI(s)`: cohort s 的首日 ROI
- `F(age)`: per-app 释放倍率曲线
- `age = (day - spend_day) + 1`

### 输出格式

```csv
日期,应用ID,y_true,y_pred,days_since_start
2026-03-15,30262609,4523.5,3891.2,15
2026-03-15,30744533,12800.0,14102.5,15
```

- `y_true` = 当日 `买量广告收入`（CSV 按 app+day 聚合）
- `y_pred` = 释放曲线模型预测
- `days_since_start` = 数据覆盖天数（<30 标记首月数据不足）

### 改动文件

| 文件 | 改动 |
|------|------|
| `scripts/predict_daily_revenue.py` | 新建：读取 CSV → 加载曲线 → 预测每日收入 → 输出 CSV |
| `app/web.py` | 新增每日收入验证区域，展示最近预测 vs 实际 |

### 实现细节

**脚本流程**：

1. 读取 `daily_merged.csv`，按 `(日期, 应用ID)` 聚合：
   - `spend` = sum(消耗金额)
   - `d1_revenue` = sum(首日广告收入)
   - `actual_revenue` = sum(买量广告收入)
2. 计算每日 D1_ROI = d1_revenue / spend
3. 加载 per-app 释放曲线（优先 `outputs/per_app_release_curves.json`，fallback 默认曲线）
4. 按日期顺序遍历：
   - 对每天，遍历该 app 所有历史 cohort（spend_day ≤ 当前天）
   - 累加每个 cohort 当日贡献
   - 输出 y_pred vs y_true
5. 保存到 `outputs/daily_revenue_predictions.csv`

**首月偏差处理**：
- `days_since_start < 30` 行标注数据不足
- 评估指标可按 `days_since_start >= 30` 筛选计算

**Web 面板**：
- 新增一个 tab 或折叠区域
- 展示最近 N 天的预测 vs 实际曲线图（ECharts）
- 按 app 筛选

### 评估指标

- MAPE（按 app，筛选 days>=30）
- 整体 MAPE
- 可视化：预测 vs 实际时间序列
