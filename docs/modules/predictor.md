# 预测模块（PredictorService）

## 位置

- `app/services/predictor.py`

## 职责

- 估计当日“历史 cohort 释放回收”
- 预测月末 ROI 与月末规模
- 输出偏差归因（SPEND / D1_ANCHOR / CURVE）

## 月末 ROI 口径（当前实现）

- 分母：当月累计消耗 + 剩余期预测消耗
- 分子：当月 cohort 在月内可释放回收（最多观察 30 天增量）
- 跨月处理：只计“当月内可归属部分”
- 不使用“买量总收入”作为核心评估分子

## 与离线 roi_d1 脚本对齐

- 离线默认：`run_parallel_models_roi_d1_by_flow.py` 使用细粒度零消耗清洗后，在 **应用ID + 推广流量名称** 上训练/评估；在线服务中槽位级信号经 **消耗加权** 融合后，与上述「按流量权重看 D1」一致。
- 求解器内产品内参考 D1 汇总方式见 `docs/modules/solver.md`（`solver_slot_d1_normalize_mode`）。

## 核心组件

- D1锚点：近期真实D1 + 预测D1 + 槽位加权信号融合
- 释放曲线：可读取外部曲线 CSV，缺失回退默认模板
- 花费轨迹：结合日历窗口（节假日/暑寒假/周末）调整

## 输出

- `month_end_roi`
- `month_total_scale_forecast`
- `roi_forecast_attribution`

## 已落地价值

- 已能解释“为什么偏差大”，而非只给单点结果。

## 待优化方向

- 增加“产品族群模板曲线”而非单模板。
- 引入不确定性区间（P25/P50/P75）用于风险边界策略。
