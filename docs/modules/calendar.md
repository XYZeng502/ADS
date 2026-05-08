# 时间层模块（CalendarService）

## 位置

- `app/core/calendar.py`

## 职责

- 识别日类型并输出时间窗口强度
- 为求解与预测提供“何时放量”的先验

## 分类优先级

1. holiday
2. shopping_festival
3. summer_winter
4. weekend
5. workday

## 输出

- `classify_day(date) -> day_type`
- `get_scale_factors(day_type) -> scale`

## 已落地价值

- 统一了策略与预测对“时间窗口”的理解，避免口径不一致。

## 待优化方向

- 外置节日配置（按业务线可自定义）
- 引入“活动日标签”与“版本更新标签”
