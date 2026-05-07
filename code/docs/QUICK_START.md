# 智能预算决策系统 - 快速参考指南

## 📋 系统8大模块概览

### 模块功能对照表

| # | 模块名称 | 文件 | 核心功能 | 输入 | 输出 |
|---|--------|------|--------|------|------|
| 1 | 数据加载与预处理 | data_loader.py | CSV加载、清洗、验证 | CSV文件 | 清洗后的DataFrame |
| 2 | LTV曲线与回收预测 | ltv_predictor.py | 用户生命周期价值、D1-D30倍率计算 | 消耗和变现数据 | LTV曲线字典 |
| 3 | 节假日识别与标注 | holiday_identifier.py | 节假日识别、scale因子、高价值窗口 | 日期范围 | 节假日表、scale因子 |
| 4 | 产品分级与诊断 | product_diagnosis.py | A/B/C级分级、效率诊断、风险预警 | 历史数据、LTV | 产品诊断字典 |
| 5 | ROI预测与消耗规划 | roi_planner.py | 月度ROI预测、可用预算计算 | 历史数据、LTV | ROI预测结果 |
| 6 | 多产品联合优化 | portfolio_optimizer.py | 线性规划求解、最优预算分配 | 诊断结果、预算 | 分配方案 |
| 7 | 报告输出 | report_generator.py | 日报生成、诊断表、建议输出 | 各模块结果 | HTML/CSV报告 |
| 8 | ML D1预测 | ml_predictor.py | LightGBM基于投放维度的D1预测 | 历史数据、节假日 | D1预测字典 |

---

## 🚀 快速使用

### 最简单的用法

```python
from budget_system_main import IntelligentBudgetSystem

# 一行代码启动系统
system = IntelligentBudgetSystem('daily_20260421_120112.csv')

# 生成最新日期的报告
report = system.daily_forecast()

# 生成月度总结
march = system.month_summary(2026, 3)
april = system.month_summary(2026, 4)
```

### 自定义KPI

```python
# 改为100%的KPI
system = IntelligentBudgetSystem('data.csv', kpi_target=1.00)

# 改为110%的KPI
system = IntelligentBudgetSystem('data.csv', kpi_target=1.10)
```

### 指定预测日期

```python
# 查看2026-04-15的预算建议
report = system.daily_forecast(forecast_date='2026-04-15')
```

---

## 📊 各模块详细API

### Module 1: DataLoader

```python
from data_loader import DataLoader

# 初始化
loader = DataLoader('data.csv')
data = loader.load_data()  # 加载并清洗

# 获取聚合数据
daily = loader.get_aggregated_data(level='day')      # 日度聚合
product = loader.get_aggregated_data(level='product') # 产品聚合
app = loader.get_aggregated_data(level='app')        # 应用聚合

# 获取日期范围
min_date, max_date = loader.get_date_range()
```

### Module 2: LTVPredictor

```python
from ltv_predictor import LTVPredictor

# 初始化
predictor = LTVPredictor(data)
ltv_curves = predictor.calculate_ltv_curves()
# 返回结构（产品ID = 应用ID）：
# {
#   'app_001': {  # 应用ID
#     'curve': {'D1': 1.0, 'D3': 1.29, 'D7': 1.38, ...},
#     'd1_rate': 0.69,
#     'avg_roi': 1.05,
#     'total_spend': 100000,
#     'total_revenue': 105000
#   },
#   ...  # 其他应用
# }

# 获取特定产品的信息
d1_rate = predictor.get_d1_rate('product_001')  # 返回 0.69
ltv = predictor.get_ltv_curve('product_001')    # 返回曲线字典

# 预测未来回收
future_revenue = predictor.predict_future_roi(
    today_spend=1000,        # 今天花费1000元
    d1_revenue=690,          # D1回收690元
    days_remaining_in_month=10  # 月内还剩10天
)
# 返回预期的未来总回收
```

### Module 3: HolidayIdentifier

```python
from holiday_identifier import HolidayIdentifier

holiday = HolidayIdentifier(year=2026)

# 获取特定日期信息
info = holiday.get_holiday_info('2026-04-05')
# {'name': '清明节', 'scale_factor': 1.4, 'type': 'short_holiday'}

# 获取时间范围的scale表
import pandas as pd
start = pd.to_datetime('2026-04-01')
end = pd.to_datetime('2026-04-30')
factors = holiday.get_holiday_scale_factors(start, end)
# DataFrame: date, scale_factor, holiday_type, holiday_name

# 识别高价值窗口
peaks = holiday.identify_peak_windows(start, end)
# [
#   {
#     'start_date': ...,
#     'end_date': ...,
#     'duration_days': 5,
#     'avg_scale_factor': 1.45,
#     'holiday_name': '五一劳动节'
#   },
#   ...
# ]

# 获取月度scale统计
month_stats = holiday.get_monthly_scale_factors(2026, 4)
# {
#   'weighted_avg_scale_factor': 1.05,
#   'type_distribution': {'workday': 20, 'weekend': 8, ...},
#   'details': DataFrame
# }
```

### Module 4: ProductDiagnostics

```python
from product_diagnosis import ProductDiagnostics

diagnosis = ProductDiagnostics(data, ltv_curves, kpi_target=1.05)

# 诊断所有产品（产品ID = 应用ID）
all_diag = diagnosis.diagnose_all_products(historical_data, current_date, kpi_target)
# 返回结构：
# {
#   'app_001': {  # 应用ID是产品标识
#     'efficiency_status': {
#       'grade': 'A',
#       'd1_rate': 0.72,
#       'actual_roi': 1.07,
#       'target_roi': 1.05
#     },
#     'roi_gap': {'absolute': 0.02, 'percentage_points': 2.0},
#     'optimization': {'direction': '加量', 'suggestion': '...'},
#     'risk_alerts': [...]
#   },
#   ...
# }

# 获取组合诊断摘要
summary = diagnosis.get_portfolio_summary(all_diag, kpi_target=1.05)
# {
#   'total_products': 150,
#   'product_grades': {'A': 30, 'B': 60, 'C': 60},
#   'portfolio_roi': 1.06,
#   'roi_gap': 0.01,
#   'products_at_risk': 15
# }
```

### Module 5: ROIPlanner

```python
from roi_planner import ROIPlanner

planner = ROIPlanner(data, ltv_curves, kpi_target=1.05)

# 预测月度ROI
forecast = planner.forecast_monthly_roi(
    historical_data=month_data,
    current_date='2026-04-21',
    month_start='2026-04-01',
    month_end='2026-04-30',
    kpi_target=1.05
)
# 返回：
# {
#   'actual_spend': 540000,          # 已发生消耗
#   'actual_revenue': 570000,        # 已发生回收
#   'predicted_total_spend': 610000, # 月度预计总消耗
#   'predicted_total_revenue': 650000, # 月度预计总回收
#   'predicted_roi': 1.066,          # 月度预测ROI
#   'roi_gap': 0.016,                # ROI缺口（正数=盈余）
#   'days_remaining': 9
# }

# 评估可达性
reachability = planner.calculate_roi_reachability(
    current_roi=1.05,
    target_roi=1.05,
    days_remaining=9,
    avg_daily_spend=20000
)

# 计算跨月价值
cross_month = planner.calculate_cross_month_value(
    today_spend=10000,
    d1_rate=0.69,
    m30_multiple=1.44
)
# {'today_spend': 10000, 'current_month_impact': -0.305, 'next_month_contribution': 8500}
```

### Module 6: PortfolioOptimizer

```python
from portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()

# 优化预算分配
allocation = optimizer.optimize_allocation(
    product_diagnoses=all_diag,
    available_budget=50000,          # 可用预算50k
    holiday_info=holiday_factors_df,
    days_remaining=9,
    kpi_target=1.05
)
# 返回：
# {
#   'allocations': {
#     'product_001': {'allocated_budget': 15000, 'daily_budget': 1667, ...},
#     'product_002': {'allocated_budget': 12000, 'daily_budget': 1333, ...},
#     ...
#   },
#   'total_allocated': 48500,
#   'utilization_rate': 0.97,
#   'status': 'Optimal',
#   'solver_status': 'SUCCESS'
# }

# 按时间窗口分配
time_alloc = optimizer.allocate_by_time_window(
    allocation,
    holiday_factors_df,
    days_remaining=9
)
# 返回：
# {
#   'product_001': {
#     datetime(2026,4,22): 2000,  # 周一
#     datetime(2026,4,23): 2000,  # 周二
#     ...
#     datetime(2026,4,30): 3000   # 周三（周末，scale更高）
#   },
#   ...
# }
```

### Module 7: ReportGenerator

```python
from report_generator import ReportGenerator

reporter = ReportGenerator()

# 生成每日综合报告
report = reporter.generate_daily_report(
    forecast_date='2026-04-21',
    monthly_stats=monthly_stats,
    product_diagnoses=all_diag,
    roi_forecast=roi_forecast,
    budget_allocation=allocation,
    holiday_info=holiday_factors_df,
    remaining_days=9,
    kpi_target=1.05
)
# 返回：
# {
#   'data': DataFrame,  # 可直接保存为CSV
#   'monthly_overview': str,  # 月度概览文本
#   'diagnosis': DataFrame,   # 产品诊断表
#   'allocation': DataFrame,  # 预算分配表
#   'roi_forecast': dict      # ROI预测数据
# }

# 保存报告
report['data'].to_csv('daily_report.csv', index=False)

# 生成月度总结
monthly_summary = reporter.generate_monthly_summary(
    data, year=2026, month=4, kpi_target=1.05
)
```

### Module 8: MLPredictor

```python
from ml_predictor import MLPredictor

# 初始化并训练LightGBM模型
ml = MLPredictor(data, holiday_identifier=holiday_obj)
trained = ml.train()  # 返回 True（训练成功）或 False（不可用时降级）

# 获取某个产品在特定时间段的平均D1预测
product_ids = ['app_001', 'app_002', 'app_003', ...]
d1_rates = ml.get_average_d1_for_period(
    product_ids=product_ids,
    start_date=pd.to_datetime('2026-04-20'),
    end_date=pd.to_datetime('2026-04-30'),
    historical_data=data,
    holiday_info=holiday_factors_df
)
# 返回：{'app_001': 0.72, 'app_002': 0.65, 'app_003': 0.68, ...}

# 获取特定日期的详细D1预测（包含所有投放维度）
predictions = ml.predict(historical_data=data, date='2026-04-21')
# 返回：DataFrame，包含target_d1列（0-2.0范围的预测D1）

# 获取产品基线（中位数作为兜底）
baselines = ml.product_medians  # {product_id: {median_d1, std_d1, mean_d1, ...}}
```

**特征说明**：
- `day_of_week`: 0-6（周一到周日）
- `is_weekend`: 1（周末）或0（工作日）
- `day_of_month`: 1-31
- `month`: 1-12
- `holiday_scale`: 1.0-1.8（根据节假日）
- `holiday_type_*`: One-hot编码的6种假日类型
- `推广流量`, `流量场景`, `创意规格`, `转化类型`: 投放维度代码
- `product_median_d1`: 该产品历史D1中位数
- `recent_3d_d1`: 最近3天的平均D1
- `d1_trend_3d`: 最近3天D1与产品中位数的偏离

---

## 🎯 核心业务逻辑

### 产品分级标准

```
达标D1 = KPI目标 ÷ M30倍率
         = 1.05 ÷ 1.44
         = 0.729 (72.9%)

A级：D1 ≥ 达标D1 + 3pp  →  D1 ≥ 75.9%  →  高效，建议加量
B级：D1 ∈ [70.9%, 75.9%)  →  中等，建议维持
C级：D1 < 70.9%  →  低效，建议减量或观察
```

### ROI倍率衰减（根据剩余时间）

```
距月末天数  回收窗口  倍率    说明
>20天      D30      1.44   用户有完整回收周期
7-20天     D7       1.38   能释放约7天的回收
3-7天      D3       1.29   只能释放3天的回收
<3天       D1       1.0    只能计入首日回收
```

### 风险预警触发

```
产品能力下滑：
  - 连续3天 D1 < 历史平均 D1 × 90%
  - 自动降权，建议缩减预算

流量异常波动：
  - 当日消耗偏离平均值 > 2σ（2个标准差）
  - 检查媒体侧投放是否正常
```

### 优化约束条件

```
约束类型          约束条件                说明
─────────────────────────────────────────
总预算上限        Σbudget ≤ 可用预算      不超过可用金额
ROI约束          加权ROI ≥ KPI目标       硬约束，不可违反
A级产品上限      budget_A ≤ 现有×50%     高效产品，允许最高增加50%
B级产品上限      budget_B ≤ 现有×20%     中等产品，小幅增加
C级产品上限      budget_C ≤ 现有×10%     低效产品，基本不增加
```

---

## 📈 关键指标速查表

### 消耗和回收指标

```
消耗金额(spend)       = 花出去的广告费用
首日回收(D1)         = 新增用户当天变现
3日累计回收(D3)      = 新增用户3天内总变现
7日累计回收(D7)      = 新增用户7天内总变现
30日累计回收(D30)    = 新增用户30天内总变现
```

### ROI计算

```
D1 ROI         = 首日回收 ÷ 消耗
M30倍率        = 30日回收 ÷ 首日回收
月度ROI        = 当月总回收 ÷ 当月总消耗
达标D1         = KPI目标 ÷ M30倍率
ROI缺口(pp)    = (KPI - 当前ROI) × 100
```

### 时间相关

```
days_elapsed        = 当前日期 - 月初日期
days_remaining      = 月末日期 - 当前日期
improvement_feasible = days_remaining能否追上缺口
critical_point      = days_remaining ≤ 3（月末冲刺）
```

---

## 🔧 常见调整

### 修改KPI目标

```python
# 在 budget_system_main.py 中修改
system = IntelligentBudgetSystem(csv_path, kpi_target=1.08)  # 改为108%
```

### 修改节假日scale因子

```python
# 在 holiday_identifier.py 的 _build_holiday_calendar 中添加
holidays[datetime(2026, 5, 20)] = {
    'name': '自定义假期', 
    'scale_factor': 1.5,  # 改为你需要的倍数
    'type': 'custom'
}
```

### 修改产品分级标准

```python
# 在 product_diagnosis.py 的 _grade_product 中修改
if d1_gap >= 0.05:  # 改为超过5pp为A级
    return 'A', ...
```

### 修改优化约束（预算上限）

```python
# 在 portfolio_optimizer.py 的 optimize_allocation 中修改
if grade == 'A':
    self.problem += product_budgets[product_id] <= current_spend * 0.8  # 改为80%
```

---

## 📌 常见问题排查

### 报错：ModuleNotFoundError

```
错误：ImportError: No module named 'pulp'
解决：pip install pulp
```

### 报错：文件不存在

```
错误：FileNotFoundError: daily_20260421_120112.csv
解决：确保CSV文件在同目录，或提供完整路径
```

### ROI预测为NaN

```
可能原因：数据中消耗为0或回收为0
解决：检查数据完整性，移除无效行
```

### 预算分配不足

```
可能原因：所有产品都是C级，或ROI缺口大
解决：1) 检查产品效率是否普遍下降
     2) 考虑降低KPI目标
     3) 检查数据是否有异常波动
```

---

## 💾 文件结构

```
项目目录/
├── budget_system_main.py         ← 主程序（从这里启动）
├── data_loader.py                ← 模块1
├── ltv_predictor.py              ← 模块2
├── holiday_identifier.py         ← 模块3
├── product_diagnosis.py          ← 模块4
├── roi_planner.py                ← 模块5
├── portfolio_optimizer.py        ← 模块6
├── report_generator.py           ← 模块7
├── README.md                      ← 详细文档
├── QUICK_START.md                ← 本文件
├── daily_20260421_120112.csv     ← 输入数据
├── daily_report_*.csv            ← 输出：每日报告
├── march_summary_*.csv           ← 输出：3月总结
└── april_summary_*.csv           ← 输出：4月总结
```

---

## 🎓 学习路径

1. **快速理解**（30分钟）
   - 阅读本文件的"系统7大模块概览"
   - 跑一遍 `budget_system_main.py`

2. **深入理解**（2小时）
   - 阅读 README.md 的详细说明
   - 查看每个模块的API文档

3. **定制开发**（根据需要）
   - 修改参数或约束条件
   - 扩展功能（如添加新的约束）

---

## 📞 支持与反馈

如有问题，请检查：
1. 数据文件完整性和格式
2. Python版本是否 >= 3.8
3. 依赖包是否完整安装
4. 是否有权限写入输出目录

---

**Happy Optimizing! 🚀**
