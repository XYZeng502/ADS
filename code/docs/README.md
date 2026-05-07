# 智能预算决策系统 - 完整实现指南

## 一、系统概述

根据《智能预算决策系统_业务场景与问题定义》文档第五章的要求，我们实现了一个完整的多模块智能预算决策系统。

**核心目标**：在月度活跃 ROI ≥ KPI（硬约束）的前提下，最大化该客户的月度广告消耗规模。

### 系统特点

- ✅ **8个独立模块**，各司其职（包含ML增强D1预测）
- ✅ **LightGBM D1预测**，基于投放维度的回收率预测
- ✅ **每日自动更新**，滚动优化决策
- ✅ **多产品联合优化**，整体达标即可（ROI池化）
- ✅ **节假日自适应**，动态scale因子调整
- ✅ **实时产品诊断**，A/B/C级分级和风险预警
- ✅ **线性规划求解**，数学最优方案

---

## 二、系统架构

```
智能预算决策系统
│
├─ [模块1] 数据加载与预处理 (data_loader.py)
│   └─ 功能：CSV数据加载、清洗、验证
│
├─ [模块2] LTV曲线与回收预测 (ltv_predictor.py)
│   └─ 功能：计算用户生命周期价值、D1-D30回收倍率
│
├─ [模块3] 节假日识别与标注 (holiday_identifier.py)
│   └─ 功能：识别节假日、计算scale因子、找出高价值窗口
│
├─ [模块4] 产品分级与诊断 (product_diagnosis.py)
│   └─ 功能：A/B/C级分级、效率诊断、风险预警、达标缺口计算
│
├─ [模块5] ROI预测与消耗规划 (roi_planner.py)
│   └─ 功能：预测月度ROI、计算剩余可用预算、评估跨期价值
│
├─ [模块6] 多产品联合优化 (portfolio_optimizer.py)
│   └─ 功能：线性规划求解、最优预算分配、按时间窗口分配
│
├─ [模块7] 报告输出 (report_generator.py)
│   └─ 功能：生成日报、产品诊断、预算建议、风险预警
│
├─ [模块8] ML D1预测 (ml_predictor.py)
│   └─ 功能：LightGBM基于投放维度的D1回收率预测、特征工程
│
└─ [主程序] 整合调度 (budget_system_main.py)
    └─ 功能：模块整合、MPC滚动优化、每日流程编排、报告生成
```

---

## 三、各模块详细说明

### 模块1：数据加载与预处理 (data_loader.py)

**职责**：
- 从CSV加载原始数据
- 数据清洗（日期格式转换、缺失值处理、重复值去除）
- 数据验证（范围检查、列完整性检查）

**关键类**：`DataLoader`

**输入**：
- `daily_20260421_120112.csv` - 包含以下关键列：
  - `日期` - 投放日期
  - `应用ID` - 产品唯一标识（产品ID）
  - `推广流量` - 推广渠道（同一产品可有多个渠道）
  - `流量场景` - 流量场景（同一产品同一渠道可有多个场景）
  - `创意规格` - 创意规格（同一产品的不同创意规格产生不同D1）
  - `转化类型` - 转化类型（不同转化类型的回收不同）
  - `消耗金额` - 当日广告花费
  - `首日广告收入` - D1回收
  - `3日/7日/30日累计变现金额` - 长期回收

**关键概念**：
- **产品ID** = 应用ID（唯一标识一个应用/产品）
- **同一产品的多条数据**：同一应用在一天内会因投放维度不同而产生多条数据
  - 不同渠道（推广流量）：开屏、底部、中部等
  - 不同场景（流量场景）：信息流、搜索、推荐等
  - 不同规格（创意规格）：大图、视频、素材等
  - 不同转化类型：激活、注册、购买等
- **广告主ID** = 管理该产品的广告主账户（一个广告主可管理多个产品）
- **数据聚合粒度** = (日期, 应用ID, 推广流量, 流量场景, 创意规格, 转化类型)

**输出**：
- 清洗后的DataFrame，包含时间范围、产品数量等元数据

**关键方法**：
```python
data = DataLoader(csv_path).load_data()
daily_agg = data_loader.get_aggregated_data(level='day')  # 按日聚合
product_agg = data_loader.get_aggregated_data(level='product')  # 按产品聚合
```

---

### 模块2：LTV曲线与回收预测 (ltv_predictor.py)

**职责**：
- 计算每个产品（应用ID）的用户生命周期价值曲线
- 根据不同投放维度的数据聚合计算产品级别的D1-D30倍率
- 为未来回收预测提供基础数据

**核心概念**：
- **产品粒度** = 应用ID（同一应用在不同渠道/场景/创意的数据会合并计算）
- **标准LTV曲线**（文档中验证的倍率）：
  ```
  D1: 1.00 (基准)
  D3: 1.29
  D7: 1.38
  D14: 1.44
  D30: 1.44 (最终稳定)
  ```

- **产品个性化曲线**：基于历史数据计算，用加权平均与标准曲线平滑

**关键类**：`LTVPredictor`

**关键方法**：
```python
ltv_curves = LTVPredictor(data).calculate_ltv_curves()
# 返回：{product_id: {'curve': {...}, 'd1_rate': ..., 'avg_roi': ...}}

d1_rate = predictor.get_d1_rate(product_id)  # 获取D1回收率
ltv_curve = predictor.get_ltv_curve(product_id)  # 获取完整LTV曲线

# 预测未来回收
future_revenue = predictor.predict_future_roi(today_spend, d1_revenue, days_remaining)
```

---

### 模块3：节假日识别与标注 (holiday_identifier.py)

**职责**：
- 根据日历自动识别节假日
- 为不同类型的时间窗口分配scale因子
- 识别高价值投放窗口

**scale因子设定**（根据文档）：
- 普通工作日：1.0
- 周末：1.25
- 小长假（清明/端午）：1.4
- 五一劳动节：1.5
- 暑假：1.5
- 电商节（618/双11）：1.7-1.8

**关键类**：`HolidayIdentifier`

**关键方法**：
```python
holiday = HolidayIdentifier()

# 获取特定日期的节假日信息
info = holiday.get_holiday_info(date)  
# 返回：{'name': '...', 'scale_factor': 1.25, 'type': 'weekend'}

# 获取时间范围的scale因子表
factors_df = holiday.get_holiday_scale_factors(start_date, end_date)

# 识别高价值投放窗口
peaks = holiday.identify_peak_windows(start_date, end_date)
```

---

### 模块4：产品分级与诊断 (product_diagnosis.py)

**职责**：
- 对所有产品（应用ID）进行A/B/C级分级
- 诊断产品效率、达标缺口、改善空间
- 检测风险警告（能力下滑、流量异常）

**分级标准**：
```
A级（高效）：D1 > 达标D1 + 3pp
B级（中等）：达标D1 - 2pp ≤ D1 ≤ 达标D1 + 3pp
C级（低效）：D1 < 达标D1 - 2pp

达标D1 = KPI ÷ M30倍率
```

**诊断指标**：
- `efficiency_status` - 效率状态（等级、D1、ROI）
- `roi_gap` - 缺口（绝对值和百分点）
- `optimization` - 调优方向（加量/维持/减量）
- `improvement_window` - 改善窗口（剩余天数）
- `risk_alerts` - 风险警告列表

**关键类**：`ProductDiagnostics`

**关键方法**：
```python
diagnosis_obj = ProductDiagnostics(data, ltv_curves, kpi_target=1.05)

# 诊断所有产品
all_diagnoses = diagnosis_obj.diagnose_all_products(historical_data, current_date, kpi_target)
# 返回：{product_id: {efficiency_status, roi_gap, optimization, ...}}

# 获取组合总体诊断
summary = diagnosis_obj.get_portfolio_summary(all_diagnoses, kpi_target)
```

---

### 模块5：ROI预测与消耗规划 (roi_planner.py)

**职责**：
- 预测月度总ROI和消耗规模
- 计算剩余可用预算
- 评估是否能达到ROI目标
- 量化跨月价值（月末蓄力的效果）

**核心公式**：
```
月度ROI = (已发生回收 + 剩余回收预测) / (已发生消耗 + 剩余消耗预测)

剩余回收预测 = Σ(剩余每日消耗 × 该日的ROI倍率)

ROI倍率随剩余时间衰减：
  - 剩余>20天：1.44（D30）
  - 剩余7-20天：1.38（D7）
  - 剩余3-7天：1.29（D3）
  - 剩余<3天：1.0（D1）
```

**关键类**：`ROIPlanner`

**关键方法**：
```python
planner = ROIPlanner(data, ltv_curves, kpi_target=1.05)

# 预测月度ROI
roi_forecast = planner.forecast_monthly_roi(
    historical_data, current_date, month_start, month_end, kpi_target
)
# 返回：{
#   'predicted_roi': 0.106,
#   'predicted_total_spend': 540000,
#   'predicted_total_revenue': 572400,
#   'roi_gap': 0.001,  # 盈余
#   'days_remaining': 7
# }

# 评估ROI可达性
reachability = planner.calculate_roi_reachability(
    current_roi, target_roi, days_remaining, avg_daily_spend
)

# 计算跨月价值
cross_month = planner.calculate_cross_month_value(today_spend, d1_rate, m30_multiple)
```

---

### 模块6：多产品联合优化 (portfolio_optimizer.py)

**职责**：
- 使用线性规划（PuLP库）求解最优预算分配
- 在ROI约束下最大化总消耗规模
- 遵守产品级约束（等级限制、上限等）

**优化模型**：
```
目标：max Σ budget(p)

约束：
1. Σ budget(p) ≤ 可用预算
2. 整体ROI ≥ KPI目标
3. A级产品：budget(p) ≤ 现有消耗 × 50%
4. B级产品：budget(p) ≤ 现有消耗 × 20%
5. C级产品：budget(p) ≤ 现有消耗 × 10%
6. budget(p) ≥ 0
```

**求解方法**：
- 优先使用PuLP库的CBC求解器（免费开源）
- 失败时降级到启发式分配（按等级权重分配）

**关键类**：`PortfolioOptimizer`

**关键方法**：
```python
optimizer = PortfolioOptimizer()

# 优化预算分配
allocation = optimizer.optimize_allocation(
    product_diagnoses, 
    available_budget=50000,
    holiday_info=factors_df,
    days_remaining=7,
    kpi_target=1.05
)
# 返回：{
#   'allocations': {product_id: {'allocated_budget': ..., 'daily_budget': ...}},
#   'total_allocated': 48500,
#   'status': 'Optimal',
#   'solver_status': 'SUCCESS'
# }

# 按时间窗口分配
time_allocation = optimizer.allocate_by_time_window(allocation, holiday_info, days_remaining)
# 返回：{product_id: {date: daily_budget, ...}}
```

---

### 模块7：报告输出 (report_generator.py)

**职责**：
- 生成每日综合报告
- 生成产品诊断表
- 生成预算建议
- 生成月度总结

**输出报告类型**：

1. **月度概览**
   - 累计消耗 vs 预测消耗
   - 累计回收 vs 预测回收
   - 当前ROI vs 预测ROI vs 目标ROI

2. **产品分级诊断**
   - A/B/C级分布
   - 每个产品的诊断表（等级、ROI、缺口、建议）

3. **预算分配建议**
   - 优化方法（线性规划 vs 启发式）
   - 各产品的分配预算和日均预算

4. **风险预警**
   - 产品能力下滑（D1下降>10%）
   - 流量异常波动（偏离平均±20%）

5. **节假日窗口分析**
   - 识别高价值投放窗口
   - Scale因子排序

6. **ROI预测详情**
   - 已发生的实际数据
   - 剩余天数的预测
   - 月度总体预测

**关键类**：`ReportGenerator`

**关键方法**：
```python
reporter = ReportGenerator()

# 生成每日报告
report = reporter.generate_daily_report(
    forecast_date='2026-04-21',
    monthly_stats=stats_dict,
    product_diagnoses=diagnoses_dict,
    roi_forecast=forecast_dict,
    budget_allocation=allocation_dict,
    holiday_info=factors_df,
    remaining_days=7,
    kpi_target=1.05
)
# 返回：{
#   'data': DataFrame,
#   'monthly_overview': str,
#   'diagnosis': DataFrame,
#   'allocation': DataFrame,
#   'roi_forecast': dict
# }

# 生成月度总结
monthly_summary = reporter.generate_monthly_summary(data, year=2026, month=4, kpi_target=1.05)
```

---

### 模块8：ML D1预测 (ml_predictor.py)

**职责**：
- 基于LightGBM机器学习预测各产品（应用ID）在不同投放维度的D1回收率
- 特征工程：日期特征、节假日特征、产品趋势、投放维度
- 产品中位数兜底预测（LightGBM不可用时）

**核心概念**：
- **训练粒度** = (日期, 应用ID, 推广流量, 流量场景, 创意规格, 转化类型)
- **产品标识** = 应用ID
- **投放维度** = 4个独立变量，导致同一产品有多种情况
  - 时间特征：周一到周日、是否周末、月份、月内第几天
  - 节假日特征：scale_factor、holiday_type（6类）
  - 投放特征：推广流量、流量场景、创意规格、转化类型
  - 产品趋势：近3天D1、近7天D1、与产品中位数的偏离

**依赖**：
- `lightgbm`库（可选，不安装时使用产品中位数兜底）
- `scikit-learn`库

**关键类**：`MLPredictor`

**关键方法**：
```python
from ml_predictor import MLPredictor

# 初始化并训练
ml = MLPredictor(data, holiday_identifier=holiday_obj)
trained = ml.train()  # 返回 True/False

# 获取某个产品在特定时间段的平均D1预测
product_ids = ['app_001', 'app_002', ...]
d1_rates = ml.get_average_d1_for_period(
    product_ids=product_ids,
    start_date='2026-04-20',
    end_date='2026-04-30',
    historical_data=data,
    holiday_info=factors_df
)
# 返回：{'app_001': 0.72, 'app_002': 0.65, ...}

# 获取特定日期的D1预测（包含所有投放维度）
d1_predictions = ml.predict(data_on_date='2026-04-21')  # 返回 DataFrame，列含 target_d1
```

**特征列表**：
```
日期维度：day_of_week, is_weekend, day_of_month, month
节假日维度：holiday_scale, holiday_type_*（6种one-hot编码）
投放维度：推广流量, 流量场景, 创意规格, 转化类型
产品维度：product_median_d1, product_std_d1, product_days_active
趋势维度：recent_3d_d1, recent_7d_d1, d1_trend_3d, d1_trend_7d, recent_3d_spend, recent_7d_spend
月度进度：days_elapsed_in_month
```

**在主系统中的角色**：
- ROIPlanner 使用 MLPredictor 的D1预测优化ROI预测精度
- 当LightGBM不可用时自动降级到产品中位数
- MPC校准框架可用ML预测错误来优化后续预测

---

## 四、完整工作流程

### 每日执行流程（文档第五章第6.2节）

```
上午8:00 → 数据更新
  │
  ├─[1] DataLoader 加载前日数据
  │
  ├─[2] LTVPredictor 更新LTV曲线
  │
  ├─[3] HolidayIdentifier 检查节假日
  │
  ├─[4] ProductDiagnostics 诊断所有产品
  │
  ├─[5] ROIPlanner 预测月度ROI和可用预算
  │
  ├─[6] PortfolioOptimizer 求解最优预算分配
  │
  └─[7] ReportGenerator 生成日报和建议

中午12:00 → 运营人员查看报告
  │
  ├─ 阅读月度概览
  ├─ 查看产品诊断
  ├─ 获取预算建议
  └─ 注意风险预警

下午 → 手动调整
  │
  └─ 在广告平台后台按建议调整各产品预算
```

### 代码调用示例

```python
from budget_system_main import IntelligentBudgetSystem

# 1. 初始化系统
system = IntelligentBudgetSystem('daily_20260421_120112.csv', kpi_target=1.05)

# 2. 生成今日报告
report = system.daily_forecast(forecast_date='2026-04-21')

# 3. 生成月度总结
march_summary = system.month_summary(year=2026, month=3)
april_summary = system.month_summary(year=2026, month=4)

# 4. 访问报告数据
print(f"月度预测ROI: {report['roi_forecast']['predicted_roi']:.2%}")
print(f"预算分配: {report['allocation']}")
```

---

## 五、关键输出文件

### 生成的CSV报告

1. **daily_report_YYYYMMDD_HHMMSS.csv**
   - 每日报告汇总
   - 包含：月度统计、产品诊断、预算建议

2. **march_summary_YYYYMMDD_HHMMSS.csv**
   - 3月份日度统计
   - 包含：日期、消耗、回收、ROI、产品数

3. **april_summary_YYYYMMDD_HHMMSS.csv**
   - 4月份日度统计（到目前为止）
   - 同上格式

### 控制台输出（完整诊断）

运行系统时会输出：

```
================================================================================
📊 2026-04-21 智能预算决策日报
================================================================================

📈 【第一部分】月度概览
...

🔍 【第二部分】产品分级诊断
...

💰 【第三部分】预算分配建议
...

⚠️  【第四部分】风险预警
...

🎯 【第五部分】节假日窗口分析
...

📊 【第六部分】ROI预测详情
...
```

---

## 六、系统特色和优势

### ✅ 完全遵循文档第五章要求

| 文档要求 | 系统实现 |
|---------|--------|
| 节假日自动识别 | ✓ HolidayIdentifier模块 |
| 滚动预测 | ✓ 每日重新计算，MPC思想 |
| 多产品联合求解 | ✓ PortfolioOptimizer，线性规划 |
| 分版位建模 | ~ 为简化，目前模型粒度为产品级 |
| KPI动态响应 | ✓ 参数化设计，可任意修改 |
| 风控内置 | ✓ 产品诊断中集成 |
| 跨期评估 | ✓ ROIPlanner中计算 |
| 数据接口对接 | ✓ DataLoader支持CSV，易扩展 |

### 🚀 超过文档要求的功能

1. **产品能力下滑检测** - 自动识别最近N天效率恶化
2. **流量异常波动告警** - 基于标准差的异常检测
3. **跨期飞轮量化** - 计算月末蓄力对下月的贡献
4. **启发式降级** - 求解器失败时自动切换到启发式算法
5. **详细诊断报告** - 每个产品的完整诊断和建议

### 📊 数据驱动的决策

- 所有约束和倍率均从历史数据计算
- LTV曲线用加权平均与标准值平滑，防止过度拟合
- ROI倍率根据回收窗口长度动态衰减

---

## 七、后续改进方向

### Phase 1（当前）：基础系统 ✅
- [x] 7个核心模块实现
- [x] 每日预报告生成
- [x] 产品诊断和风险预警
- [x] 预算优化求解

### Phase 2（建议）：增强能力
- [ ] 对接广告平台API，自动拉取数据
- [ ] 分版位（商店/联盟/智能投放）建模
- [ ] 自动执行优化建议（对接竞价系统）
- [ ] Web Dashboard展示报告
- [ ] 历史数据验证LTV曲线假设

### Phase 3（高级）：智能优化
- [ ] 多产品人群竞争建模
- [ ] A/B测试统计显著性检验
- [ ] 贝叶斯参数更新（提高预测精度）
- [ ] 强化学习长期优化

---

## 八、使用说明

### 环境要求

```
Python >= 3.8
pandas >= 1.0
numpy >= 1.15
scipy >= 1.5
pulp >= 2.4
```

### 安装依赖

```bash
pip install pandas numpy scipy pulp matplotlib
```

### 快速开始

```bash
# 运行系统
python budget_system_main.py

# 输出文件保存在当前目录：
# - daily_report_*.csv
# - march_summary_*.csv
# - april_summary_*.csv
```

### 自定义KPI

```python
# 修改KPI目标（默认105%）
system = IntelligentBudgetSystem('data.csv', kpi_target=1.08)  # 改为108%
```

---

## 九、文件清单

```
d:\博士项目\广告投放项目\智能决策模块\实验\New\
│
├─ budget_system_main.py          [主程序]
├─ data_loader.py                 [模块1：数据加载]
├─ ltv_predictor.py               [模块2：LTV预测]
├─ holiday_identifier.py          [模块3：节假日识别]
├─ product_diagnosis.py           [模块4：产品诊断]
├─ roi_planner.py                 [模块5：ROI规划]
├─ portfolio_optimizer.py         [模块6：投资组合优化]
├─ report_generator.py            [模块7：报告生成]
│
├─ daily_20260421_120112.csv      [输入数据]
├─ daily_report_*.csv             [每日报告]
├─ march_summary_*.csv            [3月总结]
├─ april_summary_*.csv            [4月总结]
│
├─ README.md                       [本文档]
└─ 智能预算决策系统_业务场景与问题定义(1).md  [原需求文档]
```

---

## 十、常见问题

**Q1: 如何修改KPI目标?**

在 `budget_system_main.py` 中修改：
```python
system = IntelligentBudgetSystem(csv_path, kpi_target=1.08)  # 改为108%
```

**Q2: 如何添加新的节假日?**

在 `holiday_identifier.py` 的 `_build_holiday_calendar` 方法中添加：
```python
holidays[datetime(2026, 5, 20)] = {
    'name': '自定义节日', 
    'scale_factor': 1.5, 
    'type': 'custom'
}
```

**Q3: 预算为什么没有完全分配?**

可能原因：
- 所有产品都是C级，分配上限有限
- ROI预测缺口大，需要保留安全边际
- 求解器认为增加消耗会违反ROI约束

**Q4: 如何对接真实数据源?**

修改 `data_loader.py`，在 `load_data` 方法中改为：
```python
# 从API拉取数据
data = fetch_from_api('endpoint_url')
```

---

## 附录：数据字典

### 输入数据（CSV）

| 列名 | 类型 | 说明 |
|-----|-----|------|
| 日期 | date | 投放日期 (YYYY-MM-DD) |
| 应用ID | str | 应用标识 |
| 广告主ID | str | 产品标识（主键之一） |
| 消耗金额 | float | 当日广告花费（元） |
| 首日广告收入 | float | D1回收金额 |
| 3日累计变现金额 | float | D3累计回收金额 |
| 7日累计变现金额 | float | D7累计回收金额 |
| 30日累计变现金额 | float | D30累计回收金额 |
| ... | | 其他字段自动忽略 |

### 关键输出指标

| 指标 | 定义 | 例值 |
|-----|-----|-----|
| D1 | 首日回收率 = 首日回收 / 消耗 | 0.69 (69%) |
| M30 | 30日倍率 = D30回收 / D1回收 | 1.44 |
| ROI | 月度ROI = 总回收 / 总消耗 | 1.06 (106%) |
| 缺口 | KPI - 当前ROI | -0.01 (-1pp，盈余) |
| 分级 | A(高效) / B(中等) / C(低效) | A |

---

**系统实现完毕，已生成可用的报告和建议。** 🎉
