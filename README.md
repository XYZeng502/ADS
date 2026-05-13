# 智能预算决策系统（框架版）

这是一个面向“多客户独立预算池”的业务系统骨架，重点实现**完整流程和模块边界**，便于后续替换预测/优化算法。

## 1. 已落地能力

- 每日滚动重算主流程（数据输入 -> 识别日类型 -> 求解 -> 预测 -> 预警 -> 输出）
- KPI 动态重算接口
- 风险预警接口（产品能力下滑、流量异常）
- 产品分级诊断（A/B/C）
- 分产品分版位预算建议输出
- 对齐第5章的执行日报字段框架（累计消耗预测、D1目标、分版位建议、KPI响应等）
- 输出月末ROI偏差来源分解（spend / D1锚点 / 曲线）

## 2. 项目结构

- `app/main.py`: FastAPI 接口层
- `app/schemas.py`: 输入/输出数据模型
- `app/core/calendar.py`: 节假日与时间窗口识别
- `app/services/predictor.py`: 回收与月末预测（占位实现）
- `app/services/solver.py`: 预算分配求解器（启发式，可替换）
- `app/services/risk.py`: 风控与告警
- `app/services/orchestrator.py`: 每日执行编排器
- `sample_input.json`: 示例请求数据
- `scripts/run_daily.py`: 本地单次演示脚本

## 3. 快速启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

接口文档：`http://127.0.0.1:8000/docs`
Web 看板：`http://127.0.0.1:8000/web`

## 4. 示例调用

```bash
curl -X POST "http://127.0.0.1:8000/v1/optimize/daily-plan" \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

执行日报接口（第5章框架）：

```bash
curl -X POST "http://127.0.0.1:8000/v1/report/daily-execution" \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

本地脚本演示：

```bash
PYTHONPATH=. python scripts/run_daily.py
```

离线回放测试（基于历史CSV）：

```bash
PYTHONPATH=. python scripts/offline_backtest.py \
  --input daily_20260421_120112.csv \
  --output-dir outputs \
  --kpi 1.05
```

应用层级（前部训练 + 最后一天预测告警与建议）：

```bash
PYTHONPATH=. python scripts/offline_backtest.py \
  --task app_last_day \
  --input daily_20260421_120112.csv \
  --output-dir outputs \
  --kpi 1.05
```

输出文件：

- `outputs/offline_backtest_summary.json`
- `outputs/offline_backtest_detail.csv`
- `outputs/app_level_last_day_prediction.json`
- `outputs/app_level_last_day_suggestions.csv`（每个应用1行汇总）

口径说明（重要）：

- 离线评估中不再使用“买量广告收入”作为实际收入参考（该字段混合新老用户总量）
- 当前分子采用 cohort 可观测口径（D1 回收）用于对齐新增用户贡献

产品每日 ROI 成长曲线（D1/D3/D7/D30/买量ROI）：

```bash
PYTHONPATH=. python scripts/plot_product_roi_curves.py \
  --input daily_20260421_120112.csv \
  --output-dir outputs/roi_curves \
  --top-n 30 \
  --min-days 7
```

说明：

- `--top-n <= 0` 可切换为全量产品出图
- 每个产品输出一张 PNG 曲线图，并生成 `product_roi_curve_summary.csv`

单 Cohort ROI 成长曲线（你说的“每一行一个cohort”场景）：

```bash
PYTHONPATH=. python scripts/plot_single_cohort_curve.py \
  --input daily_20260421_120112.csv \
  --cohort-date 2026-04-21 \
  --app-id 31440303 \
  --advertiser-id 1000331898 \
  --plan-id 229791384 \
  --adgroup-id 356610903 \
  --creative-id 663048761 \
  --output-dir outputs/cohort_curves
```

说明：

- 也可用 `--row-index` 直接按原始CSV行号选中一个cohort
- 默认仅输出成熟样本（`age_days >= max_day`）；如需强制看未成熟样本，追加 `--allow-immature`
- 默认使用 `monotone_cubic` 插值（更平滑有弧度）；如需线性可加 `--interp-method linear`
- 输出包含：曲线图 PNG、日龄曲线 CSV、元信息 JSON

同类竞品 ROI 曲线预测（类别倍率模板迁移）：

```bash
PYTHONPATH=. python scripts/predict_competitor_by_template.py \
  --input daily_20260421_120112.csv \
  --name competitor_demo \
  --d1-roi 0.65 \
  --app-ids 36655893,36238406 \
  --min-spend 100 \
  --output-dir outputs/competitor_forecast
```

说明：

- 仅用成熟样本（age>=30）构建倍率模板
- 默认使用 `monotone_cubic` 插值让日级曲线更平滑；可切换 `--interp-method linear`
- 输出 P50 主曲线 + P25/P75 区间带
- 关键节点预测：D1/D3/D7/D30

应用维度聚合 + 类别特征混合编码分析（one-hot/embedding）：

```bash
PYTHONPATH=. python scripts/app_level_aggregate_and_encode.py \
  --input daily_20260421_120112.csv \
  --output-dir outputs/app_feature_analysis \
  --onehot-max-card 8 \
  --embed-dim 4
```

说明：

- 先按 `应用ID+日期` 聚合，再分析 D1 波动目标（`abs_delta_roi_d1`）
- 低基数类别（唯一值<=阈值）自动 one-hot
- 高基数类别自动做统计向量 SVD embedding

多模型并行预测（T+1 ROI_D1）：

```bash
PYTHONPATH=. python scripts/run_parallel_models_roi_d1.py \
  --input daily_20260421_120112.csv \
  --output-dir outputs/model_parallel_roi_d1 \
  --alpha 0.4 \
  --min-train-days 20
```

模型：

- EWMA
- RandomForestRegressor
- GradientBoostingRegressor (GBDT)

月末 ROI 预测已接入“释放倍率曲线”校准：

- 文件：`app/services/predictor.py`
- 逻辑：根据“距月末可释放天数”映射倍率，替代固定比例外推
- 默认读取：`outputs/competitor_forecast/competitor_demo_curved_competitor_roi_curve.csv`
- 若文件缺失，自动回退到内置默认倍率曲线

## 5.5 业务调整安全护栏（误差不漂移）

为了在业务调整（清洗规则、聚合维度、特征改造、口径变化等）期间，确保 ROI_D1 误差不退化：

1. 基线锁定：`outputs/baseline_roi_d1_metrics.json`
  - 配置：训练实体 `应用ID + 推广流量名称`、清洗组 `应用ID/推广流量/推广流量名称/流量场景/流量场景名称/创意规格/创意规格名称`
  - 默认相对容差：MAE/RMSE/MAPE 各 10%
2. 跑出新指标到任意目录（如 `outputs/run_xxx`）后执行护栏：

```bash
PYTHONPATH=. python scripts/check_metrics_against_baseline.py \
  --baseline outputs/baseline_roi_d1_metrics.json \
  --candidate-dir outputs/run_xxx
```

1. 输出 `outputs/metrics_drift_report.json`，并在终端打印超阈违规清单。
2. CI/护栏强制：加 `--fail-on-violation`，超阈直接非零退出，阻断上线。

适用场景：

- 切换清洗规则
- 切换训练实体维度
- 替换/新增模型
- 业务字段口径调整

## 6. Web 阶段成果看板（强交互）

- 入口：`/web`
- 支持能力：
  - 填写 CSV 路径与 KPI，一键触发离线回放
  - 任务切换：
    - `应用层最后一天预测`（前部训练，仅输出最后一天）
    - `全量回放`
  - 页签化看板：核心指标 / 阶段成果对比 / 明细数据
  - 动态图表：模式分布、ROI对比、偏差主因
  - 客户端多维筛选：应用ID/广告主ID/日期/展示条数
  - 阶段对比卡：启发式、联合求解初版、联合求解调优版（若对应结果文件存在）

模块级详细文档：

- `docs/modules/README.md`
- `docs/modules/orchestrator.md`
- `docs/modules/solver.md`
- `docs/modules/predictor.md`
- `docs/modules/risk.md`
- `docs/modules/rule_engine.md`
- `docs/modules/calendar.md`
- `docs/modules/web_dashboard.md`

## 5. 后续增强建议

1. 将 `solver.py` 替换为 Gurobi/PuLP 的联合优化（产品 x 时间）
2. 将 `predictor.py` 替换为多模型融合（节假日/版位/产品分层）
3. 接入真实节假日服务与电商节配置中心
4. 打通数据中台 API，落地每天 T+1 自动重算任务
5. 增加运营看板（建议、风险、诊断、KPI 变更影响）

