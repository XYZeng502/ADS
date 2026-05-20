# 模块文档

## 文档清单

| 文档 | 说明 |
|------|------|
| `predictor.md` | 多维预测（Spend T+1 / ROI T+1 / 释放曲线 / CQR区间） |
| `solver.md` | 预算分配求解器 |
| `risk.md` | 风控与诊断 |
| `calendar.md` | 节假日识别与 scale 因子 |
| `web_dashboard.md` | Web 看板功能说明 |
| `business_trace.md` | 业务链路解释 |
| `rule_engine.md` | 规则引擎 |
| `orchestrator.md` | 每日编排流程 |

## 系统入口

```bash
# Web 看板（主入口）
PYTHONPATH=. bash scripts/run_web_8000.sh
# 访问 http://localhost:8000/web

# 每日重训练编排
PYTHONPATH=. python scripts/daily_retrain.py
```
