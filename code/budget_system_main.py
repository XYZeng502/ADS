"""
智能预算决策系统 - 主程序
第五章：我们期望的系统是什么样子 - 完整实现

系统架构：
1. 数据加载与预处理模块 (data_loader.py)
2. LTV曲线与回收预测模块 (ltv_predictor.py)
3. 节假日识别与标注模块 (holiday_identifier.py)
4. 产品分级与诊断模块 (product_diagnosis.py)
5. ROI预测与消耗规划模块 (roi_planner.py)
6. 多产品联合优化模块 (portfolio_optimizer.py)
7. 报告输出模块 (report_generator.py)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# 导入各个模块
import os
import sys
# 确保 code 目录在 Python 路径中（支持从项目根目录或 code/ 目录运行）
_code_dir = os.path.dirname(os.path.abspath(__file__))
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

from data_loader import DataLoader
from ltv_predictor import LTVPredictor
from holiday_identifier import HolidayIdentifier
from product_diagnosis import ProductDiagnostics
from roi_planner import ROIPlanner
from portfolio_optimizer import PortfolioOptimizer
from report_generator import ReportGenerator
from ml_predictor import MLPredictor, PLACEMENT_DIMS
from model_validator import ModelValidator


class IntelligentBudgetSystem:
    """智能预算决策系统的核心类"""
    
    def __init__(self, csv_path, kpi_target=1.05):
        """
        初始化系统
        
        参数：
        - csv_path: 数据文件路径
        - kpi_target: KPI目标ROI（默认105%）
        """
        self.kpi_target = kpi_target
        self.current_date = None
        self.month_start = None
        self.month_end = None
        
        # 初始化各个模块
        print("=" * 80)
        print("智能预算决策系统启动")
        print("=" * 80)
        
        print("\n[1/7] 加载数据...")
        self.data_loader = DataLoader(csv_path)
        self.raw_data = self.data_loader.load_data()
        print(f"✓ 数据加载完成，共{len(self.raw_data)}条记录")
        
        print("\n[2/7] 初始化LTV预测模块...")
        self.ltv_predictor = LTVPredictor(self.raw_data)
        self.ltv_curves = self.ltv_predictor.calculate_ltv_curves()
        print(f"✓ LTV曲线计算完成，共{len(self.ltv_curves)}种产品的曲线")
        
        print("\n[3/7] 初始化节假日识别模块...")
        self.holiday_identifier = HolidayIdentifier()
        print("✓ 节假日识别模块初始化完成")
        
        print("\n[4/7] 初始化产品诊断模块...")
        self.product_diagnosis = ProductDiagnostics(
            self.raw_data, 
            self.ltv_curves, 
            kpi_target=kpi_target
        )
        print("✓ 产品诊断模块初始化完成")
        
        print("\n[5/7] 初始化ROI规划模块...")
        self.roi_planner = ROIPlanner(self.raw_data, self.ltv_curves, kpi_target)
        print("✓ ROI规划模块初始化完成")
        
        print("\n[6/7] 初始化多产品联合优化模块...")
        self.portfolio_optimizer = PortfolioOptimizer()
        print("✓ 联合优化模块初始化完成")
        
        print("\n[7/7] 初始化报告输出模块...")
        self.report_generator = ReportGenerator()
        print("✓ 报告输出模块初始化完成")

        # MPC滚动优化状态
        self.mpc_state = self._mpc_load_state()
        print(f"  MPC状态: 校准因子={self.mpc_state.get('calibration_factor_d1', 1.0):.3f}, "
              f"历史快照={len(self.mpc_state.get('daily_snapshots', []))}条")

        # LightGBM D1预测器
        print("\n[8/8] 初始化ML预测模块...")
        self.ml_predictor = MLPredictor(self.raw_data, self.holiday_identifier)
        self.ml_trained = self.ml_predictor.train()
        if self.ml_trained:
            print("✓ LightGBM D1预测器训练完成")
        else:
            print("✓ 使用产品中位数兜底预测（LightGBM不可用或数据不足）")

        print("\n系统启动完成！\n")
    
    # ============================================================
    #  MPC 滚动优化框架
    #  每天比较"昨日预测 vs 今日实际"，用误差修正 D1 预测
    # ============================================================

    def _mpc_calibrate(self, forecast_date, roi_forecast):
        """MPC校准：用历史预测误差修正当前D1预测"""
        if not self.mpc_state.get('daily_snapshots'):
            return roi_forecast

        yesterday = pd.to_datetime(forecast_date) - timedelta(days=1)
        yesterday_str = yesterday.strftime('%Y-%m-%d')

        yesterday_snap = None
        for snap in self.mpc_state['daily_snapshots']:
            if snap['date'] == yesterday_str:
                yesterday_snap = snap
                break

        if yesterday_snap is None:
            return roi_forecast

        # 从实际数据获取昨天的真实D1
        yesterday_data = self.raw_data[self.raw_data['日期'] == yesterday]
        if len(yesterday_data) == 0:
            return roi_forecast

        actual_spend = yesterday_data['消耗金额'].sum()
        actual_d1 = yesterday_data['首日广告收入'].sum() / max(actual_spend, 1)
        yesterday_snap['actual_d1_rate'] = actual_d1

        predicted_d1 = yesterday_snap.get('predicted_d1_rate', 0)
        if predicted_d1 > 0 and actual_d1 > 0:
            error_ratio = actual_d1 / predicted_d1
            alpha = 0.3
            old_factor = self.mpc_state.get('calibration_factor_d1', 1.0)
            new_factor = alpha * error_ratio + (1 - alpha) * old_factor
            self.mpc_state['calibration_factor_d1'] = float(np.clip(new_factor, 0.7, 1.3))

            self.mpc_state.setdefault('error_history', []).append({
                'date': yesterday_str,
                'predicted_d1': round(predicted_d1, 4),
                'actual_d1': round(actual_d1, 4),
                'error_ratio': round(error_ratio, 4),
                'calibration_factor': round(self.mpc_state['calibration_factor_d1'], 4)
            })

            cal = self.mpc_state['calibration_factor_d1']
            print(f"  MPC校准: 昨日预测D1={predicted_d1:.3f}, 实际={actual_d1:.3f}, "
                  f"误差比={error_ratio:.3f}, 校准因子更新为{cal:.3f}")

        return roi_forecast

    def _mpc_record_snapshot(self, forecast_date, roi_forecast):
        """记录每日快照，供次日MPC校准使用"""
        forecast_dt = pd.to_datetime(forecast_date)
        date_str = forecast_dt.strftime('%Y-%m-%d')

        day_data = self.raw_data[self.raw_data['日期'] == forecast_dt]
        if len(day_data) > 0:
            actual_spend = day_data['消耗金额'].sum()
            actual_d1 = day_data['首日广告收入'].sum() / max(actual_spend, 1)
        else:
            actual_spend = 0
            actual_d1 = 0

        snapshot = {
            'date': date_str,
            'actual_spend': float(actual_spend),
            'actual_d1_rate': round(actual_d1, 4),
            'predicted_roi': round(roi_forecast['predicted_roi'], 4),
            'predicted_total_spend': round(roi_forecast['predicted_total_spend'], 2),
            'predicted_d1_rate': round(roi_forecast.get('d1_roi_avg', 0), 4),
            'days_remaining': roi_forecast['days_remaining'],
            'calibration_factor': round(self.mpc_state.get('calibration_factor_d1', 1.0), 4)
        }

        # 同一天只保留最新快照
        self.mpc_state['daily_snapshots'] = [
            s for s in self.mpc_state.get('daily_snapshots', [])
            if s['date'] != snapshot['date']
        ]
        self.mpc_state['daily_snapshots'].append(snapshot)

        self._mpc_save_state()

    def _mpc_save_state(self):
        """持久化MPC状态到磁盘"""
        state_path = os.path.join(_code_dir, '.mpc_state.json')
        state_copy = {
            'calibration_factor_d1': float(self.mpc_state.get('calibration_factor_d1', 1.0)),
            'daily_snapshots': self.mpc_state.get('daily_snapshots', [])[-30:],
            'error_history': self.mpc_state.get('error_history', [])[-30:]
        }
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state_copy, f, ensure_ascii=False, indent=2, default=str)

    def _mpc_load_state(self):
        """从磁盘加载MPC状态"""
        state_path = os.path.join(_code_dir, '.mpc_state.json')
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {'calibration_factor_d1': 1.0, 'daily_snapshots': [], 'error_history': []}

    def daily_forecast(self, forecast_date=None):
        """
        每日预测和优化
        
        参数：
        - forecast_date: 预测日期（YYYY-MM-DD格式），默认为最新日期
        
        返回：每日报告和建议
        """
        # 确定预测日期
        if forecast_date is None:
            forecast_date = self.raw_data['日期'].max()
        else:
            forecast_date = pd.to_datetime(forecast_date).strftime('%Y-%m-%d')
        
        print("\n" + "=" * 80)
        print(f"生成 {forecast_date} 的每日预算决策报告")
        print("=" * 80)
        
        # 提取该日期及之前的数据
        cutoff_date = pd.to_datetime(forecast_date)
        historical_data = self.raw_data[self.raw_data['日期'] <= cutoff_date].copy()
        
        # 1. 获取月度基本信息
        month_start = pd.to_datetime(f"{cutoff_date.year}-{cutoff_date.month:02d}-01")
        if cutoff_date.month == 12:
            month_end = pd.to_datetime(f"{cutoff_date.year + 1}-01-01") - timedelta(days=1)
        else:
            month_end = pd.to_datetime(f"{cutoff_date.year}-{cutoff_date.month + 1:02d}-01") - timedelta(days=1)
        
        # 2. 计算月度累计数据
        month_data = historical_data[
            (historical_data['日期'] >= month_start) & 
            (historical_data['日期'] <= cutoff_date)
        ].copy()
        
        monthly_stats = self._calculate_monthly_stats(month_data)
        
        # 3. 获取节假日信息
        remaining_days = (month_end - cutoff_date).days + 1
        holiday_info = self.holiday_identifier.get_holiday_scale_factors(
            cutoff_date, 
            month_end
        )
        
        # 4. 产品诊断
        product_diagnoses = self.product_diagnosis.diagnose_all_products(
            historical_data,
            cutoff_date,
            self.kpi_target
        )
        
        # 5. ROI预测和预算规划（ML增强）
        # 产品ID = 应用ID，获取LightGBM预测的各产品D1
        product_ids = list(historical_data['应用ID'].unique())
        ml_d1_rates = self.ml_predictor.get_average_d1_for_period(
            product_ids, cutoff_date, month_end, historical_data, holiday_info
        )
        roi_forecast = self.roi_planner.forecast_monthly_roi(
            historical_data,
            cutoff_date,
            month_start,
            month_end,
            self.kpi_target,
            product_d1_rates=ml_d1_rates
        )

        # 5.5 MPC滚动校准：用历史预测误差修正D1预测
        roi_forecast = self._mpc_calibrate(cutoff_date, roi_forecast)

        # 6. 多产品联合优化
        unit_forecasts = self._build_unit_forecasts(
            historical_data=historical_data,
            product_ids=product_ids,
            cutoff_date=cutoff_date,
            month_end=month_end,
            holiday_info=holiday_info,
            product_diagnoses=product_diagnoses
        )

        monthly_state = {
            'actual_spend': roi_forecast['actual_spend'],
            'actual_revenue': roi_forecast['actual_revenue'],
            'actual_roi': roi_forecast['actual_roi'],
            'historical_tail_revenue': roi_forecast.get('_debug_historical_tail', 0),
            'avg_daily_spend': roi_forecast.get('avg_daily_spend', 0),
            'days_elapsed': roi_forecast.get('days_elapsed', 1),
            'days_remaining': roi_forecast.get('days_remaining', remaining_days)
        }

        if unit_forecasts is not None and len(unit_forecasts) > 0:
            # 最优算法：不再用启发式提前截断预算，让联合优化器在ROI硬约束下求最大可投规模。
            remaining_budget = float(unit_forecasts['cap'].sum())
        else:
            remaining_budget = self._calculate_remaining_budget(
                monthly_stats,
                roi_forecast,
                self.kpi_target
            )
        
        budget_allocation = self.portfolio_optimizer.optimize_allocation(
            product_diagnoses,
            remaining_budget,
            holiday_info,
            remaining_days,
            self.kpi_target,
            avg_roi_multiplier=roi_forecast.get('avg_roi_multiplier', 1.0),
            historical_tail_revenue=roi_forecast.get('_debug_historical_tail', 0),
            unit_forecasts=unit_forecasts,
            monthly_state=monthly_state
        )
        
        # 7. 生成报告
        # 注：产品ID统一为应用ID（同一应用在不同渠道/场景/创意会有多条数据）
        # 不再需要广告主ID → 应用ID 映射
        id_mapping = {}  # 留作扩展使用
        report = self.report_generator.generate_daily_report(
            forecast_date=forecast_date,
            monthly_stats=monthly_stats,
            product_diagnoses=product_diagnoses,
            roi_forecast=roi_forecast,
            budget_allocation=budget_allocation,
            holiday_info=holiday_info,
            remaining_days=remaining_days,
            kpi_target=self.kpi_target,
            mpc_calibration_factor=self.mpc_state.get('calibration_factor_d1', 1.0),
            ml_d1_rates=ml_d1_rates,
            id_mapping=id_mapping
        )

        # 8. MPC快照：记录今日状态供明日校准
        self._mpc_record_snapshot(forecast_date, roi_forecast)

        return report

    def _build_unit_forecasts(self, historical_data, product_ids, cutoff_date,
                              month_end, holiday_info, product_diagnoses):
        """构建联合优化器需要的投放单元×日期预测表。"""
        unit_forecasts = self.ml_predictor.predict_remaining_days_quantiles(
            product_ids=product_ids,
            start_date=cutoff_date,
            end_date=month_end,
            historical_data=historical_data,
            holiday_info_df=holiday_info
        )

        if unit_forecasts is None or unit_forecasts.empty:
            return pd.DataFrame()

        calibration = float(self.mpc_state.get('calibration_factor_d1', 1.0))
        for col in ['d1_p20', 'd1_p50', 'd1_p80']:
            unit_forecasts[col] = (unit_forecasts[col] * calibration).clip(lower=0.01, upper=2.0)

        risk_map = {
            pid: diag.get('risk_control', {}).get('risk_multiplier', 1.0)
            for pid, diag in product_diagnoses.items()
        }

        multipliers = []
        risk_multipliers = []
        for _, row in unit_forecasts.iterrows():
            product_id = row['product_id']
            days_until_month_end = max(1, (pd.to_datetime(month_end) - pd.to_datetime(row['date'])).days + 1)
            unit_key = tuple([product_id] + [row.get(dim, 0) for dim in PLACEMENT_DIMS])
            multiplier = self.ltv_predictor.get_multiplier(
                product_id=product_id,
                days_lived=days_until_month_end,
                unit_key=unit_key
            )
            multipliers.append(multiplier)
            risk_multipliers.append(risk_map.get(product_id, 1.0))

        unit_forecasts['ltv_multiplier'] = multipliers
        unit_forecasts['risk_multiplier'] = risk_multipliers
        unit_forecasts['expected_roi_p20'] = unit_forecasts['d1_p20'] * unit_forecasts['ltv_multiplier']
        unit_forecasts['expected_roi_p50'] = unit_forecasts['d1_p50'] * unit_forecasts['ltv_multiplier']
        unit_forecasts['expected_roi_p80'] = unit_forecasts['d1_p80'] * unit_forecasts['ltv_multiplier']
        unit_forecasts['p20_margin_pp'] = (unit_forecasts['expected_roi_p20'] - self.kpi_target) * 100
        unit_forecasts['p50_margin_pp'] = (unit_forecasts['expected_roi_p50'] - self.kpi_target) * 100
        unit_forecasts['candidate_rank'] = unit_forecasts['expected_roi_p20'].rank(method='first', ascending=False).astype(int)

        return unit_forecasts
    
    def _calculate_monthly_stats(self, month_data):
        """计算月度统计信息"""
        stats = {
            'total_spend': month_data['消耗金额'].sum(),
            'total_revenue': month_data['30日累计变现金额'].sum(),
            'daily_revenue': month_data['首日广告收入'].sum(),
            'roi': None,
            'products': month_data.groupby('应用ID').agg({
                '消耗金额': 'sum',
                '30日累计变现金额': 'sum',
                '首日广告收入': 'sum'
            }).reset_index()
        }
        
        # 计算ROI
        if stats['total_spend'] > 0:
            stats['roi'] = stats['total_revenue'] / stats['total_spend']
        else:
            stats['roi'] = 0
        
        return stats
    
    def _calculate_remaining_budget(self, monthly_stats, roi_forecast, kpi_target):
        """计算剩余可用预算 - 升级版：允许ROI改善预算"""
        # 策略：根据ROI缺口大小动态调整预算分配
        roi_gap = roi_forecast['roi_gap']
        current_roi = roi_forecast['actual_roi']
        base_spend = monthly_stats['total_spend']
        
        if roi_gap >= 0:
            # ROI充足，可释放预算用于规模最大化
            cushion = roi_gap
            remaining_budget = base_spend * (cushion / kpi_target) * 0.5
        else:
            # ROI不足（缺口为负），但允许投入来改善
            # 策略：允许投入可以使ROI提升至99%的金额，更保守
            abs_gap = abs(roi_gap)  # 缺口绝对值，单位是ROI百分比
            
            if abs_gap <= 0.05:  # 缺口较小（5%以内），保守分配
                # 计算使ROI改善到99%所需的额外支出
                target_roi = 0.99
                if current_roi > 0 and current_roi < target_roi:
                    # 粗估：额外收入需求 = (target_roi * total_spend) - current_revenue
                    # 假设新增支出的ROI为current_roi（保守）
                    future_revenue = roi_forecast['predicted_total_revenue']
                    future_spend = roi_forecast['predicted_total_spend']
                    future_roi = future_revenue / future_spend if future_spend > 0 else 0
                    
                    # 允许增加的消耗：使最终ROI达到99%
                    needed_revenue = target_roi * future_spend
                    deficit = max(needed_revenue - future_revenue, 0)
                    
                    if future_roi > 0:
                        additional_spend = deficit / future_roi
                        remaining_budget = additional_spend * 0.3  # 30%保守系数
                    else:
                        remaining_budget = 0
                else:
                    remaining_budget = 0
            else:  # 缺口较大（>5%），无法通过调增预算改善
                remaining_budget = 0
        
        return max(remaining_budget, 0)
    
    def month_summary(self, year, month):
        """
        月度总结
        """
        print("\n" + "=" * 80)
        print(f"月度总结: {year}年{month}月")
        print("=" * 80)
        
        # 提取该月数据
        month_data = self.raw_data[
            (self.raw_data['日期'].dt.year == year) & 
            (self.raw_data['日期'].dt.month == month)
        ].copy()
        
        # 计算月度统计
        daily_stats = []
        for date in sorted(month_data['日期'].unique()):
            day_data = month_data[month_data['日期'] == date]
            daily_stats.append({
                '日期': date,
                '消耗': day_data['消耗金额'].sum(),
                '回收': day_data['30日累计变现金额'].sum(),
                '产品数': day_data['应用ID'].nunique()
            })
        
        summary_df = pd.DataFrame(daily_stats)
        
        print("\n日均统计：")
        print(summary_df.to_string(index=False))
        
        print(f"\n月度总计：")
        print(f"  总消耗: ¥{summary_df['消耗'].sum():,.2f}")
        print(f"  总回收: ¥{summary_df['回收'].sum():,.2f}")
        total_roi = summary_df['回收'].sum() / summary_df['消耗'].sum() if summary_df['消耗'].sum() > 0 else 0
        print(f"  月度ROI: {total_roi:.2%}")
        print(f"  日均消耗: ¥{summary_df['消耗'].mean():,.2f}")
        
        return summary_df


def main():
    """主程序入口"""
    # 设置UTF-8输出编码
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # 数据文件路径
    csv_path = 'daily_20260421_120112.csv'
    
    # 创建系统实例
    system = IntelligentBudgetSystem(csv_path, kpi_target=1.05)
    
    # 生成最新日期的预算建议
    print("\n📊 生成最新日期的每日预算建议...")
    latest_report = system.daily_forecast()
    
    # 生成3月和4月的月度总结
    print("\n📈 生成月度总结...")
    march_summary = system.month_summary(2026, 3)
    april_summary = system.month_summary(2026, 4)
    
    # 保存报告
    print("\n💾 保存报告文件...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(_code_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(latest_report, dict) and latest_report.get('data') is not None:
        daily_path = os.path.join(output_dir, f"daily_report_{timestamp}.csv")
        latest_report['data'].to_csv(daily_path, index=False)
        print(f"✓ 日报告已保存: {daily_path}")

        unit_df = latest_report.get('unit_allocation')
        if isinstance(unit_df, pd.DataFrame) and not unit_df.empty:
            unit_path = os.path.join(output_dir, f"daily_report_units_{timestamp}.csv")
            unit_df.to_csv(unit_path, index=False)
            print(f"✓ 投放单元级建议已保存: {unit_path}")

        summary_df = latest_report.get('optimization_summary')
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            summary_path = os.path.join(output_dir, f"optimization_summary_{timestamp}.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"✓ 优化效果摘要已保存: {summary_path}")

    march_path = os.path.join(output_dir, f"march_summary_{timestamp}.csv")
    april_path = os.path.join(output_dir, f"april_summary_{timestamp}.csv")
    march_summary.to_csv(march_path, index=False)
    april_summary.to_csv(april_path, index=False)
    print(f"✓ 月度报告已保存至 output/")

    print("\n🧪 划分数据集并生成算法验证报告...")
    validator = ModelValidator(system.raw_data, kpi_target=system.kpi_target, validation_days=7,
                               holiday_identifier=system.holiday_identifier)
    validation_result = validator.run(output_dir=output_dir, timestamp=timestamp)
    print(f"✓ 验证报告已保存: {validation_result['report_path']}")
    print(f"✓ 验证明细已保存: {validation_result['prediction_detail_path']}")
    print(f"✓ 排序对比明细已保存: {validation_result['allocation_detail_path']}")
    
    print("\n✅ 系统运行完成！")


if __name__ == "__main__":
    main()
