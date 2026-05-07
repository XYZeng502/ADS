"""
模块7：报告输出模块
生成每日建议、产品诊断、月度总结等多种报告
"""

import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO


class ReportGenerator:
    """报告生成类"""
    
    def __init__(self):
        """初始化报告生成器"""
        self.report = None
    
    def generate_daily_report(self, forecast_date, monthly_stats, product_diagnoses,
                             roi_forecast, budget_allocation, holiday_info,
                             remaining_days, kpi_target, mpc_calibration_factor=1.0,
                             ml_d1_rates=None, id_mapping=None):
        """
        生成每日综合报告

        返回：包含多个表单的报告字典和DataFrame
        """
        print("  - 生成报告...")
        
        report_data = []
        
        # 报告头信息
        print("\n" + "="*80)
        print(f"📊 {forecast_date} 智能预算决策日报")
        print("="*80)
        
        # 1. 月度概览
        print(f"\n📈 【第一部分】月度概览")
        print("-" * 80)
        
        monthly_overview = self._generate_monthly_overview(
            monthly_stats, roi_forecast, kpi_target, forecast_date
        )
        print(monthly_overview)
        
        # 2. 产品分级诊断
        print(f"\n🔍 【第二部分】产品分级诊断")
        print("-" * 80)
        
        diagnosis_report, diagnosis_df = self._generate_diagnosis_report(
            product_diagnoses, kpi_target
        )
        print(diagnosis_report)
        
        # 3. 预算分配建议
        print(f"\n💰 【第三部分】预算分配建议")
        print("-" * 80)
        
        allocation_report, allocation_df = self._generate_allocation_report(
            budget_allocation, product_diagnoses, remaining_days
        )
        print(allocation_report)
        
        # 4. 风险预警
        print(f"\n⚠️  【第四部分】风险预警")
        print("-" * 80)
        
        risk_report = self._generate_risk_report(product_diagnoses)
        print(risk_report)
        
        # 5. 节假日窗口分析
        print(f"\n🎯 【第五部分】节假日窗口分析")
        print("-" * 80)
        
        holiday_report = self._generate_holiday_report(holiday_info, remaining_days)
        print(holiday_report)
        
        # 6. ROI预测详情
        print(f"\n📊 【第六部分】ROI预测详情")
        print("-" * 80)
        
        roi_report = self._generate_roi_report(roi_forecast, kpi_target)
        print(roi_report)
        
        print("\n" + "="*80)
        print("✅ 报告生成完成\n")
        
        # 整合所有报告到DataFrame
        combined_report_df = self._combine_reports_to_dataframe(
            monthly_stats, product_diagnoses, roi_forecast,
            allocation_df, forecast_date, mpc_calibration_factor,
            ml_d1_rates, id_mapping, budget_allocation
        )
        unit_allocation_df = self._build_unit_allocation_dataframe(
            budget_allocation, forecast_date, kpi_target
        )
        optimization_summary_df = self._build_optimization_summary_dataframe(
            budget_allocation, roi_forecast, forecast_date, kpi_target
        )
        
        return {
            'data': combined_report_df,
            'unit_allocation': unit_allocation_df,
            'optimization_summary': optimization_summary_df,
            'monthly_overview': monthly_overview,
            'diagnosis': diagnosis_df,
            'allocation': allocation_df,
            'roi_forecast': roi_forecast
        }
    
    def _generate_monthly_overview(self, monthly_stats, roi_forecast, kpi_target, date):
        """生成月度概览"""
        output = StringIO()
        
        output.write(f"\n日期: {date}\n")
        output.write(f"{'指标':<20} {'当前':<20} {'预测':<20} {'目标':<20}\n")
        output.write("-" * 80 + "\n")
        
        output.write(f"{'累计消耗':<20} ¥{monthly_stats['total_spend']:>18,.0f} ¥{roi_forecast['predicted_total_spend']:>18,.0f}\n")
        output.write(f"{'累计回收':<20} ¥{monthly_stats['total_revenue']:>18,.0f} ¥{roi_forecast['predicted_total_revenue']:>18,.0f}\n")
        
        current_roi = monthly_stats['roi']
        output.write(f"{'当前ROI':<20} {current_roi:>19.2%} {roi_forecast['predicted_roi']:>19.2%} {kpi_target:>19.2%}\n")
        
        roi_status = "✓ 达标" if roi_forecast['predicted_roi'] >= kpi_target else "✗ 未达标"
        roi_gap = roi_forecast['roi_gap']
        output.write(f"{'ROI缺口':<20} {roi_gap:>19.2%} {roi_status:>19}\n")
        
        output.write(f"\n剩余天数: {roi_forecast['days_remaining']} 天\n")
        output.write(f"日均消耗: ¥{roi_forecast['avg_daily_spend']:,.0f}\n")
        
        return output.getvalue()
    
    def _generate_diagnosis_report(self, product_diagnoses, kpi_target):
        """生成产品诊断报告"""
        output = StringIO()
        
        # 统计等级分布
        grades = {'A': 0, 'B': 0, 'C': 0}
        total_at_risk = 0
        
        for product_id, diagnosis in product_diagnoses.items():
            grade = diagnosis['efficiency_status']['grade']
            grades[grade] += 1
            if len(diagnosis['risk_alerts']) > 0:
                total_at_risk += 1
        
        output.write(f"\n产品等级分布:\n")
        output.write(f"  A级（高效）: {grades['A']} 个\n")
        output.write(f"  B级（中等）: {grades['B']} 个\n")
        output.write(f"  C级（低效）: {grades['C']} 个\n")
        output.write(f"  风险产品: {total_at_risk} 个\n\n")
        
        # 构建产品诊断表
        diagnosis_list = []
        for product_id, diagnosis in product_diagnoses.items():
            row = {
                '产品ID': product_id,
                '等级': diagnosis['efficiency_status']['grade'],
                '当前ROI': f"{diagnosis['efficiency_status']['actual_roi']:.2%}",
                '目标ROI': f"{kpi_target:.2%}",
                '缺口': f"{diagnosis['roi_gap']['percentage_points']:+.1f}pp",
                '消耗': f"¥{diagnosis['statistics']['total_spend']:,.0f}",
                '建议': diagnosis['optimization']['suggestion']
            }
            diagnosis_list.append(row)
        
        diagnosis_df = pd.DataFrame(diagnosis_list)
        output.write(diagnosis_df.to_string(index=False))
        output.write(f"\n\n详细诊断说明:\n")
        
        for product_id, diagnosis in sorted(product_diagnoses.items(), 
                                           key=lambda x: x[1]['efficiency_status']['grade']):
            output.write(f"\n{product_id}:\n")
            output.write(f"  - 等级: {diagnosis['efficiency_status']['grade']} "
                        f"({diagnosis['efficiency_status']['grade_reason']})\n")
            output.write(f"  - 调优方向: {diagnosis['optimization']['direction']}\n")
            output.write(f"  - 建议: {diagnosis['optimization']['suggestion']}\n")
            if len(diagnosis['risk_alerts']) > 0:
                output.write(f"  - 风险警告:\n")
                for alert in diagnosis['risk_alerts']:
                    output.write(f"    • [{alert['severity']}] {alert['type']}: {alert['description']}\n")
        
        return output.getvalue(), diagnosis_df
    
    def _generate_allocation_report(self, budget_allocation, product_diagnoses, days_remaining):
        """生成预算分配报告"""
        output = StringIO()
        
        solver_status = budget_allocation.get('solver_status')
        allocation_status = budget_allocation.get('status', '')
        if solver_status == 'SUCCESS':
            status = "✓ 鲁棒联合LP最优"
        elif allocation_status == 'RecoveryBudget':
            status = "⚠️  ROI修复预算模式（严格KPI约束不可行，仅推荐可提升当前ROI的受限预算）"
        elif str(solver_status).startswith('FALLBACK') and allocation_status == 'GreedyRecovery':
            status = "⚠️  鲁棒恢复模式（当前P20约束不可达，优先选择改善ROI的预算段）"
        elif str(solver_status).startswith('FALLBACK'):
            status = "⚠️  鲁棒贪心兜底"
        else:
            status = "⚠️  启发式分配"
        
        output.write(f"\n优化方法: {status}\n")
        output.write(f"建议释放预算: ¥{budget_allocation.get('total_allocated', 0):,.0f}\n")
        output.write(f"利用率: {budget_allocation.get('utilization_rate', 0):.2%}\n\n")
        if 'predicted_roi_p20' in budget_allocation:
            output.write(f"联合优化P20 ROI: {budget_allocation.get('predicted_roi_p20', 0):.2%}\n")
            output.write(f"联合优化P50 ROI: {budget_allocation.get('predicted_roi_p50', 0):.2%}\n\n")
        recovery_mode = budget_allocation.get('recovery_mode')
        if isinstance(recovery_mode, dict):
            output.write(f"恢复模式基准ROI: {recovery_mode.get('baseline_roi', 0):.2%}\n")
            output.write(f"恢复预算上限: ¥{recovery_mode.get('recovery_budget_limit', 0):,.0f}\n")
            output.write(f"可选恢复预算段: {recovery_mode.get('eligible_segments', 0)} 个\n")
            output.write(f"说明: {recovery_mode.get('mode_note', '')}\n\n")
        
        # 构建分配表
        allocation_list = []
        for product_id, alloc in budget_allocation.get('allocations', {}).items():
            row = {
                '产品ID': product_id,
                '等级': alloc.get('grade', ''),
                '分配预算': f"¥{alloc['allocated_budget']:,.0f}",
                '日均预算': f"¥{alloc['daily_budget']:,.0f}",
                '优化方向': alloc.get('optimization', '')
            }
            allocation_list.append(row)
        
        allocation_df = pd.DataFrame(allocation_list)
        output.write(allocation_df.to_string(index=False))

        unit_alloc = budget_allocation.get('unit_allocations')
        if isinstance(unit_alloc, pd.DataFrame) and not unit_alloc.empty:
            top_units = unit_alloc.head(30).copy()
            display_cols = [
                'product_id', 'date', '推广流量', '流量场景', '创意规格', '转化类型',
                'allocated_budget', 'segment_roi_p20', 'recovery_lift_score',
                'objective_score', 'cap_utilization'
            ]
            display_cols = [c for c in display_cols if c in top_units.columns]
            top_units = top_units[display_cols]
            if 'date' in top_units.columns:
                top_units['date'] = pd.to_datetime(top_units['date']).dt.strftime('%Y-%m-%d')
            if 'allocated_budget' in top_units.columns:
                top_units['allocated_budget'] = top_units['allocated_budget'].map(lambda x: f"¥{x:,.0f}")
            if 'segment_roi_p20' in top_units.columns:
                top_units['segment_roi_p20'] = top_units['segment_roi_p20'].map(lambda x: f"{x:.2%}")
            if 'recovery_lift_score' in top_units.columns:
                top_units['recovery_lift_score'] = top_units['recovery_lift_score'].map(lambda x: f"{x:.2%}")
            if 'objective_score' in top_units.columns:
                top_units['objective_score'] = top_units['objective_score'].map(lambda x: f"{x:.4f}")
            if 'cap_utilization' in top_units.columns:
                top_units['cap_utilization'] = top_units['cap_utilization'].fillna(0).map(lambda x: f"{x:.1%}")

            output.write("\n\n投放单元级Top建议（前30条）:\n")
            output.write(top_units.to_string(index=False))
        else:
            candidates = budget_allocation.get('candidate_units')
            if isinstance(candidates, pd.DataFrame) and not candidates.empty:
                top_candidates = candidates.head(30).copy()
                display_cols = [
                    'product_id', 'date', '推广流量', '流量场景', '创意规格', '转化类型',
                    'expected_roi_p20', 'expected_roi_p50', 'expected_roi_p80',
                    'p20_margin_pp', 'p50_margin_pp', 'candidate_rank',
                    'effective_cap', 'risk_multiplier', 'holiday_scale'
                ]
                display_cols = [c for c in display_cols if c in top_candidates.columns]
                top_candidates = top_candidates[display_cols]
                if 'date' in top_candidates.columns:
                    top_candidates['date'] = pd.to_datetime(top_candidates['date']).dt.strftime('%Y-%m-%d')
                for col in ['expected_roi_p20', 'expected_roi_p50', 'expected_roi_p80']:
                    if col in top_candidates.columns:
                        top_candidates[col] = top_candidates[col].map(lambda x: f"{x:.2%}")
                for col in ['p20_margin_pp', 'p50_margin_pp']:
                    if col in top_candidates.columns:
                        top_candidates[col] = top_candidates[col].map(lambda x: f"{x:.1f}pp")
                output.write("\n\n候选投放单元Top排序（当前未释放预算时用于解释模型判断）:\n")
                output.write(top_candidates.to_string(index=False))
        
        return output.getvalue(), allocation_df
    
    def _generate_risk_report(self, product_diagnoses):
        """生成风险预警报告"""
        output = StringIO()
        
        has_risks = False
        for product_id, diagnosis in product_diagnoses.items():
            alerts = diagnosis['risk_alerts']
            if len(alerts) > 0:
                has_risks = True
                output.write(f"\n【{product_id}】\n")
                for alert in alerts:
                    severity_icon = "🔴" if alert['severity'] == 'high' else "🟠"
                    output.write(f"  {severity_icon} [{alert['severity']}] {alert['type']}\n")
                    output.write(f"    说明: {alert['description']}\n")
                    output.write(f"    建议: {alert['action']}\n")
        
        if not has_risks:
            output.write("\n✓ 暂无风险警告\n")
        
        return output.getvalue()
    
    def _generate_holiday_report(self, holiday_info, days_remaining):
        """生成节假日窗口分析"""
        output = StringIO()
        
        # 找出高价值窗口（scale_factor > 1.2）
        if len(holiday_info) > 0:
            high_value = holiday_info[holiday_info['scale_factor'] > 1.2]
            
            if len(high_value) > 0:
                output.write(f"\n识别到 {len(high_value)} 个高价值投放窗口:\n")
                output.write(f"{'日期':<15} {'节假日':<20} {'Scale因子':<15}\n")
                output.write("-" * 50 + "\n")
                
                for _, row in high_value.iterrows():
                    output.write(f"{str(row['date'].date()):<15} {row['holiday_name']:<20} {row['scale_factor']:>6.2f}\n")
            else:
                output.write("\n当前周期无特殊节假日窗口\n")
        
        return output.getvalue()
    
    def _generate_roi_report(self, roi_forecast, kpi_target):
        """生成ROI预测详情"""
        output = StringIO()
        
        output.write(f"\n预测维度: {roi_forecast['days_elapsed']} 天已发生 + {roi_forecast['days_remaining']} 天预测\n\n")
        
        output.write(f"{'指标':<25} {'数值':<25}\n")
        output.write("-" * 50 + "\n")
        output.write(f"{'已发生消耗':<25} ¥{roi_forecast['actual_spend']:>23,.0f}\n")
        output.write(f"{'已发生回收':<25} ¥{roi_forecast['actual_revenue']:>23,.0f}\n")
        output.write(f"{'已发生ROI':<25} {roi_forecast['actual_roi']:>24.2%}\n\n")
        
        output.write(f"{'预测消耗（剩余）':<25} ¥{roi_forecast['predicted_remaining_spend']:>23,.0f}\n")
        output.write(f"{'预测回收（剩余）':<25} ¥{roi_forecast['predicted_remaining_revenue']:>23,.0f}\n\n")
        
        output.write(f"{'月度总消耗预测':<25} ¥{roi_forecast['predicted_total_spend']:>23,.0f}\n")
        output.write(f"{'月度总回收预测':<25} ¥{roi_forecast['predicted_total_revenue']:>23,.0f}\n")
        output.write(f"{'月度ROI预测':<25} {roi_forecast['predicted_roi']:>24.2%}\n")
        output.write(f"{'目标ROI':<25} {kpi_target:>24.2%}\n")
        
        status = "✓ 可达标" if roi_forecast['roi_gap'] >= 0 else "✗ 缺口"
        output.write(f"{'ROI缺口':<25} {roi_forecast['roi_gap']:>24.2%} {status:>5}\n")
        
        return output.getvalue()
    
    def _combine_reports_to_dataframe(self, monthly_stats, product_diagnoses,
                                      roi_forecast, allocation_df, date,
                                      mpc_calibration_factor=1.0,
                                      ml_d1_rates=None, id_mapping=None,
                                      budget_allocation=None):
        """产品级明细宽表：每行一个产品，列覆盖诊断+预算+风险+预测+ML"""
        if ml_d1_rates is None:
            ml_d1_rates = {}
        if id_mapping is None:
            id_mapping = {}
        if budget_allocation is None:
            budget_allocation = {}
        rows = []

        # --- 解析优化分配结果 ---
        alloc_map = {}
        if allocation_df is not None and not allocation_df.empty:
            for _, row in allocation_df.iterrows():
                pid = str(row.get('产品ID', '')).strip()
                alloc_map[pid] = {
                    'daily_budget': row.get('日均预算', '-'),
                    'allocated': row.get('分配预算', '-'),
                    'direction': row.get('优化方向', '-'),
                }

        product_opt_map = self._summarize_product_optimization(budget_allocation)
        portfolio_p20 = budget_allocation.get('predicted_roi_p20')
        portfolio_p50 = budget_allocation.get('predicted_roi_p50')
        solver_status = budget_allocation.get('solver_status', '')
        optimizer_status = budget_allocation.get('status', '')
        roi_constraint = budget_allocation.get('roi_constraint', {})
        recovery_mode = budget_allocation.get('recovery_mode', {})

        # 大盘兜底值
        avg_daily = roi_forecast.get('avg_daily_spend', 10000)
        num_products = max(len(product_diagnoses), 1)
        base_budget_per_product = avg_daily / num_products

        # --- 逐产品明细行 ---
        for product_id, diag in product_diagnoses.items():
            pid = str(product_id).strip()
            eff = diag['efficiency_status']
            stats = diag['statistics']
            gap = diag['roi_gap']
            opt = diag['optimization']
            window = diag['improvement_window']
            risks = diag['risk_alerts']

            alloc = alloc_map.get(pid, {})

            # 建议预算（分配表有则用，否则兜底）
            suggested = alloc.get('daily_budget')
            if not suggested or suggested == '-':
                grade = eff['grade']
                hist_spend = stats['total_spend']
                prod_days = max(stats['days_active'], 1)
                prod_daily = hist_spend / prod_days
                if prod_daily < 100:
                    prod_daily = base_budget_per_product
                if grade == 'A':
                    suggested = f"¥{prod_daily * 1.5:,.0f}*"
                elif grade == 'B':
                    suggested = f"¥{prod_daily * 1.0:,.0f}*"
                elif grade == 'C':
                    suggested = f"¥{prod_daily * 0.5:,.0f}*"
                else:
                    suggested = '¥0*'

            # 风险简述
            risk_str = '; '.join([f"[{r['type']}] {r['description']}" for r in risks]) if risks else ''

            # ML预测D1（product_id是广告主ID，需通过mapping找到应用ID）
            app_id = id_mapping.get(product_id, product_id)
            ml_d1 = ml_d1_rates.get(app_id, None)
            ml_d1_str = round(ml_d1, 4) if ml_d1 is not None else '-'
            ml_delta_str = round(ml_d1 - eff['d1_rate'], 4) if ml_d1 is not None else '-'
            product_opt = product_opt_map.get(str(product_id), {})
            risk_control = diag.get('risk_control', {})

            rows.append({
                '日期': date,
                '产品ID': pid,
                '等级': eff['grade'],
                '优化器状态': optimizer_status,
                '求解器状态': solver_status,
                'D1回收率(历史)': round(eff['d1_rate'], 4),
                'D1回收率(ML预测)': ml_d1_str,
                'ML vs 历史差值': ml_delta_str,
                'D1预测P20(分配单元加权)': product_opt.get('d1_p20_weighted', '-'),
                'D1预测P50(分配单元加权)': product_opt.get('d1_p50_weighted', '-'),
                'D1预测P80(分配单元加权)': product_opt.get('d1_p80_weighted', '-'),
                '达标D1': round(eff['target_d1'], 4),
                '当前ROI': round(eff['actual_roi'], 4),
                'KPI': round(eff['target_roi'], 4),
                'ROI缺口(pp)': round(gap['percentage_points'], 1),
                '组合预测ROI_P20': round(portfolio_p20, 4) if portfolio_p20 is not None else '-',
                '组合预测ROI_P50': round(portfolio_p50, 4) if portfolio_p50 is not None else '-',
                'P20安全垫(pp)': round((portfolio_p20 - eff['target_roi']) * 100, 2) if portfolio_p20 is not None else '-',
                '恢复模式基准ROI': round(recovery_mode.get('baseline_roi', np.nan), 4) if recovery_mode else '-',
                '恢复预算上限': round(recovery_mode.get('recovery_budget_limit', np.nan), 2) if recovery_mode else '-',
                '可选恢复预算段': recovery_mode.get('eligible_segments', '-') if recovery_mode else '-',
                '累计消耗': round(stats['total_spend'], 2),
                '累计回收': round(stats['total_revenue'], 2),
                '活跃天数': stats['days_active'],
                '历史日均消耗': round(stats['daily_avg_spend'], 2),
                '建议日均预算': suggested,
                '分配总额': alloc.get('allocated', '-'),
                '分配单元数': product_opt.get('allocated_unit_count', 0),
                '分配覆盖日期数': product_opt.get('allocated_date_count', 0),
                '分配Cap': product_opt.get('effective_cap_total', '-'),
                'Cap使用率': product_opt.get('cap_utilization', '-'),
                '分配预算占总建议': product_opt.get('budget_share', '-'),
                '分配单元P20边际ROI': product_opt.get('segment_roi_p20_weighted', '-'),
                '分配单元P50边际ROI': product_opt.get('segment_roi_p50_weighted', '-'),
                'LTV倍率(分配加权)': product_opt.get('ltv_multiplier_weighted', '-'),
                '恢复提升分(分配加权)': product_opt.get('recovery_lift_score_weighted', '-'),
                '分配质量分(分配加权)': product_opt.get('objective_score_weighted', '-'),
                '风险预算折扣': round(risk_control.get('risk_multiplier', 1.0), 4),
                '风险预算动作': risk_control.get('budget_action', '正常'),
                '调优方向': opt['direction'],
                '改善可行性': '是' if opt['feasibility'] else '否',
                '剩余改善天': window['days_remaining'],
                '临近截止': '⚠️ 是' if window['critical_point'] else '否',
                '风险告警': risk_str,
                '诊断说明': opt['suggestion'],
                '等级判定依据': eff['grade_reason'],
            })

        # --- 月度汇总行 ---
        cal = mpc_calibration_factor
        roi_st = '盈余' if roi_forecast['roi_gap'] >= 0 else '缺口'
        portfolio_p20_text = f"{portfolio_p20:.2%}" if portfolio_p20 is not None else "-"
        rows.append({
            '日期': date,
            '产品ID': '【月度汇总】',
            '等级': '-',
            '优化器状态': optimizer_status,
            '求解器状态': solver_status,
            'D1回收率(历史)': round(roi_forecast.get('d1_roi_avg', 0), 4),
            'D1预测P20(分配单元加权)': '-',
            'D1预测P50(分配单元加权)': '-',
            'D1预测P80(分配单元加权)': '-',
            '达标D1': '-',
            '当前ROI': round(roi_forecast['actual_roi'], 4),
            'KPI': round(roi_forecast['target_roi'], 4),
            'ROI缺口(pp)': round(roi_forecast['roi_gap'] * 100, 1),
            '组合预测ROI_P20': round(portfolio_p20, 4) if portfolio_p20 is not None else '-',
            '组合预测ROI_P50': round(portfolio_p50, 4) if portfolio_p50 is not None else '-',
            'P20安全垫(pp)': round((portfolio_p20 - roi_forecast['target_roi']) * 100, 2) if portfolio_p20 is not None else '-',
            '恢复模式基准ROI': round(recovery_mode.get('baseline_roi', np.nan), 4) if recovery_mode else '-',
            '恢复预算上限': round(recovery_mode.get('recovery_budget_limit', np.nan), 2) if recovery_mode else '-',
            '可选恢复预算段': recovery_mode.get('eligible_segments', '-') if recovery_mode else '-',
            '累计消耗': round(roi_forecast['actual_spend'], 2),
            '累计回收': round(roi_forecast['actual_revenue'], 2),
            '活跃天数': '-',
            '历史日均消耗': round(roi_forecast['avg_daily_spend'], 2),
            '建议日均预算': '-',
            '分配总额': round(budget_allocation.get('total_allocated', 0), 2),
            '分配单元数': sum(v.get('allocated_unit_count', 0) for v in product_opt_map.values()),
            '分配覆盖日期数': '-',
            '分配Cap': round(sum(v.get('effective_cap_total_raw', 0) for v in product_opt_map.values()), 2),
            'Cap使用率': round(budget_allocation.get('utilization_rate', 0), 4),
            '分配预算占总建议': 1.0 if budget_allocation.get('total_allocated', 0) > 0 else 0,
            '分配单元P20边际ROI': '-',
            '分配单元P50边际ROI': '-',
            'LTV倍率(分配加权)': '-',
            '恢复提升分(分配加权)': '-',
            '分配质量分(分配加权)': '-',
            '风险预算折扣': '-',
            '风险预算动作': '-',
            '调优方向': '-',
            '改善可行性': '-',
            '剩余改善天': roi_forecast['days_remaining'],
            '临近截止': '-',
            '风险告警': '-',
            '诊断说明': (f"预测月末ROI={roi_forecast['predicted_roi']:.2%}, "
                        f"预测总消耗=¥{roi_forecast['predicted_total_spend']:,.0f}, "
                        f"预测总回收=¥{roi_forecast['predicted_total_revenue']:,.0f}, "
                        f"历史长尾回收=¥{roi_forecast.get('_debug_historical_tail', 0):,.0f}, "
                        f"优化器建议释放=¥{budget_allocation.get('total_allocated', 0):,.0f}, "
                        f"优化P20 ROI={portfolio_p20_text}, "
                        f"未来P20回收=¥{roi_constraint.get('future_revenue_p20', 0):,.0f}, "
                        f"MPC校准因子={cal:.3f}, "
                        f"已过{roi_forecast['days_elapsed']}天/共{roi_forecast['days_total']}天"),
            '等级判定依据': f"ROI状态={roi_st}, 平均倍率={roi_forecast.get('avg_roi_multiplier', 1.0):.3f}",
        })

        return pd.DataFrame(rows)

    def _summarize_product_optimization(self, budget_allocation):
        unit_df = budget_allocation.get('unit_allocations') if isinstance(budget_allocation, dict) else None
        if not isinstance(unit_df, pd.DataFrame) or unit_df.empty:
            return {}

        result = {}
        total_allocated = unit_df['allocated_budget'].sum()
        for product_id, grp in unit_df.groupby('product_id'):
            product_key = str(int(product_id)) if isinstance(product_id, float) and product_id.is_integer() else str(product_id)
            weights = grp['allocated_budget'].clip(lower=0)
            weight_sum = weights.sum()

            def weighted(col):
                if col not in grp.columns or weight_sum <= 0:
                    return '-'
                return round(float(np.average(grp[col], weights=weights)), 4)

            effective_cap_total = float(grp.get('effective_cap', pd.Series(dtype=float)).sum())
            allocated = float(grp['allocated_budget'].sum())
            result[product_key] = {
                'allocated_unit_count': int(len(grp)),
                'allocated_date_count': int(pd.to_datetime(grp['date']).dt.date.nunique()) if 'date' in grp.columns else 0,
                'effective_cap_total': round(effective_cap_total, 2),
                'effective_cap_total_raw': effective_cap_total,
                'cap_utilization': round(allocated / effective_cap_total, 4) if effective_cap_total > 0 else '-',
                'budget_share': round(allocated / total_allocated, 4) if total_allocated > 0 else 0,
                'segment_roi_p20_weighted': weighted('segment_roi_p20'),
                'segment_roi_p50_weighted': weighted('segment_roi_p50'),
                'd1_p20_weighted': weighted('d1_p20'),
                'd1_p50_weighted': weighted('d1_p50'),
                'd1_p80_weighted': weighted('d1_p80'),
                'ltv_multiplier_weighted': weighted('ltv_multiplier'),
                'recovery_lift_score_weighted': weighted('recovery_lift_score'),
                'objective_score_weighted': weighted('objective_score'),
            }
        return result

    def _build_unit_allocation_dataframe(self, budget_allocation, date, kpi_target):
        unit_df = budget_allocation.get('unit_allocations') if isinstance(budget_allocation, dict) else None
        output_type = '实际分配'
        if not isinstance(unit_df, pd.DataFrame) or unit_df.empty:
            unit_df = budget_allocation.get('candidate_units') if isinstance(budget_allocation, dict) else None
            output_type = '候选排序'
        if not isinstance(unit_df, pd.DataFrame) or unit_df.empty:
            return pd.DataFrame()

        df = unit_df.copy()
        if 'allocated_budget' not in df.columns:
            df['allocated_budget'] = 0.0
        if 'daily_budget' not in df.columns:
            df['daily_budget'] = df['allocated_budget']
        rename_map = {
            'product_id': '产品ID',
            'date': '建议日期',
            'allocated_budget': '建议预算',
            'daily_budget': '建议日预算',
            'd1_p20': 'D1预测P20',
            'd1_p50': 'D1预测P50',
            'd1_p80': 'D1预测P80',
            'ltv_multiplier': '月内LTV倍率',
            'segment_roi_p20': '边际ROI_P20',
            'segment_roi_p50': '边际ROI_P50',
            'expected_roi_p20': '单元ROI_P20',
            'expected_roi_p50': '单元ROI_P50',
            'expected_roi_p80': '单元ROI_P80',
            'p20_margin_pp': '单元P20安全垫(pp)',
            'p50_margin_pp': '单元P50安全垫(pp)',
            'candidate_rank': '候选排序',
            'recovery_lift_score': '恢复提升分',
            'objective_score': '分配质量分',
            'cap': '原始Cap',
            'effective_cap': '风险后Cap',
            'cap_utilization': 'Cap使用率',
            'risk_multiplier': '风险折扣',
            'holiday_scale': '节假日Scale',
            'historical_weight': '历史消耗权重'
        }
        df = df.rename(columns=rename_map)
        df.insert(0, '报告日期', date)
        df.insert(1, '输出类型', output_type)
        df['KPI'] = kpi_target
        if '建议日期' in df.columns:
            df['建议日期'] = pd.to_datetime(df['建议日期']).dt.strftime('%Y-%m-%d')
        if '边际ROI_P20' in df.columns:
            df['P20安全垫(pp)'] = (df['边际ROI_P20'] - kpi_target) * 100
        if '边际ROI_P50' in df.columns:
            df['P50安全垫(pp)'] = (df['边际ROI_P50'] - kpi_target) * 100

        preferred = [
            '报告日期', '输出类型', '建议日期', '产品ID', '推广流量', '流量场景', '创意规格', '转化类型',
            '建议预算', 'D1预测P20', 'D1预测P50', 'D1预测P80',
            '月内LTV倍率', '边际ROI_P20', '边际ROI_P50', 'P20安全垫(pp)', 'P50安全垫(pp)',
            '单元ROI_P20', '单元ROI_P50', '单元ROI_P80', '单元P20安全垫(pp)', '单元P50安全垫(pp)',
            '恢复提升分', '分配质量分', '候选排序', '原始Cap', '风险后Cap', 'Cap使用率', '风险折扣', '节假日Scale',
            '历史消耗权重', 'KPI'
        ]
        ordered = [c for c in preferred if c in df.columns]
        rest = [c for c in df.columns if c not in ordered]
        return df[ordered + rest]

    def _build_optimization_summary_dataframe(self, budget_allocation, roi_forecast, date, kpi_target):
        roi_constraint = budget_allocation.get('roi_constraint', {}) if isinstance(budget_allocation, dict) else {}
        rows = [{
            '日期': date,
            '优化器状态': budget_allocation.get('status', ''),
            '求解器状态': budget_allocation.get('solver_status', ''),
            'KPI': kpi_target,
            '当前ROI': roi_forecast.get('actual_roi', 0),
            '原预测月末ROI': roi_forecast.get('predicted_roi', 0),
            '优化后ROI_P20': budget_allocation.get('predicted_roi_p20', np.nan),
            '优化后ROI_P50': budget_allocation.get('predicted_roi_p50', np.nan),
            'P20安全垫(pp)': (budget_allocation.get('predicted_roi_p20', 0) - kpi_target) * 100
                              if budget_allocation.get('predicted_roi_p20') is not None else np.nan,
            '建议释放预算': budget_allocation.get('total_allocated', 0),
            '预算利用率': budget_allocation.get('utilization_rate', 0),
            '恢复模式基准ROI': budget_allocation.get('recovery_mode', {}).get('baseline_roi', np.nan),
            '恢复预算上限': budget_allocation.get('recovery_mode', {}).get('recovery_budget_limit', np.nan),
            '可选恢复预算段': budget_allocation.get('recovery_mode', {}).get('eligible_segments', np.nan),
            '历史长尾回收': roi_constraint.get('historical_tail_revenue', roi_forecast.get('_debug_historical_tail', 0)),
            '未来回收_P20': roi_constraint.get('future_revenue_p20', 0),
            '未来回收_P50': roi_constraint.get('future_revenue_p50', 0),
            '已发生消耗': roi_constraint.get('actual_spend', roi_forecast.get('actual_spend', 0)),
            '已发生回收': roi_constraint.get('actual_revenue', roi_forecast.get('actual_revenue', 0)),
        }]
        return pd.DataFrame(rows)

    def generate_monthly_summary(self, data, year, month, kpi_target):
        """生成月度总结"""
        print(f"\n{'='*80}")
        print(f"📋 {year}年{month}月份月度总结")
        print(f"{'='*80}\n")
        
        # 提取月度数据
        month_data = data[
            (data['日期'].dt.year == year) & 
            (data['日期'].dt.month == month)
        ].copy()
        
        # 计算统计数据
        daily_stats = []
        for date in sorted(month_data['日期'].unique()):
            day_data = month_data[month_data['日期'] == date]
            
            total_spend = day_data['消耗金额'].sum()
            total_revenue = day_data['30日累计变现金额'].sum()
            
            daily_stats.append({
                '日期': date,
                '消耗': total_spend,
                '回收': total_revenue,
                '当日ROI': total_revenue / total_spend if total_spend > 0 else 0,
                '产品数': day_data['应用ID'].nunique()
            })
        
        summary_df = pd.DataFrame(daily_stats)
        
        print("日度统计:")
        print(summary_df.to_string(index=False))
        
        print(f"\n月度总计:")
        print(f"  总消耗: ¥{summary_df['消耗'].sum():,.0f}")
        print(f"  总回收: ¥{summary_df['回收'].sum():,.0f}")
        total_roi = summary_df['回收'].sum() / summary_df['消耗'].sum() if summary_df['消耗'].sum() > 0 else 0
        print(f"  月度ROI: {total_roi:.2%} (目标: {kpi_target:.2%})")
        print(f"  达标状态: {'✓ 达标' if total_roi >= kpi_target else '✗ 未达标'}")
        print(f"  日均消耗: ¥{summary_df['消耗'].mean():,.0f}")
        
        return summary_df
