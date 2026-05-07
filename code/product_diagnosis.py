"""
模块4：产品分级与诊断模块
根据文档实现产品分级（A/B/C级）和诊断功能
- A级：D1高于达标D1（高效）
- B级：D1接近达标D1（中等）
- C级：D1低于达标D1（低效但长尾可能补回）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ProductDiagnostics:
    """产品诊断和分级类"""
    
    def __init__(self, data, ltv_curves, kpi_target=1.05):
        """
        初始化产品诊断器
        
        参数：
        - data: 原始数据
        - ltv_curves: LTV曲线数据
        - kpi_target: KPI目标（如1.05表示105%）
        """
        self.data = data
        self.ltv_curves = ltv_curves
        self.kpi_target = kpi_target
    
    def diagnose_all_products(self, historical_data, current_date, kpi_target):
        """
        对所有产品进行诊断
        产品ID = 应用ID（同一应用在不同渠道/场景/创意会有多条数据）
        
        参数：
        - historical_data: 历史数据（到current_date为止）
        - current_date: 当前日期
        - kpi_target: KPI目标
        
        返回：产品诊断字典列表
        """
        diagnoses = {}
        
        products = historical_data['应用ID'].unique()
        
        for product_id in products:
            product_data = historical_data[historical_data['应用ID'] == product_id].copy()
            diagnosis = self._diagnose_single_product(
                product_id,
                product_data,
                current_date,
                kpi_target
            )
            diagnoses[product_id] = diagnosis
        
        return diagnoses
    
    def _diagnose_single_product(self, product_id, product_data, current_date, kpi_target):
        """诊断单个产品"""

        # 基本统计
        total_spend = product_data['消耗金额'].sum()
        d1_revenue = product_data['首日广告收入'].sum()
        days_active = len(product_data)

        # D1回收率
        d1_rate = d1_revenue / total_spend if total_spend > 0 else 0

        # 达标D1（根据KPI和倍率倒推）
        # 达标D1 = KPI ÷ M30倍率
        m30_multiple = self.ltv_curves.get(product_id, {}).get('curve', {}).get('D30', 1.44)
        target_d1 = kpi_target / m30_multiple

        # 实际ROI：逐Cohort用 D1 × 存活倍率 精确计算，不再依赖30日累计变现金额
        actual_revenue = 0.0
        current_dt = pd.to_datetime(current_date)
        a_coef = (m30_multiple - 1.0) / np.log(30)
        for _, row in product_data.iterrows():
            d1_rev = row.get('首日广告收入')
            if pd.isna(d1_rev) or d1_rev <= 0:
                continue
            row_date = pd.to_datetime(row['日期'])
            days_lived = max(1, (current_dt - row_date).days + 1)
            mult = a_coef * np.log(days_lived) + 1.0
            actual_revenue += d1_rev * mult

        actual_roi = actual_revenue / total_spend if total_spend > 0 else 0
        
        # 产品分级
        grade, grade_reason = self._grade_product(d1_rate, target_d1)
        
        # 达标缺口（相对于KPI）
        roi_gap = kpi_target - actual_roi
        roi_gap_pp = roi_gap * 100  # 百分点
        
        # 改善窗口（剩余天数内能否追上）
        days_in_month = 30  # 假设30天月度
        current_day = (current_date.day) if hasattr(current_date, 'day') else 1
        days_remaining = max(days_in_month - current_day, 0)
        
        if days_remaining > 0 and total_spend > 0:
            # 评估是否还有改善空间
            required_daily_roi = (actual_revenue + roi_gap * total_spend) / (total_spend + total_spend * days_remaining)
            improvement_feasible = (required_daily_roi < d1_rate * m30_multiple)
        else:
            improvement_feasible = False
        
        # 调优方向
        if grade == 'A':
            optimization_direction = '加量'
            suggestion = '该产品效率高，建议增加预算分配'
        elif grade == 'B':
            optimization_direction = '维持'
            suggestion = '该产品效率中等，建议保持当前投放节奏'
        else:  # C级
            if improvement_feasible:
                optimization_direction = '减量或维持观察'
                suggestion = '该产品短期低效，但长尾可能补回，建议观察或小幅减量'
            else:
                optimization_direction = '快速收缩'
                suggestion = '剩余时间不足以改善，建议快速收缩投放'
        
        # 风险预警
        risk_alerts = self._check_risk_alerts(product_data, d1_rate)
        risk_multiplier = self._calculate_risk_multiplier(risk_alerts, grade)
        
        # 日均消耗和周期统计
        daily_avg_spend = total_spend / max(days_active, 1)
        
        return {
            'product_id': product_id,
            'efficiency_status': {
                'grade': grade,
                'grade_reason': grade_reason,
                'd1_rate': d1_rate,
                'target_d1': target_d1,
                'actual_roi': actual_roi,
                'target_roi': kpi_target
            },
            'roi_gap': {
                'absolute': roi_gap,
                'percentage_points': roi_gap_pp,
                'surplus' if roi_gap < 0 else 'deficit': abs(roi_gap_pp)
            },
            'optimization': {
                'direction': optimization_direction,
                'suggestion': suggestion,
                'feasibility': improvement_feasible
            },
            'improvement_window': {
                'days_remaining': days_remaining,
                'days_in_month': days_in_month,
                'critical_point': days_remaining <= 3
            },
            'statistics': {
                'total_spend': total_spend,
                'total_revenue': actual_revenue,
                'd1_revenue': d1_revenue,
                'days_active': days_active,
                'daily_avg_spend': daily_avg_spend,
                'last_update': product_data['日期'].max()
            },
            'risk_alerts': risk_alerts,
            'risk_control': {
                'risk_multiplier': risk_multiplier,
                'budget_action': self._risk_budget_action(risk_multiplier)
            }
        }
    
    def _grade_product(self, d1_rate, target_d1):
        """
        对产品进行A/B/C级分级
        
        参数：
        - d1_rate: 实际D1回收率
        - target_d1: 达标D1（=KPI÷M30）
        
        返回：(grade, reason)
        """
        d1_gap = d1_rate - target_d1
        
        if d1_gap >= 0.03:  # 超过达标D1 3个百分点
            return 'A', f'D1={d1_rate:.2%} 超过达标D1 {d1_gap*100:.1f}pp'
        elif d1_gap >= -0.02:  # 在达标D1 ±2pp范围内
            return 'B', f'D1={d1_rate:.2%} 接近达标D1'
        else:
            return 'C', f'D1={d1_rate:.2%} 低于达标D1 {abs(d1_gap)*100:.1f}pp'
    
    def _check_risk_alerts(self, product_data, current_d1_rate):
        """
        检查产品是否存在风险
        
        返回：风险告警列表
        """
        alerts = []
        
        # 检查最近N天的D1趋势（产品能力下滑）
        recent_days = min(3, len(product_data))
        if recent_days > 1:
            recent_data = product_data.tail(recent_days)
            
            # 计算最近几天的平均D1
            recent_spend = recent_data['消耗金额'].sum()
            recent_d1_revenue = recent_data['首日广告收入'].sum()
            recent_d1_rate = recent_d1_revenue / recent_spend if recent_spend > 0 else 0
            
            # 如果最近D1下降超过10%
            if recent_d1_rate < current_d1_rate * 0.9:
                alerts.append({
                    'type': '产品能力下滑',
                    'severity': 'high',
                    'description': f'最近{recent_days}天D1下降至{recent_d1_rate:.2%}，低于整体平均{current_d1_rate:.2%}',
                    'action': '关注流量质量，考虑调整创意或定向'
                })
        
        # 检查流量异常（单日消耗波动）
        if len(product_data) > 1:
            daily_spend = product_data['消耗金额']
            avg_spend = daily_spend.mean()
            std_spend = daily_spend.std()
            
            # 获取最后一天的消耗
            last_spend = daily_spend.iloc[-1]
            
            # 如果最后一天消耗偏离平均值超过2个标准差
            if std_spend > 0 and abs(last_spend - avg_spend) > 2 * std_spend:
                deviation_pct = abs(last_spend - avg_spend) / avg_spend * 100
                alerts.append({
                    'type': '流量异常波动',
                    'severity': 'medium',
                    'description': f'最后一天消耗{last_spend:.0f}元，偏离平均值{deviation_pct:.1f}%',
                    'action': '检查媒体侧流量投放是否正常'
                })
        
        return alerts

    def _calculate_risk_multiplier(self, alerts, grade):
        """把风险诊断转成优化器可使用的预算上限折扣。"""
        multiplier = 1.0
        for alert in alerts:
            if alert.get('severity') == 'high':
                multiplier *= 0.5
            elif alert.get('severity') == 'medium':
                multiplier *= 0.75

        if grade == 'C':
            multiplier *= 0.85

        return float(np.clip(multiplier, 0.0, 1.0))

    def _risk_budget_action(self, risk_multiplier):
        if risk_multiplier <= 0.3:
            return '暂停或极小预算观察'
        if risk_multiplier <= 0.6:
            return '明显降权'
        if risk_multiplier < 1.0:
            return '轻度降权'
        return '正常'
    
    def get_portfolio_summary(self, diagnoses, kpi_target):
        """
        获取产品组合总体诊断
        
        参数：
        - diagnoses: 所有产品的诊断结果
        - kpi_target: KPI目标
        
        返回：组合诊断摘要
        """
        grades = {}
        total_spend = 0
        total_revenue = 0
        products_at_risk = []
        
        for product_id, diagnosis in diagnoses.items():
            grade = diagnosis['efficiency_status']['grade']
            grades[grade] = grades.get(grade, 0) + 1
            
            total_spend += diagnosis['statistics']['total_spend']
            total_revenue += diagnosis['statistics']['total_revenue']
            
            if len(diagnosis['risk_alerts']) > 0:
                products_at_risk.append({
                    'product_id': product_id,
                    'alerts': diagnosis['risk_alerts']
                })
        
        portfolio_roi = total_revenue / total_spend if total_spend > 0 else 0
        roi_gap = kpi_target - portfolio_roi
        
        return {
            'total_products': len(diagnoses),
            'product_grades': {
                'A': grades.get('A', 0),
                'B': grades.get('B', 0),
                'C': grades.get('C', 0)
            },
            'portfolio_roi': portfolio_roi,
            'target_roi': kpi_target,
            'roi_gap': roi_gap,
            'roi_status': 'surplus' if roi_gap < 0 else 'deficit',
            'total_spend': total_spend,
            'total_revenue': total_revenue,
            'products_at_risk': len(products_at_risk),
            'at_risk_details': products_at_risk
        }
