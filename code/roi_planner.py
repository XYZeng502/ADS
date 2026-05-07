"""
模块5：ROI预测与消耗规划模块 (V2 架构升级版)
升级点：
1. 淘汰 if-else 阶梯，引入连续对数曲线拟合 M(t) = a * ln(t) + b
2. 引入 Cohort Matrix 长尾预测，计算历史老客在当月剩余天数的长尾贡献
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ROIPlanner:
    """ROI预测和消耗规划类"""
    
    def __init__(self, data, ltv_curves=None, kpi_target=1.05):
        """
        初始化ROI规划器
        
        参数：
        - data: 历史数据 DataFrame
        - ltv_curves: LTV曲线参数字典，例如 {'default': {'a': 0.1293, 'b': 1.0}} 
        - kpi_target: KPI目标ROI
        """
        self.data = data
        # 如果未传入曲线参数，则使用默认的对数拟合参数 (基于 D1=1.0, D30=1.44 拟合)
        self.ltv_curves = ltv_curves if ltv_curves else {'default': {'a': 0.1293, 'b': 1.0}}
        self.kpi_target = kpi_target
        
    def _get_continuous_multiplier(self, days_lived, product_id='default'):
        """
        核心升级：连续曲线获取 LTV 倍率，并向后兼容 LTVPredictor 传入的离散点数据
        """
        if days_lived <= 0:
            return 0.0
        days_lived = min(days_lived, 30)
        if days_lived == 1:
            return 1.0 # 首日基础倍率恒为 1.0
            
        params = None
        # 1. 尝试直接获取对数参数 a 和 b
        if isinstance(self.ltv_curves, dict):
            if product_id in self.ltv_curves and 'a' in self.ltv_curves[product_id]:
                params = self.ltv_curves[product_id]
            elif 'default' in self.ltv_curves and 'a' in self.ltv_curves['default']:
                params = self.ltv_curves['default']
                
        # 2. 如果没有 a 和 b，说明传入的是离散曲线（从 LTVPredictor 传来），则动态推导
        if not params:
            d30_mult = 1.44 # 默认兜底倍率
            if isinstance(self.ltv_curves, dict) and product_id in self.ltv_curves:
                prod_data = self.ltv_curves[product_id]
                # 兼容 {product_id: {'curve': {'D30': 1.44}}} 的数据结构
                if 'curve' in prod_data and 'D30' in prod_data['curve']:
                    d30_mult = prod_data['curve']['D30']
            
            # 动态计算系数 a: a * ln(30) + 1.0 = d30_mult => a = (d30_mult - 1.0) / ln(30)
            a = (d30_mult - 1.0) / np.log(30)
            b = 1.0
        else:
            a = params['a']
            b = params.get('b', 1.0)
            
        # 3. 计算对数增长
        multiplier = a * np.log(days_lived) + b
        return max(1.0, multiplier)

    def forecast_monthly_roi(self, historical_data, current_date, month_start, month_end,
                             kpi_target, product_d1_rates=None):
        """
        预测月度ROI (队列矩阵滚动版 + ML增强)

        参数：
        - product_d1_rates: dict {product_id: predicted_d1_rate}, ML预测的各产品D1
        """
        month_data = historical_data[
            (historical_data['日期'] >= month_start) & 
            (historical_data['日期'] <= current_date)
        ].copy()
        
        current_dt = pd.to_datetime(current_date)
        month_end_dt = pd.to_datetime(month_end)
        month_start_dt = pd.to_datetime(month_start)
        
        actual_spend = month_data['消耗金额'].sum()

        days_elapsed = (current_dt - month_start_dt).days + 1
        days_total = (month_end_dt - month_start_dt).days + 1
        days_remaining = days_total - days_elapsed

        avg_daily_spend = actual_spend / days_elapsed if days_elapsed > 0 else 0
        predicted_remaining_spend = avg_daily_spend * max(days_remaining, 0)

        # ---------------------------------------------------------
        # 核心升级 1：逐Cohort精确计算实际回收 + 历史长尾预测
        # 不再依赖30日累计变现金额（对不足30天的Cohort可能为空/预估值）
        # 统一用 D1 × 已存活倍率 计算实际已释放回收
        # ---------------------------------------------------------
        actual_revenue = 0.0
        historical_tail_revenue = 0.0
        d1_roi_avg = 0.0

        if len(month_data) > 0 and '首日广告收入' in month_data.columns:
            d1_total = month_data['首日广告收入'].sum()
            d1_roi_avg = d1_total / actual_spend if actual_spend > 0 else 0.65

            for index, row in month_data.iterrows():
                spend = row['消耗金额']
                if pd.isna(spend) or spend <= 0:
                    continue

                app_id = row.get('应用ID')
                day_dt = pd.to_datetime(row['日期'])
                days_lived_so_far = max(1, (current_dt - day_dt).days + 1)
                days_lived_by_month_end = (month_end_dt - day_dt).days + 1

                current_mult = self._get_continuous_multiplier(days_lived_so_far, app_id)
                end_month_mult = self._get_continuous_multiplier(days_lived_by_month_end, app_id)

                d1_revenue = row.get('首日广告收入')
                if pd.isna(d1_revenue):
                    # ML增强：优先用产品级(应用ID)D1预估，其次用全局均值
                    if product_d1_rates and app_id in product_d1_rates:
                        d1_revenue = spend * product_d1_rates[app_id]
                    else:
                        d1_revenue = spend * d1_roi_avg

                actual_revenue += d1_revenue * current_mult
                tail_revenue = d1_revenue * max(0, (end_month_mult - current_mult))
                historical_tail_revenue += tail_revenue

        # --- ML增强：计算产品消耗加权的 D1 均值，用于未来预测 ---
        if product_d1_rates:
            weighted_d1 = 0.0
            total_weight = 0.0
            for app_id, rate in product_d1_rates.items():
                prod_spend = month_data[month_data['应用ID'] == app_id]['消耗金额'].sum()
                weighted_d1 += rate * max(prod_spend, 0)
                total_weight += max(prod_spend, 0)
            ml_d1_avg = weighted_d1 / total_weight if total_weight > 0 else d1_roi_avg
        else:
            ml_d1_avg = d1_roi_avg

        # ---------------------------------------------------------
        # 核心升级 2：平滑预测未来新客的回收 (逐日计算)
        # ---------------------------------------------------------
        predicted_remaining_revenue = 0
        day_multipliers = []

        if days_remaining > 0:
            for day_offset in range(1, days_remaining + 1):
                days_until_end = days_remaining - day_offset + 1
                mult = self._get_continuous_multiplier(days_until_end)
                day_multipliers.append(mult)
                # 用 ML 增强后的 D1 预测未来回收
                predicted_remaining_revenue += avg_daily_spend * max(ml_d1_avg, 0.65) * mult

        avg_roi_multiplier = np.mean(day_multipliers) if day_multipliers else 1.0
        
        # ---------------------------------------------------------
        # 汇总：总回收 = 已发生实际回收 + 历史长尾预测 + 未来新客预测
        # ---------------------------------------------------------
        predicted_total_spend = actual_spend + predicted_remaining_spend
        predicted_total_revenue = actual_revenue + historical_tail_revenue + predicted_remaining_revenue
        
        predicted_roi = predicted_total_revenue / predicted_total_spend if predicted_total_spend > 0 else 0
        roi_gap = predicted_roi - kpi_target
        
        # 严格保持原输出格式字典
        return {
            'actual_spend': actual_spend,
            'actual_revenue': actual_revenue,
            'actual_roi': actual_revenue / actual_spend if actual_spend > 0 else 0,
            'predicted_remaining_spend': predicted_remaining_spend,
            'predicted_remaining_revenue': predicted_remaining_revenue + historical_tail_revenue, # 将长尾合并输出防报错
            'predicted_total_spend': predicted_total_spend,
            'predicted_total_revenue': predicted_total_revenue,
            'predicted_roi': predicted_roi,
            'target_roi': kpi_target,
            'roi_gap': roi_gap,
            'roi_status': 'surplus' if roi_gap >= 0 else 'deficit',
            'days_elapsed': days_elapsed,
            'days_remaining': days_remaining,
            'days_total': days_total,
            'avg_daily_spend': avg_daily_spend,
            'avg_roi_multiplier': avg_roi_multiplier,
            'd1_roi_avg': d1_roi_avg,
            '_debug_historical_tail': historical_tail_revenue
        }

    def calculate_roi_reachability(self, current_roi, target_roi, days_remaining, avg_daily_spend):
        """
        评估是否能达到ROI目标 (升级曲线版)
        """
        if current_roi >= target_roi:
            return {
                'reachable': True, 'status': '已达标',
                'required_daily_roi': 0, 'feasibility_score': 1.0,
                'surplus_roi': current_roi - target_roi
            }
        
        required_multiplier = target_roi / current_roi if current_roi > 0 else 0
        
        # 使用连续曲线预估剩余极限倍率
        possible_multiplier = self._get_continuous_multiplier(days_remaining)
        
        feasibility = min(possible_multiplier / required_multiplier, 1.0) if required_multiplier > 0 else 1.0
        
        return {
            'reachable': feasibility >= 0.9,
            'status': '可达' if feasibility >= 0.9 else '困难',
            'required_multiplier': required_multiplier,
            'possible_multiplier': possible_multiplier,
            'feasibility_score': feasibility,
            'days_remaining': days_remaining
        }
        
    def calculate_cross_month_value(self, today_spend, d1_rate, m30_multiple):
        # 此处逻辑本身基于数学推演，无需修改，保持原状
        today_revenue_in_current_month = today_spend * d1_rate * 1.0
        user_lifetime_value = today_spend * d1_rate * m30_multiple
        current_month_recovery = today_revenue_in_current_month
        next_month_recovery = (user_lifetime_value - current_month_recovery) * 0.9
        
        return {
            'today_spend': today_spend,
            'current_month_impact': -d1_rate * (m30_multiple - 1.0),
            'next_month_contribution': next_month_recovery,
            'next_month_roi_lift': next_month_recovery / today_spend if today_spend > 0 else 0,
            'cross_month_roi_ratio': next_month_recovery / current_month_recovery if current_month_recovery > 0 else 0
        }
    
    def estimate_monthly_consumption(self, daily_data, method='linear'):
        """
        估计月度消耗规模
        
        参数：
        - daily_data: 日度数据汇总
        - method: 'linear'(线性外推) / 'trend'(趋势外推)
        
        返回：月度消耗预测
        """
        if method == 'linear':
            # 线性外推：用最近7天的平均日消耗推算
            if len(daily_data) >= 7:
                recent_avg = daily_data['消耗金额'].tail(7).mean()
            else:
                recent_avg = daily_data['消耗金额'].mean()
            
            estimated_monthly = recent_avg * 30
        
        elif method == 'trend':
            # 趋势外推：用最小二乘法拟合趋势
            x = np.arange(len(daily_data))
            y = daily_data['消耗金额'].values
            
            # 简单线性回归
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # 预测未来30天
            future_days = np.arange(len(daily_data), 30)
            future_spend = p(future_days).sum()
            estimated_monthly = daily_data['消耗金额'].sum() + max(future_spend, 0)
        
        else:
            estimated_monthly = daily_data['消耗金额'].sum()
        
        return estimated_monthly
