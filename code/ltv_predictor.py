"""
模块2：LTV曲线与回收预测模块
负责计算用户生命周期价值曲线、预测回收
根据文档：
- D1: 基准 = 1.00
- D3: ~1.29
- D7: ~1.38
- D14: ~1.44
- D30: ~1.44
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


PLACEMENT_DIMS = ['推广流量', '流量场景', '创意规格', '转化类型']


class LTVPredictor:
    """LTV预测和回收曲线计算类"""
    
    # 标准LTV倍率曲线（来自文档和历史数据验证）
    STANDARD_LTV_CURVE = {
        'D1': 1.00,
        'D3': 1.29,
        'D7': 1.38,
        'D14': 1.44,
        'D30': 1.44
    }
    MAX_LTV_MULTIPLIER = 2.20
    
    def __init__(self, data):
        """
        初始化LTV预测器
        
        参数：
        - data: 包含消耗和变现数据的DataFrame
        """
        self.data = data
        self.ltv_curves = {}
        self.unit_curves = {}
        self.global_curve = self.STANDARD_LTV_CURVE.copy()
        self.daily_d1_rates = {}
        self.max_observed_date = pd.to_datetime(data['日期']).max() if '日期' in data.columns else None
    
    def calculate_ltv_curves(self):
        """
        计算每个产品的LTV曲线
        产品ID = 应用ID（同一应用在不同渠道/场景/创意会有多条数据）
        
        返回：{product_id: {'curve': {...}, 'd1_rate': ..., 'avg_roi': ...}}
        """
        print("  - 计算产品LTV曲线...")
        
        self.global_curve = self._fit_ltv_curve(self.data, self.STANDARD_LTV_CURVE)

        products = self.data['应用ID'].unique()
        
        for product_id in products:
            product_data = self.data[self.data['应用ID'] == product_id].copy()
            
            # 计算D1回收率
            d1_rate = self._calculate_d1_rate(product_data)
            
            # 计算该产品的LTV曲线（相对于标准曲线的调整）
            ltv_curve = self._calculate_product_ltv_curve(product_data, d1_rate)
            
            # 计算产品的平均ROI
            avg_roi = self._calculate_avg_roi(product_data)
            
            self.ltv_curves[product_id] = {
                'curve': ltv_curve,
                'd1_rate': d1_rate,
                'avg_roi': avg_roi,
                'total_spend': product_data['消耗金额'].sum(),
                'total_revenue': product_data['30日累计变现金额'].sum(),
                'last_update': product_data['日期'].max()
            }
        self._calculate_unit_ltv_curves()

        print(f"  - 完成{len(self.ltv_curves)}个产品、{len(self.unit_curves)}个投放单元的LTV曲线计算")
        return self.ltv_curves
    
    def _calculate_d1_rate(self, product_data):
        """
        计算D1回收率（首日广告收入 / 消耗金额）
        
        参数：
        - product_data: 该产品的数据
        
        返回：D1回收率（如 0.69 表示69%）
        """
        total_spend = product_data['消耗金额'].sum()
        total_d1_revenue = product_data['首日广告收入'].sum()
        
        if total_spend > 0:
            d1_rate = total_d1_revenue / total_spend
        else:
            d1_rate = 0
        
        return d1_rate
    
    def _calculate_product_ltv_curve(self, product_data, d1_rate):
        """
        计算单个产品的LTV曲线
        
        通过计算D1、D3、D7、D30的实际倍率，
        与标准曲线进行对比，得出该产品的个性化曲线
        """
        return self._fit_ltv_curve(product_data, self.global_curve)

    def _calculate_unit_ltv_curves(self):
        """按产品+投放维度拟合投放单元级LTV曲线，小样本自动收缩到产品曲线。"""
        group_keys = ['应用ID'] + [d for d in PLACEMENT_DIMS if d in self.data.columns]
        if len(group_keys) <= 1:
            return

        for key, unit_data in self.data.groupby(group_keys, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            product_id = key[0]
            fallback = self.get_ltv_curve(product_id)
            unit_curve = self._fit_ltv_curve(unit_data, fallback)
            self.unit_curves[key] = unit_curve

    def _fit_ltv_curve(self, subset, fallback_curve=None):
        """
        拟合单调LTV倍率曲线，并处理未成熟cohort的右截尾。

        关键点：D7/D30 只使用已经存活到对应天数的cohort，避免把未释放完的
        近期数据当成完整回收。
        """
        fallback = (fallback_curve or self.STANDARD_LTV_CURVE).copy()
        if subset is None or len(subset) == 0:
            return fallback

        data = subset.copy()
        data['日期'] = pd.to_datetime(data['日期'])
        max_date = self.max_observed_date or data['日期'].max()

        curve = {'D1': 1.0}
        horizon_columns = {
            'D3': (3, '3日累计变现金额'),
            'D7': (7, '7日累计变现金额'),
            'D30': (30, '30日累计变现金额'),
        }

        for label, (horizon, revenue_col) in horizon_columns.items():
            if revenue_col not in data.columns:
                curve[label] = fallback.get(label, self.STANDARD_LTV_CURVE[label])
                continue

            age_days = (max_date - data['日期']).dt.days + 1
            mature = data[age_days >= horizon]
            fallback_value = fallback.get(label, self.STANDARD_LTV_CURVE[label])

            d1_revenue = mature['首日广告收入'].sum() if len(mature) else 0
            horizon_revenue = mature[revenue_col].sum() if len(mature) else 0

            if d1_revenue > 0 and horizon_revenue > 0:
                observed = horizon_revenue / d1_revenue
            else:
                observed = fallback_value
            observed = float(np.clip(observed, 1.0, self.MAX_LTV_MULTIPLIER))

            weight = self._curve_shrinkage_weight(mature)
            curve[label] = observed * weight + fallback_value * (1 - weight)

        # 没有D14标签时，用D7到D30之间的对数插值估计，再向标准曲线收缩。
        d7 = curve.get('D7', fallback.get('D7', self.STANDARD_LTV_CURVE['D7']))
        d30 = curve.get('D30', fallback.get('D30', self.STANDARD_LTV_CURVE['D30']))
        if d30 > d7:
            ratio = (np.log(14) - np.log(7)) / (np.log(30) - np.log(7))
            d14_est = d7 + (d30 - d7) * ratio
        else:
            d14_est = d7
        curve['D14'] = 0.6 * d14_est + 0.4 * fallback.get('D14', self.STANDARD_LTV_CURVE['D14'])

        # 单调化，防止异常数据导致 D7 < D3 之类的曲线反常。
        ordered = ['D1', 'D3', 'D7', 'D14', 'D30']
        previous = 1.0
        for label in ordered:
            curve[label] = min(max(float(curve.get(label, previous)), previous), self.MAX_LTV_MULTIPLIER)
            previous = curve[label]

        return curve

    def _curve_shrinkage_weight(self, mature_data):
        """根据成熟样本天数、行数和消耗规模确定相信历史数据的程度。"""
        if mature_data is None or len(mature_data) == 0:
            return 0.0

        days = mature_data['日期'].nunique()
        rows = len(mature_data)
        spend = mature_data['消耗金额'].sum()

        day_weight = min(days / 14.0, 1.0)
        row_weight = min(rows / 120.0, 1.0)
        spend_weight = min(spend / 10000.0, 1.0)
        return float(np.clip(0.5 * day_weight + 0.25 * row_weight + 0.25 * spend_weight, 0.0, 1.0))
    
    def _calculate_avg_roi(self, product_data):
        """计算产品的平均ROI"""
        total_spend = product_data['消耗金额'].sum()
        total_revenue = product_data['30日累计变现金额'].sum()
        
        if total_spend > 0:
            return total_revenue / total_spend
        else:
            return 0
    
    def get_ltv_curve(self, product_id):
        """获取指定产品的LTV曲线"""
        if product_id in self.ltv_curves:
            return self.ltv_curves[product_id]['curve']
        else:
            # 返回标准曲线
            return self.global_curve.copy()

    def get_unit_ltv_curve(self, product_id, placement=None):
        """获取投放单元级LTV曲线；缺失时回退到产品级曲线。"""
        if placement is None:
            return self.get_ltv_curve(product_id)

        if isinstance(placement, dict):
            key = tuple([product_id] + [placement.get(d) for d in PLACEMENT_DIMS])
        elif isinstance(placement, tuple):
            key = placement if len(placement) == len(PLACEMENT_DIMS) + 1 else tuple([product_id] + list(placement))
        else:
            key = tuple([product_id, placement])

        return self.unit_curves.get(key, self.get_ltv_curve(product_id))
    
    def get_d1_rate(self, product_id):
        """获取指定产品的D1回收率"""
        if product_id in self.ltv_curves:
            return self.ltv_curves[product_id]['d1_rate']
        else:
            # 返回标准D1率
            return 0.66  # 标准D1率约66%
    
    def _continuous_multiplier(self, days):
        """使用全局曲线进行连续插值，保留旧接口。"""
        return self.get_multiplier('default', days)

    def get_multiplier(self, product_id, days_lived, placement=None, unit_key=None):
        """按产品/投放单元曲线获取任意存活天数的连续LTV倍率。"""
        days = float(days_lived)
        if days <= 1:
            return 1.0

        if unit_key is not None:
            curve = self.unit_curves.get(unit_key, self.get_ltv_curve(product_id))
        elif placement is not None:
            curve = self.get_unit_ltv_curve(product_id, placement)
        elif product_id in self.ltv_curves:
            curve = self.get_ltv_curve(product_id)
        else:
            curve = self.global_curve

        points = [
            (1.0, curve.get('D1', 1.0)),
            (3.0, curve.get('D3', self.STANDARD_LTV_CURVE['D3'])),
            (7.0, curve.get('D7', self.STANDARD_LTV_CURVE['D7'])),
            (14.0, curve.get('D14', self.STANDARD_LTV_CURVE['D14'])),
            (30.0, curve.get('D30', self.STANDARD_LTV_CURVE['D30'])),
        ]

        if days >= 30:
            return float(min(points[-1][1], self.MAX_LTV_MULTIPLIER))

        for (d0, m0), (d1, m1) in zip(points[:-1], points[1:]):
            if d0 <= days <= d1:
                if d0 == d1:
                    return float(m1)
                ratio = (np.log(days) - np.log(d0)) / (np.log(d1) - np.log(d0))
                return float(min(m0 + (m1 - m0) * ratio, self.MAX_LTV_MULTIPLIER))

        return 1.0

    def predict_future_roi(self, today_spend, d1_revenue, days_remaining_in_month):
        """
        预测未来收益（用于月度ROI计算）
        """
        if today_spend == 0:
            return 0

        d1_rate = d1_revenue / today_spend
        future_multiple = self._continuous_multiplier(days_remaining_in_month)
        return today_spend * d1_rate * future_multiple

    def predict_tail_revenue(self, row, current_date, target_date):
        """预测某条cohort从current_date到target_date之间仍会释放的长尾收入。"""
        spend = row.get('消耗金额', 0)
        d1_revenue = row.get('首日广告收入', 0)
        if pd.isna(spend) or spend <= 0 or pd.isna(d1_revenue) or d1_revenue <= 0:
            return 0.0

        product_id = row.get('应用ID')
        placement = {dim: row.get(dim) for dim in PLACEMENT_DIMS}
        start_date = pd.to_datetime(row['日期'])
        current_dt = pd.to_datetime(current_date)
        target_dt = pd.to_datetime(target_date)

        age_now = max(1, (current_dt - start_date).days + 1)
        age_target = max(age_now, (target_dt - start_date).days + 1)

        current_mult = self.get_multiplier(product_id, age_now, placement=placement)
        target_mult = self.get_multiplier(product_id, age_target, placement=placement)
        return float(d1_revenue * max(0.0, target_mult - current_mult))
    
    def get_monthly_roi_multiplier(self, spend_day_offset):
        """
        获取月度ROI倍率（连续对数曲线，与 roi_planner 一致）

        参数：
        - spend_day_offset: 距月末的天数（1表示月末，30表示月初）
        """
        return self._continuous_multiplier(spend_day_offset)
