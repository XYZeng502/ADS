"""
模块3：节假日识别与标注模块
根据文档，不同时间点的scale因子不同：
- 普通工作日：1.0
- 周末：1.2-1.3（用户活跃，变现好）
- 小长假（五一/端午/清明）：1.3-1.5（流量持续多天）
- 暑假/寒假：1.4-1.6（持续时间长）
- 电商节（618/双11/双12）：1.5-1.8（广告主多，ECPM高）
"""

import pandas as pd
from datetime import datetime, timedelta


class HolidayIdentifier:
    """节假日识别和scale因子标注类"""
    
    def __init__(self, year=2026):
        """
        初始化节假日识别器
        
        参数：
        - year: 当前年份
        """
        self.year = year
        self.holidays = self._build_holiday_calendar(year)
    
    def _build_holiday_calendar(self, year):
        """构建节假日日历"""
        holidays = {}
        
        # 春节（假设为2月）
        holidays[datetime(year, 2, 10)] = {'name': '春节', 'scale_factor': 1.6, 'type': 'long_holiday'}
        holidays[datetime(year, 2, 11)] = {'name': '春节', 'scale_factor': 1.6, 'type': 'long_holiday'}
        holidays[datetime(year, 2, 12)] = {'name': '春节', 'scale_factor': 1.6, 'type': 'long_holiday'}
        
        # 清明节（4月）
        holidays[datetime(year, 4, 4)] = {'name': '清明节', 'scale_factor': 1.4, 'type': 'short_holiday'}
        holidays[datetime(year, 4, 5)] = {'name': '清明节', 'scale_factor': 1.4, 'type': 'short_holiday'}
        holidays[datetime(year, 4, 6)] = {'name': '清明节', 'scale_factor': 1.4, 'type': 'short_holiday'}
        
        # 五一劳动节（5月）
        holidays[datetime(year, 5, 1)] = {'name': '五一劳动节', 'scale_factor': 1.5, 'type': 'long_holiday'}
        holidays[datetime(year, 5, 2)] = {'name': '五一劳动节', 'scale_factor': 1.5, 'type': 'long_holiday'}
        holidays[datetime(year, 5, 3)] = {'name': '五一劳动节', 'scale_factor': 1.5, 'type': 'long_holiday'}
        holidays[datetime(year, 5, 4)] = {'name': '五一劳动节', 'scale_factor': 1.5, 'type': 'long_holiday'}
        holidays[datetime(year, 5, 5)] = {'name': '五一劳动节', 'scale_factor': 1.5, 'type': 'long_holiday'}
        
        # 端午节（6月）
        holidays[datetime(year, 6, 10)] = {'name': '端午节', 'scale_factor': 1.4, 'type': 'short_holiday'}
        holidays[datetime(year, 6, 11)] = {'name': '端午节', 'scale_factor': 1.4, 'type': 'short_holiday'}
        holidays[datetime(year, 6, 12)] = {'name': '端午节', 'scale_factor': 1.4, 'type': 'short_holiday'}
        
        # 暑假（7-8月）
        for day in range(1, 32):
            if day <= 31:
                holidays[datetime(year, 7, day)] = {'name': '暑假', 'scale_factor': 1.5, 'type': 'vacation'}
        for day in range(1, 32):
            if day <= 31:
                holidays[datetime(year, 8, day)] = {'name': '暑假', 'scale_factor': 1.5, 'type': 'vacation'}
        
        # 中秋节（9月）
        holidays[datetime(year, 9, 15)] = {'name': '中秋节', 'scale_factor': 1.4, 'type': 'short_holiday'}
        holidays[datetime(year, 9, 16)] = {'name': '中秋节', 'scale_factor': 1.4, 'type': 'short_holiday'}
        holidays[datetime(year, 9, 17)] = {'name': '中秋节', 'scale_factor': 1.4, 'type': 'short_holiday'}
        
        # 国庆节（10月）
        for day in range(1, 8):
            holidays[datetime(year, 10, day)] = {'name': '国庆节', 'scale_factor': 1.6, 'type': 'long_holiday'}
        
        # 电商节 - 618（6月）
        for day in range(15, 25):
            holidays[datetime(year, 6, day)] = {'name': '618购物节', 'scale_factor': 1.7, 'type': 'ecommerce'}
        
        # 电商节 - 双11（11月）
        for day in range(8, 15):
            holidays[datetime(year, 11, day)] = {'name': '双11购物节', 'scale_factor': 1.8, 'type': 'ecommerce'}
        
        # 电商节 - 双12（12月）
        for day in range(9, 16):
            holidays[datetime(year, 12, day)] = {'name': '双12购物节', 'scale_factor': 1.7, 'type': 'ecommerce'}
        
        return holidays
    
    def get_holiday_info(self, date):
        """
        获取指定日期的节假日信息
        
        参数：
        - date: datetime对象
        
        返回：{'name': '...', 'scale_factor': ..., 'type': '...'}
               如果不是节假日，返回普通日期信息
        """
        date = pd.to_datetime(date)
        
        # 检查是否在节假日日历中
        if date in self.holidays:
            return self.holidays[date]
        
        # 检查是否是周末
        weekday = date.weekday()  # 0=Monday, 6=Sunday
        if weekday >= 5:  # Saturday=5, Sunday=6
            return {'name': '周末', 'scale_factor': 1.25, 'type': 'weekend'}
        
        # 普通工作日
        return {'name': '普通工作日', 'scale_factor': 1.0, 'type': 'workday'}
    
    def get_holiday_scale_factors(self, start_date, end_date):
        """
        获取一个时间范围内各天的scale因子
        
        参数：
        - start_date: 开始日期（包含）
        - end_date: 结束日期（包含）
        
        返回：{'日期': [dates], 'scale_factor': [factors], 'holiday_type': [types]}
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        factors = {
            'date': [],
            'scale_factor': [],
            'holiday_type': [],
            'holiday_name': []
        }
        
        for date in dates:
            info = self.get_holiday_info(date)
            factors['date'].append(date)
            factors['scale_factor'].append(info['scale_factor'])
            factors['holiday_type'].append(info['type'])
            factors['holiday_name'].append(info['name'])
        
        return pd.DataFrame(factors)
    
    def identify_peak_windows(self, start_date, end_date):
        """
        识别高价值投放窗口（scale_factor > 1.3的期间）
        
        参数：
        - start_date: 开始日期
        - end_date: 结束日期
        
        返回：高价值窗口列表
        """
        scale_factors = self.get_holiday_scale_factors(start_date, end_date)
        peak_windows = []
        
        # 按scale_factor排序
        high_value = scale_factors[scale_factors['scale_factor'] > 1.3]
        
        if len(high_value) > 0:
            # 分组连续的日期
            high_value = high_value.reset_index(drop=True)
            high_value['group'] = (
                (high_value['date'] - high_value['date'].shift()).dt.days != 1
            ).cumsum()
            
            for group_id in high_value['group'].unique():
                group_data = high_value[high_value['group'] == group_id]
                window = {
                    'start_date': group_data['date'].min(),
                    'end_date': group_data['date'].max(),
                    'duration_days': len(group_data),
                    'avg_scale_factor': group_data['scale_factor'].mean(),
                    'holiday_name': group_data['holiday_name'].iloc[0],
                    'holiday_type': group_data['holiday_type'].iloc[0]
                }
                peak_windows.append(window)
        
        return peak_windows
    
    def get_monthly_scale_factors(self, year, month):
        """
        获取某月的scale因子统计
        
        返回：月度加权平均scale因子、各类型的日数统计
        """
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        factors = self.get_holiday_scale_factors(start_date, end_date)
        
        # 计算加权平均
        weighted_avg = (factors['scale_factor'] * 1).mean()
        
        # 统计各类型的日数
        type_counts = factors['holiday_type'].value_counts().to_dict()
        
        return {
            'weighted_avg_scale_factor': weighted_avg,
            'type_distribution': type_counts,
            'details': factors
        }
