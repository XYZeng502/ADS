"""
模块1：数据加载与预处理模块
负责从CSV加载数据、数据清洗、数据验证
"""

import pandas as pd
import numpy as np
from datetime import datetime


class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, csv_path):
        """初始化数据加载器"""
        self.csv_path = csv_path
        self.data = None
    
    def load_data(self):
        """
        从CSV加载数据

        返回：清洗+聚合后的DataFrame
        """
        print(f"  - 从 {self.csv_path} 加载数据...")

        # 读取CSV
        data = pd.read_csv(self.csv_path)

        # 数据清洗
        data = self._clean_data(data)

        # 数据聚合：同应用ID下，推广流量/流量场景/创意规格/转化类型相同则合并
        data = self._aggregate_data(data)

        # 数据验证
        self._validate_data(data)

        self.data = data
        return data
    
    def _clean_data(self, data):
        """数据清洗"""
        print("  - 执行数据清洗...")
        
        # 转换日期列为datetime
        data['日期'] = pd.to_datetime(data['日期'])
        
        # 处理缺失值
        numeric_columns = [
            '消耗金额', '首日广告收入', '3日累计变现金额', 
            '7日累计变现金额', '30日累计变现金额', '买量广告收入',
            '曝光量', '点击量', '下载量', '激活人数(快应用新增用户数)', '注册人数'
        ]
        
        for col in numeric_columns:
            if col in data.columns:
                # 用0填充缺失值
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # 移除全为0的行（无效数据）
        data = data[data[numeric_columns].sum(axis=1) > 0]
        
        # 移除重复数据（基于关键字段）
        key_columns = ['日期', '应用ID', '广告主ID', '计划ID', '广告组ID', '创意ID']
        data = data.drop_duplicates(subset=key_columns, keep='first')
        
        # 按日期排序
        data = data.sort_values('日期')
        
        print(f"  - 清洗完成，保留{len(data)}条有效记录")

        return data

    def _aggregate_data(self, data):
        """
        数据聚合：同一应用ID（产品）下，
        推广流量、流量场景、创意规格、转化类型 四字段相同 → 合并为一条

        合并规则：
        - 数值列：求和
        - 描述列：取第一条
        - 计划ID/广告组ID/创意ID：丢弃（粒度太细）
        """
        print("  - 执行数据聚合...")

        # 分组键
        group_keys = ['日期', '应用ID', '推广流量', '流量场景', '创意规格', '转化类型']

        # 确保分组键存在
        missing_keys = [k for k in group_keys if k not in data.columns]
        if missing_keys:
            print(f"  ⚠️  缺少聚合字段: {missing_keys}，跳过聚合")
            return data

        n_before = len(data)

        # 数值列 → sum
        numeric_cols_to_sum = [
            '消耗金额', '首日广告收入', '3日累计变现金额',
            '7日累计变现金额', '30日累计变现金额', '买量广告收入',
            '曝光量', '点击量', '下载量', '激活人数(快应用新增用户数)', '注册人数'
        ]
        numeric_cols = [c for c in numeric_cols_to_sum if c in data.columns]

        agg_dict = {col: 'sum' for col in numeric_cols}

        # 描述列 → first（聚合后保持语义）
        description_cols = [
            '广告主ID', '推广流量名称', '流量场景名称',
            '创意规格名称', '计费方式', '转化类型名称',
            '深度转化类型', '深度转化类型名称'
        ]
        for col in description_cols:
            if col in data.columns:
                agg_dict[col] = 'first'

        # 执行聚合
        data = data.groupby(group_keys, as_index=False).agg(agg_dict)

        n_after = len(data)
        unique_apps_after = data['应用ID'].nunique()

        # 按日期排序
        data = data.sort_values('日期')

        print(f"  - 聚合完成: {n_before}条 → {n_after}条 "
              f"({n_before - n_after}条被合并), "
              f"产品数: {unique_apps_after}个")

        return data

    def _validate_data(self, data):
        """数据验证"""
        print("  - 执行数据验证...")
        
        # 检查关键列
        required_columns = ['日期', '消耗金额', '首日广告收入', '30日累计变现金额']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"缺失必需列: {missing_cols}")
        
        # 检查数据范围
        if (data['消耗金额'] < 0).any():
            print("  ⚠️  警告: 发现负的消耗金额，将转换为0")
            data.loc[data['消耗金额'] < 0, '消耗金额'] = 0
        
        if (data['首日广告收入'] < 0).any():
            print("  ⚠️  警告: 发现负的收入，将转换为0")
            data.loc[data['首日广告收入'] < 0, '首日广告收入'] = 0
        
        # 检查日期范围
        min_date = data['日期'].min()
        max_date = data['日期'].max()
        print(f"  - 数据范围: {min_date.date()} 至 {max_date.date()}")
        
        # 检查产品数量
        # 注：产品ID = 应用ID，同一应用在不同渠道/场景/创意会有多条数据
        num_apps = data['应用ID'].nunique()
        num_placement_dims = data.groupby('应用ID')[['推广流量', '流量场景', '创意规格', '转化类型']].nunique().sum().sum()
        print(f"  - 覆盖 {num_apps} 个产品(应用)，涉及 {num_placement_dims} 种投放维度组合")
    
    def get_aggregated_data(self, level='day'):
        """
        获取聚合数据
        
        参数：
        - level: 'day'(日) / 'product'(产品) / 'app'(应用)
        
        返回：聚合后的DataFrame
        """
        if self.data is None:
            return None
        
        if level == 'day':
            return self.data.groupby('日期').agg({
                '消耗金额': 'sum',
                '首日广告收入': 'sum',
                '3日累计变现金额': 'sum',
                '7日累计变现金额': 'sum',
                '30日累计变现金额': 'sum',
                '应用ID': 'nunique'
            }).reset_index().rename(columns={'应用ID': '产品数'})
        
        elif level == 'product':
            # 按应用ID(产品ID)聚合，同一产品可能有多个投放维度
            return self.data.groupby('应用ID').agg({
                '消耗金额': 'sum',
                '首日广告收入': 'sum',
                '3日累计变现金额': 'sum',
                '7日累计变现金额': 'sum',
                '30日累计变现金额': 'sum',
                '日期': 'count'
            }).reset_index().rename(columns={'日期': '数据行数'})
        
        elif level == 'app':
            # 与'product'相同，应用ID就是产品ID
            return self.data.groupby('应用ID').agg({
                '消耗金额': 'sum',
                '首日广告收入': 'sum',
                '30日累计变现金额': 'sum',
                '推广流量': 'nunique',
                '流量场景': 'nunique',
                '创意规格': 'nunique',
                '转化类型': 'nunique'
            }).reset_index().rename(columns={
                '推广流量': '推广流量数', '流量场景': '流量场景数',
                '创意规格': '创意规格数', '转化类型': '转化类型数'
            })
        
        else:
            raise ValueError(f"不支持的聚合级别: {level}")

    def get_unit_level_data(self, current_date=None):
        """
        获取投放单元级数据，并标注cohort成熟度。

        粒度：日期 + 应用ID + 推广流量 + 流量场景 + 创意规格 + 转化类型。
        """
        if self.data is None:
            return None

        data = self.data.copy()
        current_dt = pd.to_datetime(current_date) if current_date is not None else data['日期'].max()
        data['age_days'] = (current_dt - data['日期']).dt.days + 1
        data['has_d3_label'] = data['age_days'] >= 3
        data['has_d7_label'] = data['age_days'] >= 7
        data['has_d30_label'] = data['age_days'] >= 30
        data['unit_id'] = data[
            ['应用ID', '推广流量', '流量场景', '创意规格', '转化类型']
        ].astype(str).agg('|'.join, axis=1)
        return data
    
    def get_date_range(self):
        """获取数据的日期范围"""
        if self.data is None:
            return None
        return self.data['日期'].min(), self.data['日期'].max()
