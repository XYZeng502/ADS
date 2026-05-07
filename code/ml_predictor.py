"""
模块8：基于 LightGBM 的 D1 回收率预测模块
训练粒度 = data_loader 输出粒度: (应用ID, 推广流量, 流量场景, 创意规格, 转化类型)
产品标识 = 应用ID，投放维度作为特征参与建模
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# data_loader 聚合后的投放维度（对应 CSV 列名）
PLACEMENT_DIMS = ['推广流量', '流量场景', '创意规格', '转化类型']


class MLPredictor:
    """LightGBM D1 回收率预测器 — 产品ID=应用ID，投放维度为特征"""

    def __init__(self, data, holiday_identifier=None):
        self.data = data.copy()
        self.data['日期'] = pd.to_datetime(self.data['日期'])
        self.holiday_identifier = holiday_identifier
        self.model = None
        self.quantile_models = {}
        self.feature_names = []
        self.product_medians = {}
        self.unit_medians = {}

        self._compute_product_baselines()

    # ================================================================
    #  特征工程
    # ================================================================

    def _compute_product_baselines(self):
        """按应用ID计算产品基线（跨所有投放维度汇总）"""
        for pid, grp in self.data.groupby('应用ID'):
            d1_series = grp['首日广告收入'] / grp['消耗金额'].replace(0, np.nan)
            d1_series = d1_series.replace([np.inf, -np.inf], np.nan).dropna()
            self.product_medians[pid] = {
                'median_d1': d1_series.median() if len(d1_series) > 0 else 0.65,
                'std_d1': d1_series.std() if len(d1_series) > 1 else 0.05,
                'mean_d1': d1_series.mean() if len(d1_series) > 0 else 0.65,
                'days_active': len(grp['日期'].unique()),
                'total_spend': grp['消耗金额'].sum(),
            }

        available_dims = [d for d in PLACEMENT_DIMS if d in self.data.columns]
        if not available_dims:
            return

        for key, grp in self.data.groupby(['应用ID'] + available_dims, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            d1_series = grp['首日广告收入'] / grp['消耗金额'].replace(0, np.nan)
            d1_series = d1_series.replace([np.inf, -np.inf], np.nan).dropna()
            product_id = key[0]
            prod_fallback = self.product_medians.get(product_id, {})
            median = d1_series.median() if len(d1_series) > 0 else prod_fallback.get('median_d1', 0.65)
            std = d1_series.std() if len(d1_series) > 1 else prod_fallback.get('std_d1', 0.05)
            self.unit_medians[key] = {
                'median_d1': median,
                'p20_d1': d1_series.quantile(0.2) if len(d1_series) >= 5 else max(median - std, 0.01),
                'p80_d1': d1_series.quantile(0.8) if len(d1_series) >= 5 else median + std,
                'std_d1': std,
                'days_active': len(grp['日期'].unique()),
                'total_spend': grp['消耗金额'].sum(),
            }

    def _unit_key_from_row(self, pid, row):
        """统一投放单元key: 应用ID + 推广流量 + 流量场景 + 创意规格 + 转化类型。"""
        return tuple([pid] + [row.get(dim, 0) for dim in PLACEMENT_DIMS])

    def build_training_data(self):
        """构建训练集：每行 = 一个投放单元在某一天的特征 + 实际 D1
        粒度 = (日期, 应用ID, 推广流量, 流量场景, 创意规格, 转化类型)
        """
        rows = []
        df = self.data.sort_values('日期')
        all_dates = sorted(df['日期'].unique())

        for i, current_date in enumerate(all_dates):
            if i < 7:
                continue

            current_dt = pd.to_datetime(current_date)
            day_data = df[df['日期'] == current_dt]

            for _, row in day_data.iterrows():
                pid = row['应用ID']
                spend = row['消耗金额']
                if pd.isna(spend) or spend <= 0:
                    continue

                d1_revenue = row.get('首日广告收入', 0)
                if pd.isna(d1_revenue) or d1_revenue <= 0:
                    continue
                target = d1_revenue / spend
                if target <= 0 or target > 2.0:
                    continue

                features = self._build_single_features(pid, current_dt, row)
                if features is None:
                    continue

                features['target_d1'] = target
                rows.append(features)

        df_out = pd.DataFrame(rows)
        if len(df_out) == 0:
            return None
        return df_out

    def _build_single_features(self, pid, current_dt, row):
        """为单个投放单元构建特征向量

        pid = 应用ID (产品标识)
        row = 当天该投放单元的数据行（含推广流量/流量场景/创意规格/转化类型）
        """
        bl = self.product_medians.get(pid)
        if bl is None:
            return None

        # --- 日期特征 ---
        dow = current_dt.weekday()
        is_weekend = 1 if dow >= 5 else 0

        # 节假日特征
        holiday_scale = 1.0
        holiday_type = 'workday'
        if self.holiday_identifier is not None:
            info = self.holiday_identifier.get_holiday_info(current_dt)
            holiday_scale = info['scale_factor']
            holiday_type = info['type']

        # --- 产品级近期趋势（跨所有投放维度，只按应用ID过滤，防止泄漏） ---
        prod_hist = self.data[
            (self.data['应用ID'] == pid) &
            (self.data['日期'] < current_dt)
            ]
        recent_3d = prod_hist[prod_hist['日期'] >= current_dt - timedelta(days=3)]
        recent_7d = prod_hist[prod_hist['日期'] >= current_dt - timedelta(days=7)]

        def safe_d1(data_subset):
            s = data_subset['消耗金额'].sum()
            if s <= 0:
                return bl['median_d1']
            return data_subset['首日广告收入'].sum() / s

        recent_3d_d1 = safe_d1(recent_3d) if len(recent_3d) >= 1 else bl['median_d1']
        recent_7d_d1 = safe_d1(recent_7d) if len(recent_7d) >= 1 else bl['median_d1']

        recent_3d_spend = recent_3d['消耗金额'].mean() if len(recent_3d) >= 1 else 0
        recent_7d_spend = recent_7d['消耗金额'].mean() if len(recent_7d) >= 1 else 0

        # --- 投放维度特征（从当前行提取） ---
        placement = {}
        for dim in PLACEMENT_DIMS:
            placement[dim] = int(row.get(dim, 0))

        # --- 产品级月内进度 ---
        month_start = pd.to_datetime(f"{current_dt.year}-{current_dt.month:02d}-01")
        month_hist = prod_hist[prod_hist['日期'] >= month_start]
        days_elapsed_in_month = len(month_hist['日期'].unique())

        return {
            # 日期
            'day_of_week': dow,
            'is_weekend': is_weekend,
            'day_of_month': current_dt.day,
            'month': current_dt.month,
            # 节假日
            'holiday_scale': holiday_scale,
            'holiday_type_workday': 1 if holiday_type == 'workday' else 0,
            'holiday_type_weekend': 1 if holiday_type == 'weekend' else 0,
            'holiday_type_short': 1 if holiday_type == 'short_holiday' else 0,
            'holiday_type_long': 1 if holiday_type == 'long_holiday' else 0,
            'holiday_type_vacation': 1 if holiday_type == 'vacation' else 0,
            'holiday_type_ecommerce': 1 if holiday_type == 'ecommerce' else 0,
            # 投放维度
            '推广流量': placement['推广流量'],
            '流量场景': placement['流量场景'],
            '创意规格': placement['创意规格'],
            '转化类型': placement['转化类型'],
            # 产品基线
            'product_median_d1': bl['median_d1'],
            'product_std_d1': bl['std_d1'],
            'product_days_active': bl['days_active'],
            # 近期趋势（产品级）
            'recent_3d_d1': recent_3d_d1,
            'recent_7d_d1': recent_7d_d1,
            'd1_trend_3d': recent_3d_d1 - bl['median_d1'],
            'd1_trend_7d': recent_7d_d1 - bl['median_d1'],
            'recent_3d_spend': recent_3d_spend,
            'recent_7d_spend': recent_7d_spend,
            # 月内进度
            'days_elapsed_in_month': days_elapsed_in_month,
        }

    # ================================================================
    #  模型训练
    # ================================================================

    def train(self):
        """训练 LightGBM 回归模型"""
        if not HAS_LIGHTGBM:
            print("  ⚠️  LightGBM 未安装，将使用产品中位数作为兜底预测")
            print("     安装: pip install lightgbm scikit-learn")
            return False

        print("  - 构建训练特征...")
        df = self.build_training_data()
        if df is None or len(df) < 50:
            print(f"  ⚠️  训练数据不足 (n={len(df) if df is not None else 0})，使用兜底预测")
            return False

        self.feature_names = [c for c in df.columns if c != 'target_d1']
        X = df[self.feature_names]
        y = df['target_d1']

        # 将投放维度列声明为 categorical
        cat_cols = [c for c in PLACEMENT_DIMS if c in self.feature_names]
        for c in cat_cols:
            X[c] = X[c].astype('category')

        split_idx = int(len(df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"  - 训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
        print(f"  - 特征数: {len(self.feature_names)} (含{len(cat_cols)}个投放维度)")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=7,
                num_leaves=63,
                learning_rate=0.05,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_jobs=1,
                random_state=42,
                verbose=-1,
            )
            self.model.fit(X_train, y_train,
                           eval_set=[(X_val, y_val)],
                           eval_metric='mae',
                           categorical_feature=cat_cols)

            self.quantile_models = {}
            for alpha, name in [(0.2, 'p20'), (0.5, 'p50'), (0.8, 'p80')]:
                q_model = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=alpha,
                    n_estimators=180,
                    max_depth=7,
                    num_leaves=63,
                    learning_rate=0.05,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    n_jobs=1,
                    random_state=42 + int(alpha * 100),
                    verbose=-1,
                )
                q_model.fit(X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            eval_metric='quantile',
                            categorical_feature=cat_cols)
                self.quantile_models[name] = q_model

        y_pred = self.model.predict(X_val)
        mae = np.mean(np.abs(y_pred - y_val.values))
        r2 = 1 - np.sum((y_val.values - y_pred) ** 2) / np.sum((y_val.values - y_val.mean()) ** 2)

        baseline_preds = X_val['product_median_d1'].values
        baseline_mae = np.mean(np.abs(baseline_preds - y_val.values))

        print(f"  - LightGBM MAE: {mae:.4f} | 基线(中位数) MAE: {baseline_mae:.4f} | "
              f"提升: {(baseline_mae - mae) / baseline_mae * 100:.1f}% | R²: {r2:.3f}")

        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"  - Top5 特征: {', '.join(importances.head(5)['feature'].values)}")

        return True

    # ================================================================
    #  预测
    # ================================================================

    def _build_prediction_features(self, pid, date, placement_row=None,
                                   recent_context=None):
        """构建预测用特征向量（与训练特征对齐）"""
        if isinstance(date, str):
            date = pd.to_datetime(date)

        bl = self.product_medians.get(pid)
        if bl is None:
            return None

        dow = date.weekday()
        is_weekend = 1 if dow >= 5 else 0

        holiday_scale = 1.0
        holiday_type = 'workday'
        if self.holiday_identifier is not None:
            info = self.holiday_identifier.get_holiday_info(date)
            holiday_scale = info['scale_factor']
            holiday_type = info['type']

        # 近期趋势
        if recent_context:
            r3d = recent_context.get('recent_3d_d1', bl['median_d1'])
            r7d = recent_context.get('recent_7d_d1', bl['median_d1'])
            r3s = recent_context.get('recent_3d_spend', 0)
            r7s = recent_context.get('recent_7d_spend', 0)
        else:
            r3d = bl['median_d1']
            r7d = bl['median_d1']
            r3s = 0
            r7s = 0

        # 投放维度
        placement = {}
        for dim in PLACEMENT_DIMS:
            if placement_row is not None:
                placement[dim] = int(placement_row.get(dim, 0))
            else:
                placement[dim] = 0

        return {
            'day_of_week': dow,
            'is_weekend': is_weekend,
            'day_of_month': date.day,
            'month': date.month,
            'holiday_scale': holiday_scale,
            'holiday_type_workday': 1 if holiday_type == 'workday' else 0,
            'holiday_type_weekend': 1 if holiday_type == 'weekend' else 0,
            'holiday_type_short': 1 if holiday_type == 'short_holiday' else 0,
            'holiday_type_long': 1 if holiday_type == 'long_holiday' else 0,
            'holiday_type_vacation': 1 if holiday_type == 'vacation' else 0,
            'holiday_type_ecommerce': 1 if holiday_type == 'ecommerce' else 0,
            '推广流量': placement['推广流量'],
            '流量场景': placement['流量场景'],
            '创意规格': placement['创意规格'],
            '转化类型': placement['转化类型'],
            'product_median_d1': bl['median_d1'],
            'product_std_d1': bl['std_d1'],
            'product_days_active': bl['days_active'],
            'recent_3d_d1': r3d,
            'recent_7d_d1': r7d,
            'd1_trend_3d': r3d - bl['median_d1'],
            'd1_trend_7d': r7d - bl['median_d1'],
            'recent_3d_spend': r3s,
            'recent_7d_spend': r7s,
            'days_elapsed_in_month': date.day,
        }

    def predict_d1(self, pid, date, placement_row=None, recent_context=None):
        """预测单个投放单元的 D1 回收率"""
        if isinstance(date, str):
            date = pd.to_datetime(date)

        bl = self.product_medians.get(pid)
        if bl is None:
            return 0.65

        features = self._build_prediction_features(pid, date, placement_row, recent_context)
        if features is None:
            return bl['median_d1']

        if self.model is not None and len(self.feature_names) > 0:
            X = pd.DataFrame([features])[self.feature_names]
            for c in PLACEMENT_DIMS:
                if c in X.columns:
                    X[c] = X[c].astype('category')
            pred = float(self.model.predict(X)[0])
            return max(pred, 0.01)
        else:
            holiday_scale = features.get('holiday_scale', 1.0)
            return bl['median_d1'] * (holiday_scale if holiday_scale > 1.0 else 1.0)

    def _fallback_quantiles(self, pid, features, placement_row=None):
        """数据不足或模型不可用时的层级分位数兜底。"""
        bl = self.product_medians.get(pid, {
            'median_d1': 0.65,
            'std_d1': 0.05,
        })

        unit_key = None
        if placement_row is not None:
            unit_key = self._unit_key_from_row(pid, placement_row)

        if unit_key in self.unit_medians:
            unit_bl = self.unit_medians[unit_key]
            p20 = unit_bl.get('p20_d1', unit_bl['median_d1'] - unit_bl.get('std_d1', 0.05))
            p50 = unit_bl.get('median_d1', bl['median_d1'])
            p80 = unit_bl.get('p80_d1', unit_bl['median_d1'] + unit_bl.get('std_d1', 0.05))
        else:
            p50 = bl.get('median_d1', 0.65)
            std = bl.get('std_d1', 0.05)
            p20 = p50 - std
            p80 = p50 + std

        holiday_scale = features.get('holiday_scale', 1.0) if features else 1.0
        scale = holiday_scale if holiday_scale > 1.0 else 1.0
        p20, p50, p80 = p20 * scale, p50 * scale, p80 * scale

        p20 = max(float(p20), 0.01)
        p50 = max(float(p50), p20)
        p80 = max(float(p80), p50)
        return {'p20': p20, 'p50': p50, 'p80': min(p80, 2.0)}

    def predict_d1_quantiles(self, pid, date, placement_row=None, recent_context=None):
        """预测单个投放单元的 D1 分位数，用于鲁棒优化。"""
        if isinstance(date, str):
            date = pd.to_datetime(date)

        bl = self.product_medians.get(pid)
        if bl is None:
            return {'p20': 0.55, 'p50': 0.65, 'p80': 0.75}

        features = self._build_prediction_features(pid, date, placement_row, recent_context)
        if features is None:
            return self._fallback_quantiles(pid, {}, placement_row)

        if self.quantile_models and len(self.feature_names) > 0:
            X = pd.DataFrame([features])[self.feature_names]
            for c in PLACEMENT_DIMS:
                if c in X.columns:
                    X[c] = X[c].astype('category')

            preds = {}
            for name in ['p20', 'p50', 'p80']:
                if name in self.quantile_models:
                    preds[name] = float(self.quantile_models[name].predict(X)[0])

            if len(preds) == 3:
                p20 = max(preds['p20'], 0.01)
                p50 = max(preds['p50'], p20)
                p80 = max(preds['p80'], p50)
                return {'p20': p20, 'p50': p50, 'p80': min(p80, 2.0)}

        return self._fallback_quantiles(pid, features, placement_row)

    def _same_unit_mask(self, data, pid, placement_row):
        mask = data['应用ID'] == pid
        if placement_row is None:
            return mask
        for dim in PLACEMENT_DIMS:
            if dim in data.columns and dim in placement_row.index:
                mask &= data[dim] == placement_row.get(dim)
        return mask

    def _estimate_unit_cap(self, pid, placement_row, date, historical_data, holiday_scale):
        """估计投放单元在某天的可投上限，作为边际优化的物理约束。"""
        unit_hist = historical_data[self._same_unit_mask(historical_data, pid, placement_row)]
        if len(unit_hist) == 0:
            prod_hist = historical_data[historical_data['应用ID'] == pid]
            prod_daily = prod_hist.groupby('日期')['消耗金额'].sum()
            fallback = prod_daily.mean() / max(len(self.unit_medians), 1) if len(prod_daily) else 100
            return max(float(fallback), 100.0)

        daily_spend = unit_hist.groupby('日期')['消耗金额'].sum().sort_index()
        p90 = daily_spend.quantile(0.90) if len(daily_spend) >= 3 else daily_spend.max()
        recent_7 = daily_spend.tail(7).mean() if len(daily_spend) > 0 else 0
        last_spend = daily_spend.iloc[-1] if len(daily_spend) > 0 else 0
        active_avg = daily_spend.mean() if len(daily_spend) > 0 else 0

        base = max(p90, recent_7 * 1.4, last_spend * 1.25, active_avg, 100.0)
        scale = min(max(float(holiday_scale), 1.0), 1.8)
        return float(base * scale)

    def predict_remaining_days_quantiles(self, product_ids, start_date, end_date,
                                         historical_data, holiday_info_df=None):
        """
        预测剩余日期内每个投放单元的 D1 分位数和cap。

        返回粒度 = 应用ID + 推广流量 + 流量场景 + 创意规格 + 转化类型 + 日期。
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        hist = historical_data.copy()
        hist['日期'] = pd.to_datetime(hist['日期'])

        holiday_map = {}
        if holiday_info_df is not None and len(holiday_info_df) > 0:
            for _, row in holiday_info_df.iterrows():
                holiday_map[pd.to_datetime(row['date']).normalize()] = float(row['scale_factor'])

        recent_contexts = {}
        for pid in product_ids:
            prod_hist = hist[hist['应用ID'] == pid]
            recent_3d = prod_hist[prod_hist['日期'] >= start_dt - timedelta(days=3)]
            recent_7d = prod_hist[prod_hist['日期'] >= start_dt - timedelta(days=7)]

            def safe_d1(subset):
                s = subset['消耗金额'].sum()
                return subset['首日广告收入'].sum() / s if s > 0 else self.product_medians.get(pid, {}).get('median_d1', 0.65)

            recent_contexts[pid] = {
                'recent_3d_d1': safe_d1(recent_3d) if len(recent_3d) > 0 else self.product_medians.get(pid, {}).get('median_d1', 0.65),
                'recent_7d_d1': safe_d1(recent_7d) if len(recent_7d) > 0 else self.product_medians.get(pid, {}).get('median_d1', 0.65),
                'recent_3d_spend': recent_3d['消耗金额'].mean() if len(recent_3d) > 0 else 0,
                'recent_7d_spend': recent_7d['消耗金额'].mean() if len(recent_7d) > 0 else 0,
            }

        results = []
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        dim_cols = [d for d in PLACEMENT_DIMS if d in hist.columns]

        for pid in product_ids:
            ctx = recent_contexts.get(pid, {})
            pdata = hist[hist['应用ID'] == pid]
            if len(pdata) == 0:
                continue

            if dim_cols:
                unique_units = pdata.drop_duplicates(subset=dim_cols)
            else:
                unique_units = pdata.tail(1)

            product_total_spend = pdata['消耗金额'].sum()

            for _, unit_row in unique_units.iterrows():
                unit_mask = self._same_unit_mask(pdata, pid, unit_row)
                unit_total_spend = pdata[unit_mask]['消耗金额'].sum()
                historical_weight = unit_total_spend / product_total_spend if product_total_spend > 0 else 0

                for date in date_range:
                    normalized = pd.to_datetime(date).normalize()
                    if normalized in holiday_map:
                        holiday_scale = holiday_map[normalized]
                    elif self.holiday_identifier is not None:
                        holiday_scale = self.holiday_identifier.get_holiday_info(date)['scale_factor']
                    else:
                        holiday_scale = 1.0

                    q = self.predict_d1_quantiles(pid, date, placement_row=unit_row, recent_context=ctx)
                    cap = self._estimate_unit_cap(pid, unit_row, date, hist, holiday_scale)

                    row = {
                        'product_id': pid,
                        'date': pd.to_datetime(date),
                        'd1_p20': q['p20'],
                        'd1_p50': q['p50'],
                        'd1_p80': q['p80'],
                        'cap': cap,
                        'historical_weight': historical_weight,
                        'holiday_scale': holiday_scale,
                    }
                    for dim in PLACEMENT_DIMS:
                        row[dim] = unit_row.get(dim, 0) if dim in unit_row.index else 0
                    results.append(row)

        return pd.DataFrame(results)

    def predict_remaining_days(self, product_ids, start_date, end_date,
                               historical_data, holiday_info_df):
        """预测每产品每投放单元在剩余日期的 D1 → 再按历史消耗加权聚合到产品

        返回 DataFrame 列: product_id, date, predicted_d1
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 各产品近期趋势（产品级，跨投放维度）
        recent_contexts = {}
        for pid in product_ids:
            prod_hist = historical_data[historical_data['应用ID'] == pid]
            recent_3d = prod_hist[prod_hist['日期'] >= start_dt - timedelta(days=3)]
            recent_7d = prod_hist[prod_hist['日期'] >= start_dt - timedelta(days=7)]

            def safe_d1(subset):
                s = subset['消耗金额'].sum()
                return subset['首日广告收入'].sum() / s if s > 0 else self.product_medians.get(pid, {}).get('median_d1', 0.65)

            recent_contexts[pid] = {
                'recent_3d_d1': safe_d1(recent_3d) if len(recent_3d) > 0 else self.product_medians.get(pid, {}).get('median_d1', 0.65),
                'recent_7d_d1': safe_d1(recent_7d) if len(recent_7d) > 0 else self.product_medians.get(pid, {}).get('median_d1', 0.65),
                'recent_3d_spend': recent_3d['消耗金额'].mean() if len(recent_3d) > 0 else 0,
                'recent_7d_spend': recent_7d['消耗金额'].mean() if len(recent_7d) > 0 else 0,
            }

        # 各产品内，每个投放单元的历史消耗权重
        unit_weights = {}
        for pid in product_ids:
            pdata = historical_data[historical_data['应用ID'] == pid]
            if len(pdata) == 0:
                unit_weights[pid] = {}
                continue
            # 按投放维度分组，计算历史总消耗作为权重
            group_keys = [d for d in PLACEMENT_DIMS if d in pdata.columns]
            if group_keys:
                w = pdata.groupby(group_keys)['消耗金额'].sum()
                w = w / w.sum()  # 归一化
                unit_weights[pid] = w.to_dict()
            else:
                unit_weights[pid] = {}

        results = []
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

        for pid in product_ids:
            ctx = recent_contexts.get(pid, {})
            weights = unit_weights.get(pid, {})
            unique_units = historical_data[historical_data['应用ID'] == pid].drop_duplicates(
                subset=[d for d in PLACEMENT_DIMS if d in historical_data.columns]
            )

            for date in date_range:
                if len(unique_units) > 0 and len(weights) > 0:
                    # 逐投放单元预测，加权聚合
                    weighted_d1 = 0.0
                    total_w = 0.0
                    for _, unit_row in unique_units.iterrows():
                        key = tuple(unit_row.get(d, 0) for d in PLACEMENT_DIMS if d in unit_row.index)
                        # w.to_dict() 多key分组返回 {tuple: weight}，单key返回 {scalar: weight}
                        if len(key) == 1:
                            w = weights.get(key[0], 0)
                        else:
                            w = weights.get(key, 0)

                        if w <= 0:
                            continue
                        pred = self.predict_d1(pid, date, placement_row=unit_row, recent_context=ctx)
                        weighted_d1 += pred * w
                        total_w += w

                    if total_w > 0:
                        final_d1 = weighted_d1 / total_w
                    else:
                        final_d1 = self.predict_d1(pid, date, placement_row=None, recent_context=ctx)
                else:
                    final_d1 = self.predict_d1(pid, date, placement_row=None, recent_context=ctx)

                results.append({
                    'product_id': pid,
                    'date': date,
                    'predicted_d1': final_d1,
                })

        return pd.DataFrame(results)

    def get_average_d1_for_period(self, product_ids, start_date, end_date,
                                  historical_data, holiday_info_df):
        """获取剩余期间每产品的平均预测 D1（供 roi_planner 使用）"""
        preds = self.predict_remaining_days_quantiles(
            product_ids, start_date, end_date, historical_data, holiday_info_df
        )
        if preds.empty:
            return {pid: self.product_medians.get(pid, {}).get('median_d1', 0.65)
                    for pid in product_ids}

        weighted = {}
        for pid, grp in preds.groupby('product_id'):
            weights = grp['historical_weight'].replace(0, np.nan)
            if weights.notna().any():
                weighted[pid] = np.average(grp['d1_p50'], weights=weights.fillna(0))
            else:
                weighted[pid] = grp['d1_p50'].mean()
        return weighted
